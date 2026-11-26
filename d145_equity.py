# -*- coding: utf-8 -*-
"""
d145 最终审判：日频资金曲线（现金 2.5% 生息，复利），不用线性加权
对比：
  H0 一直空仓生息
  H1 买入持有
  H2 骨架：离底<=10% 一进带就满仓，持 250 日
  H3 骨架 + 当月收阴才买（月线级别确认）
  H4 骨架 + 连续 2 月收阴才买
  H5 骨架 + 从 250 日高点跌 15% 才买
  H6 H2 + 「前一年涨幅>40% 否决」
并做去重敏感性（冷却 180 / 250 日）
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load

pd.set_option('display.width', 250)
RF = 0.025
DAILY_RF = (1 + RF) ** (1 / 243) - 1


def equity(d, signals, hold=250):
    """signals: 买入日索引列表。规则：空仓时遇信号则满仓，持满 hold 卖出；不重叠"""
    c = d['close'].values
    n = len(c)
    nav = np.ones(n)
    cash = 1.0
    sh = 0.0
    entry = -1
    sig = set(signals)
    for i in range(n):
        if sh > 0:
            nav[i] = sh * c[i]
            if i - entry >= hold:
                cash = sh * c[i]
                sh = 0.0
        else:
            nav[i] = cash
            if i in sig and i > 0:
                sh = cash / c[i]
                cash = 0.0
                entry = i
        if sh == 0 and i < n - 1:
            cash *= (1 + DAILY_RF)
    return nav


def stats(nav, yrs):
    cagr = nav[-1] ** (1 / yrs) - 1
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, dd


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    ra = d['rA'].values
    pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])
    dd250 = np.array([c[i] / c[max(0, i - 250):i + 1].max() - 1 for i in range(n)])

    # 月线状态：当月是否为阴线（用上一个已收盘月的涨跌，避免未来函数）
    per = d['d'].dt.to_period('M')
    gb = d.groupby(per)
    last_i = gb.apply(lambda x: int(x.index[-1]), include_groups=False).tolist()
    mv = c[last_i]
    mon_ym = [str(p) for p in gb.groups.keys()]
    pmap = {str(p): k for k, p in enumerate(gb.groups.keys())}
    # 每日 -> 上一个完整月的阴线计数
    prev_down = np.zeros(n, dtype=int)      # 紧邻的上一个完整月是否收阴（1/0）
    consec_down = np.zeros(n, dtype=int)    # 连续阴线月数
    for i in range(n):
        k = pmap[str(per.iloc[i])]
        if k == 0:
            continue
        cnt = 0
        t = k - 1
        while t >= 1 and mv[t] < mv[t - 1]:
            cnt += 1
            t -= 1
        consec_down[i] = cnt
        prev_down[i] = 1 if cnt >= 1 else 0

    def trig(cond, cool):
        ev, last = [], -999
        for i in range(300, n):
            if i - last <= cool:
                continue
            if cond(i):
                ev.append(i)
                last = i
        return ev

    base = lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10
    S = {
        'H2 骨架：进带即买': trig(base, 180),
        'H3 + 上月收阴': trig(lambda i: base(i) and prev_down[i] == 1, 180),
        'H4 + 连续2月收阴': trig(lambda i: base(i) and consec_down[i] >= 2, 180),
        'H5 + 跌15%': trig(lambda i: base(i) and dd250[i] <= -0.15, 180),
        'H6 H2 + 否决(前一年>40%)': trig(lambda i: base(i) and pre[i] <= 0.40, 180),
        'H7 H3 + 否决(前一年>40%)': trig(lambda i: base(i) and prev_down[i] == 1 and pre[i] <= 0.40, 180),
    }

    print('=' * 122)
    print(f'样本 {d.d.iloc[0].date()}~{d.d.iloc[-1].date()}   {yrs:.1f} 年   '
          f'现金年化 {RF*100:.1f}%（日复利）   持有期 250 日')
    print('=' * 122)
    print(f'{"策略":<28} {"触发":>4} {"年化":>8} {"最大回撤":>9} {"期末净值":>9} {"在场占比":>8} '
          f'{"Calmar":>7}')

    nav0 = np.ones(n) * (1 + DAILY_RF) ** np.arange(n)
    g0, d0 = stats(nav0, yrs)
    print(f'{"H0 一直空仓生息":<28} {0:4d} {g0*100:+7.2f}% {d0*100:+8.1f}% {nav0[-1]:9.2f} '
          f'{100:7.0f}% {"—":>7}')
    nav1 = c / c[0]
    g1, d1 = stats(nav1, yrs)
    print(f'{"H1 买入持有":<28} {1:4d} {g1*100:+7.2f}% {d1*100:+8.1f}% {nav1[-1]:9.2f} '
          f'{100:7.0f}% {abs(g1/d1):7.2f}')

    res = {}
    for nm, sig in S.items():
        nav = equity(d, sig)
        g, dd = stats(nav, yrs)
        occ = sum(min(250, n - 1 - i) for i in sig) / n
        calmar = abs(g / dd) if dd < 0 else np.nan
        res[nm] = (g, dd, nav[-1], occ)
        print(f'{nm:<28} {len(sig):4d} {g*100:+7.2f}% {dd*100:+8.1f}% {nav[-1]:9.2f} '
              f'{occ*100:7.0f}% {calmar:7.2f}')

    # ---------- 去重敏感性 ----------
    print()
    print('=' * 122)
    print('去重敏感性：把冷却期改成 250 / 500 日，看排序是否稳定')
    print('=' * 122)
    print(f'{"策略":<28} ' + ' '.join([f'{"冷"+str(cc):>10}' for cc in (120, 180, 250, 500)]))
    for nm, fn in [('骨架 进带即买', lambda i: base(i)),
                   ('骨架 + 上月收阴', lambda i: base(i) and prev_down[i] == 1),
                   ('骨架 + 连续2月收阴', lambda i: base(i) and consec_down[i] >= 2),
                   ('骨架 + 跌15%', lambda i: base(i) and dd250[i] <= -0.15)]:
        row = []
        for cc in (120, 180, 250, 500):
            sig = trig(fn, cc)
            nav = equity(d, sig)
            g, _ = stats(nav, yrs)
            row.append(f'{len(sig):3d}/{g*100:+5.1f}%')
        print(f'{nm:<28} ' + ' '.join([f'{x:>10}' for x in row]))
    print('（格式：触发次数/年化）')

    # ---------- 平均浮亏 ----------
    print()
    print('=' * 122)
    print('平均过程浮亏（决定「能不能拿住」）')
    print('=' * 122)
    for nm, sig in S.items():
        lows = [min(c[i:min(n, i + 251)]) / c[i] - 1 for i in sig]
        print(f'{nm:<28} 触发 {len(sig):2d} 次   平均最深浮亏 {np.mean(lows)*100:+6.1f}%   '
              f'中位 {np.median(lows)*100:+6.1f}%   最差 {np.min(lows)*100:+6.1f}%')

    # 导出
    out = {nm: dict(g=float(v[0]), dd=float(v[1]), nav=float(v[2]), occ=float(v[3]))
           for nm, v in res.items()}
    out['H0'] = dict(g=float(g0), dd=float(d0), nav=float(nav0[-1]), occ=1.0)
    out['H1'] = dict(g=float(g1), dd=float(d1), nav=float(nav1[-1]), occ=1.0)
    import json
    json.dump(dict(out=out, sig={k: [str(d.d.iloc[i].date()) for i in v] for k, v in S.items()}),
              open('d145_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n>> d145_result.json')


if __name__ == '__main__':
    main()
