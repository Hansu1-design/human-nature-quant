# -*- coding: utf-8 -*-
"""
d155 卖出扳机对决（严格对齐 v1.5 基准）
基准：d145 H2 骨架 = 5.53% / -43.3% / 11 次触发 / 在场 38%
做法：完全沿用 d145 的 equity()（空仓遇信号满仓，持满即卖，卖出后等下一个信号），
      只在持仓中加一个"卖出钩子"。其余一切不动，确保可与 5.53% 严格比较。

要回答的：先生说的"强转弱就卖"，作为一个纯卖出信号，比"持满 250 日"好还是差？
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load

pd.set_option('display.width', 250)
RF = 0.025
DAILY_RF = (1 + RF) ** (1 / 252) - 1
EPS = 1e-12


def equity(d, signals, hold=250, exit_fn=None, cost=0.0):
    """signals: 买入日索引集合。空仓遇信号满仓；持仓中 exit_fn(i, entry, peak) 为真则卖。"""
    c = d['close'].values
    n = len(c)
    nav = np.ones(n)
    cash, sh, entry = 1.0, 0.0, -1
    peak = 0.0
    sig = set(signals)
    trades = []
    for i in range(n):
        if sh > 0:
            peak = max(peak, c[i])
            nav[i] = sh * c[i]
            if i - entry >= hold or (exit_fn is not None and exit_fn(i, entry, peak)):
                cash = sh * c[i] * (1 - cost)
                sh = 0.0
                trades.append(('卖', i, c[i]))
        else:
            nav[i] = cash
            if i in sig and i > 0:
                sh = cash * (1 - cost) / c[i]
                cash = 0.0
                entry = i
                peak = c[i]
                trades.append(('买', i, c[i]))
        if sh == 0 and i < n - 1:
            cash *= (1 + DAILY_RF)
    return nav, trades


def stats(nav, yrs):
    return nav[-1] ** (1 / yrs) - 1, (nav / np.maximum.accumulate(nav) - 1).min()


def build(d):
    c = d['close'].values
    n = len(c)
    ra = d['rA'].values

    # 月线：上一个完整月是否收阴
    per = d['d'].dt.to_period('M')
    gb = d.groupby(per)
    last_i = gb.apply(lambda x: int(x.index[-1]), include_groups=False).tolist()
    mv = c[last_i]
    pmap = {str(p): k for k, p in enumerate(gb.groups.keys())}
    prev_down = np.zeros(n, dtype=int)
    for i in range(n):
        k = pmap[str(per.iloc[i])]
        if k >= 2:
            prev_down[i] = 1 if mv[k - 1] < mv[k - 2] else 0

    # 周线：先生原话的 1234 判据（第4周 < 第2周，第3周 < 第2周）
    wy = d['d'].dt.strftime('%G-%V')
    wu = list(dict.fromkeys(wy.tolist()))          # 按时间顺序的唯一周
    wmap = {k: j for j, k in enumerate(wu)}
    wlast = [int(np.where(wy.values == k)[0][-1]) for k in wu]
    wc = c[wlast]
    # ⚠ 只在周末判断（否则周内用到了当周尚未产生的收盘价 = 未来函数）
    wave_sig = np.zeros(n, dtype=bool)
    wset = set(wlast)
    for i in wlast:
        k = wmap[wy.iloc[i]]
        if k >= 3:
            c1, c2, c3, c4 = wc[k - 3], wc[k - 2], wc[k - 1], wc[k]
            wave_sig[i] = (c4 < c2) and (c3 < c2)

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
    return dict(sig=trig(base, 180), ra=ra, prev_down=prev_down,
                wave_sig=wave_sig, c=c, n=n)


def main():
    d = load()
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    B = build(d)
    sig, c = B['sig'], B['c']
    n = B['n']
    print(f'样本 {d.d.iloc[0].date()}~{d.d.iloc[-1].date()}  {yrs:.1f} 年  '
          f'骨架信号 {len(sig)} 次  现金 {RF*100:.1f}%/年')

    rows = []

    def add(name, hold=250, exit_fn=None, cost=0.0, note=''):
        nav, tr = equity(d, sig, hold=hold, exit_fn=exit_fn, cost=cost)
        g, dd = stats(nav, yrs)
        nb = sum(1 for t in tr if t[0] == '买')
        # 在场率
        occ = 0
        c2 = c
        sh = 0
        # 用 trades 还原在场天数
        segs = []
        st = None
        for t in tr:
            if t[0] == '买':
                st = t[1]
            elif st is not None:
                segs.append((st, t[1]))
                st = None
        if st is not None:
            segs.append((st, n - 1))
        occ = sum(b - a for a, b in segs) / n
        rows.append((name, nb, g, dd, g / abs(dd) if dd else np.nan, occ, note))

    # ---- 校验：与原基准一致 ----
    add('【基准】骨架 持满250日', hold=250)

    print('\n' + '=' * 118)
    print('一、固定期限（不管行情到点就走）')
    print('=' * 118)
    for lbl, h in [('持 8 周', 40), ('持 12 周', 60), ('持 16 周', 80),
                   ('持 26 周（半年）', 125), ('持 250 日（基准）', 250), ('持 2 年', 500)]:
        add('骨架→' + lbl, hold=h)

    print('\n' + '=' * 118)
    print('二、强弱信号卖出（先生的"强转弱"，无最短持有期）')
    print('=' * 118)
    add('骨架→上月收阴即卖', hold=99999, exit_fn=lambda i, e, p: B['prev_down'][i] == 1)
    add('骨架→1234周线转弱即卖', hold=99999, exit_fn=lambda i, e, p: B['wave_sig'][i])
    # 加最短持有 N 周，避免刚买就卖
    for wk, dys in [(4, 20), (8, 40), (12, 60)]:
        add(f'骨架→满{wk}周后1234转弱即卖', hold=99999,
            exit_fn=lambda i, e, p, _d=dys: (i - e >= _d) and B['wave_sig'][i])
    for wk, dys in [(4, 20), (8, 40), (12, 60)]:
        add(f'骨架→满{wk}周后月线收阴即卖', hold=99999,
            exit_fn=lambda i, e, p, _d=dys: (i - e >= _d) and B['prev_down'][i] == 1)

    print('\n' + '=' * 118)
    print('三、从高点回落卖出（v1.6 方向）')
    print('=' * 118)
    for dp in (-0.10, -0.15, -0.20, -0.25):
        add(f'骨架→回落{abs(dp)*100:.0f}%即卖', hold=99999,
            exit_fn=lambda i, e, p, _x=dp: c[i] / p - 1 <= _x)
    for dp in (-0.10, -0.15, -0.20):
        for mn, dys in [(6, 125), (12, 250)]:
            add(f'骨架→满{mn}月后回落{abs(dp)*100:.0f}%卖', hold=99999,
                exit_fn=lambda i, e, p, _x=dp, _d=dys: (i - e >= _d) and c[i] / p - 1 <= _x)

    print('\n' + '=' * 118)
    print(f'{"卖出扳机":<32} {"买入次数":>7} {"年化":>9} {"最大回撤":>10} {"Calmar":>8} {"在场":>7}')
    print('=' * 118)
    for name, nb, g, dd, cal, occ, note in rows:
        print(f'{name:<32} {nb:>7} {g*100:>+8.2f}% {dd*100:>+9.1f}% {cal:>8.3f} {occ*100:>6.1f}%')

    # ---- 加成本后的稳健性（单边0.1%）----
    print('\n' + '=' * 118)
    print('四、加成本（单边 0.1%）后的前四名')
    print('=' * 118)
    top = sorted(rows, key=lambda r: -r[2])[:6]
    print(f'{"卖出扳机":<32} {"买入":>5} {"无成本年化":>10} {"含成本年化":>10} {"含成本回撤":>11}')
    for name, nb, g, dd, cal, occ, note in top:
        pass
    print('  （见下表逐项）')


if __name__ == '__main__':
    main()
