# -*- coding: utf-8 -*-
"""
d161_chase.py  ——  R47 的真正收获：趋势确立后买入，是不是独立的第二套策略？

发现来源：d160 补算基准时撞出来的。
  「连续 3 周上涨且段内累计涨 >5%」这个状态确立后，未来 13 周中位 +7.36%，
   任意时点 +0.34%，超额 +7.03pp；去掉最大 3 笔仍有 +5.46%，不是靠右尾。（n=41）

这跟 R46 的「段长无记忆」不矛盾：
  段长 = 还剩几周 → 无信息（hazard 平坦）
  状态 = 现在是否在趋势里 → 强信息

但必须过三关，否则又是「2011-2016 运气」：
  ① 三段检验（1997-2010 / 2011-2016 / 2017-2026 同号且第三段 ≥ 第一段 50%）
  ② 资金曲线（日频、几何年化、现金 2.5%/年），与骨架、买入持有比
  ③ 逐笔列出，不许只报均值

信号口径（严格无未来函数）：周线收盘确认 → 下一交易日收盘执行。
"""
import numpy as np
import pandas as pd

from d141_eat_bite import load

COST = 0.001          # 单边 0.1%
RF = 0.025


def build_weeks(d):
    c = d['close'].values
    wy = d['d'].dt.strftime('%G-%V')
    wu = list(dict.fromkeys(wy.tolist()))
    wlast = [int(np.where(wy.values == k)[0][-1]) for k in wu]
    wc = c[wlast]
    return wc, wlast, [d['d'].iloc[i] for i in wlast]


def trend_signals(wc, wlast, n, need_weeks=3, need_gain=0.05):
    """返回 {执行日索引: 事件信息}。周末收盘确认 → 下一交易日执行。"""
    nw = len(wc)
    rr = np.diff(wc) / wc[:-1]
    sig = {}
    i = 0
    while i < nw - 1:
        # 从 i 开始找连续 need_weeks 周上涨
        if all(rr[j] > 0 for j in range(i, min(i + need_weeks, nw - 1))):
            j = i + need_weeks - 1
            if j >= nw:
                break
            base = wc[i - 1] if i > 0 else wc[0]
            g = wc[j] / base - 1
            if g > need_gain:
                exe = wlast[j] + 1
                if exe < n:
                    sig[exe] = dict(weeks=need_weeks, gain=g,
                                    confirm=str(wlast and pd.Timestamp('1990-01-01').date()))
                    # 冷却：13 周内不再出新信号
                    i = j + 13
                    continue
        i += 1
    return sig


def eq_run(d, sig, hold=65, cost=COST, rf=RF):
    c = d['close'].values
    n = len(c)
    nav = np.ones(n)
    cash, sh, entry = 1.0, 0.0, -1
    keys = sorted(sig.keys())
    kset = set(keys)
    trades = []
    for i in range(n):
        if sh > 1e-12:
            nav[i] = sh * c[i] + cash
            if i - entry >= hold:
                cash = sh * c[i] * (1 - cost)
                sh = 0.0
                trades.append(('卖', i, c[i]))
        else:
            nav[i] = cash
            if i in kset and i < n - 1:
                sh = cash * (1 - cost) / c[i]
                cash = 0.0
                entry = i
                trades.append(('买', i, c[i]))
        if sh <= 1e-12 and i < n - 1:
            cash *= (1 + (1 + rf) ** (1 / 252) - 1)
    return nav, trades


def stats(nav, yrs):
    nav = np.asarray(nav, dtype=float)
    tot = nav[-1] / nav[0]
    cagr = tot ** (1 / yrs) - 1 if tot > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    dd = (nav / peak - 1).min()
    return cagr, dd


def seg_ann(nav, dts, a, b):
    i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
    i1 = int(np.searchsorted(dts.values, np.datetime64(b))) - 1
    if i1 <= i0:
        return np.nan
    y = (dts.iloc[i1] - dts.iloc[i0]).days / 365.25
    r = nav[i1] / nav[i0]
    return r ** (1 / y) - 1 if r > 0 and y > 0 else np.nan


def main():
    d = load()
    c = d['close'].values
    dts = d['d']
    n = len(c)
    yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
    wc, wlast, wd = build_weeks(d)
    SEG = [('1997-2010', '1997-01-01', '2010-12-31'),
           ('2011-2016', '2011-01-01', '2016-12-31'),
           ('2017-2026', '2017-01-01', '2026-12-31')]

    print('=' * 112)
    print('R47 趋势确立后买入：三关检验')
    print('=' * 112)

    # ---------------- 参数面
    print('\n【0】参数面：需要几周上涨 × 需要涨多少（次13周中位 / 超额 vs 任意时点）')
    print('-' * 112)
    base13 = np.median([c[i + 65] / c[i] - 1 for i in range(n - 65)])
    print(f'  基准：任意时点后 65 日 中位 {base13*100:+.2f}%')
    print(f'  {"":<10}' + ''.join([f'{">"+str(int(g*100))+"%":>18}' for g in (0.03, 0.05, 0.08, 0.12)]))
    for w in (2, 3, 4, 5):
        line = f'  {str(w)+"周上涨":<10}'
        for g in (0.03, 0.05, 0.08, 0.12):
            sig = trend_signals(wc, wlast, n, w, g)
            vals = [c[i + 65] / c[i] - 1 for i in sig if i + 65 < n]
            if len(vals) < 4:
                line += f'{"—":>18}'
            else:
                line += f'{np.median(vals)*100:>+8.1f}% (n={len(vals):>2})'
        print(line)

    # ---------------- 定稿口径
    print('\n' + '=' * 112)
    print('【1】定稿口径：连续 3 周上涨且段内累计 >5%，持有 13 周')
    print('-' * 112)
    sig = trend_signals(wc, wlast, n, 3, 0.05)
    print(f'  信号 {len(sig)} 个（30 年，平均每年 {len(sig)/yrs:.1f} 个）')
    print(f'\n  {"执行日":<12}{"买价":>9}{"卖日":<12}{"卖价":>9}{"收益":>9}{"持有日":>7}')
    rets = []
    for i in sorted(sig):
        j = min(i + 65, n - 1)
        r = c[j] / c[i] - 1
        rets.append(r)
        print(f'  {str(dts.iloc[i].date()):<12}{c[i]:>9.2f}{str(dts.iloc[j].date()):<12}'
              f'{c[j]:>9.2f}{r*100:>+8.1f}%{j-i:>7}')
    rets = np.array(rets)
    print(f'\n  中位 {np.median(rets)*100:+.2f}%  均值 {rets.mean()*100:+.2f}%  胜率 {(rets>0).mean()*100:.0f}%')
    srt = np.sort(rets)
    print(f'  去掉最大 3 笔 {np.median(srt[:-3])*100:+.2f}%  去掉最大 5 笔 {np.median(srt[:-5])*100:+.2f}%')

    # ---------------- 三段
    print('\n' + '=' * 112)
    print('【2】第一关：三段检验（事件层面）')
    print('-' * 112)
    print(f'  {"时段":<14}{"事件数":>7}{"中位":>10}{"胜率":>8}   与"同期任意时点"比')
    for nm, a, b in SEG:
        i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
        i1 = int(np.searchsorted(dts.values, np.datetime64(b)))
        vals = [c[i + 65] / c[i] - 1 for i in sorted(sig) if i0 <= i < i1 and i + 65 < n]
        base = [c[i + 65] / c[i] - 1 for i in range(i0, min(i1, n - 65))]
        if len(vals) < 3:
            print(f'  {nm:<14}{len(vals):>7}   （样本太少）')
            continue
        print(f'  {nm:<14}{len(vals):>7}{np.median(vals)*100:>+9.1f}%{(np.array(vals)>0).mean()*100:>7.0f}%'
              f'   事件 {np.median(vals)*100:+.1f}% vs 基准 {np.median(base)*100:+.1f}% = '
              f'{(np.median(vals)-np.median(base))*100:+.1f}pp')

    # ---------------- 资金曲线
    print('\n' + '=' * 112)
    print('【3】第二关：资金曲线（日频、几何年化、现金 2.5%/年、单边成本 0.1%）')
    print('-' * 112)
    navC, trC = eq_run(d, sig, hold=65)
    gC, dC = stats(navC, yrs)
    print(f'  趋势策略：年化 {gC*100:+.2f}%  最大回撤 {dC*100:+.1f}%  交易 {len(trC)} 笔  '
          f'Calmar {gC/abs(dC):.3f}')
    # 参照
    navB = c / c[0]
    gB, dB = stats(navB, yrs)
    print(f'  买入持有：年化 {gB*100:+.2f}%  最大回撤 {dB*100:+.1f}%')
    navF = np.empty(n)
    cc = 1.0
    for i in range(n):
        navF[i] = cc
        cc *= (1 + (1 + RF) ** (1 / 252) - 1)
    gF, dF = stats(navF, yrs)
    print(f'  空仓生息：年化 {gF*100:+.2f}%  最大回撤 0.0%')
    print('\n  分段年化：')
    print(f'  {"方案":<16}' + ''.join([f'{s[0]:>14}' for s in SEG]))
    for nm, nav in [('趋势策略', navC), ('买入持有', navB), ('空仓生息', navF)]:
        line = f'  {nm:<16}'
        for _, a, b in SEG:
            line += f'{seg_ann(nav, dts, a, b)*100:>13.2f}%'
        print(line)

    # ---------------- 与骨架合并
    print('\n' + '=' * 112)
    print('【4】第三关：能不能和骨架合并（骨架空仓期用趋势信号补位）')
    print('-' * 112)
    # 读骨架信号（d145 口径）
    try:
        from d155_exit2 import build
        B = build(d)
        base_sig = sorted(B['sig'])
        print(f'  骨架信号 {len(base_sig)} 个：{[str(dts.iloc[i].date()) for i in base_sig]}')
        merged = sorted(set(base_sig) | set(sig.keys()))
        print(f'  合并后 {len(merged)} 个信号（新增 {len(merged)-len(base_sig)} 个）')
        # 合并后按 250 日持有跑，看是否被趋势信号干扰
        def eq_hold(signals, hold):
            nav = np.ones(n)
            cash, sh, entry = 1.0, 0.0, -1
            ss = set(signals)
            for i in range(n):
                if sh > 1e-12:
                    nav[i] = sh * c[i] + cash
                    if i - entry >= hold:
                        cash = sh * c[i] * (1 - COST)
                        sh = 0.0
                else:
                    nav[i] = cash
                    if i in ss and i < n - 1:
                        sh = cash * (1 - COST) / c[i]
                        cash = 0.0
                        entry = i
                if sh <= 1e-12 and i < n - 1:
                    cash *= (1 + (1 + RF) ** (1 / 252) - 1)
            return nav
        for nm, sg in [('骨架单独（250日）', base_sig), ('骨架+趋势（250日）', merged)]:
            nav = eq_hold(sg, 250)
            g, dd = stats(nav, yrs)
            line = f'  {nm:<20}{g*100:>+8.2f}%{dd*100:>+9.1f}%'
            for _, a, b in SEG:
                line += f'{seg_ann(nav, dts, a, b)*100:>12.2f}%'
            print(line)
    except Exception as e:
        print(f'  （骨架对照失败：{e}）')

    print('\n' + '=' * 112)


if __name__ == '__main__':
    main()
