# -*- coding: utf-8 -*-
"""
d159_constraint.py  ——  R47：分布两端到底能约束什么？

先生：「两三天是肯定好不了的，十几天还不好肯定是不正常的。」
Hermes 判：短端该约束的是"最小持有期"（算术），长端该约束的是"风险暴露"（不是卖点），
         并且「两三天/十几天」很可能是自然日语言 → 该测日历，不是段长。

本脚本三件事：
  1. 最小持有期：持有 T 天，扣 0.2% 双边成本后的中位净收益与胜率（算术，不需要预测）
  2. 长端逐笔：≥11 周且涨幅 >30% 那一格只有 6 个观测，逐个列出来，看是否集中于某一年
  3. 日历效应：周内 / 节前节后 / 月内节奏 / 交割日 —— 三段检验
"""
import numpy as np
import pandas as pd

from d141_eat_bite import load

COST = 0.002      # 双边成本 0.2%


def seg3(dts):
    return [('1997-2010', '1997-01-01', '2010-12-31'),
            ('2011-2016', '2011-01-01', '2016-12-31'),
            ('2017-2026', '2017-01-01', '2026-12-31')]


def idx_of(dts, s):
    return int(np.searchsorted(dts.values, np.datetime64(s)))


def main():
    d = load()
    c = d['close'].values
    dts = d['d']
    n = len(c)
    yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25

    # ============================================================ 1. 最小持有期
    print('=' * 104)
    print('【1】最小持有期 —— 「两三天肯定好不了」到底对不对（扣 0.2% 双边成本后的净收益）')
    print('-' * 104)
    print(f'  {"持有天数":>8} {"样本n":>7} {"中位净收益":>11} {"均值净收益":>11} {"胜率":>8} {"正收益占比(扣成本)":>18}')
    for T in (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 120, 250):
        fwd = c[T:] / c[:-T] - 1 - COST
        if len(fwd) < 100:
            continue
        win = (fwd > 0).mean()
        print(f'  {T:>8} {len(fwd):>7} {np.median(fwd)*100:>+10.3f}% {fwd.mean()*100:>+10.3f}% '
              f'{win*100:>7.1f}% {win*100:>17.1f}%')

    # 用日波动率归一化后重做（这样才有普适性）
    r = np.diff(np.log(c))
    sig_all = np.std(r) * np.sqrt(252)
    print(f'\n  全期年化波动 {sig_all*100:.1f}%，对应日波动 {np.std(r)*100:.3f}%')
    print(f'  成本 0.2% = {COST/(np.std(r)):.2f} 个日标准差')
    # 不同波动档下的最小持有期
    print('\n  不同波动环境下，"净收益中位转正"所需的持有天数：')
    rv = pd.Series(np.abs(r)).rolling(20).mean().values
    qs = np.nanpercentile(rv, [33, 67])
    print(f'  {"波动档":<10} {"日均|r|":>9} {"T*=3天":>10} {"T*=5天":>10} {"T*=10天":>10} {"T*=20天":>10}')
    for nm, cond in [('低', lambda v: v <= qs[0]), ('中', lambda v: (v > qs[0]) & (v <= qs[1])),
                     ('高', lambda v: v > qs[1])]:
        row = f'  {nm:<10}'
        sel = [i for i in range(19, n - 1) if not np.isnan(rv[i]) and cond(rv[i])]
        m = np.mean([rv[i] for i in sel])
        row = f'  {nm:<10} {m*100:>8.3f}%'
        for T in (3, 5, 10, 20):
            vals = [c[i + T] / c[i] - 1 - COST for i in sel if i + T < n]
            row += f'{np.median(vals)*100:>+9.3f}%'
        print(row + f'   (n={len(sel)})')

    # ============================================================ 2. 长端逐笔
    print('\n' + '=' * 104)
    print('【2】长端那一格：已涨 ≥11 周 且 累计涨幅 >30%，历史上到底有几次、后来怎样')
    print('-' * 104)
    # 重建周序列（与 d158 同口径）
    wy = dts.dt.strftime('%G-%V')
    wu = list(dict.fromkeys(wy.tolist()))
    wmap = {k: j for j, k in enumerate(wu)}
    wlast = [int(np.where(wy.values == k)[0][-1]) for k in wu]
    wc = c[wlast]
    wd = [dts.iloc[i] for i in wlast]
    nw = len(wc)
    rr = np.diff(wc) / wc[:-1]
    sg = np.sign(rr)
    segs = []
    cur, s0 = 1, 0
    for i in range(1, len(sg)):
        if sg[i] == sg[i - 1] and sg[i] != 0:
            cur += 1
        else:
            segs.append((sg[i - 1], s0, s0 + cur, cur))
            s0, cur = s0 + cur, 1
    segs.append((sg[-1], s0, s0 + cur, cur))

    hits = []
    for s_ in segs:
        sign, a0, a1, L = s_
        if sign <= 0:
            continue
        base = wc[a0 - 1] if a0 > 0 else wc[0]
        for k in range(11, L + 1):
            ek = a0 + k - 1
            if ek + 13 >= nw:
                continue
            g = wc[ek] / base - 1
            if g > 0.30:
                hits.append((k, g, ek, wc[ek + 13] / wc[ek] - 1,
                             wc[ek:ek + 14].min() / wc[ek] - 1))
    print(f'  共 {len(hits)} 个观测点（{len(set(h[2] for h in hits))} 个不同的周末）')
    print(f'  {"周末":<12}{"已持续周":>8}{"段内涨幅":>10}{"次13周":>10}{"次13周最大回撤":>14}')
    for k, g, ek, f13, dd in hits:
        print(f'  {str(wd[ek].date()):<12}{k:>8}{g*100:>+9.1f}%{f13*100:>+9.1f}%{dd*100:>+13.1f}%')
    if hits:
        f13 = [h[3] for h in hits]
        print(f'\n  次13周中位 {np.median(f13)*100:+.2f}%  均值 {np.mean(f13)*100:+.2f}%  '
              f'胜率 {(np.array(f13)>0).mean()*100:.0f}%')

    # ============================================================ 3. 日历效应
    print('\n' + '=' * 104)
    print('【3】日历效应 —— 先生说的「两三天/十几天」可能是自然日，不是交易日')
    print('-' * 104)
    r1 = np.append(np.nan, r)            # 当日收益（首个为 nan）
    dow = dts.dt.dayofweek.values
    # 3.1 周内效应
    print('  3.1 周内效应（当日收益，%，全期）')
    print(f'  {"星期":<6} {"n":>6} {"均值":>9} {"中位":>9} {"胜率":>8} {"t值":>8}')
    from scipy import stats as st
    for k in range(5):
        v = r1[(dow == k) & ~np.isnan(r1)]
        t, p = st.ttest_1samp(v, 0)
        print(f'  {"一二三四五"[k]:<6} {len(v):>6} {v.mean()*100:>+8.3f}% {np.median(v)*100:>+8.3f}% '
              f'{(v>0).mean()*100:>7.1f}% {t:>8.2f}')

    # 3.2 节前节后：用"休市 ≥3 个自然日"识别长假
    gaps = np.diff(dts.values).astype('timedelta64[D]').astype(int)
    pre = []      # 长假前最后一个交易日
    post = []     # 长假后第一个交易日
    for i, g in enumerate(gaps):
        if g >= 4:
            pre.append(i)
            post.append(i + 1)
    print(f'\n  3.2 长假效应（休市 ≥4 自然日，即含周末以上，共 {len(pre)} 次）')
    for nm, lst in [('节前最后一日', pre), ('节后第一日', post)]:
        v = r1[lst]
        v = v[~np.isnan(v)]
        t, p = st.ttest_1samp(v, 0)
        print(f'    {nm:<12} n={len(v):>3}  均值 {v.mean()*100:+6.3f}%  中位 {np.median(v)*100:+6.3f}%  '
              f'胜率 {(v>0).mean()*100:4.1f}%  t={t:+5.2f} p={p:.3f}')

    # 3.3 月内节奏
    dom = dts.dt.day.values
    print('\n  3.3 月内节奏（按自然日，全期月度收益的日内分布）')
    for nm, cond in [('月初 1-5 日', lambda x: x <= 5), ('月中 11-20 日', lambda x: (x >= 11) & (x <= 20)),
                     ('月末 25 日后', lambda x: x >= 25)]:
        v = r1[cond(dom)]
        v = v[~np.isnan(v)]
        t, p = st.ttest_1samp(v, 0)
        print(f'    {nm:<14} n={len(v):>4}  均值 {v.mean()*100:+6.3f}%  中位 {np.median(v)*100:+6.3f}%  '
              f'胜率 {(v>0).mean()*100:4.1f}%  t={t:+5.2f}')

    # 3.4 交割日（每月第三个周五）
    is3fri = np.zeros(n, dtype=bool)
    for i in range(n):
        t_ = dts.iloc[i]
        if t_.dayofweek == 4 and 15 <= t_.day <= 21:
            is3fri[i] = True
    v = r1[is3fri]
    v = v[~np.isnan(v)]
    t, p = st.ttest_1samp(v, 0)
    print(f'\n  3.4 交割日（每月第三个周五） n={len(v):>3}  均值 {v.mean()*100:+6.3f}%  '
          f'中位 {np.median(v)*100:+6.3f}%  胜率 {(v>0).mean()*100:4.1f}%  t={t:+5.2f} p={p:.3f}')

    # 3.5 三段检验：胜率/均值最强的那几项
    print('\n  3.5 三段检验（每段独立估计，要求三段同号且第三段 ≥ 第一段 50%）')
    tests = [
        ('周一', lambda i: dow[i] == 0),
        ('周五', lambda i: dow[i] == 4),
        ('节前最后一日', lambda i: i in set(pre)),
        ('节后第一日', lambda i: i in set(post)),
        ('交割日', lambda i: is3fri[i]),
        ('月末5日', lambda i: dom[i] >= 25),
    ]
    print(f'  {"项目":<14}' + ''.join([f'{s[0]:>16}' for s in seg3(dts)]))
    for nm, fn in tests:
        line = f'  {nm:<14}'
        for _, a, b in seg3(dts):
            i0, i1 = idx_of(dts, a), idx_of(dts, b)
            v = [r1[i] for i in range(max(1, i0), min(i1, n)) if not np.isnan(r1[i]) and fn(i)]
            if len(v) < 20:
                line += f'{"—":>16}'
            else:
                line += f'{np.mean(v)*100:>+15.3f}%'
        print(line)

    print('\n' + '=' * 104)
    print('【当前读数】')
    print('-' * 104)
    print(f'  上证 {c[-1]:.2f} / {dts.iloc[-1].date()}   全期 {yrs:.1f} 年')


if __name__ == '__main__':
    main()
