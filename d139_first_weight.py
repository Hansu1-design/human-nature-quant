# -*- coding: utf-8 -*-
"""
d139 首笔该下多重
- 因为最低点多在进场后 20 日内（d134 已证），后续加仓必然更贵
- 测首笔权重 w1 ∈ {1/3, 1/2, 2/3, 1}
- 剩余按价格 -5% / 时间 1 月 分
"""
import numpy as np
import pandas as pd
from scipy import stats
from d130_probe_wave import load

pd.set_option('display.width', 240)


def simulate(c, n, i0, w1, mode, x=0.05, y=1, cap=756):
    """w1=首笔权重；剩余 (1-w1) 分两笔，按 mode 触发"""
    buys = [(i0, w1)]
    rest = 2
    each = (1 - w1) / rest
    for i in range(i0 + 1, min(n, i0 + cap)):
        if len(buys) >= 3:
            break
        if mode == 'price':
            hit = c[i] <= c[buys[-1][0]] * (1 - x)
        else:
            hit = (i - buys[-1][0]) >= y * 20.3
        if hit:
            buys.append((i, each))
    end = min(n - 1, i0 + cap)
    invested = sum(w for _, w in buys)
    shares = sum(w / c[b] for b, w in buys)
    # 未投入部分按现金 0 收益（保守；也可按 2.5% 生息）
    nav_end = shares * c[end] + (1 - invested)
    ret = nav_end - 1
    navs = []
    for i in range(i0, end + 1):
        bs = [(b, w) for b, w in buys if b <= i]
        inv = sum(w for _, w in bs)
        sh = sum(w / c[b] for b, w in bs)
        navs.append(sh * c[i] + (1 - inv))
    mdd = min(navs) - 1
    return ret, mdd, len(buys), invested


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    events = ['2008-09-08', '2011-10-21', '2018-08-17', '2020-02-03',
              '2022-04-26', '2023-10-23']
    idxs = []
    for ev in events:
        k = d.index[d.d >= ev]
        if len(k):
            idxs.append((ev, int(k[0])))

    print('=' * 112)
    print('首笔权重（36 个月窗口，剩余两笔按"跌5%"或"满1月"分）')
    print('=' * 112)
    print(f'{"首笔权重":<12} {"触发方式":<10} {"收益均值":>9} {"浮亏均值":>9} '
          f'{"实际投入":>8} {"vs 1/3+价格":>12} {"p值":>7} {"逐轮胜":>7}')
    base = None
    for mode, mlab in [('price', '跌5%'), ('time', '满1月')]:
        for w1 in [1 / 3, 1 / 2, 2 / 3, 1.0]:
            rs, ms, ivs = [], [], []
            for ev, i0 in idxs:
                r, m, nb, inv = simulate(c, n, i0, w1, mode)
                rs.append(r); ms.append(m); ivs.append(inv)
            rs = np.array(rs)
            if abs(w1 - 1 / 3) < 1e-9 and mode == 'price':
                base = rs
                print(f'{w1:<12.2f} {mlab:<10} {rs.mean()*100:8.1f}% {np.mean(ms)*100:8.1f}% '
                      f'{np.mean(ivs)*100:7.0f}% {"—":>12} {"—":>7} {"—":>7}')
            else:
                t, p = stats.ttest_rel(rs, base) if len(rs) == len(base) else (np.nan, np.nan)
                diff = rs - base
                print(f'{w1:<12.2f} {mlab:<10} {rs.mean()*100:8.1f}% {np.mean(ms)*100:8.1f}% '
                      f'{np.mean(ivs)*100:7.0f}% {diff.mean()*100:+11.2f}pp {p:7.3f} {(diff>0).sum()}/6')
        print()

    print('=' * 112)
    print('逐轮明细：首笔 1/3 vs 1/2 vs 1（其余按跌5%分）')
    print('=' * 112)
    print(f'{"进场":>12}' + ''.join(f' {"首笔"+w:>24}' for w in ['1/3', '1/2', '2/3', '1(全上)']))
    for ev, i0 in idxs:
        line = f'{ev:>12}'
        for w1 in [1 / 3, 1 / 2, 2 / 3, 1.0]:
            r, m, nb, inv = simulate(c, n, i0, w1, 'price')
            line += f' {r*100:+7.1f}%/{m*100:6.1f}%/{inv*100:3.0f}%  '
        print(line)

    print()
    print('=' * 112)
    print('和 v1.3 定稿对比（v1.3 = 首笔1/3 + 时间1月 × 2）')
    print('=' * 112)
    v13, new = [], []
    for ev, i0 in idxs:
        r1, m1, _, _ = simulate(c, n, i0, 1 / 3, 'time')
        r2, m2, _, _ = simulate(c, n, i0, 1 / 2, 'price')
        v13.append((r1, m1)); new.append((r2, m2))
    v13 = np.array(v13); new = np.array(new)
    print(f'  v1.3（1/3 + 时间1月）:  收益 {v13[:,0].mean()*100:5.1f}%  浮亏 {v13[:,1].mean()*100:5.1f}%')
    print(f'  新版（1/2 + 跌5%）:     收益 {new[:,0].mean()*100:5.1f}%  浮亏 {new[:,1].mean()*100:5.1f}%')
    print(f'  差:                    收益 {(new[:,0].mean()-v13[:,0].mean())*100:+5.2f}pp  '
          f'浮亏 {(new[:,1].mean()-v13[:,1].mean())*100:+5.2f}pp')
    print()
    print(f'  6 轮逐轮收益：v1.3 {np.round(v13[:,0]*100,1)}')
    print(f'                新版 {np.round(new[:,0]*100,1)}')
    print(f'  每个样本标准差 {v13[:,0].std()*100:.1f}pp，标准误 {v13[:,0].std()/np.sqrt(6)*100:.1f}pp')


if __name__ == '__main__':
    main()
