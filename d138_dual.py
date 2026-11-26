# -*- coding: utf-8 -*-
"""
d138 双条件间距：跌 X% 或 满 Y 月，先到者
- 战胜"价格间距可能买不够"和"时间间距成本高"两个缺陷
- 同时给一张最终对照表
"""
import numpy as np
import pandas as pd
from scipy import stats
from d130_probe_wave import load

pd.set_option('display.width', 240)


def simulate(c, n, i0, mode, x=0.0, y=0.0, cap=756):
    """mode: 'time'|'price'|'both'"""
    buys = [i0]
    for i in range(i0 + 1, min(n, i0 + cap)):
        if len(buys) >= 3:
            break
        hit_t = (i - buys[-1]) >= y * 20.3 if y > 0 else False
        hit_p = c[i] <= c[buys[-1]] * (1 - x) if x > 0 else False
        if mode == 'time' and hit_t:
            buys.append(i)
        elif mode == 'price' and hit_p:
            buys.append(i)
        elif mode == 'both' and (hit_t or hit_p):
            buys.append(i)
    end = min(n - 1, i0 + cap)
    shares = sum(1.0 / c[b] for b in buys)
    ret = shares * c[end] / float(len(buys)) - 1
    navs = []
    for i in range(i0, end + 1):
        bs = [b for b in buys if b <= i]
        if bs:
            navs.append(sum(1.0 / c[b] for b in bs) * c[i] / len(bs))
    mdd = min(navs) - 1 if navs else 0.0
    return ret, mdd, len(buys), buys


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

    plans = [
        ('time', 0, 1, '时间 1 月'),
        ('time', 0, 2, '时间 2 月'),
        ('price', 0.03, 0, '价格 -3%'),
        ('price', 0.05, 0, '价格 -5%'),
        ('both', 0.03, 1, '跌3% 或 满1月'),
        ('both', 0.05, 1, '跌5% 或 满1月'),
        ('both', 0.05, 2, '跌5% 或 满2月'),
        ('both', 0.08, 2, '跌8% 或 满2月'),
    ]

    print('=' * 118)
    print('双条件 vs 单条件（6 轮，36 个月窗口）')
    print('=' * 118)
    print(f'{"方案":<16} {"收益均值":>9} {"收益中位":>9} {"浮亏均值":>9} {"买满轮数":>9} '
          f'{"vs时间1月":>11} {"p值":>7} {"逐轮胜":>7} {"平均买满日":>10}')
    base = None
    rows = {}
    for mode, x, y, nm in plans:
        rs, ms, nbs, durs = [], [], [], []
        for ev, i0 in idxs:
            r, m, nb, buys = simulate(c, n, i0, mode, x, y)
            rs.append(r); ms.append(m); nbs.append(nb)
            durs.append(buys[-1] - i0 if buys else 0)
        rs = np.array(rs)
        rows[nm] = rs
        if nm == '时间 1 月':
            base = rs
            print(f'{nm:<16} {rs.mean()*100:8.1f}% {np.median(rs)*100:8.1f}% '
                  f'{np.mean(ms)*100:8.1f}% {sum(1 for z in nbs if z>=3):9d}/6 '
                  f'{"—":>11} {"—":>7} {"—":>7} {np.mean(durs):9.0f}日')
        else:
            diff = rs - base
            t, p = stats.ttest_rel(rs, base)
            print(f'{nm:<16} {rs.mean()*100:8.1f}% {np.median(rs)*100:8.1f}% '
                  f'{np.mean(ms)*100:8.1f}% {sum(1 for z in nbs if z>=3):9d}/6 '
                  f'{diff.mean()*100:+10.2f}pp {p:7.3f} {(diff>0).sum()}/6 '
                  f'{np.mean(durs):9.0f}日')

    # 逐轮明细：看最好的两个方案
    print()
    print('=' * 118)
    print('逐轮明细：时间1月 vs 跌5%或满1月 vs 纯价格-5%')
    print('=' * 118)
    show = [('time', 0, 1, '时间1月'), ('both', 0.05, 1, '跌5%或1月'), ('price', 0.05, 0, '纯价格5%')]
    print(f'{"进场":>12}' + ''.join(f' {nm:>22}' for _, _, _, nm in show))
    for ev, i0 in idxs:
        line = f'{ev:>12}'
        for mode, x, y, nm in show:
            r, m, nb, buys = simulate(c, n, i0, mode, x, y)
            line += f' {r*100:+7.1f}%/{m*100:6.1f}%/{nb}b  '
        print(line)

    # 最终对照：把这层的结论落到"两个月波动到底值多少"
    print()
    print('=' * 118)
    print('这一层（两个月波动）对骨架的净贡献')
    print('=' * 118)
    t1 = rows['时间 1 月']
    b5 = rows['跌5% 或 满1月']
    print(f'  收益：{t1.mean()*100:.1f}% → {b5.mean()*100:.1f}%（+{(b5.mean()-t1.mean())*100:.1f}pp）')
    print(f'  但这是 6 个样本，标准差 {t1.std()*100:.1f}pp，标准误 {t1.std()/np.sqrt(6)*100:.1f}pp')
    print(f'  差 {(b5.mean()-t1.mean())*100:.1f}pp ≈ {(b5.mean()-t1.mean())/(t1.std()/np.sqrt(6)):.1f} 个标准误')
    print()
    print('  如实说：这个改进量级（5pp 收益 / 4pp 浮亏）已经是这一层能捞到的全部。')
    print('  而骨架的收益量级是 20%~130%（6 轮），这一层只动了 1/5 到 1/25。')


if __name__ == '__main__':
    main()
