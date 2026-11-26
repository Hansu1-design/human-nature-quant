# -*- coding: utf-8 -*-
"""
d143 决定性对比：
A) 骨架规则：进入离底<=10% 就买（不看回撤）
B) 骨架 + 月线回调：离底<=10% 且从 250 日高点跌 >=15% 才买
C) 骨架 + 深回调：离底<=10% 且跌 >=25% 才买
D) 贪一点：任何位置跌 >=25% 就买（不看离底）
E) 前一年涨幅否决权：<=10% 且前一年涨幅 <=40%

同时测：分批止盈（保留右尾） vs 全跑
"""
import numpy as np
import pandas as pd
from scipy import stats
from d141_eat_bite import load

pd.set_option('display.width', 250)
RF = 0.025
RNG = np.random.default_rng(23)


def boot_ci(x, n=5000):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    m = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def entries(d, cond, cool=180):
    """按日遍历，cond(i) 为真且距上次 > cool 日则触发"""
    c = d['close'].values
    n = len(c)
    ev, last = [], -999
    for i in range(300, n):
        if i - last <= cool:
            continue
        if cond(i):
            ev.append(i)
            last = i
    return ev


def sim(d, ev, hold=250, tp_full=None, half=None, sl=None):
    """half=(x, y): 涨 x% 卖一半，剩一半持到期；tp_full: 全跑止盈"""
    c = d['close'].values
    n = len(c)
    rs, lows = [], []
    for i in ev:
        w = 1.0
        cash = 0.0
        low = 0.0
        done_half = False
        for j in range(i + 1, min(n, i + hold + 1)):
            ret = c[j] / c[i] - 1
            low = min(low, ret)
            if sl is not None and ret <= -sl:
                cash += w * (1 - sl)
                w = 0.0
                break
            if half is not None and not done_half and ret >= half:
                cash += 0.5 * (1 + ret)
                w = 0.5
                done_half = True
            if tp_full is not None and ret >= tp_full:
                cash += w * (1 + ret)
                w = 0.0
                break
        j = min(n - 1, i + hold)
        cash += w * (c[j] / c[i])
        rs.append(cash - 1)
        lows.append(low)
    return np.array(rs), np.array(lows)


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    ra = d['rA'].values
    dd250 = np.array([c[i] / c[max(0, i - 250):i + 1].max() - 1 for i in range(n)])
    pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])

    rules = [
        ('A 骨架：离底≤10% 就买', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10),
        ('B A + 跌15%', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10 and dd250[i] <= -0.15),
        ('C A + 跌25%', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10 and dd250[i] <= -0.25),
        ('D 只跌25%（不看离底）', lambda i: dd250[i] <= -0.25),
        ('E A + 前一年涨幅≤40%', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10 and pre[i] <= 0.40),
        ('F 离底≤15%（放宽骨架）', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.15),
        ('G 离底≤20%（再放宽）', lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.20),
    ]

    print('=' * 128)
    print('【1】入场规则对比（持有 250 日，无止损无止盈）—— 2008-01 起有 CPI 的子样本')
    print('=' * 128)
    print(f'{"规则":<26} {"触发":>4} {"均值":>8} {"CI":>20} {"中位":>8} {"胜率":>6} '
          f'{"浮亏均值":>8} {"vs生息2.57%":>10} {"p":>6}')
    for name, cond in rules:
        ev = entries(d, cond)
        if len(ev) < 2:
            print(f'{name:<26} {len(ev):4d}  样本不足')
            continue
        r, low = sim(d, ev, 250)
        lo_, hi_ = boot_ci(r)
        t, p = stats.ttest_1samp(r, RF * 250 / 243)
        print(f'{name:<26} {len(ev):4d} {r.mean()*100:+7.1f}% '
              f'[{lo_*100:+6.1f},{hi_*100:+6.1f}] {np.median(r)*100:+7.1f}% '
              f'{(r>0).mean()*100:5.0f}% {low.mean()*100:+7.1f}% '
              f'{(r.mean()-RF*250/243)*100:+9.1f}% {p:6.3f}')

    print()
    print('=' * 128)
    print('【2】规则 A vs B 的逐次对照（同处出手区，加不加「跌 15%」的差别）')
    print('=' * 128)
    evA = entries(d, rules[0][1])
    evB = entries(d, rules[1][1])
    print('A 触发：', [str(d.d.iloc[i].date()) for i in evA])
    print('B 触发：', [str(d.d.iloc[i].date()) for i in evB])
    print()
    print(f'{"A 触发日":>11} {"A点位":>7} {"A离底":>7} {"A跌15%?":>8} {"A 250日":>9} | '
          f'{"B 触发日":>11} {"B点位":>7} {"B 250日":>9}')
    m = max(len(evA), len(evB))
    for k in range(m):
        a = str(d.d.iloc[evA[k]].date()) if k < len(evA) else '—'
        av = f'{c[evA[k]]:7.0f}' if k < len(evA) else '      —'
        ar = f'{(ra[evA[k]]-1)*100:+6.1f}%' if k < len(evA) else '     —'
        ad = f'{"是" if dd250[evA[k]]<=-0.15 else "否":>8}' if k < len(evA) else '       —'
        ax = f'{(c[min(n-1,evA[k]+250)]/c[evA[k]]-1)*100:+8.1f}%' if k < len(evA) else '        —'
        b = str(d.d.iloc[evB[k]].date()) if k < len(evB) else '—'
        bv = f'{c[evB[k]]:7.0f}' if k < len(evB) else '      —'
        bx = f'{(c[min(n-1,evB[k]+250)]/c[evB[k]]-1)*100:+8.1f}%' if k < len(evB) else '        —'
        print(f'{a:>11} {av} {ar} {ad} {ax} | {b:>11} {bv} {bx}')

    print()
    print('=' * 128)
    print('【3】分批止盈能不能保留右尾（规则 A、离底≤10% 的 6 次，持有 250 日）')
    print('=' * 128)
    print(f'{"方案":<30} {"均值":>8} {"中位":>8} {"胜率":>6} {"最大回吐":>8}')
    plans = [
        ('全跑 +10%', dict(tp_full=0.10)),
        ('全跑 +15%', dict(tp_full=0.15)),
        ('全跑 +20%', dict(tp_full=0.20)),
        ('涨10%卖一半，余留到期', dict(half=0.10)),
        ('涨20%卖一半，余留到期', dict(half=0.20)),
        ('涨30%卖一半，余留到期', dict(half=0.30)),
        ('不动，持满250日', dict()),
    ]
    for nm, kw in plans:
        r, low = sim(d, evA, 250, **kw)
        print(f'{nm:<30} {r.mean()*100:+7.1f}% {np.median(r)*100:+7.1f}% '
              f'{(r>0).mean()*100:5.0f}% {low.mean()*100:+7.1f}%')

    print()
    print('=' * 128)
    print('【4】持有期拉长：这一层到底是「吃一口」还是「吃一段」（规则 A 的 6 次）')
    print('=' * 128)
    print(f'{"持有":>6} {"均值":>8} {"中位":>8} {"胜率":>6} {"浮亏均值":>9}')
    for h in [20, 60, 120, 250, 500, 750]:
        r, low = sim(d, evA, h)
        print(f'{h:5d}日 {r.mean()*100:+7.1f}% {np.median(r)*100:+7.1f}% '
              f'{(r>0).mean()*100:5.0f}% {low.mean()*100:+8.1f}%')

    print()
    print('=' * 128)
    print('【5】独立机会去重：出手区里真正独立的机会有几个')
    print('=' * 128)
    print('按 250 日冷却重算：')
    ev2 = entries(d, rules[0][1], cool=250)
    print(f'  250 日冷却 → {len(ev2)} 次：', [str(d.d.iloc[i].date()) for i in ev2])
    r, low = sim(d, ev2, 250)
    print(f'  均值 {r.mean()*100:+.1f}%  中位 {np.median(r)*100:+.1f}%  胜率 {(r>0).mean()*100:.0f}%')
    ev3 = entries(d, rules[0][1], cool=500)
    print(f'  500 日冷却 → {len(ev3)} 次：', [str(d.d.iloc[i].date()) for i in ev3])
    r3, low3 = sim(d, ev3, 250)
    print(f'  均值 {r3.mean()*100:+.1f}%  中位 {np.median(r3)*100:+.1f}%  胜率 {(r3>0).mean()*100:.0f}%')


if __name__ == '__main__':
    main()
