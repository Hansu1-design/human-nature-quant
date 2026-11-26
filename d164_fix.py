# -*- coding: utf-8 -*-
"""
d164 修正口径后的确定性检验
纠错：名义线与实际线不能直接相除。骨架出手线 = 实际点位 2,068 = rA 1.10（名义 2,977）。
      d163 里用了 rA<=1.583 是错的（那是 2,977/1,880，混了两种口径）。
本脚本：
  ① 骨架信号 = d155 的 sig（rA≤1.10，冷却180，11次），挂成本止损 —— 严格对齐 5.46%/-43.3%
  ② 状态组合加"独立事件计数"（R47 教训：每段只取一个观测点）
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d155_exit2 import build, equity, stats

RF = 0.025
DAILY_RF = (1 + RF) ** (1 / 252) - 1
d = load()
c = d['close'].values.astype(float)
n = len(c)
dts = d['d']
yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
B = build(d)
sig, ra = B['sig'], B['ra']
c_ser = pd.Series(c)
ret250 = (c / c_ser.shift(250) - 1).values
dd250 = (c / c_ser.rolling(250).max() - 1).values

SEGS = [('1997-2010', '1997-01-01', '2010-12-31'),
        ('2011-2016', '2011-01-01', '2016-12-31'),
        ('2017-2026', '2017-01-01', '2026-12-31')]


def seg_ann(nav, a, b):
    i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
    i1 = min(int(np.searchsorted(dts.values, np.datetime64(b))) - 1, n - 1)
    y = (dts.iloc[i1] - dts.iloc[i0]).days / 365.25
    r = nav[i1] / nav[i0]
    return (r ** (1 / y) - 1) * 100 if r > 0 and y > 0 else np.nan


def row(nm, nav, tr=None):
    g, dd = stats(nav, yrs)
    return (f'{nm:<30}{g*100:>+9.2f}{dd*100:>+11.1f}{g/abs(dd):>8.3f}'
            f'{(len(tr)//2 if tr else 0):>6}{seg_ann(nav, *[SEGS[0][1], SEGS[0][2]]):>+9.2f}'
            f'{seg_ann(nav, *[SEGS[1][1], SEGS[1][2]]):>+9.2f}'
            f'{seg_ann(nav, *[SEGS[2][1], SEGS[2][2]]):>+9.2f}')


print('=' * 120)
print('① 骨架（rA≤1.10，11 次信号，持满 250 日）+ 成本止损：严格对齐基准')
print('=' * 120)
print(f'{"方案":<30}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"笔数":>6}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
nav0, tr0 = equity(d, sig, hold=250, exit_fn=None, cost=0.001)
print(row('基准：骨架持满250日', nav0, tr0))
for sp, lab in [(-0.06, '骨架+亏6%走'), (-0.08, '骨架+亏8%走'), (-0.10, '骨架+亏10%走'),
                (-0.12, '骨架+亏12%走'), (-0.15, '骨架+亏15%走'), (-0.20, '骨架+亏20%走')]:
    x = sp
    fn = (lambda i, e, p, s=x: c[i] / c[e] - 1 <= s)
    nav, tr = equity(d, sig, hold=250, exit_fn=fn, cost=0.001)
    print(row(lab, nav, tr))

print()
print('② 同样条件下，骨架基础上加「前一年涨>40% 不买」（v1.5 硬规则5）')
print(f'{"方案":<30}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"笔数":>6}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
sig_v = [i for i in sig if not (np.isfinite(ret250[i]) and ret250[i] > 0.40)]
print(f'  否决后信号 {len(sig)} → {len(sig_v)} 次')
nav, tr = equity(d, sig_v, hold=250, cost=0.001)
print(row('骨架+否决权', nav, tr))
for sp, lab in [(-0.10, '骨架+否决权+亏10%走'), (-0.15, '骨架+否决权+亏15%走')]:
    x = sp
    fn = (lambda i, e, p, s=x: c[i] / c[e] - 1 <= s)
    nav, tr = equity(d, sig_v, hold=250, exit_fn=fn, cost=0.001)
    print(row(lab, nav, tr))

print()
print('=' * 120)
print('③ 状态组合 + 独立事件计数（把连续满足的日期并成 1 个事件，R47 教训）')
print('=' * 120)
fwd250 = (c_ser.shift(-250) / c_ser - 1).values


def blocks(mask):
    """把连续 True 切成块，返回每块起始索引"""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def analyze(mask, lab):
    ok = mask & np.isfinite(fwd250)
    y = fwd250[ok]
    if len(y) < 60:
        return
    bl = blocks(ok)
    # 每个独立事件取块内首个可算点
    ev = np.array([fwd250[a] for a, b in bl if np.isfinite(fwd250[a])])
    # 事件层面（每次买入一个观测）
    print(f'{lab:<34}{len(y):>6}{np.median(y)*100:>9.2f}{(y>0).mean()*100:>8.1f}'
          f'{np.percentile(y,10)*100:>9.2f}{len(bl):>8}'
          f'{(np.median(ev)*100 if len(ev)>=1 else np.nan):>10.2f}'
          f'{(ev>0).mean()*100 if len(ev)>=1 else np.nan:>9.1f}')


print(f'{"格子":<34}{"决策日":>6}{"中位%":>9}{"胜率%":>8}{"p10%":>9}{"独立事件":>8}'
      f'{"事件中位%":>10}{"事件胜率%":>9}')

fin = np.isfinite(ra)
analyze(fin, '基准：全部（rA 可用期内）')
for th, lab in [(1.10, '离底≤1.10 骨架出手区'), (1.35, '离底≤1.35'), (1.583, '离底≤1.583（观察线）')]:
    analyze(fin & (ra <= th), lab)
print()
for plo, phi, pl in [(None, 1.10, '离底≤1.10'), (1.10, 1.35, '离底1.10~1.35'),
                     (1.35, 1.90, '离底1.35~1.90'), (1.90, None, '离底>1.90')]:
    for rlo, rhi, rl in [(None, -0.25, '年跌>25%'), (-0.25, 0.12, '年-25~+12%'),
                         (0.12, 0.40, '年涨12~40%'), (0.40, None, '年涨>40%')]:
        m = fin.copy()
        m &= (ra <= phi) if phi else True
        m &= (ra > plo) if plo else True
        m &= (ret250 < rhi) if rhi else True
        m &= (ret250 >= rlo) if rlo else True
        analyze(m, f'{pl} × {rl}')

print()
print('=' * 120)
print('当前读数（修正单位）')
print('=' * 120)
i = n - 1
print(f'上证 {c[i]:.2f}｜实际 {d["real"].iloc[i]:.0f}｜离底 {ra[i]-1:+.1%}（rA={ra[i]:.3f}）')
print(f'出手线 rA 1.10 = 实际 {1880*1.10:.0f} → 需从当前实际点位跌 {ra[i]/1.10-1:+.1%}')
print(f'前一年涨幅 {ret250[i]:+.1%}｜距250日高 {dd250[i]:+.1%}')
