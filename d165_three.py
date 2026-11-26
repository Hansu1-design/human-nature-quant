# -*- coding: utf-8 -*-
"""
d165 三件事说清
① "离底≤1.10 × 前一年跌>25%" 到底是几次独立事件（逐个列出，R47 纪律）
② 底部止损为什么把 1997-2010 段从 4.69% 打到 1.61%（逐笔看 2008）
③ 趋势维度到底有没有用：一张干净的对照表
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d155_exit2 import build

RF = 0.025
d = load()
c = d['close'].values.astype(float)
n = len(c)
dts = d['d']
c_ser = pd.Series(c)
B = build(d)
ra = B['ra']
ret250 = (c / c_ser.shift(250) - 1).values
ma250 = c_ser.rolling(250).mean().values
ma_slope = pd.Series(ma250) / pd.Series(ma250).shift(20) - 1
rv20 = (c_ser / c_ser.shift(1) - 1).rolling(20).std().values * np.sqrt(252)
fwd250 = (c_ser.shift(-250) / c_ser - 1).values
fwd500 = (c_ser.shift(-500) / c_ser - 1).values

print('=' * 110)
print('① "离底≤1.10 且 前一年跌>25%" 的独立事件（用下穿交叉定义，连续窗口算 1 次）')
print('=' * 110)
cond = np.isfinite(ra) & np.isfinite(ret250) & (ra <= 1.10) & (ret250 < -0.25)
# 下穿：前一交易日不满足，当日满足
enter = np.zeros(n, dtype=bool)
for i in range(1, n):
    if cond[i] and not cond[i - 1]:
        enter[i] = True

evs = [i for i in range(n) if enter[i]]
print(f'独立事件（下穿次数）: {len(evs)} 次')
print(f'{"#":>3}{"入场日":<13}{"上证":>9}{"实际":>8}{"离底":>8}{"前年涨幅":>10}'
      f'{"250日后":>10}{"500日后":>10}  后续路径')
for k, i in enumerate(evs, 1):
    j2 = min(i + 250, n - 1)
    j5 = min(i + 500, n - 1)
    path = c[i:min(i + 251, n)]
    lo = path.min() / c[i] - 1
    hi = path.max() / c[i] - 1
    print(f'{k:>3}{str(dts.iloc[i].date()):<13}{c[i]:>9.2f}{d["real"].iloc[i]:>8.0f}'
          f'{ra[i]-1:>+7.1%}{ret250[i]:>+9.1%}{fwd250[i]:>+9.1%}{fwd500[i]:>+9.1%}'
          f'  途中最低{lo:+.1%} 最高{hi:+.1%}')
if len(evs) >= 3:
    r = np.array([fwd250[i] for i in evs])
    print(f'\n250 日结果：中位 {np.median(r)*100:+.1f}%  最差 {r.min()*100:+.1f}%  '
          f'最好 {r.max()*100:+.1f}%  全部为正 {bool((r>0).all())}')
    print(f'途中最大浮亏：中位 {np.median([c[i:min(i+251,n)].min()/c[i]-1 for i in evs])*100:+.1f}%  '
          f'最深 {min([c[i:min(i+251,n)].min()/c[i]-1 for i in evs])*100:+.1f}%')

print()
print('=' * 110)
print('② 底部止损为什么把 1997-2010 打到 1.61%：逐笔看骨架每次买入后先跌多少')
print('=' * 110)
sig = B['sig']
print(f'{"#":>3}{"买入日":<13}{"买价":>9}{"250日内最低":>12}{"最低点距买价":>13}'
      f'{"触发-8%?":>10}{"250日后":>9}{"若-8%止损卖出价":>16}{"少赚":>9}')
lost = []
for k, i in enumerate(sig, 1):
    seg = c[i:min(i + 251, n)]
    lo = seg.min() / c[i] - 1
    # 首次跌破 -8% 的日期与价格
    hit = None
    for j in range(i, min(i + 251, n)):
        if c[j] / c[i] - 1 <= -0.08:
            hit = j
            break
    r250 = fwd250[i]
    if hit is not None:
        px_stop = c[hit]
        miss = (c[min(i + 250, n - 1)] / px_stop - 1)
        lost.append((str(dts.iloc[i].date()), lo, r250, miss))
        print(f'{k:>3}{str(dts.iloc[i].date()):<13}{c[i]:>9.2f}{seg.min():>12.2f}{lo:>+12.1%}'
              f'{"是 "+str(dts.iloc[hit].date()):>10}{r250:>+9.1%}{px_stop:>16.2f}{miss:>+9.1%}')
    else:
        print(f'{k:>3}{str(dts.iloc[i].date()):<13}{c[i]:>9.2f}{seg.min():>12.2f}{lo:>+12.1%}'
              f'{"否":>10}{r250:>+9.1%}{"—":>16}{"—":>9}')
if lost:
    print(f'\n被 -8% 止损打掉的 {len(lost)} 次，止损后到第 250 日平均还涨 '
          f'{np.mean([x[3] for x in lost])*100:+.1f}%（这就是止损的代价）')

print()
print('=' * 110)
print('③ 趋势维度有没有用：同一口径下的对照（决策日层面）')
print('=' * 110)


def q(var, labels, name, mask=None):
    ok = np.isfinite(var) & np.isfinite(fwd250)
    if mask is not None:
        ok = ok & mask
    if ok.sum() < 300:
        return
    qs = np.percentile(var[ok], [20, 40, 60, 80])
    edges = [-9e9] + list(qs) + [9e9]
    for b, lab in enumerate(labels):
        m = ok & (var >= edges[b]) & (var < edges[b + 1])
        y = fwd250[m]
        if len(y) < 50:
            continue
        print(f'{name:<12}{lab:<16}{len(y):>6}{np.median(y)*100:>9.2f}{(y>0).mean()*100:>8.1f}'
              f'{np.percentile(y,10)*100:>9.2f}')


print(f'{"维度":<12}{"档":<16}{"n":>6}{"中位%":>9}{"胜率%":>8}{"p10%":>9}')
print(f'{"—基准—":<12}{"全部":<16}{int(np.isfinite(fwd250).sum()):>6}'
      f'{np.median(fwd250[np.isfinite(fwd250)])*100:>9.2f}'
      f'{(fwd250[np.isfinite(fwd250)]>0).mean()*100:>8.1f}'
      f'{np.percentile(fwd250[np.isfinite(fwd250)],10)*100:>9.2f}')
q(np.where(np.isnan(ma250), np.nan, (c > ma250).astype(float)), ['MA250下方', 'MA250上方'], '趋势位置')
q(ma_slope.values, ['斜率最低20%', '斜偏低', '斜偏中', '斜偏高', '斜率最高20%'], '趋势斜率')
q(rv20, ['波动最低20%', '波动偏低', '波动中', '波动偏高', '波动最高20%'], '波动状态')
q(ra, ['离底≤1.20', '离底1.2~1.4', '离底1.4~1.7', '离底1.7~2.1', '离底>2.1'], '离底距离')
q(ret250, ['一年跌>20%', '一年跌5~20%', '一年-5~+10%', '一年涨10~35%', '一年涨>35%'], '前一年涨幅')

print()
print('关键对照：同样是「离底≤1.20」，再看趋势位置（两个维度是否独立有效）')
print(f'{"子集":<32}{"n":>6}{"中位%":>9}{"胜率%":>8}{"p10%":>9}')
for lab, m in [('离底≤1.20 且 MA250上方', (ra <= 1.20) & (c > ma250)),
               ('离底≤1.20 且 MA250下方', (ra <= 1.20) & (c <= ma250))]:
    ok = m & np.isfinite(fwd250) & np.isfinite(ra) & np.isfinite(ma250)
    y = fwd250[ok]
    if len(y) > 60:
        print(f'{lab:<32}{len(y):>6}{np.median(y)*100:>9.2f}{(y>0).mean()*100:>8.1f}'
              f'{np.percentile(y,10)*100:>9.2f}')
