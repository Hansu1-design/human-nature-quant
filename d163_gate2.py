# -*- coding: utf-8 -*-
"""
d163 补充：确定性到底买到什么
① 离底闸门 vs 骨架（是同一件事吗）
② 状态组合（离底 × 前一年涨幅）联合分布
③ 成本止损：买入后跌 X% 就走 —— "不符合咱们可以走嘛"的直接检验（与"从高点回落"不同）
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load

RF = 0.025
d = load()
c = d['close'].values.astype(float)
dts = d['d']
n = len(c)
yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
c_ser = pd.Series(c)

ma250 = c_ser.rolling(250).mean().values
rv20 = (c_ser / c_ser.shift(1) - 1).rolling(20).std().values * np.sqrt(252)
dd250 = (c / c_ser.rolling(250).max() - 1).values
ret250 = (c / c_ser.shift(250) - 1).values
rA = d['rA'].values if 'rA' in d.columns else np.full(n, np.nan)
fwd250 = (c_ser.shift(-250) / c_ser.shift(-1) - 1).values

SEGS = [('1997-2010', '1997-01-01', '2010-12-31'),
        ('2011-2016', '2011-01-01', '2016-12-31'),
        ('2017-2026', '2017-01-01', '2026-12-31')]
SEGID = np.array([next((k for k, (nm, a, b) in enumerate(SEGS) if a <= str(dts.iloc[i].date()) <= b), -1)
                  for i in range(n)])


def boot_med(x, L=250, B=2000, seed=11):
    x = np.asarray(x)
    m = len(x)
    if m < L * 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / L))
    out = [np.median(np.concatenate([x[j:j + L] for j in rng.integers(0, m - L + 1, nb)])[:m])
           for _ in range(B)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def stat(mask, label, dim=''):
    y = fwd250[mask & np.isfinite(fwd250)]
    if len(y) < 60:
        return None
    lo, hi = boot_med(y)
    seg = []
    for k in range(3):
        yy = fwd250[mask & (SEGID == k) & np.isfinite(fwd250)]
        seg.append(np.median(yy) * 100 if len(yy) >= 25 else np.nan)
    return dict(dim=dim, lab=label, n=len(y), med=np.median(y) * 100, win=(y > 0).mean() * 100,
                p25=np.percentile(y, 25) * 100, p10=np.percentile(y, 10) * 100,
                lo=lo * 100, hi=hi * 100, s1=seg[0], s2=seg[1], s3=seg[2])


def pr(r):
    if r is None:
        return
    ci = f'[{r["lo"]:+6.2f},{r["hi"]:+6.2f}]'
    print(f'{r["dim"]:<12}{r["lab"]:<22}{r["n"]:>6}{r["med"]:>8.2f}{r["win"]:>7.1f}'
          f'{r["p25"]:>8.2f}{r["p10"]:>8.2f}{ci:>17}{r["s1"]:>8.2f}{r["s2"]:>8.2f}{r["s3"]:>8.2f}')


HDR = (f'{"维度":<12}{"档":<22}{"n":>6}{"中位%":>8}{"胜率%":>7}{"p25%":>8}{"p10%":>8}'
       f'{"95%CI":>17}{"97-10":>8}{"11-16":>8}{"17-26":>8}')

print('=' * 130)
print('① 离底闸门 vs 骨架出手区：是不是同一件事')
print('=' * 130)
print(HDR)
base = np.ones(n, dtype=bool)
pr(stat(base, '全部', '基准'))
for th, lab in [(1.05, '离底≤1.05（骨架出手区）'), (1.10, '离底≤1.10'), (1.20, '离底≤1.20'),
                (1.35, '离底≤1.35'), (1.60, '离底≤1.60')]:
    pr(stat((rA <= th) & np.isfinite(rA), lab, '离底'))
print('\n注：骨架出手线 2,977 / 底 1,880 = 1.583 —— 但 v1.5 的出手线是"实际点位≤2,977"，')
print('    换算成 rA 约 1.58。真正买入（建仓第一笔）发生在 ≤1.583，加仓在 ≤1.512。')

print()
print('=' * 130)
print('② 状态组合：位置 × 前一年涨幅（两者是否独立回答同一件事）')
print('=' * 130)
print(HDR)
retbin = [(None, -0.25, '一年跌>25%'), (-0.25, 0.12, '一年-25~+12%'), (0.12, 0.40, '一年涨12~40%'),
          (0.40, None, '一年涨>40%')]
posbin = [(None, 1.35, '离底≤1.35'), (1.35, 1.90, '离底1.35~1.9'), (1.90, None, '离底>1.9')]
for plo, phi, pl in posbin:
    for rlo, rhi, rl in retbin:
        m = np.isfinite(rA) & np.isfinite(ret250)
        m &= (rA <= phi) if phi else True
        m &= (rA > plo) if plo else True
        m &= (ret250 < rhi) if rhi else True
        m &= (ret250 >= rlo) if rlo else True
        pr(stat(m, f'{pl}', ''))

print()
print('=' * 130)
print('③ 成本止损："买入后没按预期走就走" —— 与"从高点回落"不同的东西')
print('=' * 130)


def sim_stop(stop=None, hold=250, cost=0.001, buy_cond=None, name=''):
    """buy_cond 为 None 时：每天都可买（持满 hold 或止损后空仓 20 日再看）；
       给 buy_cond 时：只在条件满足时买（骨架式）"""
    nav = np.ones(n)
    pos = 0.0
    cash = 1.0
    entry = -1
    px0 = 0.0
    cool = 0
    tr = 0
    rf_d = (1 + RF) ** (1 / 252) - 1
    stops = 0
    for i in range(n):
        if pos > 0:
            nav[i] = pos * c[i]
            # 止损
            if stop is not None and c[i] / px0 - 1 <= stop and i > entry:
                cash = pos * c[i] * (1 - cost)
                pos = 0.0
                cool = 20
                tr += 1
                stops += 1
            elif i - entry >= hold:
                cash = pos * c[i] * (1 - cost)
                pos = 0.0
                tr += 1
        else:
            nav[i] = cash
            if cool > 0:
                cool -= 1
            else:
                ok = (buy_cond is None) or buy_cond[i]
                if ok and i < n - 1:
                    pos = cash * (1 - cost) / c[i]
                    cash = 0.0
                    entry = i
                    px0 = c[i]
                    tr += 1
            if pos == 0:
                cash *= (1 + rf_d)
    g = nav[-1] ** (1 / yrs) - 1
    peak = np.maximum.accumulate(nav)
    dd = float(np.min(nav / peak - 1))
    segv = []
    for nm, a, b in SEGS:
        i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
        i1 = min(int(np.searchsorted(dts.values, np.datetime64(b))) - 1, n - 1)
        y = (dts.iloc[i1] - dts.iloc[i0]).days / 365.25
        rr = nav[i1] / nav[i0]
        segv.append(rr ** (1 / y) - 1 if rr > 0 and y > 0 else np.nan)
    return dict(name=name, ann=g * 100, dd=dd * 100, tr=tr, stops=stops,
                s1=segv[0] * 100, s2=segv[1] * 100, segv3=segv[2] * 100,
                cal=g / abs(dd))


print('【A】骨架式入场（离底≤1.583 才买，持满 250 日）+ 成本止损')
print(f'{"止损线":<14}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"交易":>6}{"止损次数":>9}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
buy = np.isfinite(rA) & (rA <= 1.583)
# 冷却：买入后 250 日内不重复买，靠 cool 机制；这里 buy_cond 用 rA 条件
for sp, lab in [(None, '不止损'), (-0.05, '亏5%走'), (-0.08, '亏8%走'), (-0.10, '亏10%走'),
                (-0.15, '亏15%走'), (-0.20, '亏20%走')]:
    r = sim_stop(stop=sp, hold=250, buy_cond=buy, name=lab)
    print(f'{lab:<14}{r["ann"]:>+9.2f}{r["dd"]:>+11.1f}{r["cal"]:>8.3f}{r["tr"]:>6}{r["stops"]:>9}'
          f'{r["s1"]:>+9.2f}{r["s2"]:>+9.2f}{r["segv3"]:>+9.2f}')

print()
print('【B】对照：不做位置筛选（任何时点都可买），止损有没有用')
print(f'{"止损线":<14}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"交易":>6}{"止损次数":>9}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
for sp, lab in [(None, '不止损'), (-0.10, '亏10%走'), (-0.15, '亏15%走'), (-0.20, '亏20%走')]:
    r = sim_stop(stop=sp, hold=250, buy_cond=None, name=lab)
    print(f'{lab:<14}{r["ann"]:>+9.2f}{r["dd"]:>+11.1f}{r["cal"]:>8.3f}{r["tr"]:>6}{r["stops"]:>9}'
          f'{r["s1"]:>+9.2f}{r["s2"]:>+9.2f}{r["segv3"]:>+9.2f}')

print()
print('【C】止损 + 否决权（前一年涨>40% 不买）组合，看能否补回左尾')
print(f'{"方案":<30}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"交易":>6}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
buy2 = np.isfinite(rA) & (rA <= 1.583) & ~(ret250 > 0.40)
for sp, lab in [(None, '骨架+否决权'), (-0.10, '骨架+否决权+亏10%走'),
                (-0.15, '骨架+否决权+亏15%走'), (-0.20, '骨架+否决权+亏20%走')]:
    r = sim_stop(stop=sp, hold=250, buy_cond=buy2, name=lab)
    print(f'{lab:<30}{r["ann"]:>+9.2f}{r["dd"]:>+11.1f}{r["cal"]:>8.3f}{r["tr"]:>6}'
          f'{r["s1"]:>+9.2f}{r["s2"]:>+9.2f}{r["segv3"]:>+9.2f}')

# 当前读数
print()
print('=' * 130)
print('当前读数')
print('=' * 130)
i = n - 1
print(f'上证 {c[i]:.2f} / 实际 {d["real"].iloc[i]:.0f} / 离底 {rA[i]:+.1%} / '
      f'前一年涨幅 {ret250[i]:+.1%} / 距250日高 {dd250[i]:+.1%} / RV20 {rv20[i]:.3f}')
print(f'MA250 {ma250[i]:.0f} → 价格在 MA250 的 {c[i]/ma250[i]-1:+.1%}')
