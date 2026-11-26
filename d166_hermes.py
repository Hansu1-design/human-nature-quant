# -*- coding: utf-8 -*-
"""
d166 按 Hermes 的判据重做：①锚的敏感性 ②在同档内部拆前一年涨幅 ③改用确定性尺子 ④小样本区间
Hermes 四条批评 + 我的核验
"""
import numpy as np
import pandas as pd
from scipy import stats as stt
from d141_eat_bite import load
from d155_exit2 import build, equity, stats

RF = 0.025
d = load()
c = d['close'].values.astype(float)
n = len(c)
dts = d['d']
yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
c_ser = pd.Series(c)
B = build(d)
ra = B['ra']
real = d['real'].values
ret250 = (c / c_ser.shift(250) - 1).values
fwd250 = (c_ser.shift(-250) / c_ser - 1).values
fwd500 = (c_ser.shift(-500) / c_ser - 1).values

print('=' * 112)
print('① Hermes 说"底 1,880 是用 2026 才知道的常数回测 1997 = 未来函数"')
print('   核验：rA 在 1997-2007 全是 NaN（CPI 从 2008-01 才有），所以 97-10 段实际只有 2008-2010 三年。')
print('   真正的问题是"参数选择偏差"（锚是事后选的中位，但用起来事先固定）→ 扫锚看敏感性')
print('=' * 112)
print(f'{"锚(实际点位)":<14}{"离底≤1.10 的 n":>14}{"中位%":>9}{"胜率%":>8}{"CVaR10%":>10}'
      f'{"<1.10 的交易日占比":>20}')
for anchor in (1600, 1700, 1800, 1880, 2000, 2200):
    r2 = real / anchor
    m = np.isfinite(r2) & (r2 <= 1.10) & np.isfinite(fwd250)
    y = fwd250[m]
    if len(y) < 30:
        print(f'{anchor:<14}{len(y):>14}  样本不足')
        continue
    cvar = y[y <= np.percentile(y, 10)].mean()
    print(f'{anchor:<14}{len(y):>14}{np.median(y)*100:>9.2f}{(y>0).mean()*100:>8.1f}'
          f'{cvar*100:>10.2f}{m.sum()/np.isfinite(r2).sum()*100:>19.1f}%')
print('  → 锚在 1,600~2,200 之间摆动时，"离底≤1.10"这一档的中位/胜率是否稳定，就是参数偏差的度量。')

print()
print('=' * 112)
print('② Hermes 说"清单A+B 比 A 差，可能只是样本被砍，不是 B 无信息"')
print('   → 在 rA 每一档内部，拆前一年涨幅高/低两组（同档内比较，样本量相同量级）')
print('=' * 112)
print(f'{"rA 档":<14}{"前一年涨幅":<14}{"n":>6}{"中位%":>9}{"胜率%":>8}{"p10%":>9}{"CVaR%":>9}')
for lo, hi, lab in [(None, 1.10, '离底≤1.10'), (1.10, 1.35, '1.10~1.35'),
                    (1.35, 1.90, '1.35~1.90'), (1.90, None, '离底>1.90')]:
    base = np.isfinite(ra) & np.isfinite(ret250) & np.isfinite(fwd250)
    base &= (ra <= hi) if hi else True
    base &= (ra > lo) if lo else True
    for rlo, rhi, rl in [(None, -0.05, '年跌>5%'), (-0.05, 0.20, '年-5~+20%'), (0.20, None, '年涨>20%')]:
        m = base.copy()
        m &= (ret250 < rhi) if rhi else True
        m &= (ret250 >= rlo) if rlo else True
        y = fwd250[m]
        if len(y) < 40:
            continue
        cvar = y[y <= np.percentile(y, 10)].mean()
        print(f'{lab:<14}{rl:<14}{len(y):>6}{np.median(y)*100:>9.2f}{(y>0).mean()*100:>8.1f}'
              f'{np.percentile(y,10)*100:>9.2f}{cvar*100:>9.2f}')

print()
print('=' * 112)
print('③ Hermes 说"你用年化判一个以确定性为目标的闸门，尺子错了"')
print('   → 同一批闸门，补上胜率 / CVaR / 收益尾部比（中位÷|CVaR|）')
print('=' * 112)
SEGS = [('1997-01-01', '2010-12-31'), ('2011-01-01', '2016-12-31'), ('2017-01-01', '2026-12-31')]
rf_d = (1 + RF) ** (1 / 252) - 1


def sa(nav, a, b):
    i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
    i1 = min(int(np.searchsorted(dts.values, np.datetime64(b))) - 1, n - 1)
    y = (dts.iloc[i1] - dts.iloc[i0]).days / 365.25
    rr = nav[i1] / nav[i0]
    return (rr ** (1 / y) - 1) * 100 if rr > 0 and y > 0 else np.nan


def gate(cond, name, cost=0.001):
    nav = np.ones(n)
    pos = 0.0
    cash = 1.0
    on = 0
    for i in range(n - 1):
        want = bool(cond[i])
        if want and pos == 0:
            pos = cash * (1 - cost) / c[i + 1]
            cash = 0.0
        elif (not want) and pos > 0:
            cash = pos * c[i + 1] * (1 - cost)
            pos = 0.0
        if pos > 0:
            nav[i + 1] = pos * c[i + 1]
            on += 1
        else:
            cash *= (1 + rf_d)
            nav[i + 1] = cash
    g, dd = stats(nav, yrs)
    m = pd.Series(cond, index=d.index) & pd.Series(np.isfinite(fwd250), index=d.index)
    y = fwd250[m.values]
    if len(y) < 40:
        return None
    cvar = y[y <= np.percentile(y, 10)].mean()
    return dict(name=name, ann=g * 100, dd=dd * 100, cal=g / abs(dd), occ=on / (n - 1) * 100,
                n=len(y), med=np.median(y) * 100, win=(y > 0).mean() * 100,
                cvar=cvar * 100, ratio=np.median(y) / abs(cvar) if cvar < 0 else np.nan,
                s1=sa(nav, *SEGS[0]), s2=sa(nav, *SEGS[1]), s3=sa(nav, *SEGS[2]))


print(f'{"方案":<28}{"年化%":>8}{"回撤%":>8}{"Calmar":>7}{"在场%":>7}'
      f'{"胜率%":>8}{"中位%":>8}{"CVaR%":>8}{"中位/|CVaR|":>12}')
cases = [('无条件买入持有', np.ones(n, bool), 0.0),
         ('清单 离底≤1.35', np.isfinite(ra) & (ra <= 1.35), 0.001),
         ('清单 离底≤1.20', np.isfinite(ra) & (ra <= 1.20), 0.001),
         ('清单 1.20 且年跌>5%', np.isfinite(ra) & (ra <= 1.20) & (ret250 < -0.05), 0.001),
         ('清单 1.35 且年跌>5%', np.isfinite(ra) & (ra <= 1.35) & (ret250 < -0.05), 0.001)]
for nm, cd, cost in cases:
    r = gate(cd, nm, cost)
    if r:
        print(f'{nm:<28}{r["ann"]:>+8.2f}{r["dd"]:>+8.1f}{r["cal"]:>7.3f}{r["occ"]:>7.1f}'
              f'{r["win"]:>8.1f}{r["med"]:>+8.2f}{r["cvar"]:>+8.2f}{r["ratio"]:>12.2f}')

# 骨架：按确定性口径算（用每次买入的事件样本，不是每天）
sig = B['sig']
nav, tr = equity(d, sig, hold=250, cost=0.001)
g, dd = stats(nav, yrs)
ev = np.array([fwd250[i] for i in sig])
cvar = ev[ev <= np.percentile(ev, 10)].mean()
print(f'{"【对照】骨架（11次信号）":<28}{g*100:>+8.2f}{dd*100:>+8.1f}{g/abs(dd):>7.3f}{0:>7.1f}'
      f'{(ev>0).mean()*100:>8.1f}{np.median(ev)*100:>+8.2f}{cvar*100:>+8.2f}'
      f'{np.median(ev)/abs(cvar):>12.2f}')

print()
print('=' * 112)
print('④ 止损：Hermes 说"Calmar 0.125→0.178 是用收益买风险，不是负价值；你用 n=1 判死，标准不一致"')
print('   → 按确定性口径重算止损（胜率、CVaR 用逐笔事件样本）')
print('=' * 112)
print(f'{"方案":<24}{"年化%":>9}{"回撤%":>9}{"Calmar":>8}{"胜率%":>8}(n)"{"最差%":>9}{"CVaR%":>9}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')


def stop_events(sig_list, sp, hold=250):
    """逐笔模拟：返回每笔的实际收益（含止损）"""
    out = []
    for e in sig_list:
        r = None
        for j in range(e, min(e + hold, n - 1)):
            if c[j] / c[e] - 1 <= sp:
                r = c[j] / c[e] - 1 - 0.002
                break
        if r is None:
            r = c[min(e + hold, n - 1)] / c[e] - 1 - 0.002
        out.append(r)
    return np.array(out)


for sp, lab in [(None, '骨架持满250日'), (-0.06, '亏6%走'), (-0.08, '亏8%走'),
                (-0.10, '亏10%走'), (-0.15, '亏15%走'), (-0.20, '亏20%走')]:
    if sp is None:
        nav, tr = equity(d, sig, hold=250, cost=0.001)
        ev = np.array([fwd250[i] for i in sig])
    else:
        x = sp
        fn = (lambda i, e, p, s=x: c[i] / c[e] - 1 <= s)
        nav, tr = equity(d, sig, hold=250, exit_fn=fn, cost=0.001)
        ev = stop_events(sig, sp)
    g, dd = stats(nav, yrs)
    cvar = ev[ev <= np.percentile(ev, 10)].mean()
    print(f'{lab:<24}{g*100:>+9.2f}{dd*100:>+9.1f}{g/abs(dd):>8.3f}'
          f'{(ev>0).mean()*100:>8.1f}({len(ev)}){ev.min()*100:>+9.1f}{cvar*100:>+9.2f}'
          f'{sa(nav,*SEGS[0]):>+9.2f}{sa(nav,*SEGS[1]):>+9.2f}{sa(nav,*SEGS[2]):>+9.2f}')

print()
print('=' * 112)
print('⑤ 当前那一格的校正：Hermes 说"25 格里挑最小，必须报校正后 p"')
print('=' * 112)
fwd_col = pd.Series(fwd250, index=d.index)
grid = []
for lo, hi, lab in [(None, 1.10, '离底≤1.10'), (1.10, 1.35, '1.10~1.35'),
                    (1.35, 1.90, '1.35~1.90'), (1.90, 2.50, '1.90~2.50'), (2.50, None, '>2.50')]:
    for rlo, rhi, rl in [(None, -0.25, '年跌>25%'), (-0.25, 0.12, '年-25~+12%'),
                         (0.12, 0.40, '年涨12~40%'), (0.40, None, '年涨>40%')]:
        base = np.isfinite(ra) & np.isfinite(ret250) & np.isfinite(fwd250)
        base &= (ra <= hi) if hi else True
        base &= (ra > lo) if lo else True
        base &= (ret250 < rhi) if rhi else True
        base &= (ret250 >= rlo) if rlo else True
        y = fwd250[base]
        if len(y) < 50:
            continue
        grid.append((f'{lab} × {rl}', len(y), np.median(y) * 100, (y > 0).mean() * 100))
grid.sort(key=lambda x: x[3])
print(f'{"格子":<34}{"n":>6}{"中位%":>9}{"胜率%":>8}{"排序":>5}')
for k, (nm, m, md, w) in enumerate(grid, 1):
    print(f'{nm:<34}{m:>6}{md:>9.2f}{w:>8.1f}{k:>5}')
print(f'\n共 {len(grid)} 个格子。当前落在「离底1.35~1.90 × 年-25~+12%」，是胜率第 {[g[0] for g in grid].index("1.35~1.90 × 年-25~+12%")+1} 低。')
print(f'Bonferroni 校正：若单格 p=0.059（等价于胜率 5.9% 相对基准 51.9%），×{len(grid)} → p×{len(grid)}，远超 0.05。')
print('→ 单格读数只能当"状态描述"，不能当"显著信号"。')

# 小样本区间
print()
print('=' * 112)
print('⑥ 小样本区间（Clopper-Pearson 95% 下界）：Hermes 指出 3/3 全对 = 真实胜率 ≥37%')
print('=' * 112)
for k, N in [(3, 3), (10, 10), (11, 9), (11, 7)]:
    lo = stt.beta.ppf(0.025, k, N - k + 1) if k < N else stt.beta.ppf(0.025, N, 1)
    print(f'  {k}/{N} 全对 → 真实胜率 95% 下界 {lo*100:.1f}%（点估计 {k/N*100:.0f}%）')
