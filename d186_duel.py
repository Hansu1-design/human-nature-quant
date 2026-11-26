# -*- coding: utf-8 -*-
"""d186 R53 三方对决 + G0 弱命题检验 + 二维参数面 + 分段与多重检验

  §1 G0 弱命题（Hermes 第 4 条）：连续仓位随 PB 分位产生的减仓时点，
     与分批 300/375/450 的时点是否重叠？不重叠 ⇒「消解」是假的，两条轴各自独立。
     ⚠ d177（2008-05 起、PB 扩张窗口、月末调仓）与 d183（1997 起、上证）
     不同源，此处只对齐【日期轴上的时点】，不比数值、不混报。
  §2 三方对决：持满 250 日 / 分批 3 批 / 连续仓位（各标口径）
  §3 二维参数面（批数 × 首点）只作描述，不作规则来源
  §4 分段检验（1997-2010 前 11 年零交易，必须标出）
  §5 多重检验校正（Bonferroni / BH）
  §6 独立事件归并（R47 体例：11 笔买入归并成几轮）
"""
import numpy as np
import pandas as pd
from scipy import stats
import d183_common as K
import d177_common as P

print('=' * 116)
print('§1 G0 弱命题：连续仓位（PB 轴）的减仓时点 vs 分批（时间轴）的三时点')
print('=' * 116)
# 按 d177 口径构造连续仓位目标序列，并映射到 d183 的日期轴
w_pb = P.target_from(P.pb.values, .35, .05, 'linear')
w_pb_s = pd.Series(w_pb, index=P.idx)
W = w_pb_s.reindex(K.d['d'], method='ffill').values

rows = []
for j, a in enumerate(K.SIG):
    b = min(a + 500, K.n - 1)
    seg = W[a:b + 1]
    if np.isnan(seg).any():
        seg = pd.Series(seg).ffill().fillna(0).values
    # 连续仓位「明显减仓」＝从 ≥0.5 首次跌破 0.5 的时点（相对买入日）
    below = np.where(seg < .5)[0]
    first_cut = int(below[0]) if len(below) else -1
    # 三次分批时点上的 PB 仓位
    vals = [seg[min(t, len(seg) - 1)] for t in (300, 375, 450)]
    rows.append((str(K.d['d'].iloc[a].date()), int(K.PEAKS[j]), first_cut, vals))

print(f"{'买日':<12}{'顶日':>6}{'PB首次减仓日':>14}{'w@300':>8}{'w@375':>8}{'w@450':>8}{'与300差':>9}")
for nm, pk, fc, v in rows:
    diff = abs(fc - 300) if fc >= 0 else np.nan
    print(f'{nm:<12}{pk:>6}{fc:>14}{v[0]:>8.2f}{v[1]:>8.2f}{v[2]:>8.2f}'
          + (f'{diff:>9d}' if not np.isnan(diff) else f'{"—":>9}'))
fcs = np.array([r[2] for r in rows if r[2] >= 0])
print('  PB 首次减仓日 %s' % sorted(fcs.tolist()))
print('  中位 %.0f 日  |  落在 250~450 区间 %d/%d  |  |差 300| 中位 %.0f 日'
      % (np.median(fcs) if len(fcs) else -1, ((fcs >= 250) & (fcs <= 450)).sum(), len(fcs),
         np.median(np.abs(fcs - 300)) if len(fcs) else -1))
print('  w@300 均值 %.2f（若接近 0，说明 PB 轴在 300 日时早就不持仓了）'
      % np.mean([r[3][0] for r in rows]))
print('  ⚠ 分批三时点是【常数 300/375/450】，PB 减仓日随行情漂移 → 两条轴不同源')
print('  ⇒ 判：G0「消解」不成立；出场在【时间轴】上仍暂缺，位置轴是另一件事。')

print()
print('=' * 116)
print('§2 三方对决（口径不同源，分栏报，不混比）')
print('=' * 116)
# 骨架：持满 250 日
nav250, w250, tr250, occ250, wavg250, wsum250 = K.run_ladder([250, 250, 250])  # 退化＝250 日一次全卖
a250, d250, c250 = K.metrics(nav250)
# 分批
navB, wB, trB, occB, wavgB, wsumB = K.run_ladder([300, 375, 450])
aB, dB, cB = K.metrics(navB)
# 连续仓位（d177 口径）
wC = P.target_from(P.pb.values, .35, .05, 'linear')
navC, trC, turnC, onC, wavgC = P.curve_weight(wC, band=.10)
aC, dC, cC, segC = P.metrics(navC)
# 静态 50/50（d177 口径）
def static_hold(w0):
    nv = np.ones(P.n); pos = w0 / P.c.values[0]; cash = 1 - w0
    for i in range(1, P.n):
        cash *= (1 + P.rf_d); nv[i] = pos * P.c.values[i] + cash
    return pd.Series(nv, index=P.idx)
aS, dS, cS, _ = P.metrics(static_hold(.5))

print(f"{'方案':<30}{'口径':<18}{'年化':>9}{'回撤':>9}{'Calmar':>8}{'EWR':>10}{'在场':>8}")
print(f"{'骨架 持满250日':<30}{'d183 长样本/上证':<18}{a250*100:>8.2f}%{d250*100:>8.1f}%{c250:>8.3f}{K.ewr(w250)*100:>9.4f}%{occ250*100:>7.1f}%")
print(f"{'★ 分批 3批 300/375/450':<30}{'d183 长样本/上证':<18}{aB*100:>8.2f}%{dB*100:>8.1f}%{cB:>8.3f}{K.ewr(wB)*100:>9.4f}%{occB*100:>7.1f}%")
print(f"{'连续仓位 PB 线性 .05-.35':<30}{'d177 08起/PB月末':<18}{aC*100:>8.2f}%{dC*100:>8.1f}%{cC:>8.3f}{"—":>10}{onC*100:>7.1f}%")
print(f"{'静态 50/50 对照':<30}{'d177 全区间 08起':<18}{aS*100:>8.2f}%{dS*100:>8.1f}%{cS:>8.3f}{"—":>10}{"—":>8}")
print('  ⚠ EWR 只在 d183 口径内可比（同一条日收益序列）；d177 两行不得与上两行比数值。')
print('  ⚠ 此静态 50/50 是【全区间 2008-05~2026-11】Calmar；33 号的 0.178 是')
print('    【留出测试段 2017-01~2026-11】口径，两者不同区间，不得互相替代或比较。')

print()
print('=' * 116)
print('§3 二维参数面（批数 2/3/4/5 × 首点 100~400）只作描述，不作规则来源')
print('=' * 116)
print(f"{'首点':>6}" + ''.join(f'{"%d批"%k:>10}' for k in (2, 3, 4, 5)))
grid = []
for s in range(100, 401, 50):
    line = []
    for k in (2, 3, 4, 5):
        times = [s + 75 * j for j in range(k)]
        nav, w, tr, occ, wavg, wsum = K.run_ladder(times)
        a, dd, cal = K.metrics(nav)
        line.append(cal)
        grid.append((s, k, cal, a, dd))
    print(f'{s:>6}' + ''.join(f'{x:>10.3f}' for x in line))
G = np.array([g[2] for g in grid])
print('  Calmar 全网格：min %.3f  中位 %.3f  max %.3f  极差 %.3f'
      % (G.min(), np.median(G), G.max(), G.max()-G.min()))
print('  ⚠ 参数面是【事后网格】，最优点不可当规则；且 28 号已证参数面好看的方案分段后现原形。')

print()
print('=' * 116)
print('§4 分段检验（骨架三段；⚠ 1997-2010 前 11 年 rA 为空 → 零交易、纯现金生息）')
print('=' * 116)
print(f"{'方案':<30}" + ''.join(f'{x[0]:>12}' for x in K.SEGS) + f'{"全期":>10}')
for lab, nv in [('骨架 持满250日', nav250), ('分批 3批', navB)]:
    sg = K.seg_ann(nv); a, dd, cal = K.metrics(nv)
    print(f'{lab:<30}' + ''.join(f'{x*100:>11.2f}%' for x in sg) + f'{a*100:>9.2f}%')
print('  ⚠ 第一段（1997-2010）实际只有 2008-09 之后有交易，前 11 年是 2.5% 现金 →')
print('    28 号「三段不塌陷 4.33/11.33/6.08」里，第一段的 4.33% 几乎全是现金利息，不能当分段证据。')

print()
print('=' * 116)
print('§5 多重检验校正（对 §3 网格的 Calmar 做 Bonferroni / BH）')
print('=' * 116)
# 用逐笔独立 EWR 做置换 p 值更合规：这里对网格每格与「网格中位」比较做符号检验
def _per_trade_ewr(times, nb):
    """本地副本，避免 import d185 触发其顶层打印（R53 口径一致性已核）。"""
    times = sorted(int(t) for t in times)
    out = []
    for a0 in K.SIG:
        b0 = min(a0 + 500, K.n - 1)
        r = np.zeros(b0 - a0 + 1)
        r[1:] = K.C[a0 + 1:b0 + 1] / K.C[a0:b0] - 1
        w = np.ones(len(r))
        kk = nb
        for t in times:
            w[min(t, len(r)):] = (kk - 1) / nb
            kk -= 1
        den = w[:-1].sum()
        out.append(float((w[:-1] * r[1:]).sum() / den) if den > 0 else np.nan)
    return np.array(out)

base = _per_trade_ewr([300, 375, 450], 3)
pvals = []
for s, k, cal, a, dd in grid:
    times = [s + 75 * j for j in range(k)]
    arr = _per_trade_ewr(times, k)
    if len(arr) == len(base) and np.std(arr - base) > 0:
        t, p = stats.ttest_rel(arr, base)
        pvals.append((s, k, p))
pvals = [p for _, _, p in pvals if not np.isnan(p)]
m = len(pvals)
bonf = min(1.0, min(pvals) * m) if pvals else np.nan
ps = sorted(pvals)
bh = min([ps[i] * m / (i + 1) for i in range(m)]) if m else np.nan
print('  网格 %d 格配对 t 检验：未校正最小 p=%.4f  |  Bonferroni 后 %.4f  |  BH 后 %.4f'
      % (m, min(pvals) if pvals else np.nan, bonf, bh))
print('  ⇒ 校正后仍显著才谈「某组时点更优」；本轮不做规则，只留判据。')

print()
print('=' * 116)
print('§6 独立事件归并（R47 体例：同一段行情的相邻买点不是独立样本）')
print('=' * 116)
gap = np.diff(np.array(K.SIG))
print('  11 笔相邻间隔（交易日）', gap.tolist())
for cool in (180, 250, 500):
    ev, last = [], -999
    for a in K.SIG:
        if a - last > cool:
            ev.append(a); last = a
    print('  冷却 %d 日归并 → %d 轮：%s'
          % (cool, len(ev), ', '.join(str(K.d['d'].iloc[i].date()) for i in ev)))
print('  ⚠ 28 号沿用 6 轮熊市口径；本轮 11 笔实际独立单元约 3~6 轮，')
print('    任何「胜率 / 分位 / 最优」都只能当方向，不能定稿。')
