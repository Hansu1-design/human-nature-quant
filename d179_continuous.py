# -*- coding: utf-8 -*-
"""d179 R52 主体：闸门从「点」改成「函数」（连续仓位）

按 Hermes R52 意见设计：
  - 统一月末决策 + 次月首日成交（消除 rA 日频 / PB 季频的信息到达节奏差异）
  - 决定性对照：永远 50% 暴露的买入持有（旋钮平均仓位≈50%，打不过它就没价值）
  - 显式扫 th_hi（rA 无上界，是隐藏自由参数）
  - band 与仓位间距合并扫；形状含四档跳跃（与先生④是替代关系，需二选一）
  - 报在场时间占比、平均仓位、三段年化
  - 0.1% 单边为【保守成本】：510300 实盘双边约 0.04~0.10%，实盘应更低
"""
import numpy as np, pandas as pd
import d177_common as K


def static_hold(w0):
    """静态买入持有，不再平衡：初始 w0 仓位，之后股数固定、现金计息。"""
    nv = np.ones(K.n); pos = w0 / K.c.values[0]; cash = 1 - w0
    for i in range(1, K.n):
        cash *= (1 + K.rf_d)
        nv[i] = pos * K.c.values[i] + cash
    return pd.Series(nv, index=K.idx)


def row(label, nv, tr=None, turn=None, on=None, wavg=None):
    ann, dd, cal, seg = K.metrics(nv)
    f = lambda x: ('%7.2f%%' % (x*100)) if not np.isnan(x) else '      -'
    extra = ''
    if tr is not None:
        extra = '%6d%9.1f%%%7.1f%%' % (tr, on*100, wavg*100)
    print(f'{label:<26}{ann*100:>8.2f}%{dd*100:>8.1f}%{cal:>8.3f}  |'
          + ''.join(f'{x*100:>7.1f}%' if not np.isnan(x) else '      -' for x in seg) + '  |' + extra)
    return cal, ann, dd


def runw(label, w, band=0.0, mh=0):
    nv, tr, turn, on, wavg = K.curve_weight(w, band=band, min_hold_months=mh)
    return row(label, nv, tr, turn, on, wavg)


HDR = (f"{'策略':<26}{'年化':>9}{'回撤':>9}{'Calmar':>8}  |{'08-10':>8}{'11-16':>8}{'17-26':>8}"
       f"  |{'换手':>6}{'在场%':>9}{'均仓':>7}")
LINE = '-' * 118

print('=' * 118)
print('§1 基准对照组（同口径重算，判据阈值必须从这里取，不能引用 d171 的日频数字）')
print('=' * 118)
print(HDR); print(LINE)
row('全程持有 100%', static_hold(1.0))
row('★ 静态 50/50 买入持有', static_hold(0.5))
row('静态 30/70 买入持有', static_hold(0.3))

# 同口径点估计
pt = {}
for lab, cond in [('PB≤15%', (K.pb <= .15).fillna(False).values),
                  ('PB≤20%', (K.pb <= .20).fillna(False).values),
                  ('rA≤1.05', (K.rA <= 1.05).fillna(False).values),
                  ('rA≤1.20', (K.rA <= 1.20).fillna(False).values)]:
    nv, tr, turn, on, wavg = K.curve_binary(cond)
    pt[lab] = row('点估计 ' + lab, nv, tr, turn, on, wavg)

TH90 = {k: v[0]*0.90 for k, v in pt.items()}
print()
print('判据阈值（点估计 Calmar 的 90%）：', {k: round(v, 3) for k, v in TH90.items()})

print()
print('=' * 118)
print('§2 连续仓位主表：线性、band=0.10（Hermes：band 是仓位间距，须与扫描合并，此处仅为展示行）')
print('=' * 118)
print(HDR); print(LINE)
CONFIG = [
    ('PB 线性 lo=.05 hi=.35', K.pb.values, .35, .05, 'linear'),
    ('PB 线性 lo=.00 hi=.40', K.pb.values, .40, .00, 'linear'),
    ('rA 线性 lo=1.00 hi=1.45', K.rA.values, 1.45, 1.00, 'linear'),
    ('rA 线性 lo=1.05 hi=1.60', K.rA.values, 1.60, 1.05, 'linear'),
]
res = {}
for lab, x, hi, lo, sh in CONFIG:
    w = K.target_from(x, hi, lo, sh)
    res[lab] = runw('连续 ' + lab, w, band=0.10)

print()
print('=' * 118)
print('§3 th_hi 显式扫描（rA 无上界，这个参数的影响可能大于整个旋钮设计）')
print('=' * 118)
print(HDR); print(LINE)
for hi in [1.30, 1.45, 1.60, 2.00, 3.00]:
    w = K.target_from(K.rA.values, hi, 1.00, 'linear')
    runw(f'rA hi={hi:.2f} lo=1.00', w, band=0.10)
print(LINE)
for hi in [.25, .35, .50, .70, 1.00]:
    w = K.target_from(K.pb.values, hi, .05, 'linear')
    runw(f'PB hi={hi:.2f} lo=.05', w, band=0.10)

print()
print('=' * 118)
print('§4 形状扫描（step4 四档跳跃 vs clip 连续：与先生④是替代关系，只能留一条）')
print('=' * 118)
print(HDR); print(LINE)
for sh in ['linear', 'concave', 'convex', 'step4']:
    w = K.target_from(K.pb.values, .35, .05, sh)
    runw(f'PB {sh}', w, band=0.10)
print(LINE)
for sh in ['linear', 'concave', 'convex', 'step4']:
    w = K.target_from(K.rA.values, 1.45, 1.00, sh)
    runw(f'rA {sh}', w, band=0.10)

print()
print('=' * 118)
print('§5 band / 仓位间距扫描（Hermes：这是同一件事，合并做）')
print('=' * 118)
print(HDR); print(LINE)
for band in [0.0, .05, .10, .15, .20, .25, .33]:
    w = K.target_from(K.pb.values, .35, .05, 'linear')
    runw(f'PB band={band:.2f}', w, band=band)

print()
print('=' * 118)
print('§6 平台稳健性：th_hi × 形状 全网格（不挑最优点，看整片区域）')
print('=' * 118)
grid = []
for hi in [.30, .35, .40, .50]:
    for sh in ['linear', 'concave', 'convex', 'step4']:
        w = K.target_from(K.pb.values, hi, .05, sh)
        nv, tr, turn, on, wavg = K.curve_weight(w, band=0.10)
        ann, dd, cal, seg = K.metrics(nv)
        grid.append((hi, sh, cal, ann, dd, tr, on, seg))
G = pd.DataFrame(grid, columns=['hi', 'shape', 'cal', 'ann', 'dd', 'tr', 'on', 'seg'])
piv = G.pivot(index='hi', columns='shape', values='cal').round(3)
print('Calmar 网格（行 th_hi、列形状）：')
print(piv.to_string())
print()
print(f'  全网格 Calmar: min {G.cal.min():.3f}  中位 {G.cal.median():.3f}  max {G.cal.max():.3f}  '
      f'达标(≥PB点估计90%={TH90["PB≤15%"]:.3f}) {int((G.cal>=TH90["PB≤15%"]).sum())}/{len(G)}')
print(f'  三段同为正的组合: {int(sum(all(not np.isnan(x) and x>0 for x in s) for s in G.seg))}/{len(G)}')
print(f'  超过静态50/50 Calmar({K.metrics(static_hold(.5))[2]:.3f})的组合: '
      f'{int((G.cal > K.metrics(static_hold(.5))[2]).sum())}/{len(G)}')

print()
print('=' * 118)
print('§7 判决')
print('=' * 118)
best = G.loc[G.cal.idxmax()]
print(f'  连续版最优: hi={best.hi} 形状={best["shape"]}  Calmar={best.cal:.3f}')
print(f'  点估计 PB≤15% Calmar={pt["PB≤15%"][0]:.3f} → 90% 阈值 {TH90["PB≤15%"]:.3f}')
print(f'  静态50/50  Calmar={K.metrics(static_hold(.5))[2]:.3f}')
ok1 = G.cal.median() >= TH90['PB≤15%']
ok2 = G.cal.median() > K.metrics(static_hold(.5))[2]
print(f'  [判据1] 平台中位 ≥ 点估计90%: {ok1}')
print(f'  [判据2] 平台中位 > 静态50/50 : {ok2}')
print('  → 两条都成立：平台为真，永久不再挑阈值；任一不成立：平台是假的，退回点估计并标注过拟合风险。')
