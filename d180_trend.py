# -*- coding: utf-8 -*-
"""d180 R52 收尾三验
Q5 单边趋势段：旋钮是否系统性跑输买入持有（Hermes C 必做）
Q6 去掉 2008-2010 段后 rA 的优势还在不在（防 n=1 再犯）
Q7 四档跳跃 vs 连续 clip：与先生④是替代关系，二选一
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

# ---- 构造若干策略的 nav ----
def nv_w(x, hi, lo, sh, band=0.10):
    return K.curve_weight(K.target_from(x, hi, lo, sh), band=band)[0]
def nv_b(cond):
    return K.curve_binary(np.asarray(cond, dtype=bool))[0]

S = {
    '持有100%': static_hold(1.0),
    '静态50/50': static_hold(0.5),
    '点估计 PB≤15%': nv_b((K.pb <= .15).fillna(False).values),
    '点估计 rA≤1.05': nv_b((K.rA <= 1.05).fillna(False).values),
    '连续 PB concave': nv_w(K.pb.values, .35, .05, 'concave'),
    '连续 PB linear': nv_w(K.pb.values, .35, .05, 'linear'),
    '连续 PB step4': nv_w(K.pb.values, .35, .05, 'step4'),
    '连续 rA concave': nv_w(K.rA.values, 1.45, 1.00, 'concave'),
    '连续 rA linear': nv_w(K.rA.values, 1.45, 1.00, 'linear'),
    '连续 rA step4': nv_w(K.rA.values, 1.45, 1.00, 'step4'),
}

UPS = [
    ('2008四万亿', '2008-11-04', '2009-08-04'),
    ('2014-15杠杆牛', '2014-03-12', '2015-06-12'),
    ('2019-21核心资产', '2019-01-03', '2021-02-10'),
    ('2024-26修复', '2024-02-05', '2026-05-13'),
]
print('=' * 100)
print('Q5 单边上行段：段内绝对收益（不报年化，短段年化会失真）')
print('=' * 100)
hdr = f"{'策略':<20}" + ''.join(f'{u[0]:>17}' for u in UPS) + f"{'4段均值':>10}{'跑输持有次数':>12}"
print(hdr); print('-' * 100)
for name, nv in S.items():
    vals = []
    for _, a, b in UPS:
        t = nv[a:b]
        vals.append(t.iloc[-1]/t.iloc[0]-1 if len(t) > 20 else np.nan)
    base = [ (nv[a:b].iloc[-1]/nv[a:b].iloc[0]-1) for _, a, b in UPS ]
    lose = sum(1 for x, y in zip(vals, base) if x < y) if name != '持有100%' else 0
    # 与"持有100%"比
    hold = [ (S['持有100%'][a:b].iloc[-1]/S['持有100%'][a:b].iloc[0]-1) for _, a, b in UPS ]
    lose2 = sum(1 for x, y in zip(vals, hold) if x < y)
    print(f'{name:<20}' + ''.join(f'{x*100:>16.1f}%' for x in vals)
          + f'{np.nanmean(vals)*100:>9.1f}%{lose2:>9d}/4')

print()
print('判读：若旋钮在多数上行段跑输 100% 持有 ⇒ 其收益依赖"市场会回来"（震荡市再平衡），')
print('      不是风险管理本身。这与先生「现象不是会突然影响股价的因素」并不冲突——旋钮不预测，只调暴露。')

print()
print('=' * 100)
print('Q6 去掉 2008-2010 后，rA 的优势还在吗（防 n=1）')
print('=' * 100)
def met2(nv, a, b='2026-12-31'):
    s = nv[a:b]
    ann = (s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1
    dd = (s/s.cummax()-1).min()
    return ann, dd, ann/abs(dd)
print(f"{'策略':<20}{'08起 Calmar':>13}{'11起 Calmar':>13}{'17起 Calmar':>13}{'08-10段年化':>13}")
print('-' * 100)
for name in ['点估计 PB≤15%', '点估计 rA≤1.05', '连续 PB concave', '连续 rA concave',
             '连续 PB linear', '连续 rA linear', '持有100%', '静态50/50']:
    nv = S[name]
    c0 = met2(nv, '2008-05-01')[2]; c1 = met2(nv, '2011-01-01')[2]; c2 = met2(nv, '2017-01-01')[2]
    s0 = met2(nv, '2008-05-01', '2010-12-31')[0]*100
    print(f'{name:<20}{c0:>13.3f}{c1:>13.3f}{c2:>13.3f}{s0:>12.1f}%')

print()
print('=' * 100)
print('Q7 四档跳跃 vs 连续 clip（先生④「首笔1/4、分4档」与连续函数是替代关系）')
print('=' * 100)
print(f"{'形状':<14}{'Calmar':>9}{'回撤':>9}{'换手':>7}{'均仓':>8}{'三段年化':>28}")
print('-' * 100)
for sh in ['linear', 'concave', 'convex', 'step4']:
    nv, tr, turn, on, wavg = K.curve_weight(K.target_from(K.pb.values, .30, .05, sh), band=0.10)
    ann, dd, cal, seg = K.metrics(nv)
    print(f'PB {sh:<10}{cal:>9.3f}{dd*100:>8.1f}%{tr:>7}{wavg*100:>7.1f}%'
          + ''.join(f'{x*100:>9.1f}%' for x in seg))
print('-' * 100)
for sh in ['linear', 'concave', 'convex', 'step4']:
    nv, tr, turn, on, wavg = K.curve_weight(K.target_from(K.rA.values, 1.30, 1.00, sh), band=0.10)
    ann, dd, cal, seg = K.metrics(nv)
    print(f'rA {sh:<10}{cal:>9.3f}{dd*100:>8.1f}%{tr:>7}{wavg*100:>7.1f}%'
          + ''.join(f'{x*100:>9.1f}%' for x in seg))

print()
print('=' * 100)
print('Q8 最终推荐配置（不挑最优点，取形状稳健 + 阈值保守的组合）')
print('=' * 100)
for lab, x, hi, lo, sh in [
    ('PB concave hi=.30', K.pb.values, .30, .05, 'concave'),
    ('PB concave hi=.35', K.pb.values, .35, .05, 'concave'),
    ('rA concave hi=1.30', K.rA.values, 1.30, 1.00, 'concave'),
    ('rA concave hi=1.45', K.rA.values, 1.45, 1.00, 'concave'),
]:
    nv, tr, turn, on, wavg = K.curve_weight(K.target_from(x, hi, lo, sh), band=0.15)
    ann, dd, cal, seg = K.metrics(nv)
    print(f'  {lab:<20} Calmar {cal:.3f}  年化 {ann*100:.2f}%  回撤 {dd*100:.1f}%  '
          f'换手 {tr}  均仓 {wavg*100:.0f}%  三段 '
          + '/'.join(f'{v*100:.1f}%' for v in seg))
