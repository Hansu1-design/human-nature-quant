# -*- coding: utf-8 -*-
"""d181 R52 补验（Hermes 第二轮五条，全部补上）
Q2'  horizon 扫描：水平项/变化项的符号随期限怎么翻转（排掉 chg20 符号矛盾）
      并给 95%CI 与"不可检出"口径，不再写"无信息"
Q9   暴露匹配：控制平均仓位后重比形状（否则"凹形最好"只是"仓位低"）
Q10  留出验证：2008-2016 定参 → 2017-2026 冻结检验（真正的事前/事后分离）
Q11  股东户数：先生「人数下来上来」的字面量，横截面离散度 + 时序波动
"""
import numpy as np, pandas as pd
import d177_common as K

# ============ Q2' ============
print('=' * 108)
print("Q2' 水平项 vs 变化项：符号随 horizon 的翻转（Hermes：chg20 系数为正是动量，不是反转）")
print('=' * 108)
c, cgo = K.c_all, K.CGO_all
D = pd.DataFrame({'c': c, 'level': cgo})
D['chg20'] = cgo - cgo.shift(20)
D['chg60'] = cgo - cgo.shift(60)
for h in [20, 60, 120, 250, 500, 750]:
    D[f'f{h}'] = D['c'].shift(-h)/D['c'] - 1
S = D.dropna()
print(f'样本 n={len(S)}  {S.index[0].date()} → {S.index[-1].date()}')
print()
print(f"{'期限':<8}{'水平项 高低差':>15}{'chg20 高低差':>14}{'β_level':>10}{'t_level':>9}{'β_chg20':>10}{'t_chg20':>9}{'方向':>10}")
print('-' * 108)
for h in [20, 60, 120, 250, 500, 750]:
    def spread(col):
        q = S[col].quantile([.2, .8]).values
        return (S[S[col] >= q[1]][f'f{h}'].median() - S[S[col] <= q[0]][f'f{h}'].median())*100
    X = S[['level', 'chg20']].copy()
    Xs = (X - X.mean())/X.std()
    A = np.column_stack([np.ones(len(Xs)), Xs.values]); y = S[f'f{h}'].values
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A))*(resid@resid)/(len(y)-3))
    t = coef[1:]/se[1:]
    tag = '动量' if spread('chg20') > 0 and spread('level') < 0 else ('反转' if spread('chg20') < 0 else '混合')
    print(f'{h:>3}日  {spread("level"):>13.1f}pp{spread("chg20"):>13.1f}pp'
          f'{coef[1]*100:>9.2f}{t[0]:>9.2f}{coef[2]*100:>10.2f}{t[1]:>9.2f}{tag:>10}')
print()
# 水平项 95%CI（250日）
X = S[['level', 'chg20']].copy(); Xs = (X-X.mean())/X.std()
A = np.column_stack([np.ones(len(Xs)), Xs.values]); y = S['f250'].values
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
resid = y - A@coef
se = np.sqrt(np.diag(np.linalg.inv(A.T@A))*(resid@resid)/(len(y)-3))
lo, hi = (coef[1]-1.96*se[1])*100, (coef[1]+1.96*se[1])*100
print(f'【口径修正】水平项对 f250 的 95%CI = [{lo:+.2f}pp, {hi:+.2f}pp]/1sd（t={coef[1]/se[1]:.2f}）')
print('  → 正确写法：**水平项不可检出**（18 年仅 4~5 个独立周期，SE 极大），**不是「无信息」**。')
print('  → 先生的「稳定分布是现象」：在 CGO 这个载体上**不可检出**；现象是否存在需换载体再测，目前无结论。')

# ============ Q9 ============
print()
print('=' * 108)
print('Q9 暴露匹配：把各形状的平均仓位调到同一水平，再看形状还有没有差别')
print('=' * 108)

def wavg_of(x, hi, lo, sh, band=.10):
    nv, tr, turn, on, w = K.curve_weight(K.target_from(x, hi, lo, sh), band=band)
    return w, K.metrics(nv)

def match_hi(x, lo, sh, target=0.40, his=None):
    if his is None:
        his = np.linspace(lo+0.05, (1.0 if x is K.pb.values else 3.0), 60)
    best = None
    for hi in his:
        w, _ = wavg_of(x, hi, lo, sh)
        if best is None or abs(w-target) < abs(best[1]-target): best = (hi, w)
    return best[0], best[1]

print(f"{'坐标':<5}{'形状':<10}{'匹配hi':>9}{'平均仓位':>10}{'Calmar':>9}{'年化':>9}{'回撤':>9}{'年化/均仓':>11}")
print('-' * 108)
for lab, x, lo in [('PB', K.pb.values, .05), ('rA', K.rA.values, 1.00)]:
    for sh in ['linear', 'concave', 'convex', 'step4']:
        hi, w = match_hi(x, lo, sh, target=.40,
                         his=np.linspace(.15, 1.0, 60) if lab == 'PB' else np.linspace(1.10, 3.0, 60))
        _, (ann, dd, cal, seg) = wavg_of(x, hi, lo, sh)
        print(f'{lab:<5}{sh:<10}{hi:>9.2f}{w*100:>9.1f}%{cal:>9.3f}{ann*100:>8.2f}%{dd*100:>8.1f}%{ann/w*100:>10.2f}%')
print()
print('判读：若匹配暴露后形状差异明显缩小 ⇒ 原先「凹形最好」主要是低暴露的影子。')

# ============ Q10 ============
print()
print('=' * 108)
print('Q10 留出验证：训练段 2008-05~2016-12 选参 → 测试段 2017-01~2026-11 完全冻结')
print('=' * 108)
def segmet(nv, a, b):
    s = nv[a:b]
    ann = (s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1
    dd = (s/s.cummax()-1).min()
    return ann, dd, (ann/abs(dd) if dd < 0 else np.nan)
TR_A, TR_B = '2008-05-01', '2016-12-31'
TE_A, TE_B = '2017-01-01', '2026-12-31'
grid = []
for sh in ['linear', 'concave', 'convex', 'step4']:
    for hi in [.25, .30, .35, .40, .50]:
        nv, tr, turn, on, w = K.curve_weight(K.target_from(K.pb.values, hi, .05, sh), band=.10)
        grid.append((sh, hi, segmet(nv, TR_A, TR_B)[2]))
grid.sort(key=lambda z: -z[2])
print(f'训练段最优 5 组（形状, hi, 训练Calmar）: {[(a,b,round(c,3)) for a,b,c in grid[:5]]}')
pick = grid[0]
print(f'→ 冻结参数: 形状={pick[0]}, hi={pick[1]}, lo=.05, band=.10（只在训练段选过）')
print()
print(f"{'策略':<24}{'训练段Calmar':>14}{'测试段年化':>13}{'测试段回撤':>13}{'测试段Calmar':>14}")
print('-' * 108)
nv, *_ = K.curve_weight(K.target_from(K.pb.values, pick[1], .05, pick[0]), band=.10)
a0, d0, c0 = segmet(nv, TR_A, TR_B); a1, d1, c1 = segmet(nv, TE_A, TE_B)
print(f'{"★ 冻结的连续仓位":<24}{c0:>14.3f}{a1*100:>12.2f}%{d1*100:>12.1f}%{c1:>14.3f}')
for lab, nv2 in [
    ('点估计 PB≤15%', K.curve_binary((K.pb <= .15).fillna(False).values)[0]),
    ('点估计 PB≤20%', K.curve_binary((K.pb <= .20).fillna(False).values)[0]),
    ('连续 PB concave .35', K.curve_weight(K.target_from(K.pb.values, .35, .05, 'concave'), band=.10)[0]),
]:
    c0 = segmet(nv2, TR_A, TR_B)[2]; a1, d1, c1 = segmet(nv2, TE_A, TE_B)
    print(f'{lab:<24}{c0:>14.3f}{a1*100:>12.2f}%{d1*100:>12.1f}%{c1:>14.3f}')
# 静态对照
def static(w0):
    nv = np.ones(K.n); pos = w0/K.c.values[0]; cash = 1-w0
    for i in range(1, K.n):
        cash *= (1+K.rf_d); nv[i] = pos*K.c.values[i]+cash
    return pd.Series(nv, index=K.idx)
for lab, w0 in [('静态 50/50', .5), ('持有 100%', 1.0)]:
    nv2 = static(w0); c0 = segmet(nv2, TR_A, TR_B)[2]; a1, d1, c1 = segmet(nv2, TE_A, TE_B)
    print(f'{lab:<24}{c0:>14.3f}{a1*100:>12.2f}%{d1*100:>12.1f}%{c1:>14.3f}')

# ============ Q11 ============
print()
print('=' * 108)
print('Q11 股东户数：先生「人数下来上来后行为趋于稳定」的字面量（外生、不依赖价格）')
print('=' * 108)
g = pd.read_parquet('G:/AStockData/supplement/youzi_history/gdhs.parquet')
g['q'] = pd.to_datetime(g['统计截止日'].astype(str), format='%Y%m%d', errors='coerce')
g['h'] = pd.to_numeric(g['股东户数-本次'], errors='coerce')
g['chg'] = pd.to_numeric(g['股东户数-增减比例'], errors='coerce')
g['mv'] = pd.to_numeric(g['总市值'], errors='coerce')
g = g.dropna(subset=['q', 'h'])
g = g[(g['h'] > 0) & (g['q'] >= '2013-01-01')]
print(f'覆盖 {g.q.min().date()} → {g.q.max().date()}  记录 {len(g)}  股票 {g["代码"].nunique()}  季度 {g.q.nunique()}')
print()
g['lh'] = np.log(g['h'])
rows = []
for q, s in g.groupby('q'):
    if len(s) < 100: continue
    # 按规模分 3 层，各层内 log 户数的横截面标准差，再取层内加总
    s = s.copy()
    s['bin'] = pd.qcut(s['mv'].rank(), 3, labels=['小', '中', '大'])
    disp = s.groupby('bin', observed=True)['lh'].std().mean()
    rows.append((q, len(s), disp, s['chg'].std(), s['chg'].median()))
R = pd.DataFrame(rows, columns=['q', 'n', 'disp', 'chg_std', 'chg_med']).sort_values('q')
print('按 3 年分段（横截面离散度=各规模层内 log户数 标准差的均值；波动=户数增减比例横截面 std）')
print(f"{'段':<16}{'季数':>6}{'均样本/季':>10}{'横截面离散度':>14}{'户数变动std':>13}{'变动中位':>11}")
for a, b in [('2013','2015'),('2016','2018'),('2019','2021'),('2022','2024'),('2025','2026')]:
    t = R[(R.q >= f'{a}-01-01') & (R.q <= f'{b}-12-31')]
    if len(t) == 0: continue
    print(f'{a}-{b:<11}{len(t):>6}{t.n.mean():>10.0f}{t.disp.mean():>14.3f}{t.chg_std.mean():>13.2f}{t.chg_med.mean():>10.2f}%')
print()
# 时序：个股户数变化率的自相关
piv = g.pivot_table(index='q', columns='代码', values='chg', aggfunc='last')
common = piv.columns[piv.notna().sum() >= 12]
ac = []
for col in common:
    s = piv[col].dropna()
    if len(s) >= 12:
        ac.append(s.autocorr(1))
print(f'个股户数变动率 一阶自相关：n(股票)={len(ac)}  中位 {np.nanmedian(ac):+.3f}  均值 {np.nanmean(ac):+.3f}')
print(f'横截面离散度 首3季均值 {R.disp.head(3).mean():.3f} → 末3季均值 {R.disp.tail(3).mean():.3f}'
      f'  变化 {(R.disp.tail(3).mean()/R.disp.head(3).mean()-1)*100:+.1f}%')
print(f'户数变动 std  首3季均值 {R.chg_std.head(3).mean():.2f} → 末3季均值 {R.chg_std.tail(3).mean():.2f}'
      f'  变化 {(R.chg_std.tail(3).mean()/R.chg_std.head(3).mean()-1)*100:+.1f}%')
print()
print('⚠ 诚实限制（必须随结论一起写）：')
print('  1. 户数下降可能是**散户被机构替代**（人少了≠剩下的人更稳），不能单向解读；')
print('  2. 季度频率、受送转/增发/停牌扰动；2013 年前缺失，仅约 4~5 个牛熊周期；')
print('  3. 这是**描述性检验**，不是因果；两条检验同向才算弱支持，任一反向即作废。')
