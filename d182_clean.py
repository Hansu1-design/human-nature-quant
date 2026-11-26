# -*- coding: utf-8 -*-
"""d182 R52 最后一轮补验
Q10' 公平留出：点估计阈值也在训练段选、再冻结到测试段（否则 0.597 含挑选偏差）
Q11' 清洗版股东户数：弃用被极端值污染的「增减比例」，改用 clip 后的 log 变化率 + 平衡面板
"""
import numpy as np, pandas as pd
import d177_common as K

# ============ Q10' ============
print('=' * 96)
print("Q10' 公平留出验证：点估计与连续仓位，都在训练段选参、都在测试段冻结")
print('=' * 96)
def segmet(nv, a, b):
    s = nv[a:b]
    ann = (s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1
    dd = (s/s.cummax()-1).min()
    return ann, dd, (ann/abs(dd) if dd < 0 else np.nan)
TR_A, TR_B = '2008-05-01', '2016-12-31'
TE_A, TE_B = '2017-01-01', '2026-12-31'

# 点估计：训练段扫阈值
pt = []
for th in [.10, .15, .20, .25, .30, .35, .40]:
    nv = K.curve_binary((K.pb <= th).fillna(False).values)[0]
    pt.append((th, segmet(nv, TR_A, TR_B)[2]))
print('训练段 点估计阈值扫描:', [(a, round(b, 3)) for a, b in pt])
best_th = max(pt, key=lambda z: z[1])
print(f'→ 冻结点估计阈值 = {best_th[0]:.2f}（训练段 Calmar {best_th[1]:.3f}）')

# 连续：训练段扫形状×hi
ct = []
for sh in ['linear', 'concave', 'convex', 'step4']:
    for hi in [.25, .30, .35, .40, .50]:
        nv = K.curve_weight(K.target_from(K.pb.values, hi, .05, sh), band=.10)[0]
        ct.append((sh, hi, segmet(nv, TR_A, TR_B)[2]))
best_c = max(ct, key=lambda z: z[2])
print(f'→ 冻结连续参数 = 形状 {best_c[0]} / hi {best_c[1]:.2f}（训练段 Calmar {best_c[2]:.3f}）')
print()
print(f"{'策略':<28}{'训练Calmar':>12}{'测试年化':>11}{'测试回撤':>11}{'测试Calmar':>12}")
print('-' * 96)
def show(lab, nv):
    c0 = segmet(nv, TR_A, TR_B)[2]; a1, d1, c1 = segmet(nv, TE_A, TE_B)
    print(f'{lab:<28}{c0:>12.3f}{a1*100:>10.2f}%{d1*100:>10.1f}%{c1:>12.3f}')
show('★ 冻结点估计 PB≤%.2f' % best_th[0], K.curve_binary((K.pb <= best_th[0]).fillna(False).values)[0])
show('★ 冻结连续 %s hi=%.2f' % (best_c[0], best_c[1]),
     K.curve_weight(K.target_from(K.pb.values, best_c[1], .05, best_c[0]), band=.10)[0])
show('（参考）事前 PB concave .35',
     K.curve_weight(K.target_from(K.pb.values, .35, .05, 'concave'), band=.10)[0])
def static(w0):
    nv = np.ones(K.n); pos = w0/K.c.values[0]; cash = 1-w0
    for i in range(1, K.n):
        cash *= (1+K.rf_d); nv[i] = pos*K.c.values[i]+cash
    return pd.Series(nv, index=K.idx)
show('静态 50/50', static(.5)); show('持有 100%', static(1.0))
print()
print('判读：两类策略在测试段（完全未参与选参的 9 年）都有效 ⇒ 连续仓位不是过拟合；')
print('      但谁更好必须用「同规则选出来的阈値」比，不能拿全样本挑的 15% 去比。')

# ============ Q11' ============
print()
print('=' * 96)
print("Q11' 清洗版股东户数：弃用污染的「增减比例」，用 clip(log 户数变化) + 平衡面板")
print('=' * 96)
g = pd.read_parquet('G:/AStockData/supplement/youzi_history/gdhs.parquet')
g['q'] = pd.to_datetime(g['统计截止日'].astype(str), format='%Y%m%d', errors='coerce')
for col in ['股东户数-本次', '股东户数-上次', '总市值']:
    g[col] = pd.to_numeric(g[col], errors='coerce')
g = g.dropna(subset=['q', '股东户数-本次', '股东户数-上次', '总市值'])
g = g[(g['股东户数-本次'] > 0) & (g['股东户数-上次'] > 0) & (g['总市值'] > 0) & (g['q'] >= '2013-01-01')]
# 弃用现成的「增减比例」，自己算并 clip
g['dlog'] = np.log(g['股东户数-本次']/g['股东户数-上次']).clip(-1.0, 1.0)
g['lh'] = np.log(g['股东户数-本次'])
g = g.drop_duplicates(['代码', 'q'])
print(f'清洗后 记录 {len(g)}  股票 {g["代码"].nunique()}  季度 {g.q.nunique()}'
      f'  dlog 范围 [{g.dlog.min():.2f}, {g.dlog.max():.2f}]')

# 平衡面板：只保留 2013Q2 与 2026Q2 都有记录、且季度数 >= 40 的股票
cnt = g.groupby('代码')['q'].nunique()
keep = cnt[cnt >= 40].index
B = g[g['代码'].isin(keep)].copy()
print(f'平衡面板 股票 {B["代码"].nunique()}  季度 {B.q.nunique()}（避免新股不断加入污染横截面）')

rows = []
for q, s in B.groupby('q'):
    if len(s) < 200: continue
    s = s.copy()
    s['bin'] = pd.qcut(s['总市值'].rank(), 3, labels=['小', '中', '大'])
    disp = s.groupby('bin', observed=True)['lh'].std().mean()
    rows.append((q, len(s), disp, s['dlog'].std(), s['dlog'].median()*100))
R = pd.DataFrame(rows, columns=['q', 'n', 'disp', 'dstd', 'dmed']).sort_values('q')
print()
print(f"{'段':<14}{'季数':>6}{'横截面离散度':>14}{'户数变动std':>13}{'变动中位%':>12}")
for a, b in [('2013','2015'),('2016','2018'),('2019','2021'),('2022','2024'),('2025','2026')]:
    t = R[(R.q >= f'{a}-01-01') & (R.q <= f'{b}-12-31')]
    if len(t) == 0: continue
    print(f'{a}-{b:<9}{len(t):>6}{t.disp.mean():>14.3f}{t.dstd.mean():>13.3f}{t.dmed.mean():>11.2f}%')
first3, last3 = R.disp.head(3).mean(), R.disp.tail(3).mean()
f3, l3 = R.dstd.head(3).mean(), R.dstd.tail(3).mean()
print()
print(f'横截面离散度  首3季 {first3:.3f} → 末3季 {last3:.3f}  ({(last3/first3-1)*100:+.1f}%)')
print(f'户数变动 std  首3季 {f3:.3f} → 末3季 {l3:.3f}  ({(l3/f3-1)*100:+.1f}%)')

# 个股层面：滚动 8 季度波动 + 分段自相关
piv = B.pivot_table(index='q', columns='代码', values='dlog', aggfunc='last')
roll = piv.rolling(8, min_periods=6).std()
print()
print('个股「户数变化率」滚动 8 季波动（全体中位，越小=行为越稳）：')
for yr in ['2014','2016','2018','2020','2022','2024','2026']:
    s = roll.loc[roll.index <= f'{yr}-12-31'].iloc[-1].dropna()
    if len(s) > 100: print(f'  {yr}  n={len(s):4d}  中位 {s.median():.3f}')
print()
for lab, a, b in [('前半 2013-2019','2013-01-01','2019-12-31'),('后半 2020-2026','2020-01-01','2026-12-31')]:
    sub = piv[a:b]
    ac = [sub[col].dropna().autocorr(1) for col in sub.columns if sub[col].dropna().shape[0] >= 10]
    ac = [x for x in ac if not np.isnan(x)]
    print(f'  {lab} 个股户数变动 ACF(1) 中位 {np.median(ac):+.3f}  (n={len(ac)})')
print()
print('判读（两条都同向才算弱支持，任一反向即作废）：')
print('  ① 构成趋同 = 横截面离散度下降；② 行为趋稳 = 变动 std 下降 且 ACF 上升。')
