# -*- coding: utf-8 -*-
"""
d112 — 标的错配终局：闸门该建在"上证离CPI底"还是"ETF自身分位"？
d111 已排除两条路：比值偏离不可预测（除 h=500 弱证据）、ETF 前复权价不能当绝对闸门。
本脚本做真正该做的对比：两种闸门对"你要买的那个东西"的未来收益，谁更有预测力。
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------- 数据 ----------
e = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
e['d'] = pd.to_datetime(e['日期'])
e['etf'] = pd.to_numeric(e['收盘'], errors='coerce')
e = e[['d', 'etf']].sort_values('d').reset_index(drop=True)

s = pd.read_parquet(os.path.join(BASE, 'idx_sse_all.parquet'))
s['d'] = pd.to_datetime(s['date'])
s['sse'] = pd.to_numeric(s['close'], errors='coerce')
s = s[['d', 'sse']].sort_values('d').reset_index(drop=True)

# CPI 定基 2008-01=100（'全国-当月' 是同比，开 12 次方当环比）
cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
cpi = cpi.sort_values('d').reset_index(drop=True)
cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
idx = [100.0]
for i in range(1, len(cpi)):
    idx.append(idx[-1] * cpi['mom'].iloc[i])
cpi['cpi_idx'] = idx
cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)

df = e.merge(s, on='d', how='inner').sort_values('d').reset_index(drop=True)
df = pd.merge_asof(df, cpi[['avail', 'cpi_idx']].dropna().sort_values('avail'),
                   left_on='d', right_on='avail', direction='backward')
df = df.dropna(subset=['cpi_idx']).reset_index(drop=True)
CPI_T = df['cpi_idx'].iloc[-1]

# 上证：CPI 不变价 + 离底距离（底 1879，v1.1）
df['sse_real'] = df['sse'] * 100.0 / df['cpi_idx']
df['dist_sse_%'] = (df['sse_real'] / 1879.0 - 1) * 100          # v1.1 口径：距真底累计距离
df['gate_10'] = df['dist_sse_%'] <= 10                          # 出手区
df['gate_20'] = df['dist_sse_%'] <= 20

# ETF 自身分位（无前视：只用到当日）
for w in [750, 1215, 1825]:
    df[f'rank_{w}'] = df['etf'].rolling(w, min_periods=int(w * 0.7)).apply(
        lambda x: (x < x[-1]).mean(), raw=True)

# 未来收益（买入并持有，含分红再投 = 前复权口径正确）
for h in [250, 500]:
    df[f'f{h}'] = (df['etf'].shift(-h) / df['etf'] - 1) * 100

OUT['meta'] = {'n': int(len(df)), 'start': str(df['d'].iloc[0].date()), 'end': str(df['d'].iloc[-1].date()),
               'cpi_T': round(float(CPI_T), 2),
               'gate10_now': round(float(1879 * 1.10 * CPI_T / 100), 0),
               'sse_last': round(float(df['sse'].iloc[-1]), 1),
               'dist_now_%': round(float(df['dist_sse_%'].iloc[-1]), 1),
               'etf_last': round(float(df['etf'].iloc[-1]), 3),
               'rank1215_now': round(float(df['rank_1215'].iloc[-1]), 3)}

# ---------- A. 跟踪质量：ETF 日收益 vs 上证 ----------
df['r_etf'] = df['etf'].pct_change()
df['r_sse'] = df['sse'].pct_change()
m = df['r_etf'].notna() & df['r_sse'].notna()
corr = float(np.corrcoef(df.loc[m, 'r_etf'], df.loc[m, 'r_sse'])[0, 1])
beta = float(np.polyfit(df.loc[m, 'r_sse'], df.loc[m, 'r_etf'], 1)[0])
track = (df.loc[m, 'r_etf'] - df.loc[m, 'r_sse']).abs()
OUT['A_track'] = {'corr_daily': round(corr, 4), 'beta': round(beta, 4),
                  'mean_abs_gap_%/d': round(float(track.mean() * 100), 4),
                  'p95_abs_gap_%': round(float(track.quantile(.95) * 100), 3),
                  'ann_track_err_%': round(float(track.std() * np.sqrt(243) * 100), 2)}

# ---------- B. 两种闸门对"ETF 未来收益"的预测力对比（累计口径分档）----------
def ladder(col, edges, h, label):
    rows = []
    for lo, hi in edges:
        mm = (df[col] >= lo) & (df[col] < hi) & df[f'f{h}'].notna()
        if mm.sum() < 30:
            continue
        rows.append({'gate': label, 'band': f'{lo}~{hi}', 'n': int(mm.sum()),
                     'fwd_%': round(float(df.loc[mm, f'f{h}'].mean()), 1),
                     'win_%': round(float((df.loc[mm, f'f{h}'] > 0).mean() * 100), 1)})
    return rows

OUT['B_ladder'] = []
# 累计口径：≤阈值
for thr in [5, 10, 15, 20, 30, 50]:
    mm = (df['dist_sse_%'] <= thr) & df['f250'].notna()
    if mm.sum() >= 30:
        OUT['B_ladder'].append({'gate': '上证离底(累计≤)', 'band': f'≤{thr}%', 'n': int(mm.sum()),
                                'fwd_%': round(float(df.loc[mm, 'f250'].mean()), 1),
                                'win_%': round(float((df.loc[mm, 'f250'] > 0).mean() * 100), 1)})
# ETF 自身分位
for thr in [0.2, 0.3, 0.5, 0.7, 0.9, 1.01]:
    mm = (df['rank_1215'] <= thr) & df['f250'].notna()
    if mm.sum() >= 30:
        OUT['B_ladder'].append({'gate': 'ETF5年分位(累计≤)', 'band': f'≤{thr}', 'n': int(mm.sum()),
                                'fwd_%': round(float(df.loc[mm, 'f250'].mean()), 1),
                                'win_%': round(float((df.loc[mm, 'f250'] > 0).mean() * 100), 1)})
# 全样本基准
mb = df['f250'].notna()
OUT['B_base'] = {'n': int(mb.sum()), 'fwd_%': round(float(df.loc[mb, 'f250'].mean()), 1),
                 'win_%': round(float((df.loc[mb, 'f250'] > 0).mean() * 100), 1)}

# ---------- C. 四轮底：上证离底 vs ETF 自身分位 ----------
bottoms = ['2012-12-04', '2013-06-25', '2014-03-12', '2016-01-27', '2019-01-03',
           '2022-04-26', '2024-02-05', '2026-09-11']
rows = []
for b in bottoms:
    sub = df[df['d'] <= b]
    if len(sub) == 0:
        continue
    r = sub.iloc[-1]
    rows.append({'d': str(r['d'].date()), 'sse': round(float(r['sse']), 1),
                 'sse_real': round(float(r['sse_real']), 0),
                 'dist_sse_%': round(float(r['dist_sse_%']), 1),
                 'etf': round(float(r['etf']), 3),
                 'rank1215': None if pd.isna(r['rank_1215']) else round(float(r['rank_1215']), 3)})
OUT['C_bottoms'] = rows

# ---------- D. 出手区内 ETF 分位分布（传导是否成立）----------
rows = []
for nm, msk in [('出手区≤10%', df['gate_10']), ('观察区≤20%', df['gate_20']), ('区外>20%', ~df['gate_20'])]:
    mm = msk & df['rank_1215'].notna()
    if mm.sum() < 5:
        rows.append({'zone': nm, 'n': int(mm.sum()), 'note': '样本不足'})
        continue
    rows.append({'zone': nm, 'n': int(mm.sum()),
                 'rank1215_mean': round(float(df.loc[mm, 'rank_1215'].mean()), 3),
                 'rank1215_p90': round(float(df.loc[mm, 'rank_1215'].quantile(.9)), 3),
                 'pct_rank_below_0.5': round(float((df.loc[mm, 'rank_1215'] < 0.5).mean() * 100), 1),
                 'fwd250_%': round(float(df.loc[mm, 'f250'].mean()), 1) if df.loc[mm, 'f250'].notna().sum() > 5 else None})
OUT['D_zone'] = rows

# ---------- E. 双条件：上证离底 ≤10% AND ETF 分位 ≤0.5 ----------
rows = []
for nm, cond in [('仅上证≤10%', df['gate_10']),
                 ('上证≤10% 且 ETF分位≤0.5', df['gate_10'] & (df['rank_1215'] <= 0.5)),
                 ('上证≤10% 且 ETF分位>0.5（冲突）', df['gate_10'] & (df['rank_1215'] > 0.5)),
                 ('仅ETF分位≤0.3', df['rank_1215'] <= 0.3)]:
    mm = cond & df['f250'].notna()
    if mm.sum() < 5:
        rows.append({'name': nm, 'n': int(mm.sum()), 'note': '样本不足'})
        continue
    rows.append({'name': nm, 'n': int(mm.sum()),
                 'days_%': round(float(mm.sum() / df['f250'].notna().sum() * 100), 1),
                 'fwd250_%': round(float(df.loc[mm, 'f250'].mean()), 1),
                 'win_%': round(float((df.loc[mm, 'f250'] > 0).mean() * 100), 1)})
OUT['E_double'] = rows

# ---------- F. 四只可投标的：同一批出手日的表现差异 ----------
etfs = {}
for code, nm in [('510300_沪深300ETF', '510300'), ('510500_中证500ETF', '510500'),
                 ('510880_红利ETF', '510880'), ('512100_中证1000ETF', '512100')]:
    p = f'G:/AStockData/supplement/etf_history/{code}.parquet'
    if not os.path.exists(p):
        continue
    t = pd.read_parquet(p)
    t['d'] = pd.to_datetime(t['日期'])
    t[nm] = pd.to_numeric(t['收盘'], errors='coerce')
    etfs[nm] = t[['d', nm]].sort_values('d').reset_index(drop=True)
    OUT.setdefault('F_cover', {})[nm] = {'start': str(t['d'].min().date()), 'end': str(t['d'].max().date()),
                                         'n': int(len(t))}

wide = df[['d', 'dist_sse_%', 'gate_10']].copy()
for nm, t in etfs.items():
    wide = wide.merge(t, on='d', how='left')
    wide[nm] = wide[nm].ffill()

rows = []
for nm in etfs:
    if wide[nm].isna().all():
        continue
    f = (wide[nm].shift(-250) / wide[nm] - 1) * 100
    for zn, msk in [('出手区≤10%', wide['gate_10']), ('观察区≤20%', wide['dist_sse_%'] <= 20), ('全样本', pd.Series(True, index=wide.index))]:
        mm = msk & f.notna()
        if mm.sum() < 20:
            continue
        rows.append({'etf': nm, 'zone': zn, 'n': int(mm.sum()),
                     'fwd250_%': round(float(f[mm].mean()), 1),
                     'win_%': round(float((f[mm] > 0).mean() * 100), 1)})
OUT['F_etfs'] = rows

df.to_parquet(os.path.join(BASE, 'd112_path.parquet'), index=False)
with open(os.path.join(BASE, 'd112_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print('== meta ==', json.dumps(OUT['meta'], ensure_ascii=False))
print('\n== A 跟踪 ==', OUT['A_track'])
print('\n== B 闸门对比（未来1年 ETF 收益，累计口径）==')
print('  基准', OUT['B_base'])
for r in OUT['B_ladder']:
    print('  ', r)
print('\n== C 关键低点 ==')
for r in OUT['C_bottoms']:
    print('  ', r)
print('\n== D 分区内 ETF 分位 ==')
for r in OUT['D_zone']:
    print('  ', r)
print('\n== E 双条件 ==')
for r in OUT['E_double']:
    print('  ', r)
print('\n== F 四只 ETF ==')
for r in OUT['F_etfs']:
    print('  ', r)
