# -*- coding: utf-8 -*-
"""
d81: 上证名义/实际/相对货币 的长期增长（多起点，暴露起点偏差）
+ 本地库覆盖率核对
"""
import os, json
import numpy as np
import pandas as pd

ROOT = 'G:/AStockData/supplement'
OUT = {}

sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)

# GDP 平减（2006 起，首年无名义增速）
g = pd.read_parquet(f'{ROOT}/macro/gdp.parquet')
g = g[g['季度'].str.contains('第1-4季度')].copy()
g['year'] = g['季度'].str[:4].astype(int)
g = g[['year', '国内生产总值-绝对值', '国内生产总值-同比增长']].rename(
    columns={'国内生产总值-绝对值': 'gdp', '国内生产总值-同比增长': 'real'})
g = g.sort_values('year').reset_index(drop=True)
g['nom'] = g['gdp'].pct_change() * 100
g['defl'] = ((1 + g['nom'] / 100) / (1 + g['real'] / 100) - 1) * 100
g['defl_idx'] = (1 + g['defl'] / 100).cumprod()
g.loc[g['year'] == 2006, 'defl_idx'] = 1.0
print(g[['year', 'gdp', 'real', 'nom', 'defl', 'defl_idx']].round(2).to_string())

# M2 月末
m2 = pd.read_parquet(f'{ROOT}/macro/money_supply.parquet')
m2['ym'] = m2['月份'].str.replace('年', '-').str.replace('月份', '')
m2['d'] = pd.to_datetime(m2['ym'], format='%Y-%m')
m2 = m2[['d', '货币和准货币(M2)-数量(亿元)']].rename(columns={'货币和准货币(M2)-数量(亿元)': 'm2'})
m2 = m2.sort_values('d').reset_index(drop=True)

# 上证月线末值
s = sse.set_index('d')['close'].resample('ME').last().dropna()
def at(ts):
    x = s[s.index <= pd.Timestamp(ts)]
    return (x.index[-1], float(x.iloc[-1])) if len(x) else (None, None)

print('\n=== 多起点：名义 / 扣平减 / 扣M2 ===')
rows = []
for st in ['2005-06-30', '2008-01-31', '2014-06-30', '2019-01-31', '2024-09-30']:
    d0, v0 = at(st)
    if d0 is None: continue
    d1, v1 = at('2026-11-11')
    yrs = (d1 - d0).days / 365.25
    nom = (v1 / v0) ** (1 / yrs) - 1
    # 平减
    y0, y1 = d0.year, d1.year
    ok = g[(g['year'] >= y0 + 1) & (g['year'] <= min(y1, 2025))]
    if len(ok) >= 2:
        df0 = g.loc[g['year'] == y0, 'defl_idx']
        df1 = g.loc[g['year'] == min(y1, 2025), 'defl_idx']
        if len(df0) and len(df1):
            infl = (float(df1.iloc[0]) / float(df0.iloc[0])) ** (1 / (min(y1, 2025) - y0)) - 1
        else:
            infl = np.nan
    else:
        infl = np.nan
    real = (1 + nom) / (1 + infl) - 1 if not np.isnan(infl) else np.nan
    # M2
    a = m2[m2['d'] <= d0]
    b = m2[m2['d'] <= d1]
    m2c = ((b['m2'].iloc[-1] / a['m2'].iloc[-1]) ** (1 / yrs) - 1) if len(a) and len(b) else np.nan
    rel = (1 + nom) / (1 + m2c) - 1 if not np.isnan(m2c) else np.nan
    rows.append({'start': str(d0.date()), 'v0': round(v0), 'v1': round(v1), 'yrs': round(yrs, 1),
                 'nom_%': round(nom * 100, 2), 'infl_%': round(infl * 100, 2) if not np.isnan(infl) else None,
                 'real_%': round(real * 100, 2) if not np.isnan(real) else None,
                 'm2_%': round(m2c * 100, 2) if not np.isnan(m2c) else None,
                 'rel_%': round(rel * 100, 2) if not np.isnan(rel) else None})
r = pd.DataFrame(rows)
print(r.to_string(index=False))
OUT['multi_start'] = rows

# 同期 M2 自身
a = m2[m2['d'] <= pd.Timestamp('2008-01-31')]
b = m2[m2['d'] <= pd.Timestamp('2026-11-11')]
print(f'\nM2 {a["m2"].iloc[-1]/1e4:.1f}万亿 -> {b["m2"].iloc[-1]/1e4:.1f}万亿')

# 覆盖率
cov = pd.read_parquet('d80_result.json') if os.path.exists('d80_result.json') else None
import glob
n_local = len([f for f in os.listdir(f'{ROOT}/daily_tencent_full') if f.endswith('.parquet')])
print(f'本地库股票数 {n_local}')
OUT['n_local'] = n_local

json.dump(OUT, open('d81_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved d81_result.json')
