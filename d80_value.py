# -*- coding: utf-8 -*-
"""
d80: A股"真实总价值" —— 证券化率、名义/实际、长期复合回报
只做骨架级：年度末一个点，不针灸。
"""
import os, json
import numpy as np
import pandas as pd

ROOT = 'G:/AStockData/supplement'
OUT = {}

# ---------- 1. 上证年度末 ----------
sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)
sse = sse[sse['d'] >= '2005-01-01'].reset_index(drop=True)

# 每年最后交易日
yr_last = sse.groupby(sse['d'].dt.year)['d'].max()
# 外加几个关键顶底
key = ['2007-10-16', '2008-10-28', '2015-06-12', '2018-12-28', '2024-09-18',
       '2025-12-31', '2026-11-11']
need = sorted(set([pd.Timestamp(x) for x in yr_last.values] +
                  [pd.Timestamp(x) for x in key]))
need_set = {x.strftime('%Y-%m-%d') for x in need}
print('需要日期', len(need_set))

# ---------- 2. 全A总市值 ----------
p = os.path.join(ROOT, 'daily_tencent_full')
fs = [f for f in os.listdir(p) if f.endswith('.parquet')]
print('股票数', len(fs))

tot = {}
cnt = {}
for k, f in enumerate(fs):
    try:
        d = pd.read_parquet(os.path.join(p, f), columns=['date', 'close', 'volume', 'amount', 'outstanding_share'])
    except Exception:
        continue
    if len(d) == 0:
        continue
    d['dstr'] = d['date'].astype(str).str[:10]
    d = d[d['dstr'].isin(need_set)]
    if len(d) == 0:
        continue
    d = d[d['volume'] > 0]
    raw = d['amount'] / d['volume']
    mv = raw * d['outstanding_share']
    for ds, v in zip(d['dstr'], mv):
        tot[ds] = tot.get(ds, 0.0) + v
        cnt[ds] = cnt.get(ds, 0) + 1
    if (k + 1) % 500 == 0:
        print('  ...', k + 1, flush=True)

mvdf = pd.DataFrame({'date': list(tot.keys()), 'mktcap_yi': [v / 1e8 for v in tot.values()],
                     'n_stock': [cnt[k] for k in tot.keys()]})
mvdf['d'] = pd.to_datetime(mvdf['date'])
mvdf = mvdf.sort_values('d').reset_index(drop=True)
print(mvdf.to_string())

# ---------- 3. GDP 年度 ----------
gdp = pd.read_parquet(os.path.join(ROOT, 'macro', 'gdp.parquet'))
g = gdp[gdp['季度'].str.contains('第1-4季度')].copy()
g['year'] = g['季度'].str[:4].astype(int)
g = g[['year', '国内生产总值-绝对值', '国内生产总值-同比增长']].rename(
    columns={'国内生产总值-绝对值': 'gdp_yi', '国内生产总值-同比增长': 'gdp_real_yoy'})
g = g.sort_values('year').reset_index(drop=True)
g['gdp_nom_yoy'] = g['gdp_yi'].pct_change() * 100
g['deflator_yoy'] = ((1 + g['gdp_nom_yoy'] / 100) / (1 + g['gdp_real_yoy'] / 100) - 1) * 100
g['deflator_idx'] = (1 + g['deflator_yoy'] / 100).cumprod()
print(g.tail(6).to_string())

# ---------- 4. 合并：证券化率 ----------
mvdf['year'] = mvdf['d'].dt.year
# 年末点用当年 GDP；月度点用最近已发布年度 GDP
gmap = dict(zip(g['year'], g['gdp_yi']))
mvdf['gdp_yi'] = mvdf['year'].map(gmap)
mvdf['sec_ratio'] = mvdf['mktcap_yi'] / mvdf['gdp_yi'] * 100
print(mvdf[['date', 'mktcap_yi', 'n_stock', 'gdp_yi', 'sec_ratio']].to_string())

OUT['mktcap'] = mvdf[['date', 'mktcap_yi', 'n_stock', 'gdp_yi', 'sec_ratio']].round(3).to_dict('records')

# ---------- 5. 名义 vs 实际 ----------
# 上证年度末
sse_y = sse.loc[sse.groupby(sse['d'].dt.year)['d'].idxmax()]
sse_y = sse_y[['d', 'close']].rename(columns={'close': 'sse'})
sse_y['year'] = sse_y['d'].dt.year
j = sse_y.merge(g[['year', 'deflator_idx', 'gdp_nom_yoy', 'gdp_real_yoy']], on='year', how='left')
j = j.dropna(subset=['deflator_idx'])
base = j.iloc[0]
j['sse_real'] = j['sse'] / j['deflator_idx'] * base['deflator_idx']
n = len(j) - 1
yrs = (j['d'].iloc[-1] - j['d'].iloc[0]).days / 365.25
nom_cagr = (j['sse'].iloc[-1] / j['sse'].iloc[0]) ** (1 / yrs) - 1
real_cagr = (j['sse_real'].iloc[-1] / j['sse_real'].iloc[0]) ** (1 / yrs) - 1
infl = (j['deflator_idx'].iloc[-1] / j['deflator_idx'].iloc[0]) ** (1 / yrs) - 1
print('\n=== 上证名义 vs 实际 ===')
print(f'区间 {j["d"].iloc[0].date()} ~ {j["d"].iloc[-1].date()}  {yrs:.1f}年')
print(f'名义 {j["sse"].iloc[0]:.0f} -> {j["sse"].iloc[-1]:.0f}  CAGR {nom_cagr*100:.2f}%')
print(f'平减 {infl*100:.2f}%/yr  实际 CAGR {real_cagr*100:.2f}%')
print(j[['year', 'sse', 'sse_real', 'deflator_yoy']].round(2).to_string())

OUT['sse_real'] = {
    'span': f'{j["d"].iloc[0].date()}~{j["d"].iloc[-1].date()}',
    'years': round(yrs, 1),
    'nom_cagr': round(nom_cagr * 100, 2),
    'real_cagr': round(real_cagr * 100, 2),
    'deflator_cagr': round(infl * 100, 2),
    'rows': j[['year', 'sse', 'sse_real', 'deflator_yoy']].round(2).to_dict('records'),
}

# ---------- 6. 总市值名义/实际 ----------
m = mvdf.dropna(subset=['gdp_yi']).copy()
m['defl'] = m['year'].map(dict(zip(g['year'], g['deflator_idx'])))
m = m.dropna(subset=['defl'])
m['mktcap_real'] = m['mktcap_yi'] / m['defl'] * m['defl'].iloc[0]
OUT['mktcap_real'] = m[['date', 'mktcap_yi', 'mktcap_real', 'sec_ratio']].round(2).to_dict('records')
print('\n=== 总市值名义/实际 ===')
print(m[['date', 'mktcap_yi', 'mktcap_real', 'sec_ratio']].round(2).to_string())

json.dump(OUT, open('d80_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved d80_result.json')
