# -*- coding: utf-8 -*-
"""
d66 — 收掉 Hermes 提出的三个洞
1) 上证"3000点"能不能跨年比较：上证/沪深300 比值漂移 + 上证点位 vs 全A PE分位 的秩相关
2) 热度出场是否只是"两个顶的事后拟合"：换阈值/换窗口，看还能不能抓到 2018-01 / 2021-02
3) 敞口归一的年化（在场时间口径）
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))

idx = pd.read_parquet(os.path.join(BASE, 'idx_sse_long.parquet'))
idx['d'] = pd.to_datetime(idx['date'])
for c in ['close', 'amount']:
    idx[c] = idx[c].astype(float)
idx = idx.sort_values('d').reset_index(drop=True)

# 沪深300 指数（月度）
hs = pd.read_parquet('G:/AStockData/supplement/valuation/index_pe_沪深300.parquet')
hs['d'] = pd.to_datetime(hs['日期'])
hs = hs[['d', '指数']].rename(columns={'指数': 'hs300'}).sort_values('d')

pe = pd.read_parquet('G:/AStockData/supplement/valuation/all_a_pe_ttm.parquet')
pe['d'] = pd.to_datetime(pe['date'])
pe = pe[['d', 'middlePETTM', 'quantileInRecent10YearsMiddlePeTtm']].rename(
    columns={'middlePETTM': 'pe_ttm', 'quantileInRecent10YearsMiddlePeTtm': 'pe_q10y'})

OUT = {}

# ---- 1a 比值漂移 ----
m = idx.set_index('d')['close'].resample('ME').last().rename('sse').to_frame()
m = m.join(hs.set_index('d')['hs300'], how='inner')
m['ratio'] = m['sse'] / m['hs300']
m = m[m.index >= '2015-01-01']
OUT['ratio_drift'] = {'first': round(float(m['ratio'].iloc[0]), 3), 'last': round(float(m['ratio'].iloc[-1]), 3),
                      'min': round(float(m['ratio'].min()), 3), 'max': round(float(m['ratio'].max()), 3),
                      'drift_%': round(float(m['ratio'].iloc[-1] / m['ratio'].iloc[0] - 1) * 100, 1),
                      'range_%': round(float((m['ratio'].max() / m['ratio'].min() - 1) * 100), 1),
                      'n': int(len(m)),
                      'by_year': {str(k): round(float(v), 3) for k, v in m['ratio'].groupby(m.index.year).mean().items()}}

# ---- 1b 上证点位 vs 全A PE 分位 的秩相关 ----
_m0 = m.reset_index(); _m0['d'] = _m0['d'].astype('datetime64[ns]')
_pe0 = pe.copy(); _pe0['d'] = _pe0['d'].astype('datetime64[ns]')
j = pd.merge_asof(_m0.sort_values('d'), _pe0.sort_values('d'),
                  on='d', direction='nearest', tolerance=pd.Timedelta('25D'))
j = j.set_index('d').dropna(subset=['pe_ttm'])
rho, p = stats.spearmanr(j['sse'], j['pe_ttm'])
rho_q, p_q = stats.spearmanr(j['sse'], j['pe_q10y'])
# 同时段"点位同比变化"vs"PE同比变化"
j['sse_yoy'] = j['sse'].pct_change(12)
j['pe_yoy'] = j['pe_ttm'].pct_change(12)
jj = j.dropna()
rho2, p2 = stats.spearmanr(jj['sse_yoy'], jj['pe_yoy'])
OUT['anchor'] = {'n_months': int(len(j)), 'spearman_sse_vs_peTtm': round(float(rho), 3), 'p': float(p),
                 'spearman_sse_vs_peQ10y': round(float(rho_q), 3), 'p_q': float(p_q),
                 'spearman_yoy': round(float(rho2), 3), 'p_yoy': float(p2),
                 'pe_range': [round(float(j['pe_ttm'].min()), 1), round(float(j['pe_ttm'].max()), 1)],
                 'sse_range': [round(float(j['sse'].min()), 0), round(float(j['sse'].max()), 0)]}

# 同一"3000点"在不同年份对应的全A PE
bands = []
for y0, y1 in [(2015, 2016), (2017, 2018), (2019, 2020), (2021, 2022), (2023, 2024), (2025, 2026)]:
    mm = (j.index.year >= y0) & (j.index.year <= y1)
    sub = j[mm]
    if len(sub) == 0:
        continue
    near = sub[(sub['sse'] > 2900) & (sub['sse'] < 3100)]
    bands.append({'years': f'{y0}-{y1}', 'n': int(len(sub)),
                  'pe_at_3000': round(float(near['pe_ttm'].mean()), 1) if len(near) else None,
                  'pe_avg': round(float(sub['pe_ttm'].mean()), 1)})
OUT['pe_at_3000'] = bands
OUT['hs300_equiv_of_3000'] = {str(k): round(3000 / float(v), 0) for k, v in m['ratio'].groupby(m.index.year).mean().items()}

# ---- 2 热度出场敏感性 ----
etf = pd.read_parquet(os.path.join('G:/AStockData/supplement/etf_history', '510300_沪深300ETF.parquet'))
etf['d'] = pd.to_datetime(etf['日期'])
etf = etf[['d', '收盘']].rename(columns={'收盘': 'etf'}).sort_values('d')
df = idx[['d', 'close', 'amount']].merge(etf, on='d', how='left')
df['etf'] = df['etf'].ffill()
df['ma20'] = df['close'].rolling(20).mean()
df = df[df['d'] >= '2015-01-01'].reset_index(drop=True)
C, E, AMT = df['close'].values, df['etf'].values, df['amount'].values
MA20 = df['ma20'].values
D = df['d'].values
N = len(C)


def heat_series(win):
    s = pd.Series(AMT).rolling(win).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True).values
    return s


def sim_heat(win=250, q=0.95, warm=20, dd=0.10, dd_days=20):
    H = heat_series(win)
    pos, peak, peak_i, below, half = 0.0, 0.0, 0, 0, False
    eps, cur = [], None
    s = np.zeros(N)
    for i in range(N):
        c = C[i]
        below = below + 1 if c < 3000 else 0
        if pos > 0:
            if c > peak:
                peak, peak_i = c, i
            hit, why = False, ''
            if c <= peak * (1 - dd) and (i - peak_i) >= dd_days:
                hit, why = True, 'dd'
            if not np.isnan(H[i]) and H[i] >= q and c > 3500:
                hit, why = True, 'heat'
            if hit:
                cur.update(exit=str(D[i])[:10], ret=round((E[i] / cur['ei'] - 1) * 100, 1), why=why)
                eps.append(cur)
                cur, pos, half = None, 0.0, False
            elif not half and c > 3500:
                pos, half = 0.5, True
        if pos == 0 and i > 60:
            if below >= warm and c < 3000 and c > MA20[i] and C[i - 1] <= MA20[i - 1]:
                pos, half, peak, peak_i = 1.0, False, c, i
                cur = {'entry': str(D[i])[:10], 'ei': E[i]}
        s[i] = pos
    s[1:] = s[:-1] * np.r_[0, np.diff(E) / E[:-1]][1:]
    eq = np.cumprod(1 + s)
    heat_dates = [e['exit'] for e in eps if e['why'] == 'heat']
    return {'total_%': round((eq[-1] - 1) * 100, 1), 'n': len(eps),
            'heat_exits': heat_dates, 'all_exits': [(e['exit'], e['why'], e['ret']) for e in eps]}


sens = []
for win in [120, 250, 500]:
    for q in [0.85, 0.90, 0.95, 0.99]:
        r = sim_heat(win=win, q=q)
        sens.append({'win': win, 'q': q, 'total_%': r['total_%'], 'n': r['n'],
                     'heat_exits': r['heat_exits']})
OUT['heat_sens'] = sens

# ---- 3 敞口归一 ----
def expo_annual(total_pct, in_days):
    return round(((1 + total_pct / 100) ** (243 / in_days) - 1) * 100, 2)


OUT['expo_norm'] = {
    'A_总收益166.3_在场天数': None,
}

with open(os.path.join(BASE, 'd66_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)

print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
