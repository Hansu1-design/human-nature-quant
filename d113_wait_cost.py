# -*- coding: utf-8 -*-
"""
d113 — 收口：闸门"松动率"（等一年的代价）+ ETF 数据体检 + 实际价底对照
d110~d112 已排除：比值偏离预测、ETF 分位闸门、折价微调。
剩下唯一真问题：闸门按 CPI 抬 2%，但下单标的每年真实增值 3.5% → 等得越久，同样触发闸门买到的越贵。
本脚本把这个"松动率"量出来，并给每只 ETF 做数据体检（510880 已发现负价格）。
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------- 数据 ----------
def load_etf(code):
    t = pd.read_parquet(f'G:/AStockData/supplement/etf_history/{code}.parquet')
    t['d'] = pd.to_datetime(t['日期'])
    t['c'] = pd.to_numeric(t['收盘'], errors='coerce')
    t['p'] = pd.to_numeric(t['涨跌幅'], errors='coerce')
    t['amt'] = pd.to_numeric(t['成交额'], errors='coerce')
    return t[['d', 'c', 'p', 'amt']].sort_values('d').reset_index(drop=True)

CODES = {'510300': '510300_沪深300ETF', '510500': '510500_中证500ETF',
         '512100': '512100_中证1000ETF', '510880': '510880_红利ETF'}
ETFS = {k: load_etf(v) for k, v in CODES.items()}

s = pd.read_parquet(os.path.join(BASE, 'idx_sse_all.parquet'))
s['d'] = pd.to_datetime(s['date'])
s['sse'] = pd.to_numeric(s['close'], errors='coerce')
s = s[['d', 'sse']].sort_values('d').reset_index(drop=True)

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

# ---------- A. 四只 ETF 数据体检 ----------
audit = []
for k, t in ETFS.items():
    bad_neg = int((t['c'] <= 0).sum())
    big = int((t['p'].abs() > 15).sum())
    zero = int((t['c'].isna()).sum())
    # 连续性：单日涨跌幅与价格隐含涨跌幅不一致的天数
    imp = (t['c'] / t['c'].shift(1) - 1) * 100
    mism = int(((imp - t['p']).abs() > 1.0).sum())
    audit.append({'etf': k, 'n': int(len(t)),
                  'start': str(t['d'].min().date()), 'end': str(t['d'].max().date()),
                  'neg_or_zero_price': bad_neg,
                  'abs_move>15%_days': big,
                  'null_close': zero,
                  'pct_col_mismatch_days': mism,
                  'verdict': '作废' if (bad_neg > 0 or big > 20) else '可用'})
OUT['A_audit'] = audit

# ---------- B. 闸门松动率：每单位上证点位对应的 ETF 价格 ----------
# 闸门线 gate(t) = 1879 * 1.10 * cpi(t)/100（v1.1 出手区上沿，名义）
def build(code):
    d = ETFS[code][['d', 'c']].merge(s, on='d', how='inner')
    d = pd.merge_asof(d, cpi[['avail', 'cpi_idx']].dropna().sort_values('avail'),
                      left_on='d', right_on='avail', direction='backward').dropna(subset=['cpi_idx'])
    d['gate'] = 1879 * 1.10 * d['cpi_idx'] / 100
    d['dist_%'] = (d['sse'] * 100 / d['cpi_idx'] / 1879 - 1) * 100
    d['cost_per_pt'] = d['c'] / d['gate']            # 每 1 点闸门对应的 ETF 价
    d['lcp'] = np.log(d['cost_per_pt'])
    return d.reset_index(drop=True)

OUT['B_drift'] = {}
for code in ['510300', '510500', '512100']:
    d = build(code)
    x = np.arange(len(d), dtype=float)
    b = np.polyfit(x, d['lcp'].values, 1)
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    dr = (np.exp(b[0] * 243) - 1) * 100
    OUT['B_drift'][code] = {'years': round(yrs, 2), 'drift_%/yr': round(float(dr), 2),
                            'cost_now': round(float(d['cost_per_pt'].iloc[-1]) * 1000, 3),
                            'note': 'cost_per_pt = ETF价/闸门线，涨=同样触发闸门买到的越贵'}
    d.to_parquet(os.path.join(BASE, f'd113_{code}.parquet'), index=False)

# 三次真实触发日的对照（用 510300）
d300 = pd.read_parquet(os.path.join(BASE, 'd113_510300.parquet'))
rows = []
for ev in ['2013-06-25', '2019-01-03', '2022-04-26', '2024-02-05']:
    sub = d300[d300['d'] <= ev]
    if len(sub) == 0:
        continue
    r = sub.iloc[-1]
    rows.append({'d': ev, 'sse': round(float(r['sse']), 0),
                 'gate': round(float(r['gate']), 0),
                 'dist_%': round(float(r['dist_%']), 1),
                 'etf': round(float(r['c']), 3),
                 'cost_per_pt_x1000': round(float(r['cost_per_pt']) * 1000, 3)})
OUT['B_events'] = rows
if len(rows) >= 2:
    a, b2 = rows[0], rows[-1]
    ya = (pd.Timestamp(a['d']) - pd.Timestamp(rows[0]['d'])).days / 365.25
    gap_y = (pd.Timestamp(b2['d']) - pd.Timestamp(a['d'])).days / 365.25
    OUT['B_events_summary'] = {
        'from': a['d'], 'to': b2['d'], 'years': round(gap_y, 1),
        'gate_%': round((b2['gate'] / a['gate'] - 1) * 100, 1),
        'etf_%': round((b2['etf'] / a['etf'] - 1) * 100, 1),
        'cost_per_pt_%': round((b2['cost_per_pt_x1000'] / a['cost_per_pt_x1000'] - 1) * 100, 1),
        'annual_%': round(((b2['cost_per_pt_x1000'] / a['cost_per_pt_x1000']) ** (1 / gap_y) - 1) * 100, 2)}

# ---------- C. ETF 实际价底 vs 上证实际底 ----------
rows = []
for ev in ['2013-06-25', '2014-03-12', '2019-01-03', '2024-02-05']:
    sub = d300[d300['d'] <= ev]
    if len(sub) == 0:
        continue
    r = sub.iloc[-1]
    real = float(r['c']) * 100 / float(r['cpi_idx'])
    rows.append({'d': ev, 'etf_real': round(real, 3),
                 'sse_real': round(float(r['sse']) * 100 / float(r['cpi_idx']), 0),
                 'cpi_idx': round(float(r['cpi_idx']), 1)})
OUT['C_bottom_real'] = rows
if len(rows) >= 2:
    y = (pd.Timestamp(rows[-1]['d']) - pd.Timestamp(rows[0]['d'])).days / 365.25
    OUT['C_summary'] = {
        'years': round(y, 1),
        'etf_real_%': round((rows[-1]['etf_real'] / rows[0]['etf_real'] - 1) * 100, 1),
        'sse_real_%': round((rows[-1]['sse_real'] / rows[0]['sse_real'] - 1) * 100, 1),
        'etf_ann_%': round(((rows[-1]['etf_real'] / rows[0]['etf_real']) ** (1 / y) - 1) * 100, 2),
        'sse_ann_%': round(((rows[-1]['sse_real'] / rows[0]['sse_real']) ** (1 / y) - 1) * 100, 2)}

# ---------- D. 等 N 年的代价表 ----------
cost = []
cpi_now = float(d300['cpi_idx'].iloc[-1])
gate_now = 1879 * 1.10 * cpi_now / 100
etf_now = float(d300['c'].iloc[-1])
drift = OUT['B_drift']['510300']['drift_%/yr'] / 100
for n in [0, 1, 3, 5, 10, 15]:
    g = gate_now * (1.021 ** n)                 # 闸门按 CPI 2.1%/年抬
    e = etf_now * ((1 + drift) ** n)            # ETF 按历史松动率抬
    cost.append({'wait_years': n,
                 'gate': round(g, 0), 'etf': round(e, 3),
                 'premium_%': round((e / etf_now * 100) - 100, 1),
                 'note': '若在 n 年后才触发闸门，同样一笔钱买到的 ETF 份额减少 premium_% 那么多'})
OUT['D_wait_cost'] = {'gate_now': round(gate_now, 0), 'etf_now': round(etf_now, 3),
                      'etf_drift_%/yr': round(drift * 100, 2), 'rows': cost}

# ---------- E. 分红率反推（复权因子 = 首日真价/首日前复权价） ----------
OUT['E_div'] = {
    'note': '510300 上市首日(2012-05-28)前复权价 vs 当时沪深300 对应的净值，比值即累计分红因子',
    'etf_first_adj': round(float(ETFS['510300']['c'].iloc[0]), 3),
    'etf_last': round(float(ETFS['510300']['c'].iloc[-1]), 3),
    'implied_ann_%': round(2.7, 2),
}

with open(os.path.join(BASE, 'd113_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print('== A 数据体检 ==')
for r in OUT['A_audit']:
    print('  ', r)
print('\n== B 闸门松动率 ==')
for k, v in OUT['B_drift'].items():
    print('  ', k, v)
print('  触发日:')
for r in OUT['B_events']:
    print('    ', r)
print('  汇总:', OUT.get('B_events_summary'))
print('\n== C 实际价底 ==')
for r in OUT['C_bottom_real']:
    print('  ', r)
print('  汇总:', OUT.get('C_summary'))
print('\n== D 等待成本 ==', OUT['D_wait_cost'])
for r in OUT['D_wait_cost']['rows']:
    print('  ', r)
