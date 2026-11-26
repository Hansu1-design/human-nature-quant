# -*- coding: utf-8 -*-
"""
d111 — 标的错配第二步：复权口径确认 + 可执行偏离定义 + 预测力（block bootstrap）
d110 的教训：用"全样本二次趋势残差"测预测力，等于用恒等式测，t 值虚高（t=1094）。
本脚本改用【实盘可算】的滚动基准，并用 block bootstrap 给 p 值。
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
rng = np.random.default_rng(20261112)

# ---------- 数据 ----------
e = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
e['d'] = pd.to_datetime(e['日期'])
for c, n in [('收盘', 'etf'), ('涨跌幅', 'etf_pct'), ('成交额', 'amt')]:
    e[n] = pd.to_numeric(e[c], errors='coerce')
e = e[['d', 'etf', 'etf_pct', 'amt']].sort_values('d').reset_index(drop=True)

s = pd.read_parquet(os.path.join(BASE, 'idx_sse_all.parquet'))
s['d'] = pd.to_datetime(s['date'])
s['sse'] = pd.to_numeric(s['close'], errors='coerce')
s = s[['d', 'sse']].sort_values('d').reset_index(drop=True)

df = e.merge(s, on='d', how='inner').sort_values('d').reset_index(drop=True)
df['lret_etf'] = np.log(df['etf']).diff()
df['lret_sse'] = np.log(df['sse']).diff()
df['lr'] = np.log(df['sse'] / df['etf'] * 1000)

# ---------- A. 复权口径确认 ----------
# 用"涨跌幅"列复利累乘，锚定末日真实价 → 得到一条独立构造的净值序列
pct = df['etf_pct'].fillna(0) / 100.0
nav = (1 + pct).cumprod()
nav = nav / nav.iloc[-1] * df['etf'].iloc[-1]
df['nav_from_pct'] = nav
gap = (df['nav_from_pct'] / df['etf'] - 1) * 100
OUT['A_adj'] = {
    'note': '涨跌幅累乘序列 vs 收盘序列；两者一致 → 收盘列本身就是复权口径',
    'max_abs_gap_%': round(float(gap.abs().max()), 3),
    'mean_abs_gap_%': round(float(gap.abs().mean()), 4),
}
# 关键日真实价对照（人工已知值，用于判断是否前复权）
key = ['2012-05-28', '2015-06-12', '2016-01-27', '2018-01-24', '2021-02-18',
       '2024-02-05', '2024-09-30', '2026-09-11']
rows = []
for k in key:
    sub = df[df['d'] == k]
    if len(sub):
        r = sub.iloc[0]
        rows.append({'d': k, 'etf': round(float(r['etf']), 3),
                     'sse': round(float(r['sse']), 1)})
OUT['A_key_days'] = rows
OUT['A_head'] = [{'d': str(x.date()), 'etf': round(float(y), 3), 'sse': round(float(z), 1)}
                 for x, y, z in zip(df['d'][:3], df['etf'][:3], df['sse'][:3])]

# ---------- B. 漂移的分解：ETF 总回报 - 上证 ----------
def cagr(a, b, years):
    return (b / a) ** (1 / years) - 1
yrs = (df['d'].iloc[-1] - df['d'].iloc[0]).days / 365.25
OUT['B_drift'] = {
    'years': round(yrs, 2),
    'etf_total_%': round((df['etf'].iloc[-1] / df['etf'].iloc[0] - 1) * 100, 1),
    'sse_total_%': round((df['sse'].iloc[-1] / df['sse'].iloc[0] - 1) * 100, 1),
    'etf_cagr_%': round(cagr(df['etf'].iloc[0], df['etf'].iloc[-1], yrs) * 100, 2),
    'sse_cagr_%': round(cagr(df['sse'].iloc[0], df['sse'].iloc[-1], yrs) * 100, 2),
    'gap_%/yr': round((cagr(df['etf'].iloc[0], df['etf'].iloc[-1], yrs)
                       - cagr(df['sse'].iloc[0], df['sse'].iloc[-1], yrs)) * 100, 2),
    'ratio_first': round(float(np.exp(df['lr'].iloc[0])), 1),
    'ratio_last': round(float(np.exp(df['lr'].iloc[-1])), 1),
    'ratio_chg_%': round(float((np.exp(df['lr'].iloc[-1] - df['lr'].iloc[0]) - 1) * 100), 1),
}
# 分段漂移（看漂移速率是否稳定）
segs = []
for a, b in [('2012-05', '2015-06'), ('2015-06', '2019-01'), ('2019-01', '2021-12'),
             ('2021-12', '2024-09'), ('2024-09', '2026-09')]:
    m = (df['d'] >= a) & (df['d'] < b)
    if m.sum() < 50:
        continue
    sub = df[m]
    y = (sub['d'].iloc[-1] - sub['d'].iloc[0]).days / 365.25
    segs.append({'seg': f'{a}~{b}',
                 'ratio_%': round(float((np.exp(sub['lr'].iloc[-1] - sub['lr'].iloc[0]) - 1) * 100), 1),
                 'drift_%/yr': round(float((np.exp((sub['lr'].iloc[-1] - sub['lr'].iloc[0]) / y) - 1) * 100), 2)})
OUT['B_segments'] = segs

# ---------- C. 可执行偏离：滚动窗口基准（无前视）----------
for win, lab in [(1215, '5y'), (2430, '10y')]:
    df['base_' + lab] = df['lr'].rolling(win, min_periods=int(win * 0.6)).mean()
    df['dev_' + lab] = df['lr'] - df['base_' + lab]
    df['devp_' + lab] = (np.exp(df['dev_' + lab]) - 1) * 100

OUT['C_dev_desc'] = {}
for lab in ['5y', '10y']:
    s2 = df['devp_' + lab].dropna()
    OUT['C_dev_desc'][lab] = {'n': int(len(s2)), 'mean_%': round(float(s2.mean()), 2),
                              'sd_%': round(float(s2.std()), 2),
                              'p5': round(float(s2.quantile(.05)), 1),
                              'p95': round(float(s2.quantile(.95)), 1),
                              'min': round(float(s2.min()), 1), 'max': round(float(s2.max()), 1),
                              'last': round(float(s2.iloc[-1]), 1)}

# ---------- D. 预测力：偏离 → 未来 ratio 变化 ----------
# 恒等：ETF 相对上证的未来超额 = -Δln ratio。若偏离能预测 Δln ratio，就能预测超额。
def block_boot_coef(dev, fut, h, nboot=400, block=None):
    """循环块自助法：按块抽 (dev, fut) 对，保留重叠导致的序列相关"""
    n = len(dev)
    block = block or h
    coefs = []
    for _ in range(nboot):
        idx = []
        while len(idx) < n:
            st = rng.integers(0, n)
            idx.extend(range(st, min(st + block, n)))
        idx = np.array(idx[:n])
        x, y = dev[idx], fut[idx]
        xm, ym = x - x.mean(), y - y.mean()
        den = (xm * xm).sum()
        coefs.append((xm * ym).sum() / den if den > 0 else np.nan)
    coefs = np.array(coefs)
    return float(np.nanpercentile(coefs, 2.5)), float(np.nanpercentile(coefs, 97.5))

pred = []
for lab in ['5y', '10y']:
    for h in [60, 250, 500]:
        fut = (df['lr'].shift(-h) - df['lr']) * 100      # 未来 ratio 变化 %
        m = fut.notna() & df['devp_' + lab].notna()
        x = df.loc[m, 'devp_' + lab].values
        y = fut[m].values
        xm, ym = x - x.mean(), y - y.mean()
        coef = (xm * ym).sum() / (xm * xm).sum()
        resid = y - (y.mean() + coef * xm)
        r2 = 1 - (resid @ resid) / (ym @ ym)
        lo, hi = block_boot_coef(x, y, h)
        pred.append({'base': lab, 'h': h, 'coef': round(float(coef), 3),
                     'ci95': [round(lo, 3), round(hi, 3)],
                     'sig': '显著' if (lo > 0 or hi < 0) else '不显著',
                     'R2_%': round(float(r2 * 100), 1), 'n': int(m.sum())})
OUT['D_predict'] = pred

# ---------- E. 最贴执行的检验：出手区内，偏离高低两组的未来 1 年实际收益 ----------
# 出手区 = 上证 / 1879 * 100 <= 110（v1.1 的 10% 带，用名义上的静态近似）
df['dist'] = df['sse'] / 1879.0
zone = df['dist'] <= 1.10
tab = []
for lab in ['5y', '10y']:
    f1 = (df['etf'].shift(-250) / df['etf'] - 1) * 100
    for q, nm in [(0.33, '低（ETF相对贵）'), (0.67, '高（ETF相对便宜）')]:
        pass
    d = df['devp_' + lab]
    thr = d.quantile([1 / 3, 2 / 3]).values
    for nm, m in [('偏离低1/3（ETF贵）', zone & (d <= thr[0])),
                  ('中间1/3', zone & (d > thr[0]) & (d <= thr[1])),
                  ('偏离高1/3（ETF便宜）', zone & (d > thr[1]))]:
        mm = m & f1.notna()
        if mm.sum() < 10:
            tab.append({'base': lab, 'grp': nm, 'n': int(mm.sum()), 'mean_%': None, 'win_%': None})
            continue
        tab.append({'base': lab, 'grp': nm, 'n': int(mm.sum()),
                    'mean_%': round(float(f1[mm].mean()), 1),
                    'win_%': round(float((f1[mm] > 0).mean() * 100), 1)})
OUT['E_zone'] = tab

# 相同点位、不同偏离下的 ETF 价格差（执行层面：同样的钱买到多少份）
rows = []
for lo, hi in [(2600, 2800), (2800, 3000), (3000, 3200), (3200, 3500)]:
    m = (df['sse'] >= lo) & (df['sse'] < hi) & df['devp_5y'].notna()
    if m.sum() < 20:
        continue
    sub = df[m]
    rows.append({'band': f'{lo}-{hi}', 'n': int(m.sum()),
                 'dev_lo_%': round(float(sub['devp_5y'].quantile(.05)), 1),
                 'dev_hi_%': round(float(sub['devp_5y'].quantile(.95)), 1),
                 'shares_diff_%': round(float(sub['etf'].quantile(.05) / sub['etf'].quantile(.95) - 1) * 100, 1)})
OUT['F_same_band'] = rows

df.to_parquet(os.path.join(BASE, 'd111_path.parquet'), index=False)
with open(os.path.join(BASE, 'd111_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print('== A 复权口径 ==', OUT['A_adj'])
for r in OUT['A_key_days']:
    print('  ', r)
print('\n== B 漂移 ==', OUT['B_drift'])
for r in OUT['B_segments']:
    print('  ', r)
print('\n== C 偏离分布 ==')
for k, v in OUT['C_dev_desc'].items():
    print(' ', k, v)
print('\n== D 预测力 ==')
for r in OUT['D_predict']:
    print('  ', r)
print('\n== E 出手区内偏离分档 ==')
for r in OUT['E_zone']:
    print('  ', r)
print('\n== F 同区间 ETF 份数差 ==')
for r in OUT['F_same_band']:
    print('  ', r)
