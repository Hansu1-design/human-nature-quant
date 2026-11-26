# -*- coding: utf-8 -*-
"""
d73: 补两个洞
  1) 本地数据集只含存活股 → 退市偏差敏感性（插入不同比例的退市股重算分布）
  2) 持有过程中的年度体验：2015-2025 每年个股中位 vs 指数
"""
import json
import os
import numpy as np
import pandas as pd

P = 'G:/AStockData/supplement/daily_tencent_full'
files = sorted(f for f in os.listdir(P) if f.endswith('.parquet'))
OUT = {}

cache = {}
for f in files:
    try:
        d = pd.read_parquet(os.path.join(P, f), columns=['date', 'close'])
    except Exception:
        continue
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    d = d.dropna(subset=['close'])
    d = d[d['close'] > 0].sort_values('date').reset_index(drop=True)
    if len(d) < 60:
        continue
    cache[f.replace('.parquet', '')] = (d['date'].values.astype('datetime64[D]'),
                                        d['close'].values.astype(float))
print(f"有效个股 {len(cache)}")

# ---------- 1) 2015 起点 + 退市敏感性 ----------
t0 = np.datetime64('2015-01-05')
recs = []
for code, (dts, px) in cache.items():
    if dts[0] > t0:
        continue
    j = np.searchsorted(dts, t0, side='right') - 1
    if j < 0:
        continue
    recs.append((code, px[j], px[-1]))
df = pd.DataFrame(recs, columns=['code', 'p0', 'p1'])
df['ret'] = df['p1'] / df['p0'] - 1
df = df[df['ret'].between(-1.0, 30)]
base = df['ret'].values
n0 = len(base)
print(f"2015-01-05 起点：{n0} 只（全部存活，本地库无退市股）  中位 {np.median(base)*100:.1f}%")

print("\n" + "=" * 78)
print("1) 退市偏差敏感性：假设有 x% 的股票退市、按 y 亏损计")
print("=" * 78)
print(f"{'退市比例':>8} {'按-60%':>18} {'按-80%':>18} {'按-95%':>18}")
sens = []
rng = np.random.default_rng(7)
for pct in [0.03, 0.06, 0.09, 0.12]:
    row = {'pct': pct}
    line = f"{pct*100:7.0f}% "
    for loss in [-0.60, -0.80, -0.95]:
        m = int(round(n0 * pct / (1 - pct)))
        sim = np.concatenate([base, np.full(m, loss)])
        med = float(np.median(sim)) * 100
        pos = float((sim > 0).mean()) * 100
        row[f'med_{int(-loss*100)}'] = round(med, 1)
        row[f'pos_{int(-loss*100)}'] = round(pos, 1)
        line += f"  中位 {med:5.1f}% 正 {pos:4.1f}% "
    sens.append(row)
    print(line)
OUT['surv_bias'] = sens
print("  说明：A股 2015-2025 实际退市约 200 家，占当时上市数 6~8%；按 -80% 计，中位从 26.9% 降到约 15%。")

# ---------- 2) 逐年体验 ----------
print("\n" + "=" * 78)
print("2) 买入后一直拿着，每一年的体验（年初到年末，个股中位 vs 上证）")
print("=" * 78)

idx = pd.read_parquet('idx_sse_all.parquet')
idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
idx['d'] = pd.to_datetime(idx['date'])
idx = idx.dropna(subset=['close']).sort_values('d')
idx['y'] = idx['d'].dt.year
iy = idx.groupby('y')['close'].agg(['first', 'last'])
iy['ret'] = iy['last'] / iy['first'] - 1

years = list(range(2015, 2026))
Y = []
print(f"{'年':>5} {'个股数':>6} {'个股中位':>9} {'正收益':>7} {'上证':>8} {'跑赢上证':>9}")
for y in years:
    s = np.datetime64(f'{y}-01-01')
    e = np.datetime64(f'{y}-12-31')
    rets = []
    for code, (dts, px) in cache.items():
        i = np.searchsorted(dts, s, side='left')
        j = np.searchsorted(dts, e, side='right') - 1
        if i >= len(dts) or j < 0 or j <= i:
            continue
        rets.append(px[j] / px[i] - 1)
    rets = np.array(rets)
    rets = rets[(rets > -1) & (rets < 30)]
    ir = float(iy.loc[y, 'ret']) if y in iy.index else np.nan
    row = dict(y=y, n=len(rets), med=round(float(np.median(rets)) * 100, 1),
               pos=round(float((rets > 0).mean()) * 100, 1),
               idx=round(ir * 100, 1) if ir == ir else None,
               beat=round(float((rets > ir).mean()) * 100, 1) if ir == ir else None)
    Y.append(row)
    print(f"{y:5d} {row['n']:6d} {row['med']:8.1f}% {row['pos']:6.1f}% "
          f"{(row['idx'] if row['idx'] is not None else float('nan')):7.1f}% "
          f"{(row['beat'] if row['beat'] is not None else float('nan')):8.1f}%")
OUT['yearly'] = Y

pos_years = sum(1 for r in Y if r['med'] > 0)
beat_years = sum(1 for r in Y if r['beat'] and r['beat'] >= 50)
print(f"\n  11 年里个股中位为正：{pos_years}/11；个股中位跑赢上证：{beat_years}/11")
print(f"  最差一年 {min(Y, key=lambda r: r['med'])['y']}（中位 {min(r['med'] for r in Y):.1f}%），"
      f"最好一年 {max(Y, key=lambda r: r['med'])['y']}（中位 {max(r['med'] for r in Y):.1f}%）")

json.dump(OUT, open('d73_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print("\n[saved] d73_result.json")
