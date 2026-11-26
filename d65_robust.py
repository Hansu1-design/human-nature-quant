# -*- coding: utf-8 -*-
"""d65 — 稳健性补刀：块自助法置信区间 / 收益贡献拆解 / 当前状态"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_parquet(os.path.join(BASE, 'idx_sse_long.parquet'))
idx['d'] = pd.to_datetime(idx['date'])
for c in ['close', 'amount']:
    idx[c] = idx[c].astype(float)
idx = idx.sort_values('d').reset_index(drop=True)
idx['heat'] = idx['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
idx['pct5y'] = idx['close'].rolling(1250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
idx = idx[idx['d'] >= '2015-01-01'].reset_index(drop=True)
C = idx['close'].values
f1 = pd.Series(C).shift(-250) / pd.Series(C) - 1

OUT = {}
# 1) 块自助法：低档 vs 高档 的1年前瞻收益之差
lo = (C >= 2500) & (C < 3000)
hi = (C >= 3500) & (C < 4000)
ok = f1.notna().values
lo, hi = lo & ok, hi & ok
gaps, dif = [], f1.values[lo].mean() - f1.values[hi].mean()
rng = np.random.default_rng(7)
n = len(C)
B, blk = 2000, 60
for _ in range(B):
    st = rng.integers(0, n - blk, size=n // blk)
    ii = np.concatenate([np.arange(s, s + blk) for s in st])
    l, h = lo[ii], hi[ii]
    if l.sum() > 10 and h.sum() > 10:
        gaps.append(f1.values[ii][l].mean() - f1.values[ii][h].mean())
gaps = np.array(gaps)
OUT['boot_gap'] = {'point_%': round(dif * 100, 1), 'ci95': [round(np.percentile(gaps, 2.5) * 100, 1),
                                                           round(np.percentile(gaps, 97.5) * 100, 1)],
                   'p_gt0': round(float((gaps > 0).mean()), 3), 'blocks': blk, 'B': B}

# 2) 收益贡献拆解（E1 七笔）
eps = [('2016-03-02', '2018-01-23', 61.6), ('2018-07-20', '2018-10-11', -12.8),
       ('2018-11-02', '2019-05-22', 12.3), ('2019-06-11', '2020-03-16', 1.9),
       ('2020-04-07', '2021-02-22', 57.2), ('2024-01-25', '2024-08-14', 1.1),
       ('2024-08-19', '2025-04-07', 8.6)]
tot = np.prod([1 + x[2] / 100 for x in eps]) - 1
without = []
for k in range(len(eps)):
    r = np.prod([1 + e[2] / 100 for j, e in enumerate(eps) if j != k]) - 1
    without.append(round(r * 100, 1))
OUT['contrib'] = {'total_%': round(tot * 100, 1),
                  'drop_one_out_%': without,
                  'note': '去掉任一笔后的总收益'}
top2 = (1.616 * 1.572 - 1) * 100
OUT['contrib']['top2_%'] = round(top2, 1)

# 3) 当前状态
last = idx.iloc[-1]
OUT['now'] = {'date': str(last['d'])[:10], 'idx': round(float(last['close']), 0),
              'pct5y': round(float(last['pct5y']), 2), 'heat': round(float(last['heat']), 2),
              'below3000_days_2026': int((idx[idx['d'] >= '2026-01-01']['close'] < 3000).sum()),
              'max2025_26': round(float(idx[idx['d'] >= '2025-01-01']['close'].max()), 0),
              'min_since_2025': round(float(idx[idx['d'] >= '2025-01-01']['close'].min()), 0)}

# 4) 每次"从3000以下涨到3500以上"的完整波段
runs = []
inb = False
for i in range(len(C)):
    if not inb and C[i] < 3000:
        inb, s_i = True, i
    if inb and C[i] > 3500:
        runs.append({'buy': str(idx['d'].iloc[s_i])[:10], 'buy_lv': round(float(C[s_i]), 0),
                     'sell': str(idx['d'].iloc[i])[:10], 'sell_lv': round(float(C[i]), 0),
                     'gain_%': round((C[i] / C[s_i] - 1) * 100, 1),
                     'months': round((i - s_i) / 20.3, 1)})
        inb = False
OUT['waves'] = runs

with open(os.path.join(BASE, 'd65_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)

print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
