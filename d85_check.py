# -*- coding: utf-8 -*-
"""
d85: 日频段内回撤 + 8笔交易的事前回撤位置 + 空仓现金修正
"""
import json
import numpy as np
import pandas as pd

sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)
sse = sse[sse['d'] >= '1996-01-01'].reset_index(drop=True)
C = sse['close'].values
D = sse['d'].values
n = len(C)
hi = np.maximum.accumulate(C)
dd = C / hi - 1

def idx(ts):
    return int(np.argmin(np.abs(D - np.datetime64(pd.Timestamp(ts)))))

segs = [('2005-07-11', '2007-10-16', 1011, 6092),
        ('2008-11-04', '2009-08-04', 1707, 3471),
        ('2013-06-27', '2015-06-12', 1950, 5166)]
print('=== 三次大段：日频段内最大回撤 ===')
rows = []
for a, b, p0, p1 in segs:
    i0, i1 = idx(a), idx(b)
    seg = C[i0:i1 + 1]
    pk = np.maximum.accumulate(seg)
    mdd = float(np.min(seg / pk - 1))
    # 最深那次回撤持续多久
    j = int(np.argmin(seg / pk - 1))
    k = int(np.argmax(pk[:j + 1]))
    rows.append({'seg': f'{a}→{b}', 'mdd_daily_%': round(mdd * 100, 1),
                 'dd_start': str(pd.Timestamp(D[i0 + k]).date()), 'dd_trough': str(pd.Timestamp(D[i0 + j]).date()),
                 'dd_days': j - k})
    print(rows[-1])

print('\n=== d63 记录的 8 笔：进场当日距前高回撤 ===')
trades = ['2016-03-02', '2018-07-06', '2018-11-05', '2019-06-10', '2020-04-01',
          '2024-01-22', '2024-08-05', '2025-04-09']
trows = []
for t in trades:
    i = idx(t)
    trows.append({'entry': t, 'sse': round(C[i]), 'dd_from_hi_%': round(dd[i] * 100, 1),
                  'hi': round(hi[i])})
    print(trows[-1])
T = pd.DataFrame(trows)
print(f'\n8 笔的距前高回撤区间：{T["dd_from_hi_%"].min():.1f}% ~ {T["dd_from_hi_%"].max():.1f}%')
print(f'三次大段起点：-54.9% / -72.0% / -68.0%')
print(f'2015-06 以来最深：{dd[D > np.datetime64("2015-06-12")].min()*100:.1f}% '
      f'(发生于 {pd.Timestamp(D[D > np.datetime64("2015-06-12")][int(np.argmin(dd[D > np.datetime64("2015-06-12")]))]).date()})')

print('\n=== 空仓期现金收益修正（年化 2%） ===')
for label, segs_sel, yr in [
    ('只做第三段 2013-06→2015-06 +165%，之后空仓到 2026-11', [(1950, 5166)], None),
]:
    pass
# 只做第三段：2013-06-27 买 1.0，2015-06-12 卖 2.65，之后现金 2%/yr 到 2026-11
t0 = pd.Timestamp('2013-06-27'); t1 = pd.Timestamp('2015-06-12'); t2 = pd.Timestamp('2026-11-11')
v = 1.0 * (5166 / 1950)
yrs = (t2 - t1).days / 365.25
v *= 1.02 ** yrs
tot_y = (t2 - t0).days / 365.25
print(f'1.00 → {v:.2f}  跨度 {tot_y:.1f} 年  CAGR {(v**(1/tot_y)-1)*100:.2f}%')
# 买入持有同期
i0, i2 = idx('2013-06-27'), n - 1
bh = C[i2] / C[i0]
print(f'买入持有同期 {bh:.2f}x  CAGR {(bh**(1/tot_y)-1)*100:.2f}%')

json.dump({'mdd_daily': rows, 'trades': trows}, open('d85_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
