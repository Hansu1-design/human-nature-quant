# -*- coding: utf-8 -*-
"""
d120 探针：把 2008-2026 的「进带事件」全部列出来
目的：在动手做参数扫描之前，先确认样本到底有几个，别又扫了个空
"""
import os
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))


def load():
    a = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
    a['d'] = pd.to_datetime(a['date'])
    a['close'] = pd.to_numeric(a['close'], errors='coerce')
    a = a[['d', 'close', 'high', 'low', 'amount']].dropna().sort_values('d').reset_index(drop=True)

    c = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
    c['ym'] = c['月份'].str.replace('年', '-').str.replace('月份', '')
    c['d'] = pd.to_datetime(c['ym'], format='%Y-%m')
    c = c.sort_values('d').reset_index(drop=True)
    c['mom'] = (pd.to_numeric(c['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
    ix = [100.0]
    for i in range(1, len(c)):
        ix.append(ix[-1] * c['mom'].iloc[i])
    c['cpi_idx'] = ix
    # 当月 CPI 次年 1 月中才发布，滞后约 45 天
    c['avail'] = c['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)

    d = pd.merge_asof(a, c[['avail', 'cpi_idx']].sort_values('avail'),
                      left_on='d', right_on='avail', direction='backward')
    d = d.dropna(subset=['cpi_idx']).reset_index(drop=True)
    d['real'] = d['close'] * 100.0 / d['cpi_idx']
    d['rA'] = d['real'] / 1880.0
    d['peak'] = d['close'].cummax()
    d['dd_peak'] = d['close'] / d['peak'] - 1.0
    return d


d = load()
N = len(d)
print(f'样本 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  {N} 交易日  {(d.d.iloc[-1]-d.d.iloc[0]).days/365.25:.1f} 年')
print(f'CPI 定基 2008-01=100，末日 {d.cpi_idx.iloc[-1]:.2f}')
print(f'rA 范围 {d.rA.min():.3f} ~ {d.rA.max():.3f}')
print()

# 进带事件：从 >1+k 变成 <=1+k 的第一天
def entries(k):
    m = (d['rA'].values <= 1 + k)
    out = []
    for i in range(1, N):
        if m[i] and not m[i - 1]:
            out.append(i)
    return out


for k in [0.05, 0.10, 0.15, 0.20]:
    es = entries(k)
    print(f'=== 离底 ≤{int(k*100)}%：{len(es)} 次 ===')
    for i in es:
        r = d.iloc[i]
        # 之后 500 日内的最低 real
        hi = min(N, i + 500)
        lo = i + int(np.argmin(d['real'].values[i:hi]))
        print(f"  {r.d.date()}  收 {r.close:7.0f}  实际 {r.real:6.0f}  rA {r.rA:.3f}  "
              f"距顶 {r.dd_peak*100:+6.1f}%  →500日内最低 {d.real.iloc[lo]:6.0f} "
              f"({d.d.iloc[lo].date()}, 再跌 {(d.real.iloc[lo]/r.real-1)*100:+5.1f}%)")
    print()

# 真实底部（每轮）
print('=== 实际值局部最低（滚动 250 日窗口）===')
v = d['real'].values
lows = []
for i in range(120, N - 120):
    w = v[i - 120:i + 121]
    if v[i] == w.min() and v[i] < v[max(0, i - 250)] * 0.95:
        if not lows or i - lows[-1][0] > 200:
            lows.append((i, v[i]))
for i, x in lows:
    print(f"  {d.d.iloc[i].date()}  实际 {x:6.0f}  rA {x/1880:.3f}  名义 {d.close.iloc[i]:7.0f}")
