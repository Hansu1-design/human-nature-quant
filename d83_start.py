# -*- coding: utf-8 -*-
"""
d83: 三次机会的起点特征 + 空窗期 + ZigZag 敏感性
"""
import json
import numpy as np
import pandas as pd

sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)
sse = sse[sse['d'] >= '2000-01-01'].reset_index(drop=True)
C = sse['close'].values
D = sse['d'].values
n = len(C)

pe = pd.read_parquet('G:/AStockData/supplement/valuation/all_a_pe_ttm.parquet')
pe['d'] = pd.to_datetime(pe['date'])
pe = pe.sort_values('d')


def zigzag(C, th):
    ups = []
    trend = 'down'
    ext_i, ext_v = 0, C[0]
    last_low = 0
    up_start = 0
    for i in range(1, len(C)):
        if trend == 'down':
            if C[i] < ext_v:
                ext_v, ext_i = C[i], i
                last_low = i
            if C[i] > ext_v * (1 + th):
                trend = 'up'
                up_start = last_low
                ext_v, ext_i = C[i], i
        else:
            if C[i] > ext_v:
                ext_v, ext_i = C[i], i
            if C[i] < ext_v * (1 - th):
                ups.append((up_start, ext_i, C[ext_i] / C[up_start] - 1))
                trend = 'down'
                ext_v, ext_i = C[i], i
                last_low = i
    if trend == 'up':
        ups.append((up_start, ext_i, C[ext_i] / C[up_start] - 1))
    return ups


print('=== ZigZag 阈值敏感性（2000-01 起） ===')
rows = []
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    ups = zigzag(C, th)
    df = pd.DataFrame(ups, columns=['i0', 'i1', 'ret'])
    df = df[df['i1'] > df['i0']]
    tot = (pd.Timestamp(D[-1]) - pd.Timestamp(D[0])).days
    for cut in [0.5, 1.0]:
        sel = df[df['ret'] >= cut]
        days = sum((pd.Timestamp(D[b]) - pd.Timestamp(D[a])).days for a, b in zip(sel['i0'], sel['i1']))
        rows.append({'zz': int(th * 100), 'cut': int(cut * 100), 'n': len(sel),
                     'share_%': round(days / tot * 100, 1),
                     'compound_%': round((np.prod(1 + sel['ret'].values) - 1) * 100, 0) if len(sel) else 0})
sens = pd.DataFrame(rows)
print(sens.pivot(index='zz', columns='cut', values=['n', 'share_%', 'compound_%']).to_string())

# 三次大段的起点特征
print('\n=== 三次 >=100% 段的起点：当时是什么样 ===')
ups25 = zigzag(C, 0.25)
big = [(a, b, r) for a, b, r in ups25 if r >= 1.0]
recs = []
for a, b, r in big:
    dd = C[a] / np.max(C[:a + 1]) - 1
    y2 = C[:a + 1][-500:].min() if a >= 500 else C[:a + 1].min()
    px = pe[pe['d'] <= pd.Timestamp(D[a])]
    per = float(px['middlePETTM'].iloc[-1]) if len(px) else np.nan
    perq = float(px['quantileInAllHistoryMiddlePeTtm'].iloc[-1]) if len(px) and pd.notna(px['quantileInAllHistoryMiddlePeTtm'].iloc[-1]) else np.nan
    recs.append({'start': str(pd.Timestamp(D[a]).date()), 'end': str(pd.Timestamp(D[b]).date()),
                 'p0': round(C[a]), 'p1': round(C[b]), 'ret_%': round(r * 100),
                 'months': round((pd.Timestamp(D[b]) - pd.Timestamp(D[a])).days / 30.44, 1),
                 'drawdown_from_hi_%': round(dd * 100, 1), 'pe_mid': round(per, 1), 'pe_q': round(perq, 2) if not np.isnan(perq) else None})
print(pd.DataFrame(recs).to_string(index=False))

# 空窗期
print('\n=== 大段之间的空窗 ===')
prev_end = None
gaps = []
for a, b, r in sorted(big, key=lambda x: x[0]):
    if prev_end is not None:
        g = (pd.Timestamp(D[a]) - pd.Timestamp(D[prev_end])).days / 30.44
        gaps.append({'from': str(pd.Timestamp(D[prev_end]).date()), 'to': str(pd.Timestamp(D[a]).date()), 'months': round(g, 1)})
    prev_end = b
print(pd.DataFrame(gaps).to_string(index=False))

# 每段的回撤过程：从起点到顶，中途最大回撤
print('\n=== 段内最大回撤（持仓体验） ===')
trav = []
for a, b, r in big:
    seg = C[a:b + 1]
    peak = np.maximum.accumulate(seg)
    mdd = float(np.min(seg / peak - 1))
    trav.append({'start': str(pd.Timestamp(D[a]).date()), 'mdd_in_seg_%': round(mdd * 100, 1)})
print(pd.DataFrame(trav).to_string(index=False))

# 现在的位置
print(f'\n当前 {D[-1]} = {C[-1]:.0f}；距 2026-05-13 高点 4243 回撤 {(C[-1]/4243-1)*100:.1f}%')
print(f'距 2007 顶 6092 回撤 {(C[-1]/6092-1)*100:.1f}%')

json.dump({'sens': sens.to_dict('records'), 'big': recs, 'gaps': gaps, 'trav': trav},
          open('d83_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
