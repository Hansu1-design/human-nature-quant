# -*- coding: utf-8 -*-
"""
d84: "跌一半就买" 这个信号，历史上触发过几次、之后赚多少、几次是假的
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
print(f'样本 {D[0]} ~ {D[-1]}  {n} 根')
print(f'当前回撤 {dd[-1]*100:.1f}%  历史最高 {hi[-1]:.0f}')

rows = []
for th in [0.30, 0.40, 0.50, 0.60]:
    trig = dd <= -th
    # 连续区段
    segs = []
    i = 0
    while i < n:
        if trig[i]:
            j = i
            while j + 1 < n and trig[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    # 合并 gap<=60 日的段
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= 60:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    for a, b in merged:
        # 段内最低
        k = a + int(np.argmin(C[a:b + 1]))
        rec = {'th': int(th * 100), 'enter': str(pd.Timestamp(D[a]).date()), 'exit': str(pd.Timestamp(D[b]).date()),
               'days': b - a + 1, 'p_enter': round(C[a]), 'low': round(C[k]), 'low_date': str(pd.Timestamp(D[k]).date()),
               'further_drop_%': round((C[k] / C[a] - 1) * 100, 1)}
        for h in [1, 2, 3, 5]:
            j = a + int(252 * h)
            rec[f'r{h}y_%'] = round((C[j] / C[a] - 1) * 100, 0) if j < n else None
        rows.append(rec)

df = pd.DataFrame(rows)
for th in [30, 40, 50, 60]:
    s = df[df['th'] == th]
    print(f'\n--- 回撤 <= -{th}% 触发 {len(s)} 次 ---')
    print(s.drop(columns=['th']).to_string(index=False))

print('\n=== 汇总 ===')
for th in [30, 40, 50, 60]:
    s = df[df['th'] == th]
    if len(s) == 0: continue
    print(f'-{th}%: n={len(s)}  平均再跌 {s["further_drop_%"].mean():.1f}%  中位再跌 {s["further_drop_%"].median():.1f}%  '
          f'1年 {s["r1y_%"].mean():.0f}%  2年 {s["r2y_%"].mean():.0f}%  3年 {s["r3y_%"].mean():.0f}%  '
          f'3年最差 {s["r3y_%"].min() if s["r3y_%"].notna().any() else "NA"}')

json.dump(df.to_dict('records'), open('d84_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved')
