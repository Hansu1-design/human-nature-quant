# -*- coding: utf-8 -*-
"""
d82: "十年三次机会" —— 勃发段统计
ZigZag 抽摆动，看 >=50/80/100/150% 的上涨段各有几次、占多少时间、贡献多少收益
"""
import json
import numpy as np
import pandas as pd

sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)
sse = sse[sse['d'] >= '2005-01-01'].reset_index(drop=True)
C = sse['close'].values
D = sse['d'].values
n = len(C)
print(f'样本 {D[0]} ~ {D[-1]}  {n} 根')


def zigzag(C, th=0.25):
    """返回 [(i_low, i_high, up_pct)] 上涨段"""
    piv = [(0, 'L')]
    trend = None
    ext_i, ext_v = 0, C[0]
    lows = [0]
    ups = []
    last_low = 0
    for i in range(1, len(C)):
        if C[i] > ext_v:
            ext_v, ext_i = C[i], i
            if trend != 'up':
                trend = 'up'
                up_start = last_low
        if C[i] < ext_v * (1 - th):
            # 转入下跌
            if trend == 'up' and ext_i > up_start:
                ups.append((up_start, ext_i, C[ext_i] / C[up_start] - 1))
            trend = 'down'
            ext_v, ext_i = C[i], i
            last_low = i
        elif trend == 'down' and C[i] < ext_v:
            ext_v, ext_i = C[i], i
            last_low = i
    if trend == 'up' and ext_i > up_start:
        ups.append((up_start, ext_i, C[ext_i] / C[up_start] - 1))
    return ups


ups = zigzag(C, 0.25)
df = pd.DataFrame([{'i0': a, 'i1': b, 'ret': r,
                    'start': str(pd.Timestamp(D[a]).date()), 'end': str(pd.Timestamp(D[b]).date()),
                    'months': round((pd.Timestamp(D[b]) - pd.Timestamp(D[a])).days / 30.44, 1),
                    'p0': round(C[a]), 'p1': round(C[b])} for a, b, r in ups])
df = df.sort_values('start').reset_index(drop=True)
print('\n=== ZigZag(25%) 全部上涨段 ===')
print(df.to_string(index=False))

OUT = {'all_ups': df.to_dict('records')}

# 按阈值分层
print('\n=== 不同阈值下"几次机会" ===')
lay = []
for th in [0.5, 0.8, 1.0, 1.5, 2.0]:
    sel = df[df['ret'] >= th]
    # 合并时间重叠（按 start 排序去重：若与前一段重叠取更长的）
    keep = []
    for _, r in sel.iterrows():
        if keep and r['i0'] <= keep[-1]['i1']:
            if r['ret'] > keep[-1]['ret']:
                keep[-1] = r
        else:
            keep.append(r)
    keep = pd.DataFrame(keep)
    if len(keep) == 0:
        lay.append({'th': int(th * 100), 'n': 0}); continue
    days = sum((pd.Timestamp(x['end']) - pd.Timestamp(x['start'])).days for _, x in keep.iterrows())
    tot = (pd.Timestamp(D[-1]) - pd.Timestamp(D[0])).days
    # 复合
    comp = float(np.prod(1 + keep['ret'].values) - 1)
    lay.append({'th': int(th * 100), 'n': len(keep), 'days': days, 'share_%': round(days / tot * 100, 1),
                'compound_%': round(comp * 100, 0), 'avg_months': round(keep['months'].mean(), 1),
                'list': [f"{x['start']}→{x['end']} {x['ret']*100:.0f}%" for _, x in keep.iterrows()]})
lay = pd.DataFrame([{k: v for k, v in x.items() if k != 'list'} for x in lay])
print(lay.to_string(index=False))
OUT['layers'] = lay.to_dict('records')
for x in lay.itertuples():
    pass
for x in [y for y in OUT['layers']]:
    pass

# 明细
for th in [0.5, 1.0]:
    sel = df[df['ret'] >= th]
    keep = []
    for _, r in sel.iterrows():
        if keep and r['i0'] <= keep[-1]['i1']:
            if r['ret'] > keep[-1]['ret']:
                keep[-1] = r
        else:
            keep.append(r)
    print(f'\n--- >= {int(th*100)}% 明细 ---')
    for r in keep:
        print(f"  {r['start']} → {r['end']}  {r['p0']}→{r['p1']}  {r['ret']*100:+.0f}%  {r['months']}月")

# 全程持有对比
bh = C[-1] / C[0] - 1
yrs = (pd.Timestamp(D[-1]) - pd.Timestamp(D[0])).days / 365.25
print(f'\n买入持有 2005-01→2026-11: {bh*100:+.0f}%  CAGR {((1+bh)**(1/yrs)-1)*100:.2f}%  ({yrs:.1f}年)')
OUT['buyhold'] = {'ret_%': round(bh * 100, 0), 'cagr_%': round(((1 + bh) ** (1 / yrs) - 1) * 100, 2), 'yrs': round(yrs, 1)}

# 确认成本：起点后延迟 N 个月才买，还剩多少
print('\n=== 确认成本（起点后延迟买入，保留的涨幅） ===')
rows = []
for th in [0.8, 1.0]:
    sel = df[df['ret'] >= th]
    keep = []
    for _, r in sel.iterrows():
        if keep and r['i0'] <= keep[-1]['i1']:
            if r['ret'] > keep[-1]['ret']:
                keep[-1] = r
        else:
            keep.append(r)
    for lag in [0, 1, 3, 6]:
        rets = []
        for r in keep:
            j = min(r['i1'], int(np.searchsorted(D, np.datetime64(pd.Timestamp(r['start']) + pd.DateOffset(months=lag)))))
            if j <= r['i1']:
                rets.append(C[r['i1']] / C[j] - 1)
        rows.append({'th': int(th * 100), 'lag_m': lag, 'n': len(rets),
                     'avg_left_%': round(np.mean(rets) * 100, 1),
                     'compound_%': round((np.prod([1 + x for x in rets]) - 1) * 100, 0)})
cf = pd.DataFrame(rows)
print(cf.to_string(index=False))
OUT['confirm_cost'] = rows

json.dump(OUT, open('d82_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved')
