# -*- coding: utf-8 -*-
"""
d74: 修正回吐锚的顶定义（滚动 1000 日，而非历史最高）
      + 分档买入（60/70/80 各 1/3）
   上一版把 peak 设成"历史最高"，2015 之后 peak 一直卡在 5166，
   导致"回吐比例"变成"距 2015 年顶的距离"，C 退化成 A 的另一种写法。
"""
import json
import numpy as np
import pandas as pd

OUT = {}

idx = pd.read_parquet('idx_sse_all.parquet')
idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
idx['amount'] = pd.to_numeric(idx['amount'], errors='coerce')
idx['d'] = pd.to_datetime(idx['date'])
idx = idx.dropna(subset=['close']).sort_values('d').reset_index(drop=True)

etf = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
etf = etf.rename(columns={'日期': 'd', '收盘': 'close'})
etf['d'] = pd.to_datetime(etf['d'])
etf['close'] = pd.to_numeric(etf['close'], errors='coerce')
etf = etf.dropna(subset=['close']).sort_values('d')[['d', 'close']].rename(columns={'close': 'etf'})

df = pd.merge_asof(idx.sort_values('d'), etf.sort_values('d'), on='d', direction='backward')
df = df.dropna(subset=['etf']).reset_index(drop=True)
df['ma20'] = df['close'].rolling(20).mean()
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
df['q5y'] = df['close'].rolling(1220).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)

C = df['close'].values
N = len(df)

# ---- 回吐：滚动 W 日内最高作顶，顶之前 W 日最低作启动 ----
def make_gb(W=1000):
    gb = np.full(N, np.nan)
    pv = np.full(N, np.nan)
    sv = np.full(N, np.nan)
    for i in range(N):
        a = max(0, i - W + 1)
        seg = C[a:i + 1]
        k = int(np.argmax(seg))
        pk = seg[k]
        gi = a + k
        sa = max(0, gi - W)
        st = C[sa:gi + 1].min()
        pv[i], sv[i] = pk, st
        if pk > st:
            gb[i] = (pk - C[i]) / (pk - st)
    return gb, pv, sv

gb, pv, sv = make_gb(1000)
df['gb'] = gb
print(f"数据 {df['d'].iloc[0].date()} ~ {df['d'].iloc[-1].date()}  n={N}")
print("\n修正后（滚动1000日顶）在各关键时点的读数：")
for dt in ['2008-10-28', '2011-09-05', '2012-12-04', '2016-03-02', '2018-12-27',
           '2024-01-22', '2024-09-18', '2025-04-07']:
    k = int(np.argmin(np.abs((df['d'] - pd.Timestamp(dt)).values)))
    print(f"   {dt}  点位 {C[k]:6.0f}  顶 {pv[k]:6.0f}  启动 {sv[k]:6.0f}  回吐 {gb[k]*100:6.1f}%")

def stats(eq):
    ret = np.diff(eq) / eq[:-1]
    cagr = eq[-1] ** (243 / len(eq)) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    sh = ret.mean() / ret.std() * np.sqrt(243) if ret.std() > 0 else 0.0
    return {'total': round((eq[-1] - 1) * 100, 1), 'cagr': round(cagr * 100, 2),
            'mdd': round(mdd * 100, 1), 'sharpe': round(float(sh), 2)}

def bt(r0, r1, mode='A', th=3000, ladder=(0.60, 0.70, 0.80),
       sell_th=3500, dd=0.10, dd_days=20, heat_q=0.95, warm=20, cost=0.0005):
    sub = df.iloc[r0:r1 + 1].reset_index(drop=True)
    n = len(sub)
    c, e = sub['close'].values, sub['etf'].values
    m20, ht, gv, q5 = sub['ma20'].values, sub['heat'].values, sub['gb'].values, sub['q5y'].values
    r = np.zeros(n)
    r[1:] = e[1:] / e[:-1] - 1
    pos, half, pk, pk_i, below = 0.0, False, 0.0, 0, 0
    ps = np.zeros(n)
    eps, cur, adds = [], None, []
    for i in range(n):
        ci = c[i]
        below = below + 1 if ci < 3000 else 0
        if pos > 0:
            if ci > pk:
                pk, pk_i = ci, i
            if (ci <= pk * (1 - dd) and i - pk_i >= dd_days) or \
               ((not np.isnan(ht[i])) and ht[i] >= heat_q and ci > sell_th):
                cur['exit'] = str(sub['d'].iloc[i])[:10]
                cur['ret'] = round((e[i] / cur['e0'] - 1) * 100, 1)
                cur['hold'] = int(i - cur['i'])
                cur['adds'] = adds
                eps.append(cur)
                ps[i] = 0.0
                pos, half, cur, adds = 0.0, False, None, []
                continue
            if not half and ci > sell_th:
                pos, half = 0.5, True
        if i + r0 > 260:
            ok, tgt = False, pos
            if mode == 'A':
                ok = below >= warm and ci < 3000 and ci > m20[i] and c[i-1] <= m20[i-1]
                tgt = 1.0
            elif mode == 'B':
                ok = (not np.isnan(q5[i])) and q5[i] <= th and ci > m20[i] and c[i-1] <= m20[i-1]
                tgt = 1.0
            elif mode == 'C1':      # 单点
                ok = (not np.isnan(gv[i])) and gv[i] >= th and ci > m20[i] and c[i-1] <= m20[i-1]
                tgt = 1.0
            elif mode == 'C2':      # 分档
                if not np.isnan(gv[i]):
                    g = gv[i]
                    t = sum(1 for x in ladder if g >= x) / len(ladder)
                    if t > 0:
                        if pos == 0:
                            ok = ci > m20[i] and c[i-1] <= m20[i-1]
                        else:
                            ok = t > pos
                        tgt = t
            if ok and tgt > pos:
                if cur is None:
                    cur = {'entry': str(sub['d'].iloc[i])[:10], 'entry_lv': round(ci, 0),
                           'e0': e[i], 'i': i}
                    pk, pk_i = ci, i
                else:
                    adds.append((str(sub['d'].iloc[i])[:10], round(ci, 0), round(tgt, 2)))
                pos = tgt
        ps[i] = pos
    sr = np.zeros(n)
    sr[1:] = ps[:-1] * r[1:]
    turn = np.abs(np.diff(np.concatenate([[0], ps])))
    sr = sr - turn * cost
    eq = np.cumprod(1 + sr)
    bh = np.cumprod(1 + r)
    return dict(stats=stats(eq), bh=stats(bh), expo=round(float((ps > 0).mean()) * 100, 1),
                n=len(eps), eps=eps, lvl0=round(float(c[0]), 0))

STARTS = ['2015-01-05', '2016-01-04', '2017-01-03', '2018-01-02', '2019-01-02',
          '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
i_end = N - 1
print("\n" + "=" * 96)
print("修正顶定义后：A(3000点) / B(5年分位25) / C1(回吐>=70单点) / C2(回吐60/70/80分三档)")
print("=" * 96)
print(f"{'起点':>11} {'买入持有':>9} | {'A':>8} {'MDD':>6} | {'B':>8} {'MDD':>6} "
      f"| {'C1':>8} {'MDD':>6} | {'C2':>8} {'MDD':>6} {'空仓':>6}")
G = {}
for sd in STARTS:
    r0 = int(np.argmin(np.abs((df['d'] - pd.Timestamp(sd)).values)))
    row = {}
    for mode, th in [('A', 3000), ('B', 0.25), ('C1', 0.70), ('C2', 0)]:
        row[mode] = bt(r0, i_end, mode, th)
    G[sd] = row
    b = row['A']['bh']
    print(f"{sd:>11} {b['total']:8.1f}% | {row['A']['stats']['total']:7.1f}% {row['A']['stats']['mdd']:5.1f}% "
          f"| {row['B']['stats']['total']:7.1f}% {row['B']['stats']['mdd']:5.1f}% "
          f"| {row['C1']['stats']['total']:7.1f}% {row['C1']['stats']['mdd']:5.1f}% "
          f"| {row['C2']['stats']['total']:7.1f}% {row['C2']['stats']['mdd']:5.1f}% {row['C2']['expo']:5.1f}%")

print()
for m in ['A', 'B', 'C1', 'C2']:
    win = sum(1 for sd in G if G[sd][m]['stats']['total'] > G[sd]['A']['bh']['total'])
    print(f"  {m:3s} 跑赢买入持有 {win}/{len(G)}")

print("\n2015-01 起逐笔：")
for m in ['C1', 'C2']:
    print(f"\n  [{m}] 总 {G['2015-01-05'][m]['stats']['total']}%  "
          f"CAGR {G['2015-01-05'][m]['stats']['cagr']}%  MDD {G['2015-01-05'][m]['stats']['mdd']}%")
    for ep in G['2015-01-05'][m]['eps']:
        print(f"     {ep['entry']} @{ep['entry_lv']:.0f} → {ep['exit']} {ep['ret']:+.1f}% "
              f"{ep['hold']}日 加仓{ep['adds']}")

print("\n阈值敏感性（2015-01 起，单点）：")
S = []
for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    r0 = int(np.argmin(np.abs((df['d'] - pd.Timestamp('2015-01-05')).values)))
    x = bt(r0, i_end, 'C1', th)
    S.append(dict(th=th, **x['stats'], expo=x['expo'], n=x['n'],
                  first=x['eps'][0]['entry'] if x['eps'] else '-'))
    print(f"  >= {th*100:.0f}%: 总 {x['stats']['total']:7.1f}%  MDD {x['stats']['mdd']:5.1f}%  "
          f"{x['n']}笔  首笔 {S[-1]['first']}")
OUT['sens_corrected'] = S

print("\n分档位置敏感性（2015-01 起）：")
for lad in [(0.50, 0.60, 0.70), (0.60, 0.70, 0.80), (0.65, 0.75, 0.85), (0.55, 0.70, 0.85)]:
    r0 = int(np.argmin(np.abs((df['d'] - pd.Timestamp('2015-01-05')).values)))
    x = bt(r0, i_end, 'C2', ladder=lad)
    print(f"  {lad}: 总 {x['stats']['total']:7.1f}%  MDD {x['stats']['mdd']:5.1f}%  "
          f"空仓 {x['expo']:5.1f}%  {x['n']}笔")

OUT['grid'] = {sd: {m: dict(stats=G[sd][m]['stats'], expo=G[sd][m]['expo'], n=G[sd][m]['n'],
                            bh=G[sd]['A']['bh']) for m in ['A', 'B', 'C1', 'C2']} for sd in G}
OUT['main_c2'] = dict(stats=G['2015-01-05']['C2']['stats'], eps=G['2015-01-05']['C2']['eps'],
                      expo=G['2015-01-05']['C2']['expo'])
json.dump(OUT, open('d74_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print("\n[saved] d74_result.json")
