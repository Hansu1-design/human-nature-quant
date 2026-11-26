# -*- coding: utf-8 -*-
"""
d72: 「只要不退市，长期持有就行」—— 个股层面检验
  指数有下有底（成分股换血 + 退市剔除），个股没有。他的操作方式是找票，不是买指数。
"""
import json
import os
import numpy as np
import pandas as pd

P = 'G:/AStockData/supplement/daily_tencent_full'
files = sorted(f for f in os.listdir(P) if f.endswith('.parquet'))
OUT = {}
print(f"股票文件 {len(files)} 个")

# 一次性读入（只保留 date / close）
cache = {}
for k, f in enumerate(files):
    code = f.replace('.parquet', '')
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
    cache[code] = (d['date'].values.astype('datetime64[D]'), d['close'].values.astype(float))
    if (k + 1) % 800 == 0:
        print(f"  {k+1}/{len(files)}")
print(f"有效个股 {len(cache)}")

idx = pd.read_parquet('idx_sse_all.parquet')
idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
idx['d'] = pd.to_datetime(idx['date'])
idx = idx.dropna(subset=['close']).sort_values('d')
etf = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
etf = etf.rename(columns={'日期': 'd', '收盘': 'close'})
etf['d'] = pd.to_datetime(etf['d'])
etf['close'] = pd.to_numeric(etf['close'], errors='coerce')
etf = etf.dropna(subset=['close']).sort_values('d')

def lvl_at(s, dt):
    x = s[s['d'] <= pd.Timestamp(dt)]
    return float(x['close'].iloc[-1]) if len(x) else np.nan

STARTS = ['2015-01-05', '2018-01-02', '2021-01-04', '2024-01-02']
CALC = {}
for sd in STARTS:
    t0 = np.datetime64(pd.Timestamp(sd).date())
    recs = []
    for code, (dts, px) in cache.items():
        if dts[0] > t0:
            continue
        j = np.searchsorted(dts, t0, side='right') - 1
        if j < 0:
            continue
        recs.append((code, px[j], px[-1], dts[-1], px.min()))
    df = pd.DataFrame(recs, columns=['code', 'p0', 'p1', 'last', 'low'])
    df['ret'] = df['p1'] / df['p0'] - 1
    df['surv'] = df['last'] >= np.datetime64('2025-06-30')
    df = df[df['ret'].between(-1.0, 30)]
    CALC[sd] = df
    print(f"{sd}: n={len(df)}  存活 {int(df['surv'].sum())} ({df['surv'].mean()*100:.1f}%)")

print("\n" + "=" * 92)
print("个股买入持有到 2025-12-31 的收益分布")
print("=" * 92)
print(f"{'起点':>11} {'n':>5} {'中位':>8} {'均值':>8} {'>0':>7} {'翻倍':>6} "
      f"{'腰斩':>6} {'-80%':>6} {'存活中位':>9} {'中断中位':>9} {'上证':>8} {'300ETF':>8}")
T = []
for sd in STARTS:
    df = CALC[sd]
    r = df['ret']
    row = dict(start=sd, n=len(df), med=round(float(r.median()) * 100, 1),
               mean=round(float(r.mean()) * 100, 1),
               pos=round(float((r > 0).mean()) * 100, 1),
               dbl=round(float((r >= 1).mean()) * 100, 1),
               half=round(float((r <= -0.5).mean()) * 100, 1),
               deep=round(float((r <= -0.8).mean()) * 100, 1),
               med_surv=round(float(df.loc[df['surv'], 'ret'].median()) * 100, 1),
               med_dead=round(float(df.loc[~df['surv'], 'ret'].median()) * 100, 1)
               if (~df['surv']).any() else None,
               idx=round((lvl_at(idx, '2025-12-31') / lvl_at(idx, sd) - 1) * 100, 1),
               etf=round((lvl_at(etf, '2025-12-31') / lvl_at(etf, sd) - 1) * 100, 1))
    T.append(row)
    print(f"{sd:>11} {row['n']:5d} {row['med']:7.1f}% {row['mean']:7.1f}% {row['pos']:6.1f}% "
          f"{row['dbl']:5.1f}% {row['half']:5.1f}% {row['deep']:5.1f}% "
          f"{row['med_surv']:8.1f}% {str(row['med_dead']):>9} {row['idx']:7.1f}% {row['etf']:7.1f}%")
OUT['hold'] = T

print("\n数据中断（最后交易日早于 2025-06-30，退市或长期停牌的代理）：")
for sd in STARTS:
    df = CALC[sd]
    dead = df[~df['surv']]
    if len(dead) == 0:
        continue
    print(f"  {sd}: {len(dead)} 只 ({len(dead)/len(df)*100:.1f}%)  "
          f"中位收益 {dead['ret'].median()*100:.1f}%  "
          f"其中 <=-50% 的占 {(dead['ret']<=-0.5).mean()*100:.0f}%  "
          f"<=-80% 的占 {(dead['ret']<=-0.8).mean()*100:.0f}%")
    OUT[f'dead_{sd}'] = dict(n=int(len(dead)), pct=round(float(len(dead) / len(df) * 100), 1),
                             med=round(float(dead['ret'].median()) * 100, 1))

print("\n个股 vs 同期 300ETF：")
for sd in STARTS:
    df = CALC[sd]
    e = lvl_at(etf, '2025-12-31') / lvl_at(etf, sd) - 1
    print(f"  {sd}: 跑赢 ETF {(df['ret'] > e).mean()*100:.1f}%   ETF 同期 {e*100:+.1f}%")
    OUT[f'beat_etf_{sd}'] = round(float((df['ret'] > e).mean()) * 100, 1)

# 2015 起点：跌幅分布（用"曾达到的最低"）
print("\n2015-01 起点：持有过程中最大回撤（用 path 内最低收盘 / 起点价）：")
df = CALC['2015-01-05']
df['mdd'] = df['low'] / df['p0'] - 1
print(f"  中位 {(df['mdd'].median())*100:.1f}%   <=-50% 的占 {(df['mdd']<=-0.5).mean()*100:.1f}%  "
      f"<=-70% 的占 {(df['mdd']<=-0.7).mean()*100:.1f}%")
OUT['mdd2015'] = dict(med=round(float(df['mdd'].median()) * 100, 1),
                      p50=round(float((df['mdd'] <= -0.5).mean()) * 100, 1),
                      p70=round(float((df['mdd'] <= -0.7).mean()) * 100, 1))

json.dump(OUT, open('d72_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print("\n[saved] d72_result.json")
