# -*- coding: utf-8 -*-
"""
d71: 把买点锚从「绝对点位 / 价格分位」换成「回吐上轮牛市涨幅的比例」
     三条锚 × 11 个起点，同一套出场规则，同一标的
"""
import json
import numpy as np
import pandas as pd

OUT = {}

# ---------- 数据 ----------
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
df['ma60'] = df['close'].rolling(60).mean()
df['ma250'] = df['close'].rolling(250).mean()
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
df['q5y'] = df['close'].rolling(1220).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)

# ---------- 回吐比例（实时可得） ----------
# 顶：历史最高（含当日）。确认：自高点回撤 > 20% 即认定该顶成立
# 启动点：确认顶之前 500 交易日内的最低收盘（近似"牛市起点"）
C = df['close'].values
NN = len(df)
start_lvl = np.full(NN, np.nan)
peak_lvl = np.full(NN, np.nan)
gb = np.full(NN, np.nan)
peak = C[0]
peak_i = 0
cur_start = np.nan
confirmed = False
for i in range(NN):
    if C[i] > peak:
        peak, peak_i = C[i], i
        confirmed = False
    if not confirmed and C[i] < peak * 0.80:
        lo = C[max(0, peak_i - 500):peak_i + 1]
        cur_start = float(lo.min())
        confirmed = True
    peak_lvl[i] = peak
    start_lvl[i] = cur_start
    if confirmed and not np.isnan(cur_start) and peak > cur_start:
        gb[i] = (peak - C[i]) / (peak - cur_start)
df['giveback'] = gb
df['peak_now'] = peak_lvl
df['bull_start'] = start_lvl

print(f"数据 {df['d'].iloc[0].date()} ~ {df['d'].iloc[-1].date()}  n={NN}")
print("回吐比例在各历史买点的读数：")
for dt in ['2011-09-05', '2012-12-04', '2016-03-02', '2018-12-27', '2024-01-22', '2024-09-18']:
    k = int(np.argmin(np.abs((df['d'] - pd.Timestamp(dt)).values)))
    print(f"   {dt}  点位 {C[k]:.0f}  顶 {peak_lvl[k]:.0f}  启动 {start_lvl[k]:.0f}  "
          f"回吐 {gb[k]*100 if not np.isnan(gb[k]) else float('nan'):.1f}%")

# ---------- 回测引擎 ----------
def stats(eq):
    ret = np.diff(eq) / eq[:-1]
    cagr = eq[-1] ** (243 / len(eq)) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    sh = ret.mean() / ret.std() * np.sqrt(243) if ret.std() > 0 else 0.0
    return {'total': round((eq[-1] - 1) * 100, 1), 'cagr': round(cagr * 100, 2),
            'mdd': round(mdd * 100, 1), 'sharpe': round(float(sh), 2)}

def backtest(r0, r1, anchor, th, sell_th=3500, dd=0.10, dd_days=20, heat_q=0.95,
             warm=20, reenter=True, cost=0.0005):
    sub = df.iloc[r0:r1 + 1].reset_index(drop=True)
    n = len(sub)
    c = sub['close'].values
    e = sub['etf'].values
    m20, m250 = sub['ma20'].values, sub['ma250'].values
    ht, gbv, q5 = sub['heat'].values, sub['giveback'].values, sub['q5y'].values
    r = np.zeros(n)
    r[1:] = e[1:] / e[:-1] - 1

    pos, half, pk, pk_i, below = 0.0, False, 0.0, 0, 0
    last_exit, need_re = 0.0, False
    ps = np.zeros(n)
    eps, cur = [], None
    for i in range(n):
        ci = c[i]
        below = below + 1 if ci < 3000 else 0
        if pos > 0:
            if ci > pk:
                pk, pk_i = ci, i
            hit_dd = (ci <= pk * (1 - dd)) and (i - pk_i >= dd_days)
            hit_ht = (not np.isnan(ht[i])) and ht[i] >= heat_q and ci > sell_th
            if hit_dd or hit_ht:
                cur.update(exit=str(sub['d'].iloc[i])[:10], ret=round((e[i] / cur['e0'] - 1) * 100, 1),
                           hold=int(i - cur['i']),
                           reason='回撤' if hit_dd else '热度')
                eps.append(cur)
                last_exit = ci
                need_re = hit_dd
                cur, pos, half = None, 0.0, False
            elif not half and ci > sell_th:
                pos, half = 0.5, True
        if pos == 0 and (r0 + i) > 260:
            ok = False
            if anchor == 'A':            # 绝对点位
                ok = below >= warm and ci < 3000 and ci > m20[i] and c[i-1] <= m20[i-1]
            elif anchor == 'B':          # 5 年价格分位
                ok = (not np.isnan(q5[i])) and q5[i] <= th and ci > m20[i] and c[i-1] <= m20[i-1]
            elif anchor == 'C':          # 回吐牛市涨幅
                ok = (not np.isnan(gbv[i])) and gbv[i] >= th and ci > m20[i] and c[i-1] <= m20[i-1]
            if need_re and reenter and (not np.isnan(m250[i])) and ci > m250[i] and ci > last_exit:
                ok = True
            if ok:
                need_re = False
                pos, half, pk, pk_i = 1.0, False, ci, i
                cur = {'entry': str(sub['d'].iloc[i])[:10], 'entry_lv': round(ci, 0),
                       'e0': e[i], 'i': i}
        ps[i] = pos
    sr = np.zeros(n)
    sr[1:] = ps[:-1] * r[1:]
    turn = np.abs(np.diff(np.concatenate([[0], ps])))
    sr = sr - turn * cost
    eq = np.cumprod(1 + sr)
    bh = np.cumprod(1 + r)
    return dict(stats=stats(eq), bh=stats(bh), expo=round(float((ps > 0).mean()) * 100, 1),
                n=len(eps), eps=eps, eq=eq, bh_eq=bh,
                lvl0=round(float(c[0]), 0), lvl1=round(float(c[-1]), 0))

# ---------- 主回测 ----------
i_end = NN - 1
start_dates = ['2015-01-05', '2015-06-15', '2016-01-04', '2017-01-03', '2018-01-02',
               '2019-01-02', '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
res = {}
print("\n" + "=" * 92)
print("三锚 × 11 起点（同一出场规则：3500 减半 / 回撤10%+距顶20日 / 热度95%分位且>3500 清仓 / 站回年线回补）")
print("=" * 92)
print(f"{'起点':>11} {'起点位':>6} {'买入持有':>8} | {'A:3000':>8} {'MDD':>6} {'空仓':>5} "
      f"| {'B:q5<=25':>9} {'MDD':>6} {'空仓':>5} | {'C:回吐>=70':>10} {'MDD':>6} {'空仓':>5}")
for sd in start_dates:
    r0 = int(np.argmin(np.abs((df['d'] - pd.Timestamp(sd)).values)))
    if r0 + 300 > i_end:
        continue
    row = {}
    bh = None
    for k, th in [('A', 3000), ('B', 0.25), ('C', 0.70)]:
        x = backtest(r0, i_end, k, th)
        row[k] = x
        if bh is None:
            bh = x['bh']
    res[sd] = row
    print(f"{sd:>11} {row['A']['lvl0']:6.0f} {bh['total']:7.1f}% "
          f"| {row['A']['stats']['total']:7.1f}% {row['A']['stats']['mdd']:5.1f}% {row['A']['expo']:4.1f}% "
          f"| {row['B']['stats']['total']:8.1f}% {row['B']['stats']['mdd']:5.1f}% {row['B']['expo']:4.1f}% "
          f"| {row['C']['stats']['total']:9.1f}% {row['C']['stats']['mdd']:5.1f}% {row['C']['expo']:4.1f}%")

# 胜率
print()
for k, lab in [('A', '3000 点'), ('B', '5年分位<=25'), ('C', '回吐>=70%')]:
    win = sum(1 for sd in res if res[sd][k]['stats']['total'] > res[sd]['A']['bh']['total'])
    print(f"  锚 {lab:12s} 跑赢买入持有 {win}/{len(res)} 次")

# 主窗口（2015-01 起）
main = res['2015-01-05']
print("\n2015-01 起主窗口逐笔：")
for k in ['A', 'B', 'C']:
    print(f"\n  [{k}] 总收益 {main[k]['stats']['total']}%  CAGR {main[k]['stats']['cagr']}%  "
          f"MDD {main[k]['stats']['mdd']}%  空仓 {main[k]['expo']}%  {main[k]['n']} 笔")
    for ep in main[k]['eps']:
        print(f"     {ep['entry']} @{ep['entry_lv']:.0f} → {ep['exit']} "
              f"{ep['ret']:+.1f}%  {ep['hold']}日  {ep['reason']}")

# 阈值敏感性
print("\n回吐阈值敏感性（2015-01 起）：")
sens = []
for th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    _r0 = int(np.argmin(np.abs((df['d'] - pd.Timestamp('2015-01-05')).values)))
    x = backtest(_r0, i_end, 'C', th)
    sens.append(dict(th=th, **x['stats'], expo=x['expo'], n=x['n'],
                     first=[e['entry'] for e in x['eps']]))
    print(f"  >= {th*100:.0f}%: 总 {x['stats']['total']:7.1f}%  CAGR {x['stats']['cagr']:5.2f}%  "
          f"MDD {x['stats']['mdd']:5.1f}%  空仓 {x['expo']:4.1f}%  {x['n']} 笔  "
          f"首笔 {x['eps'][0]['entry'] if x['eps'] else '-'}")
OUT['sens'] = sens

OUT['main'] = {k: dict(stats=v['stats'], bh=v['bh'], expo=v['expo'], n=v['n'], eps=v['eps'])
               for k, v in main.items()}
OUT['grid'] = {sd: {k: dict(stats=res[sd][k]['stats'], expo=res[sd][k]['expo'], n=res[sd][k]['n'],
                            bh=res[sd]['A']['bh']) for k in ['A', 'B', 'C']} for sd in res}

json.dump(OUT, open('d71_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
pd.DataFrame({'d': df['d'], 'close': df['close'], 'giveback': df['giveback'],
              'peak': df['peak_now'], 'start': df['bull_start']}).to_parquet('d71_gb.parquet', index=False)
print("\n[saved] d71_result.json / d71_gb.parquet")
