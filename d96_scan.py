# -*- coding: utf-8 -*-
"""
d96: 闸门细扫（2700~3450，步长 50）+ 去最大一笔的稳健性 + ETF价格错配检查
要点：如果只有 3000 一个点好 -> 过拟合；如果 2950~3050 成片 -> 真实结构
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}
m = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)

def bt(th, start='2015-01-05', up_cap=3900, drop_best=False):
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, eq, ent, entd = 0.0, False, 0.0, 0, 1.0, 0.0, None
    out, in_days = [], 0
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i-1]
            if c >= peak: peak, peak_i = c, i
            in_days += 1
        cheep = m['close'].iloc[i] <= th
        cross = m['close'].iloc[i] > m['ma20'].iloc[i] and m['close'].iloc[i-1] <= m['ma20'].iloc[i-1]
        if pos == 0:
            if cheep and cross and m['close'].iloc[i] < up_cap:
                pos, half, peak, peak_i, ent, entd = 1.0, False, c, i, c, str(m['d'].iloc[i].date())
        else:
            why = None
            if (peak-c)/peak >= .10 and i-peak_i >= 20: why = '回撤'
            elif m['amt_q'].iloc[i] >= .95 and m['close'].iloc[i] > 3500: why = '热度'
            if why and not half: half = True
            elif why and half:
                out.append(dict(buy=entd, sell=str(m['d'].iloc[i].date()), ret=round(float(c/ent-1)*100, 1)))
                pos, half = 0.0, False
    ndays = len(m) - i0
    tot = (eq - 1) * 100
    if drop_best and out:
        best = max(out, key=lambda x: x['ret'])
        # 用几何方式剔除
        factor = 1 + best['ret']/100
        tot = ((eq / factor) - 1) * 100
    return dict(total=round(float(tot), 1), n=len(out), expo=round(in_days/ndays*100, 1),
                cagr=round(float(eq ** (250.0/ndays) - 1) * 100, 2), trades=out)

print('=== 闸门细扫（名义点位，2015-01 起点）===')
print(f"{'阈值':>6}{'笔数':>5}{'在场':>8}{'总收益':>9}{'去掉最大一笔':>13}{'在场年化':>9}")
scan = []
for th in range(2700, 3451, 50):
    r = bt(th); rb = bt(th, drop_best=True)
    scan.append(dict(th=th, n=r['n'], expo=r['expo'], total=r['total'], total_db=rb['total'], cagr=r['cagr']))
    print(f"{th:>6}{r['n']:>5}{r['expo']:>7.1f}%{r['total']:>8.1f}%{rb['total']:>12.1f}%{r['cagr']:>8.2f}%")
OUT['scan'] = scan

# 平台判定
ts = [x for x in scan if 2900 <= x['th'] <= 3150]
print(f"\n2900-3150 区间：收益 {min(x['total'] for x in ts):.1f}% ~ {max(x['total'] for x in ts):.1f}%，跨度 {max(x['total'] for x in ts)-min(x['total'] for x in ts):.1f}pt")
best = max(scan, key=lambda x: x['total'])
print(f"峰值在 {best['th']}（{best['total']}%）；相邻 ±50：{bt(best['th']-50)['total']}% / {bt(best['th']+50)['total']}%")

# 去最大一笔后重排
bestdb = max(scan, key=lambda x: x['total_db'])
print(f"去掉最大一笔后，峰值移到 {bestdb['th']}（{bestdb['total_db']}%）")
OUT['peak'] = dict(th=best['th'], total=best['total'], peak_db=bestdb['th'], total_db=bestdb['total_db'],
                   range_2900_3150=[min(x['total'] for x in ts), max(x['total'] for x in ts)])

# 8 笔明细与各笔贡献
r = bt(3000)
print('\n=== 3000 点 8 笔，逐笔贡献 ===')
eq = 1.0
for x in r['trades']:
    eq *= (1 + x['ret']/100)
    print(f"  {x['buy']} → {x['sell']}  {x['ret']:+6.1f}%   累计 {(eq-1)*100:>7.1f}%")
print(f"  去掉最大一笔：{bt(3000,drop_best=True)['total']}%")

# ---------- ETF 价格错配检查 ----------
print('\n=== 错配检查：上证 2900-3100 时，510300 的价格范围 ===')
sub = m[(m['close'] >= 2900) & (m['close'] <= 3100)]
print(f"样本 {len(sub)} 天，ETF 价格 {sub['etf'].min():.3f} ~ {sub['etf'].max():.3f}，均值 {sub['etf'].mean():.3f}，标准差 {sub['etf'].std():.3f}")
print(f"变异系数 {sub['etf'].std()/sub['etf'].mean()*100:.1f}%  （越大说明错配越严重）")
corr = m[['close', 'etf']].corr().iloc[0, 1]
print(f"上证与 510300 日收盘相关系数 {corr:.4f}")
OUT['etf_mismatch'] = dict(n=int(len(sub)), lo=round(float(sub['etf'].min()), 3), hi=round(float(sub['etf'].max()), 3),
                           cv=round(float(sub['etf'].std()/sub['etf'].mean()*100), 1), corr=round(float(corr), 4))

# 用 ETF 自身价格分位做闸门（规避错配）
print('\n=== 对照：改用 510300 自身 5 年分位 <= 0.25 做闸门 ===')
m['q_etf'] = m['etf'].rolling(1250, min_periods=250).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
def bt2(col, th, start='2015-01-05', up_cap=3900):
    mm = m.dropna(subset=[col]).reset_index(drop=True)
    i0 = int(np.argmin(np.abs((mm['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, eq, ent = 0.0, False, 0.0, 0, 1.0, 0.0
    out = []
    for i in range(i0, len(mm)):
        c = mm['etf'].iloc[i]
        if pos > 0:
            eq *= c / mm['etf'].iloc[i-1]
            if c >= peak: peak, peak_i = c, i
        cheep = (not np.isnan(mm[col].iloc[i])) and mm[col].iloc[i] <= th
        cross = mm['close'].iloc[i] > mm['ma20'].iloc[i] and mm['close'].iloc[i-1] <= mm['ma20'].iloc[i-1]
        if pos == 0:
            if cheep and cross and mm['close'].iloc[i] < up_cap:
                pos, half, peak, peak_i, ent = 1.0, False, c, i, c
        else:
            why = None
            if (peak-c)/peak >= .10 and i-peak_i >= 20: why = '回撤'
            elif mm['amt_q'].iloc[i] >= .95 and mm['close'].iloc[i] > 3500: why = '热度'
            if why and not half: half = True
            elif why and half:
                out.append(round(float(c/ent-1)*100, 1)); pos, half = 0.0, False
    nd = len(mm) - i0
    return dict(total=round(float((eq-1)*100), 1), n=len(out), cagr=round(float(eq ** (250.0/nd)-1)*100, 2), trades=out)

re = bt2('q_etf', 0.25)
print(f"  510300分位<=0.25：{re['n']} 笔，合计 {re['total']}%，年化 {re['cagr']}%")
print(f"  逐笔 {re['trades']}")
rn = bt2('q_nom', 0.25)
print(f"  上证分位<=0.25：{rn['n']} 笔，合计 {rn['total']}%，年化 {rn['cagr']}%")
OUT['etf_gate'] = dict(etf=re, nom=rn)

with open(os.path.join(B, 'd96_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nsaved d96_result.json')
