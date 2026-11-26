# -*- coding: utf-8 -*-
"""
d97: "接近底"作为信号 —— 把硬阈值改成距离带
口径：
  A 绝对锚：real_cpi / 1880（五轮熊底实际值中位，样本内）
  B 事前锚：real_cpi / min(real_cpi, 过去 8 年)  —— 不依赖未来信息
检验：
  1) 分档前瞻收益（1/2/3 年），看"越接近底收益越高"是否单调
  2) 带宽内的天数、段数、段长（"准备期"有多长）
  3) 首次触带后的路径：之后是否还跌、多久见底
  4) 当前读数：离底多远
"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}
d = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)
d['d'] = pd.to_datetime(d['d'])
N = len(d)

# ---------- 1. 重建 CPI 定基与实际值 ----------
c = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
c['ym'] = c['月份'].str.replace('年', '-').str.replace('月份', '')
c['d'] = pd.to_datetime(c['ym'], format='%Y-%m')
c = c.sort_values('d').reset_index(drop=True)
c['mom'] = (pd.to_numeric(c['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
ix = [100.0]
for i in range(1, len(c)):
    ix.append(ix[-1] * c['mom'].iloc[i])
c['cpi_idx'] = ix
c['avail'] = c['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
d = pd.merge_asof(d, c[['avail', 'cpi_idx']].sort_values('avail'),
                  left_on='d', right_on='avail', direction='backward')
d['real'] = d['close'] * 100.0 / d['cpi_idx']
d = d.dropna(subset=['real']).reset_index(drop=True)
N = len(d)
CPI_T = float(d['cpi_idx'].iloc[-1])
print(f'样本 {N} 根  {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  CPI定基 {CPI_T:.1f}')

# ---------- 2. 两个锚 ----------
ANCHOR = 1880.0                      # 五轮熊底实际值中位（样本内）
d['rA'] = d['real'] / ANCHOR         # 距底倍数，1.00 = 到底

WIN = 8 * 244                        # 过去 8 年（交易日）
d['rmin8'] = d['real'].rolling(WIN, min_periods=244).min()
d['rB'] = d['real'] / d['rmin8']     # 距 8 年低点倍数，1.00 = 正在创新低

OUT['meta'] = dict(n=N, start=str(d.d.iloc[0].date()), end=str(d.d.iloc[-1].date()),
                   cpi_now=round(CPI_T, 1), anchor=ANCHOR,
                   real_now=round(float(d['real'].iloc[-1]), 0),
                   close_now=round(float(d['close'].iloc[-1]), 0))

# ---------- 3. 前瞻收益 ----------
for h, lab in [(244, '1y'), (488, '2y'), (732, '3y')]:
    d[f'f{lab}'] = d['close'].shift(-h) / d['close'] - 1

def bucket_table(col, edges, name):
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (d[col] > lo) & (d[col] <= hi)
        if m.sum() < 30:
            rows.append(dict(lo=round(lo, 2), hi=round(hi, 2), n=int(m.sum()), **{k: None for k in ['f1y', 'f2y', 'f3y', 'win1y']}))
            continue
        r = dict(lo=round(lo, 2), hi=round(hi, 2), n=int(m.sum()), days_pct=round(float(m.mean() * 100), 1))
        for k in ['f1y', 'f2y', 'f3y']:
            s = d.loc[m, k].dropna()
            r[k] = round(float(s.mean() * 100), 1) if len(s) else None
            if k == 'f1y':
                r['win1y'] = round(float((s > 0).mean() * 100), 1) if len(s) else None
                r['med1y'] = round(float(s.median() * 100), 1) if len(s) else None
        rows.append(r)
    OUT[name] = rows
    print(f'\n--- 分档：{name} ---')
    print(f"{'区间':<14}{'天数':>7}{'占比%':>7}{'1年均':>9}{'1年中位':>9}{'1年胜率':>9}{'2年均':>9}{'3年均':>9}")
    for r in rows:
        if r['n'] < 30:
            print(f'  {r["lo"]}~{r["hi"]:<8}{r["n"]:>7}  (样本不足)')
            continue
        print(f'  {r["lo"]}~{r["hi"]:<8}{r["n"]:>7}{r.get("days_pct",0):>7}'
              f'{r["f1y"]:>9}{r["med1y"]:>9}{r["win1y"]:>9}{r["f2y"]:>9}{r["f3y"]:>9}')
    return rows

bucket_table('rA', [0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.60, 2.0, 99], 'A_绝对锚')
bucket_table('rB', [0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.60, 2.0, 99], 'B_事前锚')

# ---------- 4. 带宽段统计 ----------
def episodes(col, k, min_gap=20):
    m = (d[col] <= 1 + k).values
    segs, i = [], 0
    while i < N:
        if m[i]:
            j = i
            while j + 1 < N and m[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s))
            merged[-1] = (s[0], s[1])
    return [(a, b) for a, b in merged]

ep_rows = []
for col, nm in [('rA', 'A'), ('rB', 'B')]:
    for k in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        eps = episodes(col, k)
        days = int((d[col] <= 1 + k).sum())
        lens = [(b - a + 1) for a, b in eps]
        ep_rows.append(dict(anchor=nm, k=int(k * 100), episodes=len(eps), days=days,
                            days_pct=round(days / N * 100, 1),
                            med_len=int(np.median(lens)) if lens else 0,
                            max_len=max(lens) if lens else 0))
print('\n--- 带宽段统计 ---')
print(f"{'锚':<3}{'带宽':>6}{'段数':>6}{'天数':>7}{'占比%':>7}{'中位段长':>9}{'最长':>7}")
for r in ep_rows:
    print(f"  {r['anchor']:<2}{r['k']:>4}%{r['episodes']:>6}{r['days']:>7}{r['days_pct']:>7}{r['med_len']:>9}{r['max_len']:>7}")
OUT['episodes'] = ep_rows

# ---------- 5. 触带之后：还会跌多少、多久见底 ----------
touch = []
for col, nm in [('rA', 'A'), ('rB', 'B')]:
    for k in [0.05, 0.10, 0.15, 0.20]:
        eps = episodes(col, k)
        for a, b in eps:
            # 段内最低点（真底）
            lo_i = a + int(np.argmin(d['close'].values[a:b + 1]))
            after20 = d['close'].iloc[min(N - 1, b + 20)] / d['close'].iloc[a] - 1 if b + 20 < N else np.nan
            after60 = d['close'].iloc[min(N - 1, b + 60)] / d['close'].iloc[a] - 1 if b + 60 < N else np.nan
            after244 = d['close'].iloc[min(N - 1, b + 244)] / d['close'].iloc[a] - 1 if b + 244 < N else np.nan
            touch.append(dict(anchor=nm, k=int(k * 100),
                              enter=str(d.d.iloc[a].date()), exit=str(d.d.iloc[b].date()),
                              close=round(float(d.close.iloc[a]), 0),
                              real=round(float(d['real'].iloc[a]), 0),
                              low_close=round(float(d.close.iloc[lo_i]), 0),
                              low_after=round(float(d.close.iloc[lo_i] / d.close.iloc[a] - 1) * 100, 1),
                              days_to_low=int(lo_i - a),
                              d20=None if pd.isna(after20) else round(float(after20) * 100, 1),
                              d60=None if pd.isna(after60) else round(float(after60) * 100, 1),
                              d244=None if pd.isna(after244) else round(float(after244) * 100, 1)))
OUT['touch'] = touch
print('\n--- 首日触带之后（A 锚 15% 带宽）---')
print(f"{'入场':<12}{'离场':<12}{'入场点数':>9}{'段内最低':>9}{'再跌%':>8}{'距底天':>7}{'+20日':>8}{'+60日':>8}{'+1年':>8}")
for r in touch:
    if r['anchor'] == 'A' and r['k'] == 15:
        print(f"  {r['enter']:<12}{r['exit']:<12}{r['close']:>9.0f}{r['low_close']:>9.0f}"
              f"{r['low_after']:>8}{r['days_to_low']:>7}{str(r['d20']):>8}{str(r['d60']):>8}{str(r['d244']):>8}")

# ---------- 6. 当前读数 ----------
last = d.iloc[-1]
OUT['now'] = dict(close=round(float(last['close']), 0), real=round(float(last['real']), 0),
                  rA=round(float(last['rA']), 3), rB=round(float(last['rB']), 3) if pd.notna(last['rB']) else None,
                  real_low8y=round(float(last['rmin8']), 0),
                  eq_anchor_today=round(ANCHOR * CPI_T / 100, 0))
print(f"\n--- 当前 ---")
print(f"上证 {last['close']:.0f} | 实际值 {last['real']:.0f} | 距底倍数 {last['rA']:.2f}（即离底 {last['rA']-1:+.1%}）")
print(f"过去8年实际最低 {last['rmin8']:.0f} | 距8年低点 {last['rB']:.2f}")
print(f"今天的'底'折成名义点位 = {ANCHOR * CPI_T / 100:.0f}")

# 需要跌到多少才进入各带宽
need = []
for k in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    need.append(dict(k=int(k * 100), real=round(ANCHOR * (1 + k), 0),
                     nom=round(ANCHOR * (1 + k) * CPI_T / 100, 0),
                     drop=round((ANCHOR * (1 + k) * CPI_T / 100) / float(last['close']) - 1, 3)))
OUT['need'] = need
print('\n--- 要跌到多少才进入带宽 ---')
for r in need:
    print(f"  离底 {r['k']:>2}% 内 → 实际 {r['real']:.0f} → 名义 {r['nom']:.0f}（从现在跌 {r['drop']:+.1%}）")

json.dump(OUT, open(os.path.join(B, 'd97_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved d97_result.json')
