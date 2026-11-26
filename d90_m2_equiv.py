# -*- coding: utf-8 -*-
"""
d90: 用 M2 做平减，把上证点位换算成"货币口径的等价点位"
核心：ratio_t = 上证_t / M2_t（点/万亿元）
      eq_t    = 上证_t * M2_T / M2_t   （折算到 T 日货币水平）
目的：检验"同一个 ratio = 同一个位置"是否成立；熊市底/牛市顶的 ratio 区间是多少
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------- 1. 读数据 ----------
sse = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d')[['d', 'close']].reset_index(drop=True)

m2 = pd.read_parquet('G:/AStockData/supplement/macro/money_supply.parquet')
m2['ym'] = m2['月份'].str.replace('年', '-').str.replace('月份', '')
m2['d'] = pd.to_datetime(m2['ym'], format='%Y-%m')
# 该月数据次月中旬公布 -> 从次月 15 日起可用（严格口径）
m2['avail'] = m2['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
m2 = m2[['d', 'avail', '货币和准货币(M2)-数量(亿元)']].rename(columns={'货币和准货币(M2)-数量(亿元)': 'm2'})
m2 = m2.sort_values('avail').reset_index(drop=True)
m2['m2_wy'] = m2['m2'] / 1e4  # 万亿元

df = pd.merge_asof(sse.sort_values('d'), m2[['avail', 'm2_wy', 'd']].rename(columns={'d': 'm2_month'}),
                   left_on='d', right_on='avail', direction='backward')
df = df.dropna(subset=['m2_wy']).reset_index(drop=True)
print('区间', df['d'].iloc[0].date(), '→', df['d'].iloc[-1].date(), '共', len(df), '根')

# ---------- 2. 比值与等价点位 ----------
M2_T = df['m2_wy'].iloc[-1]
df['ratio'] = df['close'] / df['m2_wy']          # 点 / 万亿元
df['eq'] = df['close'] * M2_T / df['m2_wy']      # 折算到末日货币水平
print(f'末日 M2 = {M2_T:.1f} 万亿，上证 {df["close"].iloc[-1]:.0f}，ratio = {df["ratio"].iloc[-1]:.2f}')
OUT['meta'] = dict(start=str(df['d'].iloc[0].date()), end=str(df['d'].iloc[-1].date()),
                   n=len(df), m2_T=round(M2_T, 1), last_close=round(float(df['close'].iloc[-1]), 0),
                   last_ratio=round(float(df['ratio'].iloc[-1]), 3))

# ---------- 3. 年度对照：名义 vs 等价 ----------
yr = df.groupby(df['d'].dt.year).agg(
    lo=('close', 'min'), hi=('close', 'max'), last=('close', 'last'),
    eq_lo=('eq', 'min'), eq_hi=('eq', 'max'), eq_last=('eq', 'last'),
    r_lo=('ratio', 'min'), r_hi=('ratio', 'max'), r_last=('ratio', 'last'),
    m2=('m2_wy', 'last'))
print('\n=== 年度：名义点位 / 折算到 2026-11 货币水平的等价点位 / ratio ===')
print(f"{'年':<6}{'名义低':>7}{'名义高':>7}{'等价低':>9}{'等价高':>9}{'M2万亿':>9}{'ratio低':>9}{'ratio高':>9}")
for y, r in yr.iterrows():
    print(f"{y:<6}{r.lo:>7.0f}{r.hi:>7.0f}{r.eq_lo:>9.0f}{r.eq_hi:>9.0f}{r.m2:>9.1f}{r.r_lo:>9.2f}{r.r_hi:>9.2f}")
OUT['yearly'] = {int(y): {k: round(float(v), 2) for k, v in r.items()} for y, r in yr.iterrows()}

# ---------- 4. 长期中枢 ----------
print('\n=== 长期中枢 ===')
for name, col in [('名义点位', 'close'), ('等价点位(按末日M2)', 'eq')]:
    s = df[col]
    print(f'{name}: 均值 {s.mean():.0f} | 中位 {s.median():.0f} | P20 {s.quantile(.2):.0f} | P80 {s.quantile(.8):.0f} | 区间 {s.min():.0f}~{s.max():.0f}')
print(f'ratio:   均值 {df['ratio'].mean():.2f} | 中位 {df['ratio'].median():.2f} | P20 {df['ratio'].quantile(.2):.2f} | P80 {df['ratio'].quantile(.8):.2f} | 区间 {df['ratio'].min():.2f}~{df['ratio'].max():.2f}')
OUT['center'] = dict(
    close_mean=round(float(df.close.mean())), close_med=round(float(df.close.median())),
    eq_mean=round(float(df['eq'].mean())), eq_med=round(float(df['eq'].median())),
    ratio_mean=round(float(df['ratio'].mean()), 2), ratio_med=round(float(df['ratio'].median()), 2),
    ratio_p20=round(float(df['ratio'].quantile(.2)), 2), ratio_p80=round(float(df['ratio'].quantile(.8)), 2))

# ---------- 5. ZigZag 分段：顶底的名义点位与 ratio ----------
def zigzag(c, th=0.20):
    """返回 [(idx, 'H'/'L'), ...] 交替的转折点"""
    piv = [(0, 'L' if c[1] > c[0] else 'H')]
    trend, ext_i, ext_v = piv[0][1], 0, c[0]
    for i in range(1, len(c)):
        if trend == 'L':
            if c[i] > ext_v:
                ext_i, ext_v = i, c[i]
            elif c[i] <= ext_v * (1 - th):
                piv.append((ext_i, 'H')); trend, ext_i, ext_v = 'H', i, c[i]
        else:
            if c[i] < ext_v:
                ext_i, ext_v = i, c[i]
            elif c[i] >= ext_v * (1 + th):
                piv.append((ext_i, 'L')); trend, ext_i, ext_v = 'L', i, c[i]
    piv.append((ext_i, trend))
    return piv

C = df['close'].values
piv = zigzag(C, 0.20)
print(f'\n=== ZigZag(20%) 转折点：{len(piv)} 个 ===')
rows = []
for i, t in piv:
    r = df.iloc[i]
    rows.append(dict(date=str(r['d'].date()), type=t, close=round(float(r['close'])),
                     m2=round(float(r['m2_wy']), 1), ratio=round(float(r['ratio']), 2),
                     eq=round(float(r['eq']))))
    print(f"{r['d'].date()} {t}  上证{r['close']:>6.0f}  M2{r['m2_wy']:>6.1f}万亿  ratio {r['ratio']:>5.2f}  等价 {r['eq']:>6.0f}")
OUT['pivots'] = rows

# 顶底分组
tops = [x for x in rows if x['type'] == 'H']
bots = [x for x in rows if x['type'] == 'L']
print('\n--- 顶（不止一个，含次级顶）---')
for x in tops: print(f"  {x['date']} {x['close']:>5}  ratio {x['ratio']:.2f}  等价 {x['eq']}")
print('--- 底 ---')
for x in bots: print(f"  {x['date']} {x['close']:>5}  ratio {x['ratio']:.2f}  等价 {x['eq']}")
OUT['tops'] = tops; OUT['bots'] = bots

with open(os.path.join(B, 'd90_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
df.to_parquet(os.path.join(B, 'd90_daily.parquet'), index=False)
print('\nsaved d90_result.json / d90_daily.parquet')
