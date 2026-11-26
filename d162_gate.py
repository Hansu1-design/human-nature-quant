# -*- coding: utf-8 -*-
"""
d162 入场闸门：买之前看状态，"不符合预期就不做"
先生原话：不是 3 天买卖，是"当我想买的时候，看看趋势……重要的是大概能判断一下
是不是符合咱们的预期，普适的市场规律。不符合咱们可以走嘛，咱们要的不是做对每一个，
而是只做概率最大确定性最高的"。

两层：
  层1 条件分布画像 —— 某状态下，后续 250 日收益的"形状"（中位/胜率/p25/p10）
  层2 闸门资金曲线 —— 状态满足才持，不满足空仓生息（2.5%/年）
关键：确定性提高多少 vs 在场时间损失多少
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load

RF = 0.025
pd.set_option('display.width', 220)

d = load()
c = d['close'].values.astype(float)
v = pd.to_numeric(d.get('volume', pd.Series(np.nan, index=d.index)), errors='coerce').values
dts = d['d']
n = len(c)
yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25

# ---------------- 状态变量（全部 t 日收盘可知） ----------------
S = pd.DataFrame(index=d.index)
S['close'] = c
ma250 = pd.Series(c).rolling(250).mean().values
S['ma_up'] = np.where(np.isnan(ma250), np.nan, (c > ma250).astype(float))
S['ma_slope'] = pd.Series(ma250) / pd.Series(ma250).shift(20) - 1
ret = pd.Series(np.concatenate([[np.nan], c[1:] / c[:-1] - 1]))
S['rv20'] = ret.rolling(20).std() * np.sqrt(252)
S['rv60'] = ret.rolling(60).std() * np.sqrt(252)
S['rv_q'] = pd.Series(S['rv20']).rolling(750).apply(lambda x: (x[-1] > x).mean(), raw=True)
S['dd250'] = c / pd.Series(c).rolling(250).max() - 1
S['ret250'] = c / pd.Series(c).shift(250) - 1
S['ret60'] = c / pd.Series(c).shift(60) - 1
if 'rA' in d.columns:
    S['rA'] = d['rA'].values
if np.isfinite(v).sum() > 100:
    S['vq'] = (pd.Series(v).rolling(20).mean() / pd.Series(v).rolling(250).mean()).values
else:
    S['vq'] = np.nan

# 目标：次日收盘买入，250 交易日后
c_ser = pd.Series(c)
S['fwd250'] = (c_ser.shift(-250) / c_ser.shift(-1) - 1).values

SEGS = [('1997-2010', '1997-01-01', '2010-12-31'),
        ('2011-2016', '2011-01-01', '2016-12-31'),
        ('2017-2026', '2017-01-01', '2026-12-31')]


def seg_of(i):
    for k, (nm, a, b) in enumerate(SEGS):
        if a <= str(dts.iloc[i].date()) <= b:
            return k
    return -1


SEGID = np.array([seg_of(i) for i in range(n)])


def blockboot(x, L=250, B=2000, seed=3):
    """块自助：重叠观测的置信区间"""
    x = np.asarray(x)
    m = len(x)
    if m < L * 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / L))
    out = []
    for _ in range(B):
        st = rng.integers(0, m - L + 1, nb)
        s = np.concatenate([x[j:j + L] for j in st])[:m]
        out.append(np.median(s))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def profile(mask, label, name):
    y = S.loc[mask, 'fwd250'].dropna().values
    if len(y) < 40:
        return dict(dim=name, bucket=label, n=len(y))
    lo, hi = blockboot(y)
    segs = []
    for k, (nm, a, b) in enumerate(SEGS):
        m2 = mask & (SEGID == k)
        yy = S.loc[m2, 'fwd250'].dropna().values
        segs.append(np.median(yy) * 100 if len(yy) >= 25 else np.nan)
    return dict(dim=name, bucket=label, n=len(y),
                med=np.median(y) * 100, win=(y > 0).mean() * 100,
                p25=np.percentile(y, 25) * 100, p10=np.percentile(y, 10) * 100,
                mean=y.mean() * 100, lo=lo * 100 if lo == lo else np.nan,
                hi=hi * 100 if hi == hi else np.nan,
                s1=segs[0], s2=segs[1], s3=segs[2])


rows = []


def quint(var, labels, name, q=(0.2, 0.4, 0.6, 0.8)):
    s = S[var]
    ok = s.notna() & S['fwd250'].notna()
    if ok.sum() < 300:
        return
    qs = s[ok].quantile(q).values
    edges = [-9e9] + list(qs) + [9e9]
    for b, lab in enumerate(labels):
        m = ok & (s >= edges[b]) & (s < edges[b + 1])
        r = profile(m, lab, name)
        if 'med' in r:
            rows.append(r)


print('=' * 150)
print('层1  各状态下"未来 250 日收益"的形状（次日买、250 日后算；胜率=确定性，p10=左尾厚度）')
print('=' * 150)
print(f'{"维度":<16}{"档":<14}{"n":>6}{"中位%":>9}{"胜率%":>8}{"p25%":>9}{"p10%":>9}{"均值%":>9}'
      f'{"95%CI(中位)":>20}{"97-10":>9}{"11-16":>9}{"17-26":>9}')

# 无条件基准
allm = S['fwd250'].notna()
r = profile(allm, '全部（无条件）', '基准')
rows.append(r)
print(f"{'基准':<16}{'全部':<14}{r['n']:>6}{r['med']:>9.2f}{r['win']:>8.1f}{r['p25']:>9.2f}"
      f"{r['p10']:>9.2f}{r['mean']:>9.2f}   [{r['lo']:+7.2f},{r['hi']:+7.2f}]"
      f"{r['s1']:>9.2f}{r['s2']:>9.2f}{r['s3']:>9.2f}")

quint('ma_up', ['MA250下方', 'MA250上方'], '趋势位置')
quint('ma_slope', ['MA250下行快', 'MA250下行慢', 'MA250上行慢', 'MA250上行快'], '趋势斜率')
quint('rv_q', ['波动最低20%', '波动低', '波动中', '波动高', '波动最高20%'], '波动状态')
quint('dd250', ['距高-40%以下', '距高-40~-25%', '距高-25~-12%', '距高-12~-4%', '距高-4%内'], '距250日高点')
quint('ret250', ['一年跌>25%', '一年跌5~25%', '一年-5~+12%', '一年涨12~40%', '一年涨>40%'], '前一年涨幅')
quint('ret60', ['60日跌>12%', '60日跌3~12%', '60日-3~+8%', '60日涨8~20%', '60日涨>20%'], '近60日涨幅')
if S['rA'].notna().sum() > 500:
    quint('rA', ['离底≤1.15', '离底1.15~1.35', '离底1.35~1.6', '离底1.6~1.9', '离底>1.9'], '离底距离(2008起)')
if S['vq'].notna().sum() > 500:
    quint('vq', ['量能极缩', '量能偏缩', '量能中', '量能偏放', '量能极放'], '量能')

for r in rows[1:]:
    if 'med' not in r:
        continue
    flag = ''
    if r['win'] >= rows[0]['win'] + 6 and r['p10'] > rows[0]['p10']:
        flag = '  ★胜率与左尾同时改善'
    elif r['win'] <= rows[0]['win'] - 6:
        flag = '  ⚠明确更差'
    print(f"{r['dim']:<16}{r['bucket']:<14}{r['n']:>6}{r['med']:>9.2f}{r['win']:>8.1f}{r['p25']:>9.2f}"
          f"{r['p10']:>9.2f}{r['mean']:>9.2f}   [{r['lo']:+7.2f},{r['hi']:+7.2f}]"
          f"{r['s1']:>9.2f}{r['s2']:>9.2f}{r['s3']:>9.2f}{flag}")

pd.DataFrame(rows).to_csv('d162_profile.csv', index=False, encoding='utf-8-sig')

# ---------------- 层2 闸门资金曲线 ----------------
print()
print('=' * 150)
print('层2  当闸门用：状态满足才持（次日买/次日卖），不满足空仓生息 2.5%/年，成本单边 0.1%')
print('=' * 150)


def gate_sim(cond, cost=0.001, name=''):
    """cond[i] 表示 t 日收盘后可判断的状态，t+1 日收盘执行"""
    nav = np.ones(n)
    pos = 0.0
    cash = 1.0
    px_entry = 0.0
    on_days = 0
    trades = 0
    rf_d = 1 + RF ** (1 / 252) - 1 if False else (1 + RF) ** (1 / 252) - 1
    for i in range(n - 1):
        want = bool(cond[i])
        if want and pos == 0:
            pos = cash * (1 - cost) / c[i + 1]
            cash = 0.0
            px_entry = c[i + 1]
            trades += 1
        elif (not want) and pos > 0:
            cash = pos * c[i + 1] * (1 - cost)
            pos = 0.0
            trades += 1
        if pos > 0:
            nav[i + 1] = pos * c[i + 1]
            on_days += 1
        else:
            cash *= (1 + rf_d)
            nav[i + 1] = cash
    tt = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
    g = nav[-1] ** (1 / tt) - 1
    peak = np.maximum.accumulate(nav)
    dd = float(np.min(nav / peak - 1))
    segv = []
    for nm, a, b in SEGS:
        i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
        i1 = min(int(np.searchsorted(dts.values, np.datetime64(b))) - 1, n - 1)
        y = (dts.iloc[i1] - dts.iloc[i0]).days / 365.25
        rr = nav[i1] / nav[i0]
        segv.append(rr ** (1 / y) - 1 if rr > 0 and y > 0 else np.nan)
    return dict(name=name, ann=g * 100, dd=dd * 100, occ=on_days / (n - 1) * 100,
                trades=trades, s1=segv[0] * 100, s2=segv[1] * 100, s3=segv[2] * 100,
                calmar=g / abs(dd) if dd < 0 else np.nan)


cond_all = np.ones(n, dtype=bool)
cases = [('无条件（买入持有）', cond_all)]

# 单维闸门：只做"最好"的档
s = S
cases.append(('只做 MA250 上方', (s['ma_up'] == 1).fillna(False).values))
cases.append(('只做 MA250 上方且上行', ((s['ma_up'] == 1) & (s['ma_slope'] > 0)).fillna(False).values))
cases.append(('只做 波动最低40%', (s['rv_q'] <= 0.4).fillna(False).values))
cases.append(('只做 波动最高40%', (s['rv_q'] >= 0.6).fillna(False).values))
cases.append(('只做 距高-25%以下', (s['dd250'] <= -0.25).fillna(False).values))
cases.append(('只做 前一年涨幅<40%', (s['ret250'] < 0.40).fillna(False).values))
cases.append(('只做 前一年跌幅>5%', (s['ret250'] < -0.05).fillna(False).values))
if s['rA'].notna().sum() > 500:
    cases.append(('只做 离底≤1.35', (s['rA'] <= 1.35).fillna(False).values))

print(f'{"闸门":<26}{"年化%":>9}{"最大回撤%":>11}{"Calmar":>8}{"在场%":>8}{"交易":>6}'
      f'{"97-10":>9}{"11-16":>9}{"17-26":>9}')
res = []
for nm, cd in cases:
    r = gate_sim(cd, name=nm)
    res.append(r)
    print(f'{nm:<26}{r["ann"]:>+9.2f}{r["dd"]:>+11.1f}{r["calmar"]:>8.3f}{r["occ"]:>8.1f}'
          f'{r["trades"]:>6}{r["s1"]:>+9.2f}{r["s2"]:>+9.2f}{r["s3"]:>+9.2f}')

# 空仓生息
nav_rf = np.cumprod(np.concatenate([[1.0], np.full(n - 1, (1 + RF) ** (1 / 252))]))
grf = nav_rf[-1] ** (1 / yrs) - 1
print(f'\n{"空仓生息（参照）":<26}{grf*100:>+9.2f}{0:>+11.1f}{"—":>8}{0:>8.1f}{0:>6}')

pd.DataFrame(res).to_csv('d162_gate.csv', index=False, encoding='utf-8-sig')

# ---------------- 层3 "确定性"能不能换成可用规则：胜率—暴露权衡 ----------------
print()
print('=' * 150)
print('层3  胜率 vs 在场时间：把胜率抬到 X% 需要放弃多少在场时间（这是"只做确定性最高的"的代价）')
print('=' * 150)
print(f'{"闸门":<30}{"胜率%":>8}{"在场%":>8}{"中位%":>8}{"p10%":>8}{"年化%":>9}{"评价":<30}')
for nm, cd in cases:
    mask = pd.Series(cd, index=S.index) & S['fwd250'].notna()
    y = S.loc[mask, 'fwd250'].dropna().values
    if len(y) < 100:
        continue
    r = [x for x in res if x['name'] == nm][0]
    ev = '胜率↑但年化↓' if r['ann'] > grf * 100 and r['ann'] < res[0]['ann'] else (
        '胜率↑且年化↑' if r['ann'] >= res[0]['ann'] else '不如空仓生息')
    print(f'{nm:<30}{(y>0).mean()*100:>8.1f}{r["occ"]:>8.1f}{np.median(y)*100:>8.2f}'
          f'{np.percentile(y,10)*100:>8.2f}{r["ann"]:>+9.2f}{ev:<30}')
