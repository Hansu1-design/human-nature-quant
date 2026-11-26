# -*- coding: utf-8 -*-
"""
d110 — 标的错配诊断：上证锚 vs 510300（ETF）实际价
问题：上证点位做闸门，但下单标的是 ETF。同一点位，ETF 贵贱差多少？偏离能不能用？
产出：① ratio 性质（趋势/均值回复/随机游走）② 偏离的预测力 ③ ETF 直接换锚可行性
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ============ 1. 数据 ============
e = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
e['d'] = pd.to_datetime(e['日期'])
e['etf'] = e['收盘'].astype(float)
e['etf_pct'] = e['涨跌幅'].astype(float)
e = e[['d', 'etf', 'etf_pct']].sort_values('d').reset_index(drop=True)

s = pd.read_parquet(os.path.join(BASE, 'idx_sse_all.parquet'))
s['d'] = pd.to_datetime(s['date'])
s['sse'] = s['close'].astype(float)
s = s[['d', 'sse']].sort_values('d').reset_index(drop=True)

df = e.merge(s, on='d', how='inner').reset_index(drop=True)
df['sse_pct'] = df['sse'].pct_change() * 100
df['gap'] = df['etf_pct'] - df['sse_pct']
df['r'] = df['sse'] / df['etf'] * 1000          # 归一化比值
df['lr'] = np.log(df['r'])

OUT['meta'] = {
    'etf_code': '510300', 'n': int(len(df)),
    'start': str(df['d'].iloc[0].date()), 'end': str(df['d'].iloc[-1].date()),
    'etf_first': float(df['etf'].iloc[0]), 'etf_last': float(df['etf'].iloc[-1]),
    'sse_first': float(df['sse'].iloc[0]), 'sse_last': float(df['sse'].iloc[-1]),
    'r_first': round(float(df['r'].iloc[0]), 1), 'r_last': round(float(df['r'].iloc[-1]), 1),
}

# ============ 2. 分红/跳变检测 ============
# ETF 分红除息日：ETF 跌而指数不跌，且幅度 >1.5%
div_days = df.loc[(df['gap'] < -1.5) & (df['sse_pct'] > -0.5),
                  ['d', 'etf', 'etf_pct', 'sse_pct', 'gap']].copy()
div_days['d'] = div_days['d'].dt.strftime('%Y-%m-%d')
OUT['B_div_candidates'] = {
    'n': int(len(div_days)),
    'days': div_days.round(2).to_dict('records')[:40],
    'note': 'ETF跌>1.5%而上证未跌>0.5% → 疑似除息（分红导致价格下折，非真实亏损）',
}

# ============ 3. ratio 形态 ============
yr = df.groupby(df['d'].dt.year)['r'].agg(['mean', 'min', 'max', 'std']).round(1)
OUT['C_ratio_by_year'] = [{'year': int(k), **v} for k, v in yr.to_dict('index').items()]

# ============ 4. 平稳性：自实现 ADF + VR ============
def adf(y, maxlag=20, kind='c'):
    """Augmented Dickey-Fuller。kind: 'n' 无常数, 'c' 常数, 'ct' 常数+趋势"""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    n = len(dy)
    # 选滞后：固定 20（日线约一个月），并给 AIC 选的点
    def build(p):
        rows = n - p
        Y = dy[p:]
        cols = [y[p:-1]]                      # y_{t-1}
        if kind in ('c', 'ct'):
            cols.append(np.ones(rows))
        if kind == 'ct':
            cols.append(np.arange(rows, dtype=float))
        for i in range(1, p + 1):
            cols.append(dy[p - i: n - i])
        return np.column_stack(cols), Y

    def fit(p):
        X, Y = build(p)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        k = X.shape[1]
        sigma2 = resid @ resid / (len(Y) - k)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t = beta[0] / se[0]
        aic = len(Y) * np.log(resid @ resid / len(Y)) + 2 * k
        return t, aic, beta[0]

    res = {}
    for p in [0, 5, 10, 20]:
        t, aic, g = fit(p)
        res[f'lag{p}'] = {'t': round(float(t), 3), 'gamma': round(float(g), 5)}
    # AIC 自动选阶
    cands = [(fit(p)[1], p) for p in range(0, 26)]
    pbest = min(cands)[1]
    tb, aicb, gb = fit(pbest)
    res['aic_pick'] = {'lag': int(pbest), 't': round(float(tb), 3), 'gamma': round(float(gb), 5)}
    return res

# MacKinnon (1994) 渐近临界值，n=∞
CV = {'n': {'1%': -2.58, '5%': -1.95, '10%': -1.62},
      'c': {'1%': -3.43, '5%': -2.86, '10%': -2.57},
      'ct': {'1%': -3.96, '5%': -3.41, '10%': -3.12}}

OUT['D_adf'] = {}
for kind, lab in [('n', '无常数无趋势'), ('c', '含常数'), ('ct', '含常数+趋势')]:
    OUT['D_adf'][lab] = {'crit': CV[kind], **adf(df['lr'].values, kind=kind)}

def vr(y, q):
    """Lo-MacKinlay 方差比（同方差版）"""
    y = np.asarray(y, dtype=float)
    r = np.diff(y)
    T = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (T - 1)
    rq = y[q:] - y[:-q]
    varq = ((rq - q * mu) ** 2).sum() / (T - q + 1) / q
    VR = varq / var1
    z = (VR - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    return round(float(VR), 4), round(float(z), 3)

OUT['E_vr'] = {}
for q in [2, 5, 20, 60, 120, 250]:
    v, z = vr(df['lr'].values, q)
    OUT['E_vr'][f'q{q}'] = {'VR': v, 'z': z, 'revert': '均值回复' if z < -1.96 else ('趋势延续' if z > 1.96 else '随机游走（不拒绝）')}

# ============ 5. 去趋势残差 ============
t = np.arange(len(df), dtype=float)
X = np.column_stack([np.ones(len(df)), t, t ** 2])
beta, *_ = np.linalg.lstsq(X, df['lr'].values, rcond=None)
df['trend'] = X @ beta
df['dev'] = df['lr'] - df['trend']
# 年化趋势漂移（线性项 + 二次项在末端的导数）
drift_lin = beta[1] * 243
drift_end = (beta[1] + 2 * beta[2] * t[-1]) * 243
OUT['F_trend'] = {
    'coef_lin': round(float(beta[1]), 8), 'coef_quad': round(float(beta[2]), 12),
    'drift_yr_start_%': round(float(np.exp(beta[1] * 243) - 1) * 100, 2),
    'drift_yr_end_%': round(float(np.exp(drift_end) - 1) * 100, 2),
    'note': 'ln(上证/ETF) 对时间的确定性趋势；负漂移=ETF 相对走强（分红再投+成分调整+费用）',
}
df['dev_pct'] = (np.exp(df['dev']) - 1) * 100

# 扩张窗口分位（无前视）
def exp_rank(s, min_n=250):
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(min_n, len(v)):
        out[i] = (v[:i + 1] < v[i]).mean()
    return out
df['dev_rank'] = exp_rank(df['dev'])

# ============ 6. 偏离的预测力（核心）============
def nw_se(X, resid, lags):
    """Newey-West HAC 标准误"""
    n, k = X.shape
    S = np.zeros((k, k))
    for l in range(lags + 1):
        w = 1 - l / (lags + 1)
        Xl = X[l:]
        rl = resid[l:] * resid[:n - l][:, None]
        S += w * (Xl * resid[l:][:, None]).T @ (X[:n - l] * resid[:n - l][:, None])
    XtX_inv = np.linalg.pinv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv) / n)

pred = []
for h in [60, 250, 500]:
    fut = (np.log(df['etf'].shift(-h)) - np.log(df['etf'])) - (np.log(df['sse'].shift(-h)) - np.log(df['sse']))
    m = fut.notna() & df['dev'].notna()
    Xr = np.column_stack([np.ones(m.sum()), df.loc[m, 'dev'].values])
    Yr = fut[m].values * 100
    b, *_ = np.linalg.lstsq(Xr, Yr, rcond=None)
    res = Yr - Xr @ b
    se = nw_se(Xr, res, lags=max(h // 20, 5))
    pred.append({'h_days': h,
                 'coef': round(float(b[1]), 2),
                 't_nw': round(float(b[1] / se[1]), 2),
                 'n': int(m.sum()),
                 'R2_%': round(float(1 - (res @ res) / (((Yr - Yr.mean()) ** 2).sum())), 4) * 100,
                 'read': '偏离高1σ→未来ETF相对上证超额' })
OUT['G_predict'] = pred

# ============ 7. 同点位下 ETF 贵贱极差 ============
bands = [(0, 2500), (2500, 2800), (2800, 3000), (3000, 3200), (3200, 3500), (3500, 4000), (4000, 9000)]
rows = []
for lo, hi in bands:
    m = (df['sse'] >= lo) & (df['sse'] < hi)
    if m.sum() < 20:
        continue
    sub = df[m]
    rows.append({'band': f'{lo}-{hi}', 'n': int(m.sum()),
                 'etf_min': round(float(sub['etf'].min()), 3),
                 'etf_max': round(float(sub['etf'].max()), 3),
                 'etf_cv_%': round(float(sub['etf'].std() / sub['etf'].mean() * 100), 1),
                 'dev_min_%': round(float(sub['dev_pct'].min()), 1),
                 'dev_max_%': round(float(sub['dev_pct'].max()), 1),
                 'spread_%': round(float(sub['etf'].max() / sub['etf'].min() - 1) * 100, 1)})
OUT['H_same_level'] = rows

# ============ 8. ETF 直接换锚：CPI 实际价底 ============
cpi = pd.read_parquet(os.path.join(BASE, 'cpi.parquet')) if os.path.exists(os.path.join(BASE, 'cpi.parquet')) else None
if cpi is None:
    for cand in ['G:/AStockData/supplement/cpi.parquet', 'G:/AStockData/cpi.parquet',
                 os.path.join(BASE, 'd90_daily.parquet')]:
        if os.path.exists(cand):
            cpi = pd.read_parquet(cand)
            OUT['cpi_src'] = cand
            break
OUT['cpi_cols'] = list(cpi.columns)[:10] if cpi is not None else None
if cpi is not None:
    OUT['cpi_head'] = cpi.head(3).astype(str).to_dict('records')

df.to_parquet(os.path.join(BASE, 'd110_path.parquet'), index=False)
with open(os.path.join(BASE, 'd110_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print('== meta ==', json.dumps(OUT['meta'], ensure_ascii=False))
print('\n== B 疑似除息 ==', OUT['B_div_candidates']['n'], '天')
for r in OUT['B_div_candidates']['days'][:12]:
    print('  ', r)
print('\n== C ratio 年度 ==')
for r in OUT['C_ratio_by_year']:
    print('  ', r)
print('\n== D ADF ==')
for k, v in OUT['D_adf'].items():
    print(' ', k, 'crit', v['crit'], 'aic_pick', v['aic_pick'])
print('\n== E VR ==')
for k, v in OUT['E_vr'].items():
    print(' ', k, v)
print('\n== F 趋势 ==', OUT['F_trend'])
print('\n== G 偏离预测力 ==')
for r in OUT['G_predict']:
    print('  ', r)
print('\n== H 同点位 ETF 极差 ==')
for r in OUT['H_same_level']:
    print('  ', r)
print('\n== CPI 源 ==', OUT.get('cpi_src'), OUT.get('cpi_cols'))
