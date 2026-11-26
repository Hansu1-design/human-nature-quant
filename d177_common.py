# -*- coding: utf-8 -*-
"""d177_common R52 共享数据层：所有验证脚本必须 import 本模块，杜绝两套实现不一致。

统一口径（与 d170/d171 对齐，并按 Hermes R52 意见修正）：
  - 起点 2008-05-01（PB 扩张窗口留 36 期，rA/CGO 同起点，保证可比）
  - 决策：月末取值；执行：次月首个交易日收盘成交（无前视）
  - 空仓部分 2.5%/年计息；单边成本 0.1%，按【调仓份额】计，不按全额
  - 三段改为 2008-2010 / 2011-2016 / 2017-2026（起点受 PB 数据限制）
"""
import pandas as pd, numpy as np

RF = 0.025
COST = 0.001
START = '2008-05-01'
SEGS = [('08-10', '2008-05-01', '2010-12-31'),
        ('11-16', '2011-01-01', '2016-12-31'),
        ('17-26', '2017-01-01', '2026-12-31')]

_d = pd.read_parquet('idx_sse_all.parquet')
_d['date'] = pd.to_datetime(_d['date'])
for _x in ['close', 'amount', 'volume']:
    _d[_x] = pd.to_numeric(_d[_x], errors='coerce')
_d = _d.dropna(subset=['close']).set_index('date').sort_index()
_d = _d[_d.index >= '1996-01-01']

c_all = _d['close']
amt_all = _d['amount']

# ---------- CPI 定基（d91 口径，同比开 12 次方，末值 144.0） ----------
def _cpi_index():
    x = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
    x['m'] = x['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]), int(str(s)[5:7]), 1))
    x['yoy'] = pd.to_numeric(x['全国-当月'], errors='coerce')
    x = x.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
    x['mom'] = (x['yoy'] / 100.) ** (1 / 12.)
    i0 = x.index[x['m'] == pd.Timestamp('2008-01-01')][0]
    v = [np.nan] * len(x); v[i0] = 100.
    for i in range(i0 + 1, len(x)): v[i] = v[i-1] * x.loc[i, 'mom']
    for i in range(i0 - 1, -1, -1): v[i] = v[i+1] / x.loc[i+1, 'mom']
    return pd.Series(v, index=x['m'])

_ci = _cpi_index()
_real = c_all / _ci.reindex(c_all.index, method='ffill') * 100
rA_all = _real / 1880.0

# ---------- PB 扩张窗口分位（沪深300 加权，月频） ----------
_pb = pd.read_parquet('G:/AStockData/supplement/valuation/index_pb_沪深300.parquet')
_pb['日期'] = pd.to_datetime(_pb['日期'])
_pb = _pb.set_index('日期').sort_index()
_pbv = pd.to_numeric(_pb['市净率'], errors='coerce').dropna()
_arr = _pbv.values
_pct = np.array([np.nan] * len(_arr))
for i in range(len(_arr)):
    if i >= 36:
        _pct[i] = (_arr[:i+1] < _arr[i]).mean()
PBpct_all = pd.Series(_pct, index=_pbv.index).reindex(c_all.index, method='ffill')
PBlevel_all = _pbv

# ---------- CGO（Grinblatt-Han 换手率加权参考价） ----------
_raw = amt_all / c_all
V_all = (_raw / _raw.rolling(250, min_periods=60).median() * 0.015).clip(0.0005, 0.12).fillna(0.015)
_P = c_all.values
RP = np.zeros(len(_P)); RP[0] = _P[0]
for i in range(1, len(_P)):
    RP[i] = (1 - V_all.values[i]) * RP[i-1] + V_all.values[i] * _P[i]
CGO_all = pd.Series((_P - RP) / _P, index=c_all.index)
RP_all = pd.Series(RP, index=c_all.index)

# ---------- 统一评估区间 ----------
idx = c_all.index[c_all.index >= START]
c = c_all.reindex(idx)
rA = rA_all.reindex(idx)
pb = PBpct_all.reindex(idx)
cgo = CGO_all.reindex(idx)
vol = V_all.reindex(idx)
n = len(idx)
rf_d = (1 + RF) ** (1 / 250) - 1

# 月末最后交易日 / 次月首个交易日
_per = idx.to_period('M')
last_of_month = np.array([i for i in range(n-1) if _per[i] != _per[i+1]])
first_of_month = np.array([i+1 for i in last_of_month])
# 调仓日 → 该日应使用的目标仓位（来自上月末的信号）
rebal_map = {}
for i, j in zip(last_of_month, first_of_month):
    rebal_map[j] = i


def target_from(x, th_hi, th_lo, shape='linear'):
    """把坐标 x 映射成目标仓位 w∈[0,1]。x 越小（越便宜）w 越大。"""
    u = (th_hi - x) / (th_hi - th_lo)
    u = np.clip(u, 0, 1)
    if shape == 'linear':   return u
    if shape == 'concave':  return u ** 2          # 便宜时才加速
    if shape == 'convex':   return u ** 0.5        # 稍贵就先上车
    if shape == 'step4':    return np.round(u * 4) / 4         # 四档跳跃 0/.25/.5/.75/1（先生④）
    raise ValueError(shape)


def curve_weight(w_series, band=0.0, cost=COST, min_hold_months=0):
    """连续仓位资金曲线。

    w_series : 长度 n 的目标仓位序列（当日可得，月末更新、月内保持）
    band     : |target - current| < band 则不调仓
    返回 nav, 换手次数, 累计换手额, 在场天数占比, 平均仓位
    """
    nav = np.ones(n); pos = 0.0; cash = 1.0
    tr = 0; turn = 0.0; ondays = 0; wsum = 0.0
    last_rebal_month = -999
    month_no = 0
    ws = np.asarray(w_series, dtype=float)
    for i in range(n-1):
        cash *= (1 + rf_d)
        val = pos * c.values[i+1] + cash
        # 调仓只在次月首个交易日
        if i+1 in rebal_map:
            month_no += 1
            sig_i = rebal_map[i+1]
            tgt = ws[sig_i]
            if np.isnan(tgt): tgt = 0.0
            w_now = (pos * c.values[i+1]) / val if val > 0 else 0.0
            delta = abs(tgt - w_now)
            if (month_no - last_rebal_month > min_hold_months) and delta >= band and delta > 1e-6:
                # delta>1e-6 防止 band=0 时把"仓位未变"误记为一次调仓（成本本就为 0，但换手数会虚高）
                val_after = val - delta * val * cost
                pos = tgt * val_after / c.values[i+1]
                cash = (1 - tgt) * val_after
                val = pos * c.values[i+1] + cash
                tr += 1; turn += delta
                last_rebal_month = month_no
        nav[i+1] = val
        w_now2 = (pos * c.values[i+1]) / val if val > 0 else 0.0
        if w_now2 > 0.01: ondays += 1
        wsum += w_now2
    return (pd.Series(nav, index=idx), tr, turn, ondays/n, wsum/n)


def curve_binary(cond, cost=COST, min_hold_months=0):
    """二值闸门，同样走月末决策口径，便于与连续版公平比较。"""
    w = np.where(np.asarray(cond, dtype=bool), 1.0, 0.0)
    return curve_weight(w, band=0.0, cost=cost, min_hold_months=min_hold_months)


def metrics(nv):
    s = nv[START:]
    ann = (s.iloc[-1]/s.iloc[0]) ** (365.25/(s.index[-1]-s.index[0]).days) - 1
    dd = (s/s.cummax()-1).min()
    seg = []
    for _, a, b in SEGS:
        t = nv[a:b]
        if len(t) < 120: seg.append(np.nan); continue
        seg.append((t.iloc[-1]/t.iloc[0]) ** (365.25/(t.index[-1]-t.index[0]).days) - 1)
    return ann, dd, (ann/abs(dd) if dd < 0 else np.nan), seg


def describe():
    print('共享数据层 d177_common')
    print('  区间', idx[0].date(), '→', idx[-1].date(), ' n=', n)
    print('  当前 rA=%.3f  PB分位=%.3f  CGO=%.1f%%  点位=%.0f'
          % (rA.iloc[-1], pb.iloc[-1], cgo.iloc[-1]*100, c.iloc[-1]))
    print('  月末调仓次数上限', len(last_of_month), ' rf_d=%.7f  cost=%.3f' % (rf_d, COST))


if __name__ == '__main__':
    describe()
