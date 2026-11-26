# -*- coding: utf-8 -*-
"""
d141 月线回调能不能吃一口：出场规则网格 × 事前筛选变量
核心问题：回撤 15% 触发买入，「吃一口」（小止盈快跑）到底赚不赚
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.width', 240)
RF = 0.025          # 空仓生息 2.5%/年


def load(start='1997-01-01'):
    a = pd.read_parquet('idx_sse_all.parquet')
    a['d'] = pd.to_datetime(a['date'])
    a = a[['d', 'open', 'high', 'low', 'close']].sort_values('d').reset_index(drop=True)
    a = a[a['d'] >= start].reset_index(drop=True)
    for k in ['open', 'high', 'low', 'close']:
        a[k] = pd.to_numeric(a[k], errors='coerce')
    a = a.dropna(subset=['close']).reset_index(drop=True)

    # 实际点位（CPI 定基 2008-01=100，口径同 d91），仅用于「离底多远」这一筛选变量
    # 注意：CPI 数据从 2008-01 起，2008 年之前 rA 为 NaN
    cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
    cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
    cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
    cpi = cpi.sort_values('d').reset_index(drop=True)
    cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
    idx = [100.0]
    for i in range(1, len(cpi)):
        idx.append(idx[-1] * cpi['mom'].iloc[i])
    cpi['cpi_idx'] = idx
    cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    a['d'] = a['d'].astype('datetime64[ns]')
    cpi['avail'] = cpi['avail'].astype('datetime64[ns]')
    a = pd.merge_asof(a, cpi[['avail', 'cpi_idx']].dropna().sort_values('avail'),
                      left_on='d', right_on='avail', direction='backward')
    a['real'] = a['close'] * 100.0 / a['cpi_idx']
    a['rA'] = a['real'] / 1880.0
    return a


def events(d, thr=0.15, win=250, cool=60):
    c = d['close'].values
    n = len(c)
    ev = []
    last = -999
    for i in range(win, n):
        hi = c[i - win:i + 1].max()
        if c[i] / hi - 1 <= -thr and i - last > cool:
            ev.append(i)
            last = i
    return ev


def trade(c, i, tp=None, sl=None, maxd=120, exit_ma=None, ma=None):
    """单笔：返回 (收益, 持有日, 出场原因, 期间最大浮亏)"""
    n = len(c)
    peak = -1e9
    for j in range(i + 1, min(n, i + maxd + 1)):
        ret = c[j] / c[i] - 1
        peak = max(peak, ret)
        low = min(c[i:j + 1]) / c[i] - 1
        # 止损优先（同日都触及时保守）
        if sl is not None and ret <= -sl:
            return -sl if c[j] / c[i] - 1 > -sl * 1.02 else ret, j - i, '止损', low
        if tp is not None and ret >= tp:
            return ret, j - i, '止盈', low
        if exit_ma is not None and ma is not None:
            if not np.isnan(ma[j]) and c[j] < ma[j] and c[j - 1] >= ma[j - 1]:
                return ret, j - i, '破线', low
    j = min(n - 1, i + maxd)
    return c[j] / c[i] - 1, j - i, '到期', min(c[i:j + 1]) / c[i] - 1


def run(d, ev, **kw):
    c = d['close'].values
    ma = kw.pop('_ma', None)
    rs, hold, lows, whys = [], [], [], []
    for i in ev:
        r, h, w, lo = trade(c, i, ma=ma, **kw)
        rs.append(r); hold.append(h); lows.append(lo); whys.append(w)
    return np.array(rs), np.array(hold), np.array(lows), whys


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    ma20 = pd.Series(c).rolling(20).mean().values
    ev = events(d, 0.15)
    print('=' * 118)
    print(f'样本 {d.d.iloc[0].date()}~{d.d.iloc[-1].date()}  {yrs:.1f}年  '
          f'回撤 15% 触发 {len(ev)} 次 = {len(ev)/yrs:.2f} 次/年')
    print('=' * 118)

    plans = [
        ('A +8%止盈/-8%止损/90日', dict(tp=0.08, sl=0.08, maxd=90)),
        ('B +10%止盈/-10%止损/120日', dict(tp=0.10, sl=0.10, maxd=120)),
        ('C +15%止盈/-10%止损/180日', dict(tp=0.15, sl=0.10, maxd=180)),
        ('D +20%止盈/-10%止损/250日', dict(tp=0.20, sl=0.10, maxd=250)),
        ('E 只止损-10%/持有120日', dict(tp=None, sl=0.10, maxd=120)),
        ('F 只止损-10%/持有250日', dict(tp=None, sl=0.10, maxd=250)),
        ('G 无规则/持有60日', dict(tp=None, sl=None, maxd=60)),
        ('H 无规则/持有120日', dict(tp=None, sl=None, maxd=120)),
        ('I 无规则/持有250日', dict(tp=None, sl=None, maxd=250)),
        ('J 破MA20走/-10%止损/250日', dict(tp=None, sl=0.10, maxd=250, exit_ma=1)),
    ]
    print()
    print(f'{"方案":<28} {"胜率":>6} {"均值":>8} {"中位":>8} {"平均持有":>8} '
          f'{"最大浮亏中位":>10} {"年均贡献":>9} {"t值":>7} {"p值":>7}')
    rows = []
    for name, kw in plans:
        rs, hold, lows, whys = run(d, ev, _ma=ma20 if kw.get('exit_ma') else None, **kw)
        win = (rs > 0).mean()
        t, p = stats.ttest_1samp(rs, 0)
        # 年均贡献：每笔收益 × 每年次数（不复合），这是「吃一口」对总资产的年化贡献
        contrib = rs.mean() * len(ev) / yrs
        rows.append(dict(name=name, win=float(win), mean=float(rs.mean()),
                         med=float(np.median(rs)), hold=float(hold.mean()),
                         low=float(np.median(lows)), contrib=float(contrib),
                         t=float(t), p=float(p), n=len(rs)))
        print(f'{name:<28} {win*100:5.0f}% {rs.mean()*100:+7.2f}% {np.median(rs)*100:+7.2f}% '
              f'{hold.mean():7.0f}日 {np.median(lows)*100:+9.1f}% '
              f'{contrib*100:+8.2f}% {t:7.2f} {p:7.3f}')

    print()
    print('对照：同期空仓生息 2.5%/年；买入持有全程年化 '
          f'{(c[-1]/c[0])**(1/yrs)-1:+.2%}')

    # ---------- 事前筛选：离底距离 ----------
    print()
    print('=' * 118)
    print('事前筛选 1：触发时的离底距离 rA（实际点位 / 1,880）—— 用方案 I（无规则持250日）与 F（-10%止损）')
    print('=' * 118)
    ra = d['rA'].values
    buckets = [(0, 1.10, '离底 ≤10%'), (1.10, 1.30, '10~30%'),
               (1.30, 1.60, '30~60%'), (1.60, 99, '>60%')]
    print(f'{"区间":<12} {"n":>4} {"方案I均值":>10} {"I胜率":>7} {"方案F均值":>10} '
          f'{"F胜率":>7} {"触发时点位中位":>12}')
    for lo, hi, lab in buckets:
        idx = [i for i in ev if lo < ra[i] <= hi]
        if not idx:
            print(f'{lab:<12} {0:4d}  —')
            continue
        rI, _, _, _ = run(d, idx, tp=None, sl=None, maxd=250)
        rF, _, _, _ = run(d, idx, tp=None, sl=0.10, maxd=250)
        print(f'{lab:<12} {len(idx):4d} {rI.mean()*100:+9.1f}% {(rI>0).mean()*100:6.0f}% '
              f'{rF.mean()*100:+9.1f}% {(rF>0).mean()*100:6.0f}% '
              f'{np.median([c[i] for i in idx]):12.0f}')

    # ---------- 事前筛选：前 250 日涨幅 ----------
    print()
    print('=' * 118)
    print('事前筛选 2：触发前 250 日的涨幅（涨多了才回调 vs 阴跌）')
    print('=' * 118)
    pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])
    pbs = [(-9, -0.15, '前一年跌 >15%'), (-0.15, 0.0, '前一年跌 0~15%'),
           (0.0, 0.30, '前一年涨 0~30%'), (0.30, 0.80, '前一年涨 30~80%'),
           (0.80, 99, '前一年涨 >80%')]
    print(f'{"区间":<16} {"n":>4} {"方案I均值":>10} {"I胜率":>7} {"方案F均值":>10} {"F胜率":>7}')
    for lo, hi, lab in pbs:
        idx = [i for i in ev if lo < pre[i] <= hi]
        if len(idx) < 2:
            print(f'{lab:<16} {len(idx):4d}  —（样本不足）')
            continue
        rI, _, _, _ = run(d, idx, tp=None, sl=None, maxd=250)
        rF, _, _, _ = run(d, idx, tp=None, sl=0.10, maxd=250)
        print(f'{lab:<16} {len(idx):4d} {rI.mean()*100:+9.1f}% {(rI>0).mean()*100:6.0f}% '
              f'{rF.mean()*100:+9.1f}% {(rF>0).mean()*100:6.0f}%')

    # ---------- 事后看：右尾有多大 ----------
    print()
    print('=' * 118)
    print('关键：每次回撤后 250 日的收益分布（方案 I 的实际分布，判断「吃一口」砍掉了什么）')
    print('=' * 118)
    rs, _, lows, _ = run(d, ev, tp=None, sl=None, maxd=250)
    qs = [5, 10, 25, 50, 75, 90, 95]
    print('分位：', '  '.join([f'p{q}={np.percentile(rs,q)*100:+.1f}%' for q in qs]))
    print(f'均值 {rs.mean()*100:+.1f}%  为正的比例 {(rs>0).mean()*100:.0f}%  '
          f'涨幅>30% 的比例 {(rs>0.30).mean()*100:.0f}%  '
          f'>50% 的比例 {(rs>0.50).mean()*100:.0f}%')
    print(f'如果去掉涨幅最大的 3 笔，均值变成 '
          f'{np.sort(rs)[:-3].mean()*100:+.1f}%')
    print(f'如果去掉涨幅最大的 5 笔，均值变成 '
          f'{np.sort(rs)[:-5].mean()*100:+.1f}%')

    json.dump(dict(rows=rows, n_ev=len(ev), yrs=float(yrs),
                   ev=[str(d.d.iloc[i].date()) for i in ev]),
              open('d141_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n>> d141_result.json 已写')


if __name__ == '__main__':
    main()
