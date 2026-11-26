# -*- coding: utf-8 -*-
"""
d135 波动率这个规律能拿来干什么
- 当前处在什么波动环境
- 高/低波动环境下，"跌到下一档"要多久（决定建仓间距）
- 波动率是否结构性下降（解释 MA 波段失效）
- 当前离底 +36% 到出手区 +10%，历史上类似情况要多久
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load

pd.set_option('display.width', 240)


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] - 1
    ar = np.abs(r)
    vol20 = pd.Series(ar).rolling(20).mean().values * np.sqrt(243)   # 年化
    vol60 = pd.Series(ar).rolling(60).mean().values * np.sqrt(243)

    # ============ 1. 波动率是不是结构性下降 ============
    print('=' * 92)
    print('一、A 股波动率是不是结构性下降（这解释为什么 MA 波段失效）')
    print('=' * 92)
    print(f'  {"年份":>6} {"年化波动(20日均值中位)":>22} {"年内最大单日":>12} {"|日涨跌|中位":>13}')
    for y in range(1997, 2027):
        m = d.d.dt.year == y
        if m.sum() < 100:
            continue
        x = vol20[m.values]
        x = x[np.isfinite(x)]
        rr = r[m.values]
        print(f'  {y:6d} {np.median(x)*100:20.1f}% {np.abs(rr).max()*100:11.2f}% '
              f'{np.median(np.abs(rr))*100:12.2f}%')
    print()
    # 三个时代
    for lab, a, b in [('1997-2007', '1997-01-01', '2008-01-01'),
                      ('2008-2016', '2008-01-01', '2017-01-01'),
                      ('2017-2026', '2017-01-01', '2027-01-01')]:
        m = (d.d >= a) & (d.d < b)
        x = vol20[m.values]
        x = x[np.isfinite(x)]
        print(f'  {lab}: 年化波动中位 {np.median(x)*100:5.1f}%  '
              f'|日涨跌|中位 {np.median(np.abs(r[m.values]))*100:.2f}%  '
              f'（n={m.sum()}）')

    # ============ 2. 当前读数 ============
    print()
    print('=' * 92)
    print('二、现在处在什么波动环境')
    print('=' * 92)
    cur_vol = vol20[-1]
    hist = vol20[np.isfinite(vol20)]
    pct = (hist < cur_vol).mean() * 100
    print(f'  当前 20 日年化波动 {cur_vol*100:.1f}%')
    print(f'  历史分位 {pct:.1f}%（历史中位 {np.median(hist)*100:.1f}%）')
    for q in [10, 25, 50, 75, 90]:
        print(f'    p{q}: {np.percentile(hist,q)*100:5.1f}%')
    print()
    print(f'  当前 60 日年化波动 {vol60[-1]*100:.1f}%')
    print()
    # 按当前档位，未来 20 日可能波动多大
    past20 = pd.Series(ar).rolling(20).mean().values
    f20 = pd.Series(ar).rolling(20).mean().shift(-20).values
    v = pd.DataFrame({'past': past20, 'fut': f20}).dropna()
    # 找当前值落在哪个分位
    cuts = np.percentile(v['past'], [25, 50, 75])
    lab = 0 if past20[-1] <= cuts[0] else (1 if past20[-1] <= cuts[1] else (2 if past20[-1] <= cuts[2] else 3))
    names = ['低', '偏低', '偏高', '高']
    sub = v.iloc[(np.searchsorted(np.sort(v['past'].values), v['past'].values) >= 0)]
    # 直接用同样的分位切
    v['bin'] = pd.cut(v['past'], bins=[-1] + list(cuts) + [1],
                      labels=['低', '偏低', '偏高', '高'])
    g = v.groupby('bin', observed=True)['fut'].agg(['size', 'mean', 'median',
                                                    lambda x: np.percentile(x, 5),
                                                    lambda x: np.percentile(x, 95)])
    g.columns = ['n', 'mean', 'median', 'p5', 'p95']
    print(f'  当前落在「{names[lab]}」档。历史同档未来 20 日日均波动：')
    row = g.loc[names[lab]]
    print(f'    中位 {row["median"]*100:.3f}%/日 → 年化 {row["median"]*np.sqrt(243)*100:.1f}%')
    print(f'    区间 p5~p95: {row["p5"]*100:.3f}% ~ {row["p95"]*100:.3f}%/日')
    print(f'    换成 20 日累计：约 ±{row["median"]*20*100:.1f}%（中位），'
          f'极端 ±{row["p95"]*20*100:.1f}%')

    # ============ 3. 高/低波动环境下，跌 10% 要多久 ============
    print()
    print('=' * 92)
    print('三、关键：在不同波动环境下，"跌到下一档"要多久（决定建仓间距对不对）')
    print('=' * 92)
    print('  做法：对每个交易日，看从当天起跌 10%/19% 需要多少交易日（向前找第一次触及）')
    print()
    print(f'  {"波动档":>6} {"跌10%中位耗时":>14} {"跌19%中位耗时":>14} {"n":>6}')
    for lo, hi, nm in [(0, 0.12, '低<12%'), (0.12, 0.20, '中12-20%'), (0.20, 9, '高>20%')]:
        sel = np.isfinite(vol20) & (vol20 >= lo) & (vol20 < hi)
        idxs = np.where(sel)[0]
        t10, t19 = [], []
        for i in idxs[::5]:     # 抽样加速
            p0 = c[i]
            # 跌 10%
            j = i + 1
            while j < min(n, i + 500) and c[j] > p0 * 0.90:
                j += 1
            if j < min(n, i + 500):
                t10.append(j - i)
            # 跌 19%
            j = i + 1
            while j < min(n, i + 500) and c[j] > p0 * 0.81:
                j += 1
            if j < min(n, i + 500):
                t19.append(j - i)
        if t10:
            print(f'  {nm:>6} {np.median(t10):11.0f} 日 '
                  f'({np.median(t10)/20.3:.1f}月) {np.median(t19) if t19 else np.nan:11.0f} 日 '
                  f'({np.median(t19)/20.3 if t19 else np.nan:.1f}月) {len(t10):6d}')

    # ============ 4. 当前：从 +36% 走到 +10% 的历史先例 ============
    print()
    print('=' * 92)
    print('四、当前离底 +36%，走到出手区 +10%，历史上类似情况要多久')
    print('=' * 92)
    # 用名义点位近似（这里不追求精确，只看量级）
    # 简化：找历史上距 250 日低点 +36% 左右的时点（用滚动偏离度）
    low250 = pd.Series(c).rolling(250).min().values
    dev = c / low250 - 1
    print(f'  当前：上证 {c[-1]:.0f}，250 日最低 {low250[-1]:.0f}，'
          f'高出 {dev[-1]*100:+.1f}%')
    print()
    print('  历史上"高出 250 日低点 30~40%"之后，多久回落到 +10% 以内：')
    waits = []
    for i in range(250, n - 1):
        if not np.isfinite(dev[i]):
            continue
        if 0.30 <= dev[i] <= 0.40:
            j = i + 1
            while j < min(n, i + 750) and dev[j] > 0.10:
                j += 1
            if j < min(n, i + 750):
                waits.append((i, j - i, c[j] / c[i] - 1))
    if waits:
        w = np.array([x[1] for x in waits])
        dr = np.array([x[2] for x in waits])
        print(f'    可统计样本 {len(waits)} 次')
        print(f'    耗时中位 {np.median(w):.0f} 交易日 ({np.median(w)/20.3:.1f} 月)  '
              f'p25 {np.percentile(w,25):.0f}  p75 {np.percentile(w,75):.0f}  最长 {w.max():.0f}')
        print(f'    期间跌幅中位 {np.median(dr)*100:.1f}%  '
              f'最深 {dr.min()*100:.1f}%')
        print(f'    750 日内没跌到 +10% 的：{sum(1 for i in range(250, n-1) if np.isfinite(dev[i]) and 0.30<=dev[i]<=0.40) - len(waits)} 次')


if __name__ == '__main__':
    main()
