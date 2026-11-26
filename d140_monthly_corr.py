# -*- coding: utf-8 -*-
"""
d140 探针：月线级别的回调存在吗？频率、幅度、时长、之后怎么走
- 不折 CPI（持有几个月，通胀 1~2%，可忽略；报告里标注）
- 两条口径互为校验：
  A) 日线口径：从滚动 250 日高点回撤 >= X%，且持续 >= 20 个交易日
  B) 月线口径：月收盘从月收盘高点回撤 >= 12%，连续 >= 2 个月
- 事后分类：这次回撤后来回前高了（真回调）还是没回（转熊）
"""
import numpy as np
import pandas as pd

pd.set_option('display.width', 240)


def load(start='1997-01-01'):
    a = pd.read_parquet('idx_sse_all.parquet')
    a['d'] = pd.to_datetime(a['date'])
    a = a[['d', 'open', 'high', 'low', 'close', 'amount']].sort_values('d').reset_index(drop=True)
    a = a[a['d'] >= start].reset_index(drop=True)
    for k in ['open', 'high', 'low', 'close', 'amount']:
        a[k] = pd.to_numeric(a[k], errors='coerce')
    return a.dropna(subset=['close']).reset_index(drop=True)


def dd_series(c, win=250):
    """滚动 win 日最高点的回撤（不含当日，避免未来函数）"""
    n = len(c)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - win)
        hi = c[lo:i + 1].max()
        out[i] = c[i] / hi - 1
    return out


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    print('=' * 100)
    print(f'样本 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  {n} 交易日  {yrs:.1f} 年')
    print('=' * 100)

    # ---------- A) 日线口径：回撤阈值扫描 ----------
    dd = dd_series(c, 250)
    print()
    print('【A】日线口径：从 250 日高点回撤的分布与频率（冷却 60 日去重）')
    print(f'{"阈值":>6} {"触发次数":>8} {"次/年":>7} {"之后20日":>9} {"之后60日":>9} '
          f'{"之后120日":>9} {"之后250日":>9} {"后续最大浮亏":>10} {"回前高比例":>10}')
    rowsA = []
    for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        ev = []
        last = -999
        for i in range(250, n):
            if dd[i] <= -thr and i - last > 60:
                ev.append(i)
                last = i
        if not ev:
            continue
        f20, f60, f120, f250, mdd, rec = [], [], [], [], [], []
        for i in ev:
            for arr, h in [(f20, 20), (f60, 60), (f120, 120), (f250, 250)]:
                j = min(n - 1, i + h)
                arr.append(c[j] / c[i] - 1)
            seg = c[i:min(n, i + 250)]
            mdd.append(seg.min() / c[i] - 1)
            prior = c[max(0, i - 250):i].max()
            rec.append(1.0 if seg.max() >= prior else 0.0)
        rowsA.append(dict(thr=thr, n=len(ev), per=len(ev) / yrs,
                          f20=np.median(f20), f60=np.median(f60),
                          f120=np.median(f120), f250=np.median(f250),
                          mdd=np.median(mdd), rec=np.mean(rec),
                          e20=np.mean(f20), e60=np.mean(f60),
                          e120=np.mean(f120), e250=np.mean(f250)))
        r = rowsA[-1]
        print(f'{thr*100:5.0f}% {len(ev):8d} {len(ev)/yrs:7.2f} '
              f'{np.median(f20)*100:+8.1f}% {np.median(f60)*100:+8.1f}% '
              f'{np.median(f120)*100:+8.1f}% {np.median(f250)*100:+8.1f}% '
              f'{np.median(mdd)*100:+9.1f}% {np.mean(rec)*100:9.0f}%')

    print()
    print('同上，但看【均值】（被极端值拉动）')
    print(f'{"阈值":>6} {"之后20日":>9} {"之后60日":>9} {"之后120日":>9} {"之后250日":>9}')
    for r in rowsA:
        print(f'{r["thr"]*100:5.0f}% {r["e20"]*100:+8.1f}% {r["e60"]*100:+8.1f}% '
              f'{r["e120"]*100:+8.1f}% {r["e250"]*100:+8.1f}%')

    # ---------- B) 月线口径 ----------
    m = d.set_index('d')['close'].resample('ME').last().dropna()
    mv = m.values
    nm = len(mv)
    print()
    print('=' * 100)
    print(f'【B】月线口径：月收盘从月收盘高点回撤  {m.index[0].date()} ~ {m.index[-1].date()}  {nm} 个月')
    print('=' * 100)
    print(f'{"阈值":>6} {"触发次数":>8} {"次/年":>7} {"之后3月":>9} {"之后6月":>9} '
          f'{"之后12月":>9} {"后续最大浮亏":>10} {"回前高比例":>10}')
    for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        ev = []
        last = -999
        for i in range(1, nm):
            hi = mv[:i + 1].max()
            if mv[i] / hi - 1 <= -thr and i - last >= 3:
                ev.append(i)
                last = i
        if not ev:
            continue
        fs = {3: [], 6: [], 12: []}
        mdd, rec = [], []
        for i in ev:
            for h in fs:
                j = min(nm - 1, i + h)
                fs[h].append(mv[j] / mv[i] - 1)
            seg = mv[i:min(nm, i + 12)]
            mdd.append(seg.min() / mv[i] - 1)
            prior = mv[:i + 1].max()
            rec.append(1.0 if seg.max() >= prior else 0.0)
        print(f'{thr*100:5.0f}% {len(ev):8d} {len(ev)/(nm/12):7.2f} '
              f'{np.median(fs[3])*100:+8.1f}% {np.median(fs[6])*100:+8.1f}% '
              f'{np.median(fs[12])*100:+8.1f}% {np.median(mdd)*100:+9.1f}% '
              f'{np.mean(rec)*100:9.0f}%')

    # ---------- C) 15% 回撤事件明细（事后分类） ----------
    print()
    print('=' * 100)
    print('【C】15% 回撤事件明细（250 日高点口径，冷却 60 日）')
    print('=' * 100)
    ev = []
    last = -999
    for i in range(250, n):
        if dd[i] <= -0.15 and i - last > 60:
            ev.append(i)
            last = i
    print(f'{"#":>3} {"触发日":>11} {"点位":>7} {"距250高":>8} {"之后最低":>8} '
          f'{"最低在几日":>9} {"250日后":>9} {"回前高":>6} {"当时MA250":>9}')
    ma250 = pd.Series(c).rolling(250).mean().values
    lines = []
    for k, i in enumerate(ev):
        seg = c[i:min(n, i + 250)]
        li = int(np.argmin(seg))
        prior = c[max(0, i - 250):i].max()
        rec = '是' if seg.max() >= prior else '否'
        above = '上方' if (not np.isnan(ma250[i]) and c[i] > ma250[i]) else '下方'
        f250 = c[min(n - 1, i + 250)] / c[i] - 1
        lines.append((k, d.d.iloc[i].date(), c[i], dd[i], seg.min() / c[i] - 1,
                      li, f250, rec, above))
        print(f'{k:3d} {str(d.d.iloc[i].date()):>11} {c[i]:7.0f} {dd[i]*100:+7.1f}% '
              f'{seg.min()/c[i]*100-100:+7.1f}% {li:9d} {f250*100:+8.1f}% '
              f'{rec:>6} {above:>9}')


if __name__ == '__main__':
    main()
