# -*- coding: utf-8 -*-
"""
d130 探针：两个月级别的波动存在吗？
- 不依赖 CPI（两个月内通胀可忽略），用全史名义点位
- ZigZag 多阈值切段 -> 段时长分布
- 周期图（FFT）-> 有没有 40 日近似峰
- 固定 40 日滚动收益的分布
"""
import numpy as np
import pandas as pd

pd.set_option('display.width', 200)


def load(start='1997-01-01'):
    a = pd.read_parquet('idx_sse_all.parquet')
    a['d'] = pd.to_datetime(a['date'])
    a = a[['d', 'open', 'high', 'low', 'close', 'amount']].sort_values('d').reset_index(drop=True)
    a = a[a['d'] >= start].reset_index(drop=True)
    a['close'] = pd.to_numeric(a['close'], errors='coerce')
    a['amount'] = pd.to_numeric(a['amount'], errors='coerce')
    return a.dropna(subset=['close']).reset_index(drop=True)


def zigzag(c, thr):
    """返回段列表：[(start_i, end_i, 方向, 幅度, 段内极值idx)]"""
    n = len(c)
    if n < 5:
        return []
    segs = []
    # 初始化：先定第一个转折
    last_pivot = 0
    direction = 0  # 0 未定, 1 上, -1 下
    ext_i = 0  # 当前极值位置
    ext_p = c[0]
    for i in range(1, n):
        if direction == 0:
            if c[i] > ext_p * (1 + thr):
                direction = 1
                segs.append((last_pivot, ext_i, -1, c[ext_i] / c[last_pivot] - 1, ext_i))
                last_pivot = ext_i
                ext_i, ext_p = i, c[i]
            elif c[i] < ext_p * (1 - thr):
                direction = -1
                segs.append((last_pivot, ext_i, 1, c[ext_i] / c[last_pivot] - 1, ext_i))
                last_pivot = ext_i
                ext_i, ext_p = i, c[i]
            else:
                if c[i] > ext_p:
                    ext_i, ext_p = i, c[i]
        elif direction == 1:
            if c[i] > ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] < ext_p * (1 - thr):
                segs.append((last_pivot, ext_i, 1, ext_p / c[last_pivot] - 1, ext_i))
                last_pivot = ext_i
                direction = -1
                ext_i, ext_p = i, c[i]
        else:
            if c[i] < ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] > ext_p * (1 + thr):
                segs.append((last_pivot, ext_i, -1, ext_p / c[last_pivot] - 1, ext_i))
                last_pivot = ext_i
                direction = 1
                ext_i, ext_p = i, c[i]
    segs.append((last_pivot, ext_i, direction if direction else 1,
                 c[ext_i] / c[last_pivot] - 1, ext_i))
    return segs


if __name__ == '__main__':
    d = load()
    c = d['close'].values
    print(f'样本 {len(d)} 交易日  {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  '
          f'({(d.d.iloc[-1]-d.d.iloc[0]).days/365.25:.1f} 年)')
    print(f'点位 {c[0]:.0f} -> {c[-1]:.0f}，最高 {c.max():.0f}，最低 {c.min():.0f}')
    print()

    # ---------- 1. ZigZag 段时长 ----------
    print('=' * 78)
    print('ZigZag 切段：段时长（交易日）分布 —— 找"两个月"')
    print('=' * 78)
    print(f'{"阈值":>6} {"段数":>5} {"中位":>6} {"均值":>6} {"p25":>6} {"p75":>6} '
          f'{"折算月(p50)":>10} {"≥20日占比":>9} {"8-60日占比":>9}')
    for thr in [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        segs = zigzag(c, thr)
        L = np.array([s[1] - s[0] for s in segs if s[1] > s[0]])
        if len(L) < 5:
            continue
        ratio = float(d['d'].iloc[-1].year - d['d'].iloc[0].year)  # 粗略
        print(f'{thr*100:5.0f}% {len(segs):5d} {np.median(L):6.0f} {L.mean():6.1f} '
              f'{np.percentile(L,25):6.0f} {np.percentile(L,75):6.0f} '
              f'{np.median(L)/20.3:10.1f} {(L>=20).mean()*100:8.1f}% '
              f'{((L>=8)&(L<=60)).mean()*100:8.1f}%')

    print()
    print('  说明：40 交易日 ≈ 2 个月。看中位数在哪一档阈值上落在 40 附近。')

    # ---------- 2. 段时长与幅度联合 ----------
    print()
    print('=' * 78)
    print('幅度-时长对照（阈值 5%）：每个"波动"到底多大、多久')
    print('=' * 78)
    for thr in [0.05, 0.08]:
        segs = zigzag(c, thr)
        L = np.array([s[1] - s[0] for s in segs])
        A = np.array([abs(s[3]) for s in segs])
        ok = L > 0
        L, A, segs2 = L[ok], A[ok], [s for s in segs if s[1] > s[0]]
        print(f'阈值 {thr*100:.0f}%: n={len(L)}  幅度中位 {np.median(A)*100:.1f}%  '
              f'时长中位 {np.median(L):.0f} 日  幅度×时长相关 '
              f'{np.corrcoef(np.log(A), np.log(L))[0,1]:+.2f}')
        # 按幅度分档看时长
        for lo, hi in [(0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 9)]:
            m = (A >= lo) & (A < hi)
            if m.sum() >= 3:
                print(f'   幅度 {lo*100:.0f}~{hi*100 if hi<9 else 99:.0f}%: n={m.sum():3d} '
                      f'时长中位 {np.median(L[m]):3.0f} 日 '
                      f'({np.median(L[m])/20.3:.1f} 月)')

    # ---------- 3. 周期图 ----------
    print()
    print('=' * 78)
    print('周期图：日收益率序列的频谱，找 40 日附近的能量')
    print('=' * 78)
    r = np.diff(np.log(c))
    r = r - r.mean()
    n = len(r)
    F = np.fft.rfft(r)
    P = (np.abs(F) ** 2) / n
    freq = np.fft.rfftfreq(n, d=1.0)  # 每交易日的频率
    period = 1.0 / np.maximum(freq, 1e-12)
    # 只看 5~250 日
    m = (period >= 5) & (period <= 250)
    pp, pv = period[m], P[m]
    # 按周期分桶求平均能量
    print(f'{"周期(日)":>10} {"周期(月)":>9} {"相对能量":>10} {"排名":>6}')
    order = np.argsort(-pv)
    for lo, hi in [(5, 10), (10, 15), (15, 25), (25, 35), (35, 45), (45, 55),
                   (55, 70), (70, 90), (90, 130), (130, 180), (180, 250)]:
        mm = (pp >= lo) & (pp < hi)
        if mm.sum() > 0:
            print(f'{lo:5d}~{hi:3d} {(lo+hi)/2/20.3:8.1f} {pv[mm].mean()/pv.mean():9.2f}x')
    print()
    print('  最强 12 个周期点：')
    for k in order[:12]:
        print(f'    {pp[k]:6.1f} 日 ({pp[k]/20.3:5.1f} 月)  能量 {pv[k]/pv.mean():5.2f}x 中位')

    # ---------- 4. 固定窗口收益 ----------
    print()
    print('=' * 78)
    print('固定窗口：40 交易日（2 个月）滚动收益的分布')
    print('=' * 78)
    for w in [10, 20, 40, 60]:
        rr = c[w:] / c[:-w] - 1
        print(f'{w:3d} 日: 均值 {rr.mean()*100:+5.2f}%  sd {rr.std()*100:5.2f}%  '
              f'|涨|中位 {np.median(np.abs(rr))*100:5.2f}%  '
              f'p5 {np.percentile(rr,5)*100:+6.1f}%  p95 {np.percentile(rr,95)*100:+6.1f}%  '
              f'跌>5%占比 {(rr<-0.05).mean()*100:4.1f}%  涨>5%占比 {(rr>0.05).mean()*100:4.1f}%')

    # ---------- 5. "两个月一次"这个说法验证 ----------
    print()
    print('=' * 78)
    print('验证："两个月左右就会有一次"——多久出现一次有意义的波动？')
    print('=' * 78)
    for thr in [0.05, 0.08, 0.10]:
        segs = zigzag(c, thr)
        nseg = len([s for s in segs if s[1] > s[0]])
        yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
        print(f'阈值 {thr*100:.0f}%: {nseg} 段 / {yrs:.1f} 年 = '
              f'{nseg/yrs:5.1f} 段/年 = 每 {yrs*12/nseg:4.1f} 个月一次')
