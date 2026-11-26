# -*- coding: utf-8 -*-
"""
d160_indep.py  ——  R47 的方法论返工：把"段内每个周末"改成"每段一个独立观测"。

上一版（d158 C 表）报了 787 个观测点，其实只来自 349 个上涨段 ——
同一段的相邻周末共享绝大部分未来，等于把 n 虚增了 2~10 倍。
这跟 R45 那轮"33 次虚报成 60 次"是同一个病。

本脚本：
  1. 独立事件口径重做「时间 × 幅度」：每段只取一个观测点（条件首次满足时）
  2. 块自助法（按段重采样）给中位置信区间
  3. 幅度的倒 U 型：短/中/长三个尺度分别检验（25日 / 整段 / 段末）
  4. 最小持有期 × 波动档：找每档的收益转正点
"""
import numpy as np
import pandas as pd

from d141_eat_bite import load

COST = 0.002


def build_weeks(d):
    c = d['close'].values
    wy = d['d'].dt.strftime('%G-%V')
    wu = list(dict.fromkeys(wy.tolist()))
    wlast = [int(np.where(wy.values == k)[0][-1]) for k in wu]
    return c[wlast], [d['d'].iloc[i] for i in wlast], wlast


def segs_of(wc):
    rr = np.diff(wc) / wc[:-1]
    sg = np.sign(rr)
    out = []
    cur, s0 = 1, 0
    for i in range(1, len(sg)):
        if sg[i] == sg[i - 1] and sg[i] != 0:
            cur += 1
        else:
            out.append((sg[i - 1], s0, s0 + cur, cur))
            s0, cur = s0 + cur, 1
    out.append((sg[-1], s0, s0 + cur, cur))
    return out


def boot_ci(vals, B=5000, seed=3):
    if len(vals) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    ms = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(B)]
    return (np.percentile(ms, 2.5), np.percentile(ms, 97.5))


def main():
    d = load()
    c = d['close'].values
    dts = d['d']
    n = len(c)
    wc, wd, wlast = build_weeks(d)
    nw = len(wc)
    segs = segs_of(wc)

    print('=' * 108)
    print('R47 返工：独立事件口径（每段一个观测点）')
    print('=' * 108)

    # ---------------- 1. 独立事件：每段达到「≥k周 且 >g%」的第一次
    print('\n【1】时间 × 幅度（独立事件：每段只取条件首次成立的周末）')
    print('-' * 108)
    grid_k = [3, 7, 11]
    grid_g = [0.05, 0.15, 0.30]
    print(f'  {"条件":<26} {"独立事件数":>10} {"次13周中位":>11} {"95%CI":>20} {"胜率":>8} {"次13周最大回撤中位":>18}')
    rec = {}
    for k in grid_k:
        for g in grid_g:
            f13, dd, seg_ids = [], [], []
            for si, s_ in enumerate(segs):
                sign, a0, a1, L = s_
                if sign <= 0 or L < k:
                    continue
                base = wc[a0 - 1] if a0 > 0 else wc[0]
                hit = None
                for kk in range(k, L + 1):
                    ek = a0 + kk - 1
                    if ek + 13 >= nw:
                        break
                    if wc[ek] / base - 1 > g:
                        hit = ek
                        break
                if hit and hit + 13 < nw:
                    f13.append(wc[hit + 13] / wc[hit] - 1)
                    dd.append(wc[hit:hit + 14].min() / wc[hit] - 1)
                    seg_ids.append(si)
            if len(f13) < 3:
                print(f'  ≥{k:2d}周 且 >{g*100:.0f}%{"":<10} {len(f13):>10}  （样本太少）')
                continue
            lo, hi = boot_ci(f13)
            print(f'  ≥{k:2d}周 且 >{g*100:.0f}%{"":<10} {len(f13):>10} {np.median(f13)*100:>+10.2f}% '
                  f'[{lo*100:>+7.2f}%,{hi*100:>+7.2f}%] {(np.array(f13)>0).mean()*100:>7.1f}% '
                  f'{np.median(dd)*100:>+17.2f}%')
            rec[(k, g)] = (f13, seg_ids)

    # ---------------- 2. 对照：无条件的同期基准
    print('\n  对照（同口径，不加任何条件）：')
    allf, alld = [], []
    for s_ in segs:
        sign, a0, a1, L = s_
        if sign <= 0 or L < 11:
            continue
        ek = a0 + 10
        if ek + 13 < nw:
            allf.append(wc[ek + 13] / wc[ek] - 1)
            alld.append(wc[ek:ek + 14].min() / wc[ek] - 1)
    lo, hi = boot_ci(allf)
    print(f'  上涨段第 11 周末（无条件） n={len(allf)}  中位 {np.median(allf)*100:+.2f}% '
          f'[{lo*100:+.2f}%,{hi*100:+.2f}%]  胜率 {(np.array(allf)>0).mean()*100:.0f}%  '
          f'回撤中位 {np.median(alld)*100:+.2f}%')

    # ---------------- 3. 幅度倒 U：三个尺度
    print('\n' + '=' * 108)
    print('【3】幅度是"倒 U 型"吗：三个时间尺度分别看（独立事件）')
    print('-' * 108)

    # 3a 短尺度：25 日涨幅分档 → 后续 20 日
    r25 = np.array([c[i] / c[max(0, i - 25)] - 1 if i >= 25 else np.nan for i in range(n)])
    ok = ~np.isnan(r25)
    qs = np.nanpercentile(r25[ok], [20, 40, 60, 80])
    print('  3a 短期（25 交易日涨幅分档 → 后续 20 日收益，日频非重叠采样每 20 日取一次）')
    print(f'  {"档位":<14} {"n":>6} {"后续20日中位":>13} {"CI":>20} {"胜率":>8}')
    for b, nm in enumerate(['Q1 最低', 'Q2', 'Q3', 'Q4', 'Q5 最高']):
        lo_e = -9 if b == 0 else qs[b - 1]
        hi_e = 9 if b == 4 else qs[b]
        idx = [i for i in range(26, n - 21, 20) if lo_e <= r25[i] < hi_e]
        if len(idx) < 10:
            continue
        v = [c[i + 20] / c[i] - 1 for i in idx]
        lo, hi = boot_ci(v)
        print(f'  {nm:<14} {len(idx):>6} {np.median(v)*100:>+12.2f}% [{lo*100:>+7.2f}%,{hi*100:>+7.2f}%] '
              f'{(np.array(v)>0).mean()*100:>7.1f}%')

    # 3b 中尺度：整段涨幅（段结束后 13 周）
    print('\n  3b 中尺度（整段累计涨幅分档 → 段结束后 13 周，独立事件=每段）')
    print(f'  {"档位":<14} {"n":>6} {"段后13周中位":>14} {"CI":>20} {"胜率":>8}')
    segG = []
    for s_ in segs:
        sign, a0, a1, L = s_
        if sign <= 0 or L < 4:
            continue
        base = wc[a0 - 1] if a0 > 0 else wc[0]
        end = min(a1, nw - 1)
        if end + 13 >= nw:
            continue
        segG.append((wc[end] / base - 1, wc[end + 13] / wc[end] - 1))
    SG = pd.DataFrame(segG, columns=['G', 'F'])
    gq = SG.G.quantile([0.2, 0.4, 0.6, 0.8]).values
    for b, nm in enumerate(['Q1 最小', 'Q2', 'Q3', 'Q4', 'Q5 最大']):
        lo_e = -9 if b == 0 else gq[b - 1]
        hi_e = 9 if b == 4 else gq[b]
        sub = SG[(SG.G >= lo_e) & (SG.G < hi_e)]
        if len(sub) < 8:
            continue
        v = sub.F.values
        lo, hi = boot_ci(v)
        print(f'  {nm:<14} {len(sub):>6} {np.median(v)*100:>+13.2f}% [{lo*100:>+7.2f}%,{hi*100:>+7.2f}%] '
              f'{(v>0).mean()*100:>7.1f}%')

    # 3c 段长 vs 段幅 谁携带信息（偏相关）
    print('\n  3c 段长与段幅谁带信息（段结束后 13 周收益对二者回归）')
    rows = []
    for s_ in segs:
        sign, a0, a1, L = s_
        if sign <= 0 or L < 4:
            continue
        base = wc[a0 - 1] if a0 > 0 else wc[0]
        end = min(a1, nw - 1)
        if end + 13 >= nw:
            continue
        rows.append((L, wc[end] / base - 1, wc[end + 13] / wc[end] - 1))
    RG = pd.DataFrame(rows, columns=['L', 'G', 'F'])
    X = np.column_stack([np.ones(len(RG)), RG.L.values, RG.G.values])
    y = RG.F.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - 3)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    print(f'  n={len(RG)}（独立上涨段）')
    print(f'    F = {beta[0]*100:+.2f}%  +  ({beta[1]*100:+.3f}%)×段长  +  ({beta[2]*100:+.2f}%)×段幅')
    print(f'    t(段长) = {beta[1]/se[1]:+.2f}   t(段幅) = {beta[2]/se[2]:+.2f}')

    # ---------------- 4. 最小持有期 × 波动档
    print('\n' + '=' * 108)
    print('【4】最小持有期：按"买入时的波动档"分，持有多少天净收益才转正')
    print('-' * 108)
    r = np.diff(np.log(c))
    ar = np.abs(r)
    rv = pd.Series(ar).rolling(20).mean().shift(1).values   # 只用前 20 日，避免未来
    qs = np.nanpercentile(rv[~np.isnan(rv)], [20, 40, 60, 80])
    print(f'  {"波动档":<10} {"买入样本":>8}' + ''.join([f'{"T="+str(t):>9}' for t in (2, 3, 5, 10, 20, 40)]))
    for b, nm in enumerate(['Q1 最低', 'Q2', 'Q3', 'Q4', 'Q5 最高']):
        lo_e = -9 if b == 0 else qs[b - 1]
        hi_e = 9 if b == 4 else qs[b]
        base_i = [i for i in range(21, n - 41, 5) if not np.isnan(rv[i]) and lo_e <= rv[i] < hi_e]
        if len(base_i) < 50:
            continue
        line = f'  {nm:<10} {len(base_i):>8}'
        for T in (2, 3, 5, 10, 20, 40):
            v = [c[i + T] / c[i] - 1 - COST for i in base_i]
            line += f'{np.median(v)*100:>+8.2f}%'
        print(line)
    print('\n  （每 5 个交易日取一个买入点，避免重叠；数值=扣 0.2% 成本后中位净收益）')

    # 转正点
    print('\n  各档"中位净收益首次转正"的持有天数：')
    for b, nm in enumerate(['Q1 最低', 'Q2', 'Q3', 'Q4', 'Q5 最高']):
        lo_e = -9 if b == 0 else qs[b - 1]
        hi_e = 9 if b == 4 else qs[b]
        base_i = [i for i in range(21, n - 121, 5) if not np.isnan(rv[i]) and lo_e <= rv[i] < hi_e]
        if len(base_i) < 50:
            continue
        found = None
        for T in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120):
            v = [c[i + T] / c[i] - 1 - COST for i in base_i if i + T < n]
            if np.median(v) > 0:
                found = T
                break
        print(f'    {nm:<10} {"T = " + str(found) + " 天" if found else "120 天内从未转正"}')

    print('\n' + '=' * 108)
    print(f'当前：上证 {c[-1]:.2f}  RV20 分位 {(rv[-1] > qs).sum()*20}% 档')
    print('=' * 108)


if __name__ == '__main__':
    main()
