# -*- coding: utf-8 -*-
"""
d158_tail.py  ——  R47：分布形状能不能当约束？

先生的新框架（R47）：
  「感冒平均七天好，不能让咱们第五天就说好了、确定买卖。
    但是他可以让咱们确定：两三天是肯定好不了的，十几天还不好肯定是不正常的。」

→ 不是预测时点，是用分布的**两端**做约束。本脚本把它拆成可测的四件事：

  A. 短端：段长 ≤2 周 / ≤1 月的事件频率，以及它们的后续收益与成本占卜
  B. 长端：已持续 k 周后的条件生存 S(k)、剩余期望、后续 N 周收益（是否有"异常"）
  C. 二维：已持续周数 × 段内累计涨幅 —— 联合状态是否比单一维度有信息
  D. 波动率记忆：|r| 自相关 / EWMA 最优衰减 / 半衰期 / 波动率状态 → 后续回撤

口径：上证日线 1997-2026（CPI 不变价另算在别处），周线按自然周最后交易日聚合。
"""
import numpy as np
import pandas as pd
import io as _io

from d141_eat_bite import load


# ---------------------------------------------------------------- 周线聚合
def weekly(d):
    """返回周序列：wlast(每周末的日索引)、wc(周收盘)、wy(周标签)"""
    c = d['close'].values
    n = len(c)
    wy = d['d'].dt.strftime('%G-%V')
    wu = list(dict.fromkeys(wy.tolist()))
    wmap = {k: j for j, k in enumerate(wu)}
    wlast = [int(np.where(wy.values == k)[0][-1]) for k in wu]
    wc = c[wlast]
    wcode = np.array([wmap[k] for k in wy])
    return wlast, wc, wcode, n


def segments(wc):
    """周线同向段：按相邻周涨跌符号切分。返回 [(sign, start_k, end_k, weeks)]"""
    r = np.diff(wc) / wc[:-1]
    sign = np.sign(r)
    out = []
    cur = 1
    s0 = 0
    for i in range(1, len(sign)):
        if sign[i] == sign[i - 1] and sign[i] != 0:
            cur += 1
        else:
            out.append((sign[i - 1], s0, s0 + cur, cur))
            s0 = s0 + cur
            cur = 1
    out.append((sign[-1], s0, s0 + cur, cur))
    return out


def main():
    d = load()
    c = d['close'].values
    dts = d['d']
    n = len(c)
    yrs = (dts.iloc[-1] - dts.iloc[0]).days / 365.25
    wlast, wc, wcode, _ = weekly(d)
    wd = [dts.iloc[i] for i in wlast]
    nw = len(wc)
    segs = segments(wc)

    print('=' * 108)
    print(f'R47 分布两端能不能当约束？  上证 {dts.iloc[0].date()} ~ {dts.iloc[-1].date()}  {yrs:.1f} 年')
    print(f'周数 {nw}，同向段 {len(segs)} 段')
    print('=' * 108)

    # ============================================================ A. 段长完整分布
    print('\n【A】段长的完整分布 —— 短端到底有多短，长端到底能多长')
    print('-' * 108)
    ups = [s[3] for s in segs if s[0] > 0]
    dns = [s[3] for s in segs if s[0] < 0]
    allw = [s[3] for s in segs]
    for nm, arr in [('上涨段', ups), ('下跌段', dns), ('全部段', allw)]:
        a = np.array(arr)
        print(f'  {nm} n={len(a):3d}  均值 {a.mean():5.2f} 周  中位 {np.median(a):4.0f}  '
              f'p10 {np.percentile(a,10):3.0f}  p90 {np.percentile(a,90):3.0f}  '
              f'最短 {a.min():2.0f}  最长 {a.max():3.0f}')
    a = np.array(allw)
    print()
    print('  短端（先生说的"两三天肯定好不了"）：')
    for k, lab in [(1, '1 周（≈5交易日）'), (2, '≤2 周（≈10交易日）'), (3, '≤3 周')]:
        p = (a <= k).mean()
        print(f'    {lab:<20} 出现 {int((a<=k).sum()):3d} 次   占比 {p*100:5.1f}%')
    print('  长端（先生说的"十几天还不好肯定不正常"）：')
    for k in (8, 12, 16, 20, 26):
        p = (a >= k).mean()
        print(f'    ≥{k:2d} 周（≈{k*5:3d}交易日）        出现 {int((a>=k).sum()):3d} 次   占比 {p*100:5.1f}%'
              + ('   ← 1 个月内' if k == 8 else ''))
    # 几何分布理论对照
    pu = len(ups) / max(1, len(ups) + len(dns))
    print(f'\n  对照：若周涨跌近似抛硬币（p_up={pu:.3f}），理论 P(段长≥k) = p^(k-1)')
    for k in (8, 12, 16, 20):
        th = pu ** (k - 1) if True else 0
        print(f'    ≥{k:2d} 周  理论 {th*100:5.2f}%   实测 {(a>=k).mean()*100:5.2f}%')

    # ============================================================ B. 长端条件生存
    print('\n' + '=' * 108)
    print('【B】长端的"不正常"能不能用：已涨了 k 周，接下来会怎样')
    print('-' * 108)
    print(f'  {"已持续(周)":>10} {"样本段":>7} {"剩余周数中位":>12} {"次4周收益中位":>14} {"次13周收益中位":>15} {"次13周最大回撤中位":>17}')
    # 对每个 k，取"上涨段在已持续 k 周之后"的所有观测点
    for k in (1, 3, 5, 8, 12, 16):
        fut4, fut13, dd13, rem = [], [], [], []
        for sg in segs:
            sign, s0, s1, L = sg
            if sign <= 0 or L < k:
                continue
            # 段内第 k 周末的索引（周序号 k-1 相对 s0）
            ek = s0 + k - 1
            if ek + 1 >= nw:
                continue
            rem.append(L - k)
            if ek + 4 < nw:
                fut4.append(wc[ek + 4] / wc[ek] - 1)
            if ek + 13 < nw:
                seg13 = wc[ek:ek + 14]
                fut13.append(wc[ek + 13] / wc[ek] - 1)
                dd13.append(seg13.min() / wc[ek] - 1)
        if not rem:
            continue
        m4 = np.median(fut4) * 100 if fut4 else float('nan')
        m13 = np.median(fut13) * 100 if fut13 else float('nan')
        d13 = np.median(dd13) * 100 if dd13 else float('nan')
        print(f'  {k:>10} {len(rem):>7} {np.median(rem):>12.1f} {m4:>13.2f}% {m13:>14.2f}% {d13:>16.2f}%')

    # 生存函数
    print('\n  条件生存 S(k) = 已涨 k 周后，还能再涨 1 周以上的比例：')
    for k in (1, 2, 3, 5, 8, 12, 16, 20):
        tot = [sg for sg in segs if sg[0] > 0 and sg[3] >= k]
        ext = [sg for sg in tot if sg[3] > k]
        if tot:
            print(f'    k={k:2d}  存活 {len(tot):3d} 段  继续 {len(ext):3d} 段  S={len(ext)/len(tot)*100:5.1f}%')

    # ============================================================ C. 二维
    print('\n' + '=' * 108)
    print('【C】时间 × 幅度：联合状态是否比单一维度有信息（先生框架的关键检验）')
    print('-' * 108)
    # 对所有上涨段的第 k 周末，算：已持续 k 周 + 段内累计涨幅 g
    rows = []
    for sg in segs:
        sign, s0, s1, L = sg
        if sign <= 0:
            continue
        base = wc[s0 - 1] if s0 > 0 else wc[0]
        for k in range(1, L + 1):
            ek = s0 + k - 1
            if ek + 13 >= nw:
                continue
            g = wc[ek] / base - 1
            f13 = wc[ek + 13] / wc[ek] - 1
            dd = wc[ek:ek + 14].min() / wc[ek] - 1
            rows.append((k, g, f13, dd))
    R = pd.DataFrame(rows, columns=['k', 'g', 'f13', 'dd'])
    print(f'  观测点 {len(R)} 个（每个上涨段的每个周末）')
    tb = ['≤3周', '4-6周', '7-10周', '≥11周']
    gb = ['≤5%', '5~15%', '15~30%', '>30%']
    R['kt'] = pd.cut(R.k, [0, 3, 6, 10, 999], labels=tb)
    R['gtt'] = pd.cut(R.g, [-9, .05, .15, .30, 9], labels=gb)
    print('\n  格内填：后续 13 周收益中位 / 后续 13 周最大回撤中位（样本数）')
    print(f'  {"":<10}' + ''.join([f'{x:>26}' for x in gb]))
    for kt in tb:
        line = f'  {kt:<10}'
        for gt in gb:
            sub = R[(R.kt == kt) & (R.gtt == gt)]
            if len(sub) < 5:
                line += f'{"—":>26}'
            else:
                line += f'{sub.f13.median()*100:>+9.1f}% /{sub.dd.median()*100:>+8.1f}% ({len(sub):>3})'
        print(line)
    # 边际
    print('\n  只看时间（不分幅度）：')
    for kt in tb:
        sub = R[R.kt == kt]
        print(f'    {kt:<8} n={len(sub):4d}  次13周 {sub.f13.median()*100:+6.2f}%  回撤 {sub.dd.median()*100:+6.2f}%')
    print('  只看幅度（不分时间）：')
    for gt in gb:
        sub = R[R.gtt == gt]
        print(f'    {gt:<8} n={len(sub):4d}  次13周 {sub.f13.median()*100:+6.2f}%  回撤 {sub.dd.median()*100:+6.2f}%')

    # 段幅本身有没有记忆
    print('\n  段幅 vs 段长的关系（"涨得猛的是不是持续得短"）：')
    seginfo = []
    for sg in segs:
        sign, s0, s1, L = sg
        if sign <= 0:
            continue
        base = wc[s0 - 1] if s0 > 0 else wc[0]
        seginfo.append((L, wc[min(s1, nw - 1)] / base - 1))
    SI = pd.DataFrame(seginfo, columns=['L', 'G'])
    if len(SI) > 10:
        from scipy import stats as st
        rho, p = st.spearmanr(SI.L, SI.G)
        print(f'    n={len(SI)}  Spearman ρ(段长, 段幅) = {rho:+.3f}  p={p:.4f}')

    # ============================================================ D. 波动率记忆
    print('\n' + '=' * 108)
    print('【D】收益方向无记忆，那波动率呢？（"感冒严重程度"是否可预测）')
    print('-' * 108)
    r = np.diff(np.log(c))
    ar = np.abs(r)
    print('  |日收益| 的自相关（真实 vs 打乱顺序）：')
    rng = np.random.default_rng(11)
    sh = ar.copy()
    rng.shuffle(sh)
    print(f'  {"滞后":>4} {"真实":>9} {"打乱":>9} {"倍数":>7}')
    for lag in (1, 2, 5, 10, 20, 60):
        x = ar[:-lag]
        y = ar[lag:]
        ac = np.corrcoef(x, y)[0, 1]
        x2 = sh[:-lag]
        y2 = sh[lag:]
        ac2 = np.corrcoef(x2, y2)[0, 1]
        print(f'  {lag:>4} {ac:>+9.4f} {ac2:>+9.4f} {ac/abs(ac2) if ac2 else 0:>7.1f}x')

    # EWMA 最优衰减 → 半衰期
    best = None
    for lam in np.arange(0.80, 0.999, 0.005):
        v = np.empty(n - 1)
        s2 = ar[:250].var()
        for i in range(n - 1):
            v[i] = s2
            s2 = lam * s2 + (1 - lam) * r[i] ** 2
        # 用预测 |r| 与实际的秩相关衡量
        from scipy import stats as st
        rho, _ = st.spearmanr(v[:-1], ar[1:])
        if best is None or rho > best[1]:
            best = (lam, rho)
    lam, rho = best
    hl = -np.log(2) / np.log(lam) if lam < 1 else np.inf
    print(f'\n  最优 EWMA 衰减 λ = {lam:.3f}  → 半衰期 {hl:.0f} 个交易日（≈{hl/21:.1f} 个月）')
    print(f'  预测力：预测波动 vs 次日|收益| 的 Spearman ρ = {rho:+.4f}')

    # ============================================================ E. 波动状态 → 回撤
    print('\n' + '=' * 108)
    print('【E】波动率状态能不能预测"接下来的风险"（这才是可用的那条）')
    print('-' * 108)
    # RV20
    rv = pd.Series(ar).rolling(20).mean().values
    qs = np.nanpercentile(rv, [20, 40, 60, 80])
    lab = np.full(n - 1, -1)
    for i in range(1, n - 1):
        if np.isnan(rv[i - 1]):
            continue
        lab[i] = int(np.searchsorted(qs, rv[i - 1]))
    print(f'  {"RV20 分档":<12} {"n":>6} {"次20日收益中位":>15} {"次20日最大回撤中位":>19} {"次60日回撤中位":>17} {"次60日收益中位":>15}')
    for b in range(5):
        idx = [i for i in range(1, n - 1) if lab[i] == b and i + 60 < n]
        if len(idx) < 30:
            continue
        f20 = [c[i + 20] / c[i] - 1 for i in idx]
        d20 = [c[i:i + 21].min() / c[i] - 1 for i in idx]
        f60 = [c[i + 60] / c[i] - 1 for i in idx]
        d60 = [c[i:i + 61].min() / c[i] - 1 for i in idx]
        print(f'  Q{b+1} ({["最低","低","中","高","最高"][b]})  {len(idx):>6} '
              f'{np.median(f20)*100:>14.2f}% {np.median(d20)*100:>18.2f}% '
              f'{np.median(d60)*100:>16.2f}% {np.median(f60)*100:>14.2f}%')

    # 分段时间：波动率预测力是否三段都成立
    print('\n  分段检验（波动最高档 vs 最低档，次60日最大回撤中位之差）：')
    seg3 = [('1997-2010', '1997-01-01', '2010-12-31'),
            ('2011-2016', '2011-01-01', '2016-12-31'),
            ('2017-2026', '2017-01-01', '2026-12-31')]
    for nm, a, b in seg3:
        i0 = int(np.searchsorted(dts.values, np.datetime64(a)))
        i1 = int(np.searchsorted(dts.values, np.datetime64(b)))
        lo, hi = [], []
        for i in range(max(1, i0), min(i1, n - 61)):
            if lab[i] < 0:
                continue
            dd = c[i:i + 61].min() / c[i] - 1
            if lab[i] <= 0:
                lo.append(dd)
            elif lab[i] >= 4:
                hi.append(dd)
        if lo and hi:
            print(f'    {nm}  n低={len(lo):4d} n高={len(hi):4d}  '
                  f'低波动档 {np.median(lo)*100:+6.2f}%  高波动档 {np.median(hi)*100:+6.2f}%  '
                  f'差 {(np.median(hi)-np.median(lo))*100:+6.2f}pp')

    # ============================================================ 当前状态
    print('\n' + '=' * 108)
    print('【当前状态】')
    print('-' * 108)
    rv_now = rv[-1]
    pct = (rv < rv_now).mean() * 100
    print(f'  RV20 = {rv_now*100:.3f}%/日   历史分位 {pct:.0f}%')
    # 当前上涨段已持续多少周
    last = segs[-1]
    print(f'  最近一段：{"上涨" if last[0]>0 else "下跌"}，已持续 {last[3]} 周，'
          f'自 {wd[last[1]].date()} 起')


if __name__ == '__main__':
    main()
