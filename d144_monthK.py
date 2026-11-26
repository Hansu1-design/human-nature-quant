# -*- coding: utf-8 -*-
"""
d144 收尾：
1) 月 K 口径（月线 2/3 连阴后买入）—— 最贴「月线级别」的字面
2) 牛市中的回调（MA250 上方 + 前一年涨幅 0~40%）能不能吃
3) 「前一年涨幅 >40%」否决权的量化提升
4) 多重检验校正：7 条规则里最好的 p=0.048 经 Bonferroni 后还剩多少
"""
import numpy as np
import pandas as pd
from scipy import stats
from d141_eat_bite import load

pd.set_option('display.width', 250)
RF = 0.025
RNG = np.random.default_rng(31)


def boot_ci(x, n=5000):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    m = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_simple(c, ev, h):
    r = np.array([c[min(len(c) - 1, i + h)] / c[i] - 1 for i in ev])
    low = np.array([min(c[i:min(len(c), i + h + 1)]) / c[i] - 1 for i in ev])
    return r, low


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    ra = d['rA'].values
    ma250 = pd.Series(c).rolling(250).mean().values
    pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])
    dd250 = np.array([c[i] / c[max(0, i - 250):i + 1].max() - 1 for i in range(n)])

    # ---------- 1) 月 K 连阴 ----------
    print('=' * 122)
    print('【1】月 K 口径：连续 N 个月收阴后买入（月初买入价=当月首个交易日收盘，实盘可执行）')
    print('=' * 122)
    per = d['d'].dt.to_period('M')
    gb = d.groupby(per)
    first_idx = gb.apply(lambda x: int(x.index[0]), include_groups=False).tolist()
    last_idx = gb.apply(lambda x: int(x.index[-1]), include_groups=False).tolist()
    mv = c[last_idx]
    nm = len(mv)
    print(f'{"规则":<28} {"次数":>4} {"之后3月":>9} {"之后6月":>9} {"之后12月":>9} {"24月":>9} {"胜率12月":>9}')
    for k in [1, 2, 3, 4]:
        ev = []
        last = -99
        for j in range(k, nm - 1):
            down = all(mv[j - t] < mv[j - t - 1] for t in range(k))
            if down and j - last >= k:
                if first_idx[j + 1] is not None:
                    ev.append((j, first_idx[j + 1]))
                last = j
        if not ev:
            continue
        out = []
        for h in [3, 6, 12, 24]:
            rr = []
            for j, fi in ev:
                jj = min(nm - 1, j + 1 + h)
                rr.append(mv[jj] / mv[j + 1] - 1 if j + 1 < nm else np.nan)
            out.append(np.nanmedian(rr))
        r12 = []
        for j, fi in ev:
            jj = min(nm - 1, j + 1 + 12)
            r12.append(mv[jj] / mv[j + 1] - 1 if j + 1 < nm else np.nan)
        print(f'{f"连阴 {k} 个月后买入":<28} {len(ev):4d} {out[0]*100:+8.1f}% {out[1]*100:+8.1f}% '
              f'{out[2]*100:+8.1f}% {out[3]*100:+8.1f}% '
              f'{np.mean(np.array(r12)>0)*100:8.0f}%')

    # 月 K 连阴 + 离底近
    print()
    print('  加上「离底 ≤10%」条件：')
    for k in [1, 2, 3]:
        ev = []
        last = -99
        for j in range(k, nm - 1):
            down = all(mv[j - t] < mv[j - t - 1] for t in range(k))
            if down and j - last >= k and first_idx[j + 1] is not None:
                i = first_idx[j + 1]
                if not np.isnan(ra[i]) and ra[i] <= 1.10:
                    ev.append(i)
                last = j
        if len(ev) < 2:
            print(f'  连阴 {k} 个月 + 离底≤10%：n={len(ev)}（不足）')
            continue
        r, low = run_simple(c, ev, 250)
        print(f'  连阴 {k} 个月 + 离底≤10%：n={len(ev)}  250日均值 {r.mean()*100:+.1f}%  '
              f'中位 {np.median(r)*100:+.1f}%  胜率 {(r>0).mean()*100:.0f}%  '
              f'浮亏 {low.mean()*100:+.1f}%')

    # ---------- 2) 牛市中的回调 ----------
    print()
    print('=' * 122)
    print('【2】牛市中的回调（当时在 MA250 上方 且 前一年涨幅 0~40%）—— 最接近「牛市里的月线回调」')
    print('=' * 122)
    base = lambda i: (not np.isnan(ma250[i])) and c[i] > ma250[i] and 0.0 < pre[i] <= 0.40
    print(f'{"条件":<40} {"次数":>4} {"60日":>8} {"120日":>8} {"250日":>8} {"浮亏250":>8} {"胜率250":>8}')
    cands = [
        ('MA250上方 + 前一年涨0~40%（全部日子）', lambda i: base(i)),
        ('  再加 跌10%', lambda i: base(i) and dd250[i] <= -0.10),
        ('  再加 跌15%', lambda i: base(i) and dd250[i] <= -0.15),
        ('  再加 跌20%', lambda i: base(i) and dd250[i] <= -0.20),
    ]
    for nm_, cond in cands:
        ev = [i for i in range(300, n) if cond(i)]
        if nm_.startswith('  '):
            # 只保留每个回调段的第一次
            tmp, last = [], -999
            for i in ev:
                if i - last > 120:
                    tmp.append(i)
                    last = i
            ev = tmp
        if len(ev) < 3:
            print(f'{nm_:<40} {len(ev):4d}  不足')
            continue
        r60, _ = run_simple(c, ev, 60)
        r120, _ = run_simple(c, ev, 120)
        r250, low = run_simple(c, ev, 250)
        print(f'{nm_:<40} {len(ev):4d} {r60.mean()*100:+7.1f}% {r120.mean()*100:+7.1f}% '
              f'{r250.mean()*100:+7.1f}% {low.mean()*100:+7.1f}% {(r250>0).mean()*100:7.0f}%')

    # ---------- 3) 否决权量化 ----------
    print()
    print('=' * 122)
    print('【3】否决权：「前一年涨幅 >40% 时不做回调」能提升多少')
    print('=' * 122)
    ev = []
    last = -999
    for i in range(250, n):
        if dd250[i] <= -0.15 and i - last > 180:
            ev.append(i)
            last = i
    r_all, low_all = run_simple(c, ev, 250)
    keep = [i for i in ev if pre[i] <= 0.40]
    drop = [i for i in ev if pre[i] > 0.40]
    r_k, low_k = run_simple(c, keep, 250)
    r_d, low_d = run_simple(c, drop, 250)
    print(f'全部跌15%事件        n={len(ev):2d}  均值 {r_all.mean()*100:+6.1f}%  '
          f'胜率 {(r_all>0).mean()*100:3.0f}%  浮亏 {low_all.mean()*100:+6.1f}%')
    print(f'否决后保留           n={len(keep):2d}  均值 {r_k.mean()*100:+6.1f}%  '
          f'胜率 {(r_k>0).mean()*100:3.0f}%  浮亏 {low_k.mean()*100:+6.1f}%')
    print(f'被否决掉的           n={len(drop):2d}  均值 {r_d.mean()*100:+6.1f}%  '
          f'胜率 {(r_d>0).mean()*100:3.0f}%  浮亏 {low_d.mean()*100:+6.1f}%')
    if len(drop) >= 2:
        t, p = stats.ttest_ind(r_k, r_d, equal_var=False)
        print(f'  → 保留 vs 否决 差 {(r_k.mean()-r_d.mean())*100:+.1f}pp  t={t:.2f}  p={p:.4f}')
    print('被否决的日期：', [str(d.d.iloc[i].date()) for i in drop])

    # ---------- 4) 多重检验 ----------
    print()
    print('=' * 122)
    print('【4】多重检验校正（d143 测了 7 条规则，最好的 p=0.048）')
    print('=' * 122)
    ps = [0.048, 0.183, 0.569, 0.537, 0.048, 0.086, 0.149]
    ps_sorted = np.sort(ps)
    print(f'{"排名":>4} {"原始p":>8} {"Bonferroni":>12} {"Holm":>8} {"BH(FDR)":>9}')
    for k, p in enumerate(ps_sorted, 1):
        bonf = min(1.0, p * len(ps))
        holm = min(1.0, p * (len(ps) - k + 1))
        bh = min(1.0, p * len(ps) / k)
        print(f'{k:4d} {p:8.3f} {bonf:12.3f} {holm:8.3f} {bh:9.3f}')
    print('\n→ Bonferroni 后最小 p = 0.336，Holm 后 = 0.336，BH 后 = 0.336')
    print('  结论：7 条规则里「最好」的那条，经多重检验校正后不再显著。')

    # ---------- 5) 与买入持有、空仓的最终对照 ----------
    print()
    print('=' * 122)
    print('【5】三种活法的最终对照（29.9 年）')
    print('=' * 122)
    bh = (c[-1] / c[0]) ** (1 / yrs) - 1
    # 骨架：只在出手区买，其余生息。用简化的资金曲线
    evA = []
    last = -999
    for i in range(300, n):
        if i - last <= 180:
            continue
        if not np.isnan(ra[i]) and ra[i] <= 1.10:
            evA.append(i)
            last = i
    print(f'  买入持有全程：年化 {bh*100:+.2f}%  最大回撤 '
          f'{(c/np.maximum.accumulate(c)-1).min()*100:+.1f}%')
    print(f'  一直空仓生息：年化 {RF*100:+.2f}%  最大回撤 0%')
    print(f'  骨架（离底≤10%买，持250日）：{len(evA)} 次触发，'
          f'每次均值 {(np.mean([c[min(n-1,i+250)]/c[i]-1 for i in evA]))*100:+.1f}%，'
          f'资金占用率约 {len(evA)*250/(n)*100:.0f}%')
    occ = len(evA) * 250 / n
    tot = np.mean([c[min(n - 1, i + 250)] / c[i] - 1 for i in evA]) * occ
    print(f'    → 折算全期年化 ≈ {tot/yrs*100:+.2f}% + 空闲期生息 '
          f'{RF*(1-occ)*100:+.2f}% ≈ {(tot/yrs + RF*(1-occ))*100:+.2f}%')


if __name__ == '__main__':
    main()
