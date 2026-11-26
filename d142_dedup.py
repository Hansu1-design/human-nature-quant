# -*- coding: utf-8 -*-
"""
d142 严谨化：
1) 事件去重（180 日冷却 + 中途须反弹 >=10%），避免同一轮熊市重复计数
2) 离底分层的明细与 bootstrap 置信区间
3) 增量测试：在「离底 10~30%」这个骨架不买的区间，跌 15% 就买 vs 一直空仓
4) 「前一年暴涨」否决权的独立检验
"""
import numpy as np
import pandas as pd
from scipy import stats
from d141_eat_bite import load

pd.set_option('display.width', 250)
RF = 0.025
RNG = np.random.default_rng(11)


def events_dedup(d, thr=0.15, win=250, cool=180, reb=0.10):
    """去重：冷却 >=cool 日，且两次之间须出现过 >=reb 的反弹（否则算同一次回调）"""
    c = d['close'].values
    n = len(c)
    ev = []
    last_i = -999
    last_low = None
    for i in range(win, n):
        hi = c[i - win:i + 1].max()
        if c[i] / hi - 1 > -thr:
            continue
        if i - last_i <= cool:
            # 冷却期内：只有反弹超过 reb（从上次低点算）才算新事件
            if last_low is not None and c[last_i:i].max() / c[last_i] - 1 >= reb and i - last_i > 30:
                pass
            else:
                continue
        ev.append(i)
        last_i = i
        last_low = c[i]
    return ev


def fwd(c, i, h):
    j = min(len(c) - 1, i + h)
    return c[j] / c[i] - 1


def boot_ci(x, n=5000):
    m = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    ra = d['rA'].values
    pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])

    print('=' * 120)
    print('【0】去重对比')
    print('=' * 120)
    for cool in [60, 120, 180, 250]:
        ev = events_dedup(d, 0.15, cool=cool)
        r = np.array([fwd(c, i, 250) for i in ev])
        print(f'冷却 {cool:3d} 日 → {len(ev):3d} 次 ({len(ev)/yrs:.2f}/年)  '
              f'250日均值 {r.mean()*100:+6.1f}%  中位 {np.median(r)*100:+6.1f}%  '
              f'胜率 {(r>0).mean()*100:.0f}%')
    ev = events_dedup(d, 0.15, cool=180)
    print(f'\n>> 采用 180 日冷却：{len(ev)} 次')

    print()
    print('=' * 120)
    print('【1】去重后事件明细（含离底距离、前一年涨幅）')
    print('=' * 120)
    print(f'{"#":>3} {"触发日":>11} {"点位":>7} {"离底":>7} {"前一年":>8} '
          f'{"250日":>8} {"500日":>8} {"期间最低":>8}')
    for k, i in enumerate(ev):
        seg = c[i:min(n, i + 500)]
        print(f'{k:3d} {str(d.d.iloc[i].date()):>11} {c[i]:7.0f} '
              f'{(ra[i]-1)*100 if not np.isnan(ra[i]) else float("nan"):+6.1f}% '
              f'{pre[i]*100:+7.1f}% {fwd(c,i,250)*100:+7.1f}% {fwd(c,i,500)*100:+7.1f}% '
              f'{seg.min()/c[i]*100-100:+7.1f}%')

    # ---------- 2) 离底分层（只有 2008 后有 rA） ----------
    print()
    print('=' * 120)
    print('【2】离底分层（去重后，仅 2008-01 起有 CPI 的样本），持有 250 日')
    print('=' * 120)
    subs = [(0, 1.10, '离底 ≤10%'), (1.10, 1.30, '10~30%'),
            (1.30, 1.60, '30~60%'), (1.60, 99, '>60%')]
    tot = []
    for lo, hi, lab in subs:
        idx = [i for i in ev if not np.isnan(ra[i]) and lo < ra[i] <= hi]
        if not idx:
            print(f'{lab:<10} n=0')
            continue
        r = np.array([fwd(c, i, 250) for i in idx])
        lo_, hi_ = boot_ci(r)
        t, p = stats.ttest_1samp(r, 0) if len(r) > 2 else (np.nan, np.nan)
        flag = '**' if (len(r) > 2 and p < 0.05) else ''
        print(f'{lab:<10} n={len(idx):2d}  均值 {r.mean()*100:+6.1f}% '
              f'CI[{lo_*100:+6.1f},{hi_*100:+6.1f}]  中位 {np.median(r)*100:+6.1f}%  '
              f'胜率 {(r>0).mean()*100:3.0f}%  t={t:5.2f} p={p:.3f} {flag}')
        tot.append((lab, r))
    if len(tot) >= 2:
        f, p = stats.f_oneway(*[x[1] for x in tot])
        print(f'\n四组 ANOVA: F={f:.2f}  p={p:.4f}  → '
              f'{"组间差异显著" if p < 0.05 else "组间差异不显著"}')
    # 二分：≤10% vs >10%
    a = [i for i in ev if not np.isnan(ra[i]) and ra[i] <= 1.10]
    b = [i for i in ev if not np.isnan(ra[i]) and ra[i] > 1.10]
    if a and b:
        ra_, rb_ = np.array([fwd(c, i, 250) for i in a]), np.array([fwd(c, i, 250) for i in b])
        t, p = stats.ttest_ind(ra_, rb_, equal_var=False)
        print(f'≤10% (n={len(a)}) vs >10% (n={len(b)}): '
              f'{ra_.mean()*100:+.1f}% vs {rb_.mean()*100:+.1f}%, '
              f'差 {(ra_.mean()-rb_.mean())*100:+.1f}pp, t={t:.2f}, p={p:.4f}')

    # ---------- 3) 增量测试：离底 10~30% 区间，买 vs 空仓 ----------
    print()
    print('=' * 120)
    print('【3】增量测试：骨架「不买」的区间（离底 10~30%），跌 15% 买入是否强于空仓生息')
    print('=' * 120)
    idx = [i for i in ev if not np.isnan(ra[i]) and 1.10 < ra[i] <= 1.30]
    print(f'样本 {len(idx)} 次：', [str(d.d.iloc[i].date()) for i in idx])
    for h in [60, 120, 250]:
        r = np.array([fwd(c, i, h) for i in idx])
        rf = RF * h / 243
        t, p = stats.ttest_1samp(r, rf)
        lo_, hi_ = boot_ci(r)
        print(f'  持有 {h:3d} 日：均值 {r.mean()*100:+6.2f}%  中位 {np.median(r)*100:+6.2f}%  '
              f'胜率 {(r>0).mean()*100:3.0f}%  平均浮亏 '
              f'{np.mean([min(c[i:min(n,i+h)])/c[i]-1 for i in idx])*100:+6.2f}%  '
              f'vs 生息 {rf*100:.2f}%  t={t:5.2f} p={p:.3f}')

    # ---------- 4) 前一年暴涨的否决权 ----------
    print()
    print('=' * 120)
    print('【4】否决权检验：触发时「前 250 日涨幅」是否独立地带信息（全样本 60 次）')
    print('=' * 120)
    ev60 = events_dedup(d, 0.15, cool=60)
    print(f'{"前一年涨幅":<16} {"n":>4} {"250日均值":>10} {"中位":>8} {"胜率":>6} {"期间最低均值":>12}')
    for lo, hi, lab in [(-9, 0.0, '跌'), (0.0, 0.20, '涨 0~20%'),
                        (0.20, 0.40, '涨 20~40%'), (0.40, 99, '涨 >40%')]:
        ii = [i for i in ev60 if lo < pre[i] <= hi]
        if not ii:
            continue
        r = np.array([fwd(c, i, 250) for i in ii])
        lowm = np.mean([min(c[i:min(n, i + 250)]) / c[i] - 1 for i in ii])
        print(f'{lab:<16} {len(ii):4d} {r.mean()*100:+9.1f}% {np.median(r)*100:+7.1f}% '
              f'{(r>0).mean()*100:5.0f}% {lowm*100:+11.1f}%')
    # 相关系数
    ok = [i for i in ev60 if not np.isnan(ra[i])]
    if len(ok) > 5:
        rr = np.array([fwd(c, i, 250) for i in ok])
        for nm, v in [('离底 rA', np.array([ra[i] for i in ok])),
                      ('前一年涨幅', np.array([pre[i] for i in ok]))]:
            rho, p = stats.spearmanr(v, rr)
            print(f'\n{nm} vs 之后250日收益：Spearman ρ={rho:+.3f}  p={p:.4f}  n={len(ok)}')


if __name__ == '__main__':
    main()
