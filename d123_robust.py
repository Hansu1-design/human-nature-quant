# -*- coding: utf-8 -*-
"""
d123: 稳健性 —— 参数曲面上那些差异是真信号还是 6 个样本的噪声？
三把刀：
  ① 留一事件（leave-one-out）：去掉任意一轮，最优参数会不会换人
  ② 锚扰动（1700 / 1880 / 2050）：底的位置不确定 ±13%，参数排序扛不扛得住
  ③ 组间离散 vs 抽样噪声：48 组的收益差，比不比"6 个样本估一个均值"的误差大
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats

import importlib.util
spec = importlib.util.spec_from_file_location('eng', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'd122_engine.py'))
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)

B = os.path.dirname(os.path.abspath(__file__))
COMBOS = [(n, cf, g) for n in [2, 3, 4, 5] for cf in ['none', 'ma20', 'reb8', 'mup'] for g in [1, 2, 3]]
OUT = {}


def run_all(d, ev):
    rows = []
    for n, cf, g in COMBOS:
        per = [eng.backtest(d, i0, n=n, confirm=cf, gap_m=g, max_hold=756) for i0 in ev]
        rows.append(dict(key=f'n{n}_{cf}_g{g}', n=n, cf=cf, g=g,
                         rets=[p['ret'] for p in per],
                         dds=[p['max_dd_acct'] for p in per],
                         prems=[p['cost_prem'] for p in per],
                         pos=[p['maxpos'] for p in per]))
    return rows


def main():
    d = eng.load()
    ev = eng.find_events(d)
    print(f'事件 {len(ev)} 轮：{[str(d.d.iloc[i].date()) for i in ev]}\n')

    base = run_all(d, ev)
    for r in base:
        r['mean'] = float(np.mean(r['rets']))
        r['std_dd'] = float(np.mean(r['dds']))
        r['calmar'] = r['mean'] / abs(r['std_dd']) if r['std_dd'] else np.nan

    print('=== ① 留一事件：每次去掉一轮，最优参数换不换人 ===')
    loo_best, loo_rank_corr = [], []
    for k in range(len(ev)):
        sub = [i for i in range(len(ev)) if i != k]
        m = [np.mean([r['rets'][j] for j in sub]) for r in base]
        best = int(np.argmax(m))
        loo_best.append(base[best]['key'])
        # 与全样本排名的 Spearman
        fullrank = stats.rankdata([-r['mean'] for r in base])
        subrank = stats.rankdata([-x for x in m])
        rho = stats.spearmanr(fullrank, subrank).statistic
        loo_rank_corr.append(float(rho))
        print(f'  去掉 {str(d.d.iloc[ev[k]].date())} → 最优 {base[best]["key"]:16s} 均值 {m[best]*100:5.2f}%  '
              f'与全样本排名相关 ρ={rho:.3f}')
    full_best = max(base, key=lambda r: r['mean'])['key']
    print(f'  全样本最优 {full_best}')
    print(f'  留一最优出现次数：{pd.Series(loo_best).value_counts().to_dict()}')
    print(f'  排名相关 ρ 中位 {np.median(loo_rank_corr):.3f}  最低 {min(loo_rank_corr):.3f}\n')
    OUT['loo'] = dict(best_list=loo_best, rhos=loo_rank_corr, full_best=full_best)

    print('=== ② 锚扰动：底 = 1700 / 1880 / 2050（±13~14%）===')
    anchor_res = {}
    for A in [1700.0, 1880.0, 2050.0]:
        dd = d.copy()
        dd['rA'] = dd['real'] / A
        ev2 = eng.find_events(dd)
        rows = run_all(dd, ev2)
        m = [np.mean(r['rets']) for r in rows]
        order = np.argsort(m)[::-1]
        top5 = [rows[i]['key'] for i in order[:5]]
        anchor_res[A] = dict(n_ev=len(ev2), top5=top5,
                             events=[str(dd.d.iloc[i].date()) for i in ev2],
                             means={rows[i]['key']: round(m[i] * 100, 2) for i in order[:5]})
        print(f'  锚 {A:.0f}：{len(ev2)} 轮事件  top5 {top5}')
    keys = [r['key'] for r in base]
    ranks = {}
    for A in [1700.0, 1880.0, 2050.0]:
        ranks[A] = {k: anchor_res[A]['means'].get(k, np.nan) for k in keys}
    print()
    OUT['anchor'] = anchor_res

    # 三个锚下都进 top10 的参数
    print('  三个锚下的 top10 交集：')
    top10s = []
    for A in [1700.0, 1880.0, 2050.0]:
        top10s.append(set(anchor_res[A]['top5']))
    print(f'    1700∩1880∩2050 = {set.intersection(*top10s)}')
    print()

    print('=== ③ 组间离散 vs 抽样噪声 ===')
    means = np.array([r['mean'] for r in base])
    single = np.array([x for r in base for x in r['rets']])
    print(f'  48 组均值：范围 {means.min()*100:.2f}% ~ {means.max()*100:.2f}%，标准差 {means.std()*100:.2f}pp')
    print(f'  单次收益：标准差 {single.std()*100:.2f}pp（n={len(single)}）')
    se = single.std() / np.sqrt(len(ev))
    print(f'  6 个样本估一个均值的标准误：{se*100:.2f}pp')
    print(f'  组间极差 / 标准误 = {(means.max()-means.min())/se:.2f}')
    # 组间差异里有多少是"真"的：单因素 ANOVA
    groups = [np.array(r['rets']) for r in base]
    F, p = stats.f_oneway(*groups)
    print(f'  单因素 ANOVA（48 组 × 6 事件）：F={F:.3f}  p={p:.4f}')
    # Kendall W：6 个事件对 48 组的排序一致吗
    mat = np.array([r['rets'] for r in base])       # 48 × 6
    W, chi2, pW = stats.kendalltau(mat[:, 0], mat[:, 1])[:1] + (None, None) if False else (None, None, None)
    rk = np.array([stats.rankdata(-col) for col in mat.T])   # 6 × 48
    k, nn = rk.shape
    Rsum = rk.sum(axis=0)
    W = 12 * ((Rsum - Rsum.mean()) ** 2).sum() / (k ** 2 * (nn ** 3 - nn))
    chi2 = k * (nn - 1) * W
    pW = 1 - stats.chi2.cdf(chi2, nn - 1)
    print(f'  Kendall W（6 轮事件对 48 组的排序一致性）= {W:.4f}  χ²={chi2:.1f}  p={pW:.4f}')
    print(f'    W 解读：0=完全不一致，1=完全一致。{W:.3f} 说明 6 轮事件几乎各自排序不同\n')
    OUT['noise'] = dict(grid_range=[float(means.min()), float(means.max())],
                        grid_sd=float(means.std()), single_sd=float(single.std()),
                        se=float(se), ratio=float((means.max() - means.min()) / se),
                        F=float(F), p=float(p), W=float(W), pW=float(pW))

    print('=== ④ 每轮事件各自的最优参数（看是否指向同一处）===')
    per_event_best = []
    for j in range(len(ev)):
        vals = [r['rets'][j] for r in base]
        bi = int(np.argmax(vals))
        per_event_best.append(base[bi]['key'])
        top3 = sorted(range(len(vals)), key=lambda x: -vals[x])[:3]
        print(f'  {d.d.iloc[ev[j]].date()}  {d.dd_peak.iloc[ev[j]]*100:+5.1f}%  '
              f'最优 {base[bi]["key"]:16s} {vals[bi]*100:5.1f}%  '
              f'最优3 {[base[t]["key"] for t in top3]}  '
              f'最差 {min(vals)*100:5.1f}%  极差 {(max(vals)-min(vals))*100:4.1f}pp')
    print()
    OUT['per_event_best'] = per_event_best

    print('=== ⑤ 档数边际：固定 confirm=reb8, gap=2，n 从 1 到 6 ===')
    marg = []
    for n in [1, 2, 3, 4, 5, 6]:
        per = [eng.backtest(d, i0, n=n, confirm='none' if n == 1 else 'reb8', gap_m=2, max_hold=756) for i0 in ev]
        m = np.mean([p['ret'] for p in per])
        dd = np.mean([p['max_dd_acct'] for p in per])
        pos = np.mean([p['maxpos'] for p in per])
        marg.append(dict(n=n, ret=float(m), dd=float(dd), pos=float(pos)))
        print(f'  n={n}: 收益 {m*100:5.2f}%  平均浮亏 {dd*100:6.2f}%  平均仓位 {pos*100:5.1f}%  收益/浮亏 {m/abs(dd):5.2f}')
    OUT['marginal'] = marg

    print('\n=== ⑥ 对照：v1.0 匀速加仓 vs v1.1 确认加仓 ===')
    def uniform(i0, drop=0.03, legs=5):
        lot = 1.0 / legs
        cash, sh, inv, last = 1.0, 0.0, 0.0, None
        N = len(d)
        regime = 'burst' if d['dd_peak'].iloc[i0] <= -0.50 else 'box'
        touched, pk, peak_gain = False, 0.0, -9.9
        exit_i, why = N - 1, '未出场'
        for i in range(i0, min(N, i0 + 1800)):
            px = d['close'].iloc[i]
            if inv < 1 - 1e-9:
                go = last is None or d['rA'].iloc[i] <= d['rA'].iloc[last] * (1 - drop)
                if go:
                    amt = min(lot, cash); sh += amt / px; cash -= amt; inv += amt; last = i
            if inv > 0 and exit_i == N - 1:
                pk = max(pk, px); peak_gain = max(peak_gain, px / (inv / sh) - 1)
                ma120 = d['ma120'].iloc[i]
                if not np.isnan(ma120) and px > ma120: touched = True
                tout = touched and not np.isnan(ma120) and px < ma120 and bool(d['ma120_down'].iloc[i])
                dout = peak_gain > 0.15 and px / pk - 1 <= -0.20
                g = px / (inv / sh) - 1
                months = (d['d'].iloc[i].year - d['d'].iloc[i0].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[i0].month)
                if regime == 'burst' and (tout or dout):
                    cash += sh * px; sh = 0; exit_i = i
                elif regime == 'box' and (g >= .5 or (months >= 20 and tout) or dout):
                    cash += sh * px; sh = 0; exit_i = i
            cash *= (1 + eng.DC)
        return cash + sh * d['close'].iloc[exit_i] - 1, inv
    for name, kw in [('每跌3% 共5笔', dict(drop=0.03, legs=5)),
                     ('每跌5% 共4笔', dict(drop=0.05, legs=4)),
                     ('每跌8% 共3笔', dict(drop=0.08, legs=3))]:
        rs = [uniform(i0, **kw) for i0 in ev]
        m = np.mean([x[0] for x in rs]); pos = np.mean([x[1] for x in rs])
        print(f'  {name}: 收益 {m*100:5.2f}%  平均仓位 {pos*100:5.1f}%  各次 {[round(x[0]*100,1) for x in rs]}')
        OUT.setdefault('uniform', {})[name] = dict(ret=float(m), pos=float(pos),
                                                   rets=[round(x[0] * 100, 1) for x in rs])

    # 最优 reb8 组对照
    for n, cf, g in [(3, 'reb8', 2), (3, 'ma20', 2), (4, 'none', 1)]:
        per = [eng.backtest(d, i0, n=n, confirm=cf, gap_m=g, max_hold=756) for i0 in ev]
        m = np.mean([p['ret'] for p in per]); dd = np.mean([p['max_dd_acct'] for p in per])
        print(f'  n={n} {cf} g={g}: 收益 {m*100:5.2f}%  浮亏 {dd*100:6.2f}%  各次 {[round(p["ret"]*100,1) for p in per]}')

    json.dump(OUT, open(os.path.join(B, 'd123_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\nsaved d123_result.json')


if __name__ == '__main__':
    main()
