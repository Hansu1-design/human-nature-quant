# -*- coding: utf-8 -*-
"""d190 R54 ① 缺仓画像 / 汇总口径 / ③′ 决定性对照

★ 口径全部从 d189_predeclare 读，不在本脚本另设。
★ 模块级只做静默计算（ARMS / CID），所有打印包进 main()，
  避免 R53 d186 import d185 那种「顶层执行污染输出」。
★ 比较一律 EWR = Σ(w·r)/Σw，与 d187/d188 同源、同一条上证 close。
"""
import json
import numpy as np
import pandas as pd
import d183_common as K
import d177_common as P
import d189_predeclare as D

C, n = K.C, K.n
WIN = D.WIN

# ── 与 d187 完全同源的 PB 连续仓位 ───────────────────────────────
_w = pd.Series(P.target_from(P.pb.values, D.TH_HI, D.TH_LO, 'linear'), index=P.idx)
W = _w.reindex(K.d['d'], method='ffill').fillna(0).values
PBQ = P.PBpct_all.reindex(K.d['d'], method='ffill').values
DATES = K.d['d']

ARM_NAMES = ('batch', 'never', 'pb', 'pb_bin')
LABEL = {'batch': '①分批', 'never': '③永不出场', 'pb': '④PB闸门',
         'pb_bin': "③′二值化"}


def one_arm(kind, a):
    """单笔窗口的暴露路径与 EWR 分子/分母（pooled 汇总需要 num/den）。"""
    b = min(a + WIN, n - 1)
    L = b - a + 1
    r = np.zeros(L)
    r[1:] = C[a + 1:b + 1] / C[a:b] - 1
    if kind == 'batch':
        w = np.ones(L); k = 3
        for j, t in enumerate((300, 375, 450)):
            w[min(t, L):] = (k - j - 1) / k
    elif kind == 'never':
        w = np.ones(L)
    elif kind == 'pb':
        w = W[a:b + 1].copy()
    elif kind == 'pb_bin':
        # 【7】③′：与 ④ 完全相同的在场日集合，只把仓位幅度抹平成 0/1
        w = (W[a:b + 1] > D.W_BIN_EPS).astype(float)
    else:
        raise ValueError(kind)
    den = w[:-1].sum()
    num = (w[:-1] * r[1:]).sum()
    return dict(num=float(num), den=float(den),
                ewr=float(num / den) if den > 0 else np.nan,
                w=w, L=L, a=a, b=b)


ARMS = {k: [one_arm(k, a) for a in K.SIG] for k in ARM_NAMES}
CID = np.array(D.cluster_ids(K.SIG))
CLUSTERS = {c: [i for i, x in enumerate(CID) if x == c] for c in range(D.N_INDEP)}


def pooled(keys, kind):
    num = sum(ARMS[kind][i]['num'] for i in keys)
    den = sum(ARMS[kind][i]['den'] for i in keys)
    return float(num / den) if den > 0 else np.nan


def all_keys():
    return list(range(len(K.SIG)))


def median_ewr(kind):
    return float(np.median([ARMS[kind][i]['ewr'] for i in all_keys()]))


def main():
    print('=' * 116)
    print('§0 事前声明核验（d189_predeclare v2）')
    print('=' * 116)
    print('  sha256[:16] = %s  |  n_trades=%d  n_indep=%d  m_primary=%d  m_full=%d  boot=%d'
          % (D.predeclare_stamp(), D.N_TRADES, D.N_INDEP,
             D.M_PRIMARY, D.M_FAMILY_FULL, D.N_BOOT))
    print('  轮次归属 cid = %s（500 日冷却归并，d188 §6 既有口径）' % CID.tolist())
    print('  各轮笔数 = %s' % [len(CLUSTERS[c]) for c in range(D.N_INDEP)])

    print()
    print('=' * 116)
    print('§1 11 笔逐笔画像（Σw 与 EWR，④ 为 PB 闸门）')
    print('=' * 116)
    print('  %-12s %3s %7s %10s %10s %10s %10s %10s' %
          ('买入日', '轮', 'Σw④', 'EWR①', 'EWR③', 'EWR④', 'EWR③′', '④−①'))
    rows = []
    for i, a in enumerate(K.SIG):
        e = {k: ARMS[k][i]['ewr'] * 100 for k in ARM_NAMES}
        d41 = e['pb'] - e['batch']
        print('  %-12s %3d %7.0f %9.4f%% %9.4f%% %9.4f%% %9.4f%% %+9.4f%%'
              % (DATES.iloc[a].date(), CID[i], ARMS['pb'][i]['den'],
                 e['batch'], e['never'], e['pb'], e['pb_bin'], d41))
        rows.append(dict(date=str(DATES.iloc[a].date()), cid=int(CID[i]),
                         sw=ARMS['pb'][i]['den'], **{k: e[k] for k in ARM_NAMES},
                         d41=d41))

    print()
    print('=' * 116)
    print('§2 【6】功能 vs bug：Σw 最小那笔的 PB 分位轨迹（事前判据）')
    print('=' * 116)
    j = int(np.argmin([ARMS['pb'][i]['den'] for i in all_keys()]))
    a, b = ARMS['pb'][j]['a'], ARMS['pb'][j]['b']
    q = PBQ[a:b + 1]
    w = ARMS['pb'][j]['w']
    q = q[~np.isnan(q)]
    frac_hi = float((q > D.TH_HI).mean()) if len(q) else np.nan
    frac_band = float(((q >= D.TH_LO) & (q <= D.TH_HI)).mean()) if len(q) else np.nan
    flat = float((w <= D.W_FLAT_EPS).mean())
    changes = int((np.diff(w) != 0).sum())
    ondays = int((w > D.W_BIN_EPS).sum())
    print('  对象：第 %d 笔 %s（第 %d 轮），Σw④=%.0f / 窗口 %d 日'
          % (j + 1, DATES.iloc[a].date(), CID[j], ARMS['pb'][j]['den'], w.size))
    print('  PB 分位：min %.3f  中位 %.3f  max %.3f  有效日 %d'
          % (np.nanmin(q), np.nanmedian(q), np.nanmax(q), len(q)))
    print('  >th_hi(0.35) 占比 %.1f%%  |  在 [0.05,0.35] 带内占比 %.1f%%'
          % (frac_hi * 100, frac_band * 100))
    print('  w ≤0.05 占比 %.1f%%  |  w >0.01 在场 %d 日  |  w 相邻变化 %d 次'
          % (flat * 100, ondays, changes))
    print('  EWR④ %.4f%%  EWR① %.4f%%  EWR③ %.4f%%'
          % (ARMS['pb'][j]['ewr'] * 100, ARMS['batch'][j]['ewr'] * 100,
             ARMS['never'][j]['ewr'] * 100))
    if frac_hi >= .9 and flat >= .9:
        verdict = '(a)【功能】一直很贵所以一直没进场 → 保留、不剔除成立'
    elif changes >= 10 and frac_band >= .3:
        verdict = '(b)【实现抖动/bug】分位在带内反复穿越 → 须修后重跑，不得宣称不剔除成立'
    else:
        verdict = '(c)【无结论】两条事前条件都不满足，不得据此宣称 G-A 成立'
    print('  判据输出：%s' % verdict)
    print('  ⚠ 上面的 0.9/10/0.3 只是把【6】定性条件落成的展示阈值；')
    print('    判定以声明【6】文字条件为准，原始事实已全报，不靠单一阈值翻案。')

    print()
    print('=' * 116)
    print('§3 【5】三种汇总口径对比（等权中位 / pooled / Σw 加权）')
    print('=' * 116)
    print('  %-14s %14s %14s' % ('臂', '等权中位(旧)', 'pooled EWR(新)'))
    pooled_all = {}
    for k in ARM_NAMES:
        pooled_all[k] = pooled(all_keys(), k)
        print('  %-14s %13.4f%% %13.4f%%'
              % (LABEL[k], median_ewr(k) * 100, pooled_all[k] * 100))
    d_med = median_ewr('pb') - median_ewr('batch')
    d_pool = pooled_all['pb'] - pooled_all['batch']
    print('  ④−①：等权中位差 %+.4f%%  |  pooled 差 %+.4f%%  |  相差 %.1f 倍'
          % (d_med * 100, d_pool * 100, abs(d_pool / d_med) if d_med else np.nan))
    print('  ⇒ 两种汇总结论是否同号：%s'
          % ('同号' if d_med * d_pool > 0 else '【符号翻转】'))
    print('  ⚠ pooled 每笔按其 Σw 贡献，Σw=7 那笔自动只占 7/Σw 的权重；')
    print('    等权中位让它与 Σw≈500 的笔各占 1/11 —— 这才是「缺仓」真正的害处。')

    print()
    print('=' * 116)
    print('§4 【5】Σw 分层（按窗口比例固定切 125/250/375/500，非等频）')
    print('=' * 116)
    edges = [0] + [int(x * WIN) for x in D.SIGMA_BINS] + [WIN + 1]
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [i for i in all_keys() if lo <= ARMS['pb'][i]['den'] < hi]
        if not sel:
            strata.append(dict(lo=lo, hi=hi, n=0, e4=np.nan, e1=np.nan, d=np.nan))
            continue
        e4, e1 = pooled(sel, 'pb'), pooled(sel, 'batch')
        strata.append(dict(lo=lo, hi=hi, n=len(sel), e4=e4, e1=e1, d=e4 - e1))
    print('  %-14s %4s %12s %12s %12s' % ('Σw 层', '笔数', 'pooled④', 'pooled①', '④−①'))
    for s in strata:
        print('  %-14s %4d %11.4f%% %11.4f%% %+11.4f%%'
              % ('%d~%d' % (s['lo'], s['hi'] - 1 if s['hi'] <= WIN else WIN),
                 s['n'], s['e4'] * 100 if s['n'] else np.nan,
                 s['e1'] * 100 if s['n'] else np.nan,
                 s['d'] * 100 if s['n'] else np.nan))
    print('  ⚠ 层内笔数少（≤4）只作方向描述，不做显著性判决（M_DIAG=0）。')

    print()
    print('=' * 116)
    print('§5 【7】决定性对照 ③′：分离「何时在场」与「减仓幅度」')
    print('=' * 116)
    d43 = pooled_all['pb'] - pooled_all['pb_bin']
    per = [(ARMS['pb'][i]['ewr'] - ARMS['pb_bin'][i]['ewr']) * 100 for i in all_keys()]
    print('  ③′ = ④ 的 w 二值化（w>%.2f→1，否则 0）；两臂在场日集合完全相同。'
          % D.W_BIN_EPS)
    print('  pooled EWR：④ %.4f%%  ③′ %.4f%%  ④−③′ %+.4f%%'
          % (pooled_all['pb'] * 100, pooled_all['pb_bin'] * 100, d43 * 100))
    print('  逐笔 ④−③′ 为正 %d/11：%s'
          % (sum(1 for x in per if x > 0), ' '.join('%+.4f' % x for x in per)))
    print('  ⇒ 若 ④−③′ > 0：减仓幅度本身有价值（不只是「何时在场」的功劳）')
    print('    若 ④−③′ ≈ 0：优势全部来自择时/入场选择，与减仓幅度无关')

    print()
    print('=' * 116)
    print('§6 【8】Z5 右尾依赖：去最大 1/2/3 笔后 pooled ④−① 是否翻转')
    print('=' * 116)
    d_per = np.array([(ARMS['pb'][i]['num'] - 0) for i in all_keys()])  # 占位
    order = np.argsort([-(ARMS['pb'][i]['ewr'] - ARMS['batch'][i]['ewr'])
                        for i in all_keys()])
    for k in (0, 1, 2, 3):
        keep = [i for i in all_keys() if i not in set(order[:k].tolist())]
        v = pooled(keep, 'pb') - pooled(keep, 'batch')
        print('  去掉 ④−① 最大的 %d 笔 → pooled 差 %+.4f%%（%s）'
              % (k, v * 100, '正' if v > 0 else '负/翻转'))

    out = dict(rows=rows, pooled={k: pooled_all[k] for k in ARM_NAMES},
               median={k: median_ewr(k) for k in ARM_NAMES},
               d_med=d_med, d_pool=d_pool, d43=d43,
               strata=strata, verdict=verdict,
               min_sw_pen=dict(date=str(DATES.iloc[a].date()), cid=int(CID[j]),
                               sw=ARMS['pb'][j]['den'], frac_hi=frac_hi,
                               frac_band=frac_band, flat=flat,
                               changes=changes, ondays=ondays,
                               ewr4=ARMS['pb'][j]['ewr']),
               clusters={str(c): CLUSTERS[c] for c in CLUSTERS},
               cid=CID.tolist())
    with open('G:/探讨集/_tmp/r54_d190.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n  → 已导出 _tmp/r54_d190.json')


if __name__ == '__main__':
    main()
