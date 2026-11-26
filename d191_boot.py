# -*- coding: utf-8 -*-
"""d191 R54 ② 主检验：6 轮 cluster bootstrap + n/m 分开写死后的判决

★ 口径全部从 d189_predeclare 读；共享计算从 d190_profile 静默 import。
★ 【3】主检验＝按【轮】有放回重抽、轮内笔全带，零自由度、天然处理轮内相关。
★ 【2】族大小双报：m=1（单一预设假设）与 m=6（保守全族上界）结论须一致。
★ 主假设 H0：位置维 EWR ≤ 时间维 EWR ⇔ 主比较 ④−①。
"""
import json
import numpy as np
from scipy import stats
import d189_predeclare as D
from d190_profile import (ARMS, CID, CLUSTERS, ARM_NAMES, LABEL,
                          pooled, all_keys, DATES, K)

rng = np.random.default_rng(D.BOOT_SEED)
ALL = all_keys()


def boot_pair(kx, ky, nboot=D.N_BOOT):
    """按轮有放回重抽；每轮内所有笔（num/den）全带上。返回 pooled 差分布。"""
    out = np.empty(nboot)
    for t in range(nboot):
        pick = rng.integers(0, D.N_INDEP, size=D.N_INDEP)
        keys = [i for c in pick for i in CLUSTERS[int(c)]]
        out[t] = pooled(keys, kx) - pooled(keys, ky)
    return out


def boot_p(d):
    """双边 bootstrap p：2×min(单侧尾概率)。"""
    return float(2 * min((d <= 0).mean(), (d > 0).mean()))


def corrections(p, m):
    """单个 p 参加 m 族校正时的 Holm / Bonferroni / BH。
    ⚠ 事前声明【2】v2 注：对单个 p，三法同值 = min(1, p·m)；
      这【不是巧合】，也【不构成三条独立证据】，报告必须写明。"""
    v = min(1.0, float(p) * m)
    return dict(holm=v, bonferroni=v, bh=v)


def round_level(kx, ky):
    """轮级差＝轮内所有笔 pooled，【不是】挑一笔当代表（Hermes R54-1 第 4 条）。"""
    return np.array([pooled(CLUSTERS[c], kx) - pooled(CLUSTERS[c], ky)
                     for c in range(D.N_INDEP)])


def main():
    print('=' * 116)
    print('§0 事前声明核验')
    print('=' * 116)
    print('  sha256[:16] = %s  n_indep=%d  m_primary=%d  m_full=%d  N_BOOT=%d  seed=%d'
          % (D.predeclare_stamp(), D.N_INDEP, D.M_PRIMARY,
             D.M_FAMILY_FULL, D.N_BOOT, D.BOOT_SEED))
    print('  【4】功效边界（纯算术）：(n=6,6/6) p=0.03125 | (6,5/6) 0.21875 | '
          '(8,8/8) 0.0078125 | (8,7/8) 0.0703')

    print()
    print('=' * 116)
    print('§1 主检验 ④−①：6 轮 cluster bootstrap')
    print('=' * 116)
    d_main = boot_pair('pb', 'batch')
    ci = np.percentile(d_main, [2.5, 97.5])
    p_main = boot_p(d_main)
    pooled_d = pooled(ALL, 'pb') - pooled(ALL, 'batch')
    print('  原样本 pooled ④−① = %+.4f%%' % (pooled_d * 100))
    print('  bootstrap(%d 次) 中位 %+.4f%%  均值 %+.4f%%'
          % (D.N_BOOT, np.median(d_main) * 100, d_main.mean() * 100))
    print('  95%%CI [%+.4f%%, %+.4f%%]  → %s'
          % (ci[0] * 100, ci[1] * 100, '不跨 0' if ci[0] > 0 else '【跨 0】'))
    print('  P(④−① > 0) = %.4f   |   bootstrap 双边 p = %.4f'
          % ((d_main > 0).mean(), p_main))

    print()
    print('  -- 旁证：轮级差（轮内所有笔 pooled，不挑代表） --')
    rd = round_level('pb', 'batch')
    print('  6 轮差 %s' % ' '.join('%+.4f%%' % (x * 100) for x in rd))
    k6 = int((rd > 0).sum())
    p_sign6 = stats.binomtest(k6, D.N_INDEP, .5).pvalue
    p_w6 = stats.wilcoxon(rd, alternative='two-sided').pvalue
    print('  为正 %d/6  符号检验 p=%.4f  Wilcoxon p=%.4f' % (k6, p_sign6, p_w6))
    print('  ⚠ 符号检验 n=6 时最小可达双边 p=0.03125（6/6 全同向也只到这个数）。')

    print()
    print('=' * 116)
    print('§2 【2】族大小双报：m=1 与 m=6 × 三种校正（共 6 个数全报）')
    print('=' * 116)
    res = {}
    for m, tag in ((D.M_PRIMARY, '主 m=1'), (D.M_FAMILY_FULL, '保守 m=6')):
        c = corrections(p_main, m)
        res[tag] = c
        print('  %-10s Holm %.4f  Bonferroni %.4f  BH %.4f  → %s'
              % (tag, c['holm'], c['bonferroni'], c['bh'],
                 '<0.05' if c['holm'] < D.ALPHA else '≥0.05'))
    print('  ⚠ 单个 p 校正时三法必然同值（=min(1,p·m)），')
    print('    【不构成三条独立证据】；真正起作用的是「两族结论是否一致」。')
    ok1 = res['主 m=1']['holm'] < D.ALPHA
    ok6 = res['保守 m=6']['holm'] < D.ALPHA
    ci_ok = bool(ci[0] > 0)
    print('  硬条件 (i) CI 不跨 0：%s   (ii) 两族一致且都<α：%s（m1=%s / m6=%s）'
          % ('满足' if ci_ok else '不满足',
             '满足' if (ok1 and ok6) else '不满足',
             '过' if ok1 else '未过', '过' if ok6 else '未过'))
    verdict = '可判' if (ci_ok and ok1 and ok6) else '不可判'
    print('  ⇒ 判决：【%s】' % verdict)
    if ok1 and not ok6:
        print('    m=1 过、m=6 未过 ⇒ 这正是 Hermes R54-1 第 1/2 条说的【判决翻转点】：')
        print('    结论完全由「族大小怎么定」决定，按事前声明【2】判【不可判】，不挑有利的报。')

    print()
    print('=' * 116)
    print('§3 诊断对照（M_DIAG=0：只报方向与 CI，不做显著性判决，不得升为主结论）')
    print('=' * 116)
    diag = {}
    for kx, ky in [('pb', 'never'), ('pb', 'pb_bin'), ('batch', 'never'),
                   ('pb_bin', 'never')]:
        d = boot_pair(kx, ky)
        c = np.percentile(d, [2.5, 97.5])
        key = '%s−%s' % (kx, ky)
        diag[key] = dict(med=float(np.median(d)), ci=[float(c[0]), float(c[1])],
                         p_pos=float((d > 0).mean()))
        print('  %-18s pooled %+9.4f%%  95%%CI [%+.4f%%, %+.4f%%]  P(>0)=%.3f  %s'
              % (key, (pooled(ALL, kx) - pooled(ALL, ky)) * 100,
                 c[0] * 100, c[1] * 100, (d > 0).mean(),
                 'CI 不跨 0' if c[0] > 0 else 'CI 跨 0'))

    print()
    print('=' * 116)
    print('§4 R53 三个 p 值按 6 轮口径重算（伪独立撤回核对）')
    print('=' * 116)
    d11 = np.array([ARMS['pb'][i]['ewr'] - ARMS['batch'][i]['ewr'] for i in ALL])
    k11 = int((d11 > 0).sum())
    p_sign11 = stats.binomtest(k11, len(d11), .5).pvalue
    p_w11 = stats.wilcoxon(d11, alternative='two-sided').pvalue
    p_t11 = stats.ttest_rel([ARMS['pb'][i]['ewr'] for i in ALL],
                            [ARMS['batch'][i]['ewr'] for i in ALL]).pvalue
    print('  R53 原报（11 笔）：符号 p=%.4f（%d/11）  Wilcoxon p=%.4f  配对 t p=%.4f'
          % (p_sign11, k11, p_w11, p_t11))
    print('  R54 6 轮重算    ：符号 p=%.4f（%d/6）  Wilcoxon p=%.4f'
          % (p_sign6, k6, p_w6))
    print('  ⇒ 11 笔口径把同一轮内的相邻买入当独立样本（伪独立观测，R47 铁律）；')
    print('    三个 p 全部【已撤回】，一律以 6 轮 cluster bootstrap 为准。')
    print('  ⚠ 本轮不得再写「p=0.0117 证明 PB 闸门更优」，也不得写「一格都不显著」。')

    print()
    print('=' * 116)
    print('§5 作废条件自查')
    print('=' * 116)
    print('  Z1 事后改 m/分层/口径/boot 次数：否（声明 v2 哈希 %s，跑数前定死）'
          % D.predeclare_stamp())
    print('  Z2 诊断对照升主结论：否（§3 全部标 M_DIAG=0）')
    print('  Z3 两族或同族三口径不一致：%s'
          % ('触发 ⇒ 不可判' if (ok1 != ok6) else '未触发'))
    print('  Z4 主检验 95%%CI 跨 0：%s'
          % ('未触发' if ci_ok else '触发 ⇒ 本样本不可判，连带 R53 T1'))
    # Z5 在 d190 §6 已报
    print('  Z5 右尾依赖：见 d190 §6（去最大 1/2/3 笔）')

    out = dict(stamp=D.predeclare_stamp(), n_indep=D.N_INDEP,
               m1=D.M_PRIMARY, m6=D.M_FAMILY_FULL, n_boot=D.N_BOOT,
               pooled_d=float(pooled_d), ci=[float(ci[0]), float(ci[1])],
               p_main=float(p_main), p_pos=float((d_main > 0).mean()),
               round_diff=[float(x) for x in rd], k6=k6,
               p_sign6=float(p_sign6), p_w6=float(p_w6),
               p_sign11=float(p_sign11), p_w11=float(p_w11),
               p_t11=float(p_t11),
               corrections=res, ci_ok=ci_ok, ok1=bool(ok1), ok6=bool(ok6),
               verdict=verdict, diag=diag,
               boot_hist=np.percentile(d_main, np.arange(0, 101, 5)).tolist())
    with open('G:/探讨集/_tmp/r54_d191.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n  → 已导出 _tmp/r54_d191.json')


if __name__ == '__main__':
    main()
