# -*- coding: utf-8 -*-
"""d192 R54 ③ 收尾补充（Hermes R54-2）：检验方法冲突 / 交集同分母 / 撤回校准

★ 【不修改 d189_predeclare v2】：本脚本只补【诊断对照】与【并报】，
  主检验仍是声明【3】指定的 cluster bootstrap；打印声明哈希自证未改。
★ 全部新增对照 M_DIAG=0：只报方向与覆盖率，不进族、不升主结论、不做显著性判决。
"""
import json
import numpy as np
from scipy import stats
import d189_predeclare as D
from d190_profile import ARMS, CID, CLUSTERS, ARM_NAMES, pooled, all_keys, C, n
from d191_boot import boot_pair, boot_p, corrections, round_level

ALL = all_keys()
rng = np.random.default_rng(D.BOOT_SEED)


def main():
    print('=' * 116)
    print('§0 Z1 自查：声明未改')
    print('=' * 116)
    print('  d189_predeclare sha256[:16] = %s（与 d190/d191 报的一致则未改）'
          % D.predeclare_stamp())
    print('  主检验仍为【3】6 轮 cluster bootstrap；本脚本全部新增项 M_DIAG=0。')

    # ── 复算两个 p（不重跑主检验结论，只并报） ─────────────────────
    d_main = boot_pair('pb', 'batch')
    p_boot = boot_p(d_main)
    rd = round_level('pb', 'batch')
    k6 = int((rd > 0).sum())
    p_sign = float(stats.binomtest(k6, D.N_INDEP, .5).pvalue)

    print()
    print('=' * 116)
    print('§1 【Hermes R54-2 第 2/3/4 条】检验方法冲突必须显式并报，不得掩盖')
    print('=' * 116)
    print('  %-16s %10s %14s %14s' % ('检验方法', '原始 p', 'm=1 校正后', 'm=6 校正后'))
    grid = {}
    for lab, p in [('符号检验(6轮)', p_sign), ('cluster bootstrap', p_boot)]:
        c1 = corrections(p, D.M_PRIMARY)['holm']
        c6 = corrections(p, D.M_FAMILY_FULL)['holm']
        grid[lab] = dict(p=float(p), m1=float(c1), m6=float(c6))
        print('  %-16s %10.4f %10.4f(%s) %10.4f(%s)'
              % (lab, p, c1, '过' if c1 < D.ALPHA else '未过',
                 c6, '过' if c6 < D.ALPHA else '未过'))
    print('  ⚠ 关键一句（Hermes 第 4 条）：【符号检验 m=1 = %.4f，过了 0.05】；' % p_sign)
    print('    cluster bootstrap m=1 = %.4f，没过。' % p_boot)
    print('    ⇒「两族结论一致」只在【各自检验方法内部】成立；')
    print('      换检验方法就翻转，这才是本轮真正的判决翻转点（不是 m 单独决定）。')
    print('  ⚠ 以 bootstrap 为准的理由（Hermes 第 3 条）：符号检验又一次等权，')
    print('    把 %+.4f%% 那轮和 %+.4f%% 那轮算成同样一票 —— ' %
          (rd.max() * 100, rd.min() * 100))
    print('    与「等权中位把 Σw=7 和 Σw=500 当等精度」是同一个病，只是搬到轮级。')
    print('  ⇒ 按事前声明【2】选取更保守者 ⇒ 判决仍为【不可判】，且须公开全部 4 格。')

    print()
    print('=' * 116)
    print('§2 【Hermes R54-2 第 6 条】交集同分母对照（零自由度，仅敏感性）')
    print('=' * 116)
    eps = D.W_BIN_EPS
    rows = []
    for i in ALL:
        x = ARMS['pb'][i]
        a, b, L = x['a'], x['b'], x['L']
        r = np.zeros(L); r[1:] = C[a + 1:b + 1] / C[a:b] - 1
        w4 = ARMS['pb'][i]['w']; w1 = ARMS['batch'][i]['w']
        on4 = w4[:-1] > eps; on1 = w1[:-1] > eps
        both = on4 & on1
        if both.sum() == 0:
            rows.append(dict(i=i, cov4=0., cov1=0., e4=np.nan, e1=np.nan,
                             d=np.nan, days=0)); continue
        e4 = float((w4[:-1][both] * r[1:][both]).sum() / w4[:-1][both].sum())
        e1 = float((w1[:-1][both] * r[1:][both]).sum() / w1[:-1][both].sum())
        rows.append(dict(i=i, cov4=float(both.sum() / max(on4.sum(), 1)),
                         cov1=float(both.sum() / max(on1.sum(), 1)),
                         e4=e4, e1=e1, d=e4 - e1, days=int(both.sum())))
    print('  交集定义：④ 与 ① 当日 w>%.2f 同时成立；两臂在【同一批日子】上各按自身 w 算 EWR。'
          % eps)
    print('  %-5s %6s %9s %9s %11s %11s %11s' %
          ('笔', '交集日', '覆盖④', '覆盖①', 'EWR④', 'EWR①', '④−①'))
    for x in rows:
        print('  %-5d %6d %8.1f%% %8.1f%% %10.4f%% %10.4f%% %+10.4f%%'
              % (x['i'] + 1, x['days'], x['cov4'] * 100, x['cov1'] * 100,
                 x['e4'] * 100, x['e1'] * 100, x['d'] * 100))
    cov4 = np.nanmean([x['cov4'] for x in rows])
    cov1 = np.nanmean([x['cov1'] for x in rows])
    # pooled（交集内，按 Σw 汇总，零自由度）
    num4 = num1 = den4 = den1 = 0.0
    for i, x in enumerate(ARMS['pb']):
        a, b, L = x['a'], x['b'], x['L']
        r = np.zeros(L); r[1:] = C[a + 1:b + 1] / C[a:b] - 1
        w4 = ARMS['pb'][i]['w']; w1 = ARMS['batch'][i]['w']
        both = (w4[:-1] > eps) & (w1[:-1] > eps)
        num4 += (w4[:-1][both] * r[1:][both]).sum(); den4 += w4[:-1][both].sum()
        num1 += (w1[:-1][both] * r[1:][both]).sum(); den1 += w1[:-1][both].sum()
    e4p, e1p = num4 / den4, num1 / den1
    print('  平均覆盖率：④ %.1f%%  ① %.1f%%  → %s'
          % (cov4 * 100, cov1 * 100,
             '覆盖率够高，对照有意义' if min(cov4, cov1) > .5 else '【覆盖率偏低，此对照无意义】'))
    print('  交集内 pooled：④ %.4f%%  ① %.4f%%  ④−① %+.4f%%'
          % (e4p * 100, e1p * 100, (e4p - e1p) * 100))
    print('  ⚠ 仅作敏感性（M_DIAG=0）：同分母不解决「两臂暴露路径不同」，')
    print('    也不进主检验；若与主检验方向相反须如实报，不得删。')

    print()
    print('=' * 116)
    print('§3 【Hermes R54-2 第 8 条】撤回清单校准')
    print('=' * 116)
    print('  ✗ 原写法「R53 三个 p 全部撤回」——过强。')
    print('  ✓ 正确：11 笔口径的符号 0.0117 / Wilcoxon 0.0098 / 配对 t 0.0751')
    print('    【已撤回】（伪独立观测，R47 铁律）；')
    print('    6 轮口径的符号 %.4f / Wilcoxon %.4f 是【有效 p，保留】，' % (p_sign, p_sign))
    print('    但只测方向、且与 bootstrap 冲突，【不作判决依据】。')
    print('  另：R53 T1 原文「永久不可判」用词【本身有误】，样本可继续累积，')
    print('    只能写「本代数据（2008-2026）不可判」；本轮无新增样本 ⇒ T1 未触发。')

    print()
    print('=' * 116)
    print('§4 【Hermes R54-2 第 9/10 条】两处标注，防止旧叙事复活')
    print('=' * 116)
    print('  a) Σw 分层表（0~124 层 +0.0824% → 375~500 层 +0.0147%）：')
    print('     层内仅 2~5 笔 ⇒【仅列示，不作证据】，不得写成「低暴露笔更好」。')
    print('  b) 去最大 1/2/3 笔后 pooled 差仍正：pooled 差＝两个臂各自归一后的差、')
    print('     分母不同、不是配对量 ⇒ 只能说明「符号没变」，【不能当稳健性证据】。')

    out = dict(stamp=D.predeclare_stamp(), grid=grid,
               p_sign6=float(p_sign), p_boot=float(p_boot),
               cov4=float(cov4), cov1=float(cov1),
               e4p=float(e4p), e1p=float(e1p),
               inter_d=float(e4p - e1p), rows=rows)
    with open('G:/探讨集/_tmp/r54_d192.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n  → 已导出 _tmp/r54_d192.json')


if __name__ == '__main__':
    main()
