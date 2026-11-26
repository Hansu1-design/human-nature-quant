# -*- coding: utf-8 -*-
"""d188 R53 收尾修正（Hermes R53-3 全采纳）

  ① 多重检验：四臂 6 对比较做 Bonferroni / BH，p=0.0117 必须校正后才报
  ② ①−③ 补 Wilcoxon 与配对 t（符号检验 5/11 必然近 1.0，浪费幅度信息）
  ③ 核查 ④ PB 闸门是否也混了「样本选择」：逐笔独立口径下 11 笔是否全保留、
     每笔 Σw 分布如何；并给出「扣除样本选择」的对照
  ④ 撤回清单补齐三条（分批唯一留存／九方向结构必然／6.1 个月与 341 日）
  ⑤ 作废条件四条 + 两条时限，n 一律按「笔数 + 500 日冷却归并轮次」报
"""
import numpy as np
import pandas as pd
from scipy import stats
import d183_common as K
import d177_common as P

rng = np.random.default_rng(20261113)
WIN = 500
C, n = K.C, K.n
w_pb_s = pd.Series(P.target_from(P.pb.values, .35, .05, 'linear'), index=P.idx)
W = w_pb_s.reindex(K.d['d'], method='ffill').fillna(0).values


def arms(kind, times=(300, 375, 450)):
    out, sw = [], []
    for a in K.SIG:
        b = min(a + WIN, n - 1); L = b - a + 1
        r = np.zeros(L); r[1:] = C[a + 1:b + 1] / C[a:b] - 1
        if kind == 'batch':
            w = np.ones(L); k = 3
            for j, t in enumerate(sorted(times)):
                w[min(t, L):] = (k - j - 1) / k
        elif kind == 'never':
            w = np.ones(L)
        elif kind == 'pb':
            w = W[a:b + 1].copy()
        den = w[:-1].sum()
        out.append(float((w[:-1] * r[1:]).sum() / den) if den > 0 else np.nan)
        sw.append(float(den))
    return np.array(out), np.array(sw)

A1, S1 = arms('batch')
A3, S3 = arms('never')
A4, S4 = arms('pb')

print('=' * 116)
print('§1 多重检验校正：四臂 6 对比较（Hermes R53-3 第 1 条）')
print('=' * 116)
pairs = [('①分批 − ③永不出场', A1, A3), ('①分批 − ④PB闸门', A1, A4),
         ('④PB闸门 − ③永不出场', A4, A3)]
raw = []
for lab, x, y in pairs:
    d = x - y
    nz = int((d != 0).sum()); k = int((d > 0).sum())
    p_sign = stats.binomtest(k, nz, .5).pvalue
    try:
        p_w = stats.wilcoxon(d, alternative='two-sided').pvalue
    except Exception:
        p_w = np.nan
    p_t = stats.ttest_rel(x, y).pvalue
    raw.append((lab, p_sign, p_w, p_t, np.median(d), np.mean(d)))
    print('  %-24s 符号p %.4f  Wilcoxon p %.4f  配对t p %.4f  | 中位差 %+.4f%% 均值差 %+.4f%%'
          % (lab, p_sign, p_w, p_t, np.median(d)*100, np.mean(d)*100))

# m = 3 个配对 × 2 种检验（符号 + Wilcoxon）= 6 次，两种 p 全收进数组
all_p = []
for r in raw:
    all_p.extend([r[1], r[2]])
ps = np.array(all_p, dtype=float)
m = len(ps)
bonf = min(1.0, np.nanmin(ps) * m)
sorted_p = np.sort(ps)
bh = np.nanmin([sorted_p[i] * m / (i + 1) for i in range(m)])
print('  m=%d 次检验：最小 p=%.4f → Bonferroni %.4f  |  BH %.4f'
      % (m, np.nanmin(ps), bonf, bh))
print('  其中单看符号检验 ①−④ p=0.0117 → Bonferroni %.4f' % min(1.0, 0.0117*m))
print('  ⇒ 原报 p=0.0117 是符号检验单对未校正值。')
print('  ⚠ 两个校正口径结论相反：Bonferroni(FWER) %.4f 未过、BH(FDR) %.4f 过了。' % (bonf, bh))
print('    ⇒ 正确写法「依赖校正方法、n 仅 11 笔/6 轮独立，判【不可判】」；')
print('      不得写「一格都不显著」（BH 过了），也不得写「PB 闸门已被证明更优」。')
print('    ⇒ R53 未事前指定用 FWER 还是 FDR ⇒ 本条判【不可判】，下轮事前写死。')

print()
print('=' * 116)
print('§2 ①−③ 幅度检验（Hermes R53-3 第 2 条：符号 5/11 必然近 1.0）')
print('=' * 116)
d = A1 - A3
print('  ①−③ 逐笔差 %s' % ' '.join('%+.4f%%' % (x*100) for x in d))
print('  中位 %+.4f%%  均值 %+.4f%%  std %.4f%%  最小 %+.4f%%  最大 %+.4f%%'
      % (np.median(d)*100, np.mean(d)*100, np.std(d, ddof=1)*100, d.min()*100, d.max()*100))
print('  Wilcoxon p=%.4f  配对t p=%.4f  95%%CI(均值) [%.4f%%, %.4f%%]'
      % (stats.wilcoxon(d).pvalue, stats.ttest_rel(A1, A3).pvalue,
         (np.mean(d) - 1.96*np.std(d, ddof=1)/np.sqrt(11))*100,
         (np.mean(d) + 1.96*np.std(d, ddof=1)/np.sqrt(11))*100))
print('  ⇒ 幅度含大幅正负抵消，CI 跨 0；正确写法「三者落在随机分布内、本样本不可区分」，')
print('    不是「①≈②≈③ 相等」（0.0503/0.0543/0.0587 相差 17%）。')

print()
print('=' * 116)
print('§3 核查 ④ 是否也混了「样本选择」（Hermes R53-3 第 3 条）')
print('=' * 116)
print('  逐笔独立口径下 11 笔全部保留（for a in SIG），没有跳笔：')
print('  ④ 每笔 Σw %s' % ' '.join('%.0f' % x for x in S4))
print('  ④ Σw 最小 %.0f / 最大 %.0f（全窗口 500）→ 有 %d 笔 Σw < 250（相当于半仓以下）'
      % (S4.min(), S4.max(), (S4 < 250).sum()))
print('  ⚠ 但「笔数 11→5」是【资金曲线口径】的现象（持仓中不接新信号）；')
print('    逐笔独立口径下笔数恒 11，【笔数层面】不存在样本选择。')
print('  ⚠ 然而 ④ 有一笔 Σw=7（近乎全程空仓）、4 笔 Σw<250 —— 这是【功能性缺仓】：')
print('    Hermes 第 3 条指出的机制在逐笔口径里仍以「仓位被压到近零」的形式存在，')
print('    不能说「已完全切掉」。故 0.0789% 仍须扣掉这部分后才能与 ① 公平比较；')
print('    这一扣本轮未做 ⇒ 「位置维替代时间维」维持【未测/下轮第一步】。')
# 资金曲线口径下 ④ 是否可比：给一个显式说明
print('  验证：d187 §3 已列资金曲线 s=20→500 的买入笔数 11→8→6→5，')
print('    那是骨架状态机的样本选择；本脚本 §1~§2 全部是逐笔口径，不受其污染。')

print()
print('=' * 116)
print('§4 撤回清单（补齐 Hermes R53-3 第 5 条）')
print('=' * 116)
withdraw = [
    '「分批卖出是唯一留存/最优方案」（28 号 §4.4）——现为「与不出场不可区分」，最该撤',
    '「九方向全排除是结构必然」（我 R53 第一轮 G0）——否定性证据不能建构，同源撤回',
    '「最优卖点中位 6.1 个月 / 顶日 341 日覆盖顶部」——时间轴无信息后两数一起失效',
    '「分段不塌陷 4.33/11.33/6.08」——1997-2010 前 11 年零交易、纯现金利息',
    '「+6.28% 是分批时点优势」——资金曲线口径混了样本选择，逐笔 EWR 分位 28.6%',
    '「两套口径分开报不混比」——28 号第一段作废后样本已同源（同一条 close）',
    '「暴露降低的影子」——漏了「少做 6 笔交易」这个独立机制',
    '「p=0.0117 证明 PB 闸门更优」——Bonferroni 0.0586 未过 / BH 0.0352 过了，口径依赖，判不可判',
]
for i, x in enumerate(withdraw, 1):
    print('  %d. %s 【已撤回】' % (i, x))

print()
print('=' * 116)
print('§5 改题后的作废条件（Hermes R53-3 第 6/7/8 条，事前写死）')
print('=' * 116)
print('  待办 1 判【改题】：原题「再找一组卖出时点」前提错（含一个假分段）。')
print('  新命题 H0：位置维 EWR ≤ 时间维 EWR（即「位置维减仓可替代出场层」不成立）。')
print('  推翻 H0 需【四条全中】：')
print('    a) n ≥ 20 笔独立交易（n 按「笔数 + 500 日冷却归并轮次」双报，不按天数）')
print('    b) 多重检验校正后 p < 0.05（Bonferroni 或 BH，须报比较次数 m）')
print('    c) 三段（2008-2010 / 2011-2016 / 2017-2026）同号为正')
print('    d) ④−① 扣除样本选择效应后仍 > 0')
print('  两条时限：')
print('    T1 再累积 8 笔（约 4~6 年）仍 n<20 ⇒ 判【永久不可判】，停止追加样本')
print('    T2 ④−① 在三段中任一段为负 ⇒ 判【替代不成立】，本分支关闭')
print('  ⚠ 没有 T1，这条命题会被无限追加样本拖着走；没有 T2，会挑对自己有利的段。')

print()
print('=' * 116)
print('§6 n 的双口径（Hermes R53-3 第 8 条）')
print('=' * 116)
gap = np.diff(np.array(K.SIG))
for cool in (250, 500):
    ev, last = [], -999
    for a in K.SIG:
        if a - last > cool:
            ev.append(a); last = a
    print('  笔数 %d  |  冷却 %d 日归并独立轮次 %d  |  最长间隔 %d 日'
          % (len(K.SIG), cool, len(ev), gap.max()))
print('  ⇒ 本轮所有分位、p 值、CI 一律标「11 笔 / 6 轮独立」，')
print('    不得用 7,235 个交易日冒充样本量（28 号老毛病）。')
