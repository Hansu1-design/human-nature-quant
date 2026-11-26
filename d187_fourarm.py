# -*- coding: utf-8 -*-
"""d187 R53 四臂同批对照（Hermes R53-2 第 7/8 条：补「永不出场」这一臂）

原判决「时间轴无效 / 不需要出场」缺了直接对照：所有比较都是
「分批 vs 随机时点」，从来没测过「根本不出场」。本脚本同批 11 笔、
同一 500 日窗口、全用 EWR=Σ(w·r)/Σ(w)，四臂：
  ① 分批 300/375/450（1→2/3→1/3→0）
  ② 随机时点（500 次，取中位）
  ③ 持有到窗口末（w≡1，永不出场）——「时间轴无效」的直接检验
  ④ 持有到窗口末 + PB 闸门（w=PB 连续仓位）—— 真正想立的那条

★ Hermes R53-2 第 9 条：28 号「1997-2010 段」作废后，d183 的有效交易
  （2008-09 起）与 d177（2008-05 起）已经同源、同一条上证 close 序列。
  故本脚本不再以「口径不同」回避比较，四臂在同一段样本上直接对照。
  差异只在【交易机制】（事件驱动 vs 月末连续调仓），不在数据源。

★ 两个机制分开写（Hermes R53-2 第 1 条）：
  资金曲线口径的 +6.28% 里混了「少做 6 笔交易」＝样本选择效应；
  逐笔独立口径把 11 笔全部保留，只剩【暴露水平】差异。二者不可混称。
"""
import numpy as np
import pandas as pd
import d183_common as K
import d177_common as P

rng = np.random.default_rng(20261113)
WIN = 500
C, n = K.C, K.n

# PB 连续仓位目标序列，映射到 d183 日期轴（同源、同价格序列）
w_pb_s = pd.Series(P.target_from(P.pb.values, .35, .05, 'linear'), index=P.idx)
W = w_pb_s.reindex(K.d['d'], method='ffill').fillna(0).values


def arm_weights(kind, times=(300, 375, 450), seed_times=None):
    """返回 11 笔各自的 w 序列与逐笔 EWR。w 长度＝每笔 500 日窗口（末笔截断）。"""
    out = []
    for a in K.SIG:
        b = min(a + WIN, n - 1)
        L = b - a + 1
        r = np.zeros(L)
        r[1:] = C[a + 1:b + 1] / C[a:b] - 1
        if kind == 'batch':
            w = np.ones(L); k = len(times)
            for j, t in enumerate(sorted(times)):
                w[min(t, L):] = (k - j - 1) / k
        elif kind == 'random':
            w = np.ones(L); k = len(seed_times)
            for j, t in enumerate(sorted(seed_times)):
                w[min(t, L):] = (k - j - 1) / k
        elif kind == 'never':
            w = np.ones(L)
        elif kind == 'pb':
            w = W[a:b + 1].copy()
        else:
            raise ValueError(kind)
        den = w[:-1].sum()
        out.append((float((w[:-1] * r[1:]).sum() / den) if den > 0 else np.nan,
                    float(den), w))
    return out


def report(lab, arms):
    e = np.array([x[0] for x in arms])
    sw = np.array([x[1] for x in arms])
    print('  %-26s EWR中位 %.4f%%  均值 %.4f%%  为正 %2d/11  Σw %.0f'
          % (lab, np.median(e)*100, np.mean(e)*100, int((e > 0).sum()), sw.sum()))
    return e


print('=' * 116)
print('§1 四臂同批对照（11 笔、500 日窗口、同一条上证 close、全用 EWR）')
print('=' * 116)
A1 = report('① 分批 300/375/450', arm_weights('batch'))
# ② 随机时点 500 次，取「EWR 中位」的中位
rand_meds = []
for _ in range(500):
    t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    while np.min(np.diff(t)) < 20:
        t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    e = np.array([x[0] for x in arm_weights('random', seed_times=[int(v) for v in t])])
    rand_meds.append(np.median(e))
rand_meds = np.array(rand_meds)
print('  %-26s EWR中位 %.4f%%（500 次中位的中位）  P5 %.4f%%  P95 %.4f%%'
      % ('② 随机时点', np.median(rand_meds)*100,
         np.percentile(rand_meds, 5)*100, np.percentile(rand_meds, 95)*100))
A3 = report('③ 永不出场 w≡1', arm_weights('never'))
A4 = report('④ 永不出场 + PB 闸门', arm_weights('pb'))

print()
print('=' * 116)
print('§2 配对差（逐笔相减，同笔同窗口，零自由度）')
print('=' * 116)
print(f"{'对比':<34}{'中位差':>12}{'均值差':>12}{'胜出笔数':>12}{'符号检验p':>12}")
from scipy import stats
for lab, x, y in [('① 分批 − ② 随机', A1, np.full(11, np.nan)),
                  ('① 分批 − ③ 永不出场', A1, A3),
                  ('① 分批 − ④ PB闸门', A1, A4),
                  ('④ PB闸门 − ③ 永不出场', A4, A3)]:
    if np.isnan(y).all():
        d = A1 - np.median(rand_meds)
        print('  %-34s%+11.4f%%%+11.4f%%%9d/11%12s'
              % (lab, np.median(d)*100, np.mean(d)*100, int((d > 0).sum()), '—'))
    else:
        d = x - y
        nz = int((d != 0).sum()); k = int((d > 0).sum())
        pv = stats.binomtest(k, nz, .5).pvalue if nz else np.nan
        print('  %-34s%+11.4f%%%+11.4f%%%9d/11%12.4f'
              % (lab, np.median(d)*100, np.mean(d)*100, k, pv))

print()
print('  ⚠ 关键读法：若 ①≈②≈③（三者 EWR 中位同量级、配对差不显著），')
print('    则「分批卖点」既没赢随机、也没赢根本不卖 ⇒ 时间维不构成独立扳机。')
print('    若 ④ 明显优于 ③，才有资格谈「位置维减仓在功能上替代时间维减仓」；')
print('    这一步 Hermes R53-2 第 6 条明确说【本轮未测、不得一起结案】。')

print()
print('=' * 116)
print('§3 两个机制分开写（Hermes R53-2 第 1 条）')
print('=' * 116)
for s in (20, 130, 300, 500):
    nav, w, tr, occ, wavg, wsum = K.run_ladder([s, s + 75, s + 150])
    a, dd, cal = K.metrics(nav)
    nb = sum(1 for t in tr if t[0] == '买')
    print('  资金曲线 s=%3d：年化 %.2f%%  Σw %.0f  买入 %2d 笔  在场 %.1f%%'
          % (s, a*100, wsum, nb, occ*100))
print('  → 卖点越晚，Σw 越大且【买入笔数从 11 掉到 5】：这是样本选择（少做交易），')
print('    不是「仓位低」。逐笔独立口径（§1）把 11 笔全保留，才只剩暴露水平差异。')

print()
print('=' * 116)
print('§4 同源性核对（Hermes R53-2 第 9 条）')
print('=' * 116)
print('  d183 有效交易首日 %s  |  d177 起点 %s' %
      (K.d['d'].iloc[K.SIG[0]].date(), P.idx[0].date()))
print('  两者都用 idx_sse_all.parquet 的 close（同一条价格序列）→ 样本已同源。')
print('  ⇒ 撤销「两套口径分开报、不混比」：差异只是交易机制（事件驱动 vs 月末调仓），')
print('    §1 四臂已在同一段样本、同一条价格上直接对照。')
