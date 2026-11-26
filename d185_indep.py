# -*- coding: utf-8 -*-
"""d185 R53 逐笔独立检验 —— 真正零自由度的时点置换

d184 的平移网格有个解不开的耦合：骨架「持仓中不接新信号」，卖点越晚吞掉
越多后续买入（11 笔→5 笔），Σw 从 1060 变到 2879。年化分位被暴露时长污染。

本脚本把每笔买入**单独拆出来**：固定 500 日窗口、固定 1→2/3→1/3→0 阶梯，
只变三个卖点。
⚠ Σw 并不恒定（卖得越早、暴露天数越少，实测极差 4455）——「零自由度」指的是
  EWR=Σ(w·r)/Σ(w) 自动除掉暴露水平差异、无需选匹配口径（Hermes 第 7 条），
  不是 Σw 不变。判据一律看 EWR，不看裸收益。

另做：
  §2 留出验证：前 6 笔（2008-2014）训练选起点、后 5 笔（2018-2024）完全冻结
  §3 11 笔逐笔明细 + 去最大三笔（R45 体例④右尾依赖）
  §4 二维参数面（批数 × 首点）只作描述，不作规则来源
  §5 顶日到三个卖点的距离分布
"""
import numpy as np
import d183_common as K

rng = np.random.default_rng(20261113)
WIN = 500
C = K.C
n = K.n


def per_trade_ewr(times, nb=3):
    """每笔独立：固定 500 日窗口，1→…→0 阶梯（nb 批），返回每笔 EWR 与逐笔收益。

    EWR_i = Σ(w_t·r_t)/Σ(w_t)，r_t 为当日相对收益。
    ⚠ 对任何 times，Σw 完全相同（窗口与阶梯形状固定）→ 零自由度。
    """
    times = sorted(int(t) for t in times)
    out_r, out_w = [], []
    for a in K.SIG:
        b = min(a + WIN, n - 1)
        r = np.zeros(b - a + 1)
        r[1:] = C[a + 1:b + 1] / C[a:b] - 1
        w = np.ones(len(r))
        k = nb
        for t in times:
            tt = min(t, len(r))
            w[tt:] = (k - 1) / nb
            k -= 1
        den = w[:-1].sum()
        out_r.append(float((w[:-1] * r[1:]).sum() / den) if den > 0 else np.nan)
        out_w.append(float(den))
    return np.array(out_r), np.array(out_w)


def summ(times, nb=3):
    r, w = per_trade_ewr(times, nb)
    return dict(med=float(np.median(r)), mean=float(np.mean(r)),
                pos=int((r > 0).sum()), arr=r, wsum=float(w.sum()))


print('=' * 116)
print('§0 真实时点 [300,375,450] · 逐笔独立口径（判据用 EWR）')
print('=' * 116)
real = summ([300, 375, 450])
print('  11 笔 EWR 中位 %.4f%%  均值 %.4f%%  为正 %d/11  Σw %.0f'
      % (real['med']*100, real['mean']*100, real['pos'], real['wsum']))
print('  逐笔 ' + ' '.join('%.3f%%' % (x*100) for x in real['arr']))

# ---------------- §1 平移网格（真·零自由度）----------------
print()
print('=' * 116)
print('§1 平移网格（逐笔独立）：间距 75/75 固定，只平移起点 s（Σw 随 s 变，EWR 已除掉）')
print('=' * 116)
print(f"{'s':>6}{'时点':>22}{'EWR中位':>11}{'EWR均值':>11}{'为正':>7}{'相对真实':>10}")
shift = []
for s in range(20, 501, 10):
    r = summ([s, s + 75, s + 150])
    shift.append((s, r))
for s, r in shift:
    mark = '  ★真实' if s == 300 else ''
    print(f'{s:>6}{str([s,s+75,s+150]):>22}{r["med"]*100:>10.4f}%{r["mean"]*100:>10.4f}%'
          f'{r["pos"]:>5d}/11{(r["med"]-real["med"])*100:>+9.4f}%{mark}')
meds = np.array([r['med'] for _, r in shift])
means = np.array([r['mean'] for _, r in shift])
print('-' * 116)
print('  网格 %d 格：EWR中位 [%.4f%%, %.4f%%] 中位值 %.4f%%  | 真实 s=300 分位 %.1f%%（均值分位 %.1f%%）'
      % (len(shift), meds.min()*100, meds.max()*100, np.median(meds)*100,
         (meds <= real['med']).mean()*100, (means <= real['mean']).mean()*100))
print('  Σw 极差 %.0f（卖点越早暴露越短；EWR 已除掉此差异，故不污染判据）'
      % (max(r['wsum'] for _, r in shift) - min(r['wsum'] for _, r in shift)))
print('  优于真实 EWR 中位的网格占比 %.1f%%（≈50%% ⇒ 时点零信息）'
      % ((meds > real['med']).mean()*100))

# ---------------- §2 随机置换 ----------------
print()
print('=' * 116)
print('§2 随机置换（逐笔独立）：三时点 [20,500] 随机有序抽 500 次')
print('=' * 116)
pr, pm = [], []
for _ in range(500):
    t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    while np.min(np.diff(t)) < 20:
        t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    r = summ([int(x) for x in t])
    pr.append(r['med']); pm.append(r['mean'])
pr, pm = np.array(pr), np.array(pm)
print('  随机 500 次 EWR中位：中位 %.4f%%  P5 %.4f%%  P95 %.4f%%'
      % (np.median(pr)*100, np.percentile(pr, 5)*100, np.percentile(pr, 95)*100))
print('  真实分位：EWR中位 %.1f%%  |  EWR均值 %.1f%%'
      % ((pr <= real['med']).mean()*100, (pm <= real['mean']).mean()*100))
print('  优于真实的随机组合占比：中位口径 %.1f%%  |  均值口径 %.1f%%'
      % ((pr > real['med']).mean()*100, (pm > real['mean']).mean()*100))

# ---------------- §3 留出验证 ----------------
print()
print('=' * 116)
print('§3 留出验证：前 6 笔（2008-2014）训练选 s，后 5 笔（2018-2024）完全冻结')
print('=' * 116)
train_s = [(s, summ([s, s + 75, s + 150])['med'], summ([s, s + 75, s + 150])['arr'][:6].mean())
           for s in range(20, 501, 10)]
best_s = max(train_s, key=lambda x: x[2])[0]
print('  训练段（前 6 笔）最优起点 s* = %d  |  事前真实 s = 300' % best_s)


def frozen_eval(s, sl=slice(6, 11)):
    r = summ([s, s + 75, s + 150])['arr'][sl]
    return float(np.median(r)), float(np.mean(r)), int((r > 0).sum())


for lab, s in [('训练选参 s*=%d' % best_s, best_s), ('事前真实 s=300', 300),
               ('网格起点中位数 s=260', 260)]:
    m, a, p = frozen_eval(s)
    print('  %-24s 测试段 EWR中位 %.4f%%  均值 %.4f%%  为正 %d/5' % (lab, m*100, a*100, p))
print('  ⚠ 训练/测试各只有 6/5 笔独立事件，Clopper-Pearson 下界极宽，只读方向不读数值')

# ---------------- §4 11 笔逐笔 + 去最大三笔 ----------------
print()
print('=' * 116)
print('§4 11 笔逐笔 EWR 与右尾依赖（真实时点 vs 平移网格中位）')
print('=' * 116)
grid_med = np.array([summ([s, s + 75, s + 150])['arr'] for s in range(20, 501, 10)])
grid_col_med = np.median(grid_med, axis=0)
print(f"{'买日':<12}{'顶日':>6}{'真实EWR':>11}{'网格中位':>11}{'差':>10}")
for i, a in enumerate(K.SIG):
    print('%s%6d%10.3f%%%10.3f%%%+9.3f%%'
          % (str(K.d['d'].iloc[a].date()), K.PEAKS[i], real['arr'][i]*100,
             grid_col_med[i]*100, (real['arr'][i]-grid_col_med[i])*100))
order = np.argsort(-real['arr'])
for k in (0, 2, 3):
    keep = order[k:]
    print('  去掉最好的 %d 笔后：真实中位 %.4f%%  vs 网格中位(同笔) %.4f%%'
          % (k+1, np.median(real['arr'][keep])*100, np.median(grid_col_med[keep])*100))

# ---------------- §5 顶日距离 ----------------
print()
print('=' * 116)
print('§5 顶日到三个卖点的距离（负＝顶在卖点之前，正＝卖点之后才见顶）')
print('=' * 116)
for t in (300, 375, 450):
    dist = K.PEAKS - t
    print('  卖点 %d：顶在此前 %d/11，此后 %d/11；距离中位 %+.0f 日，范围 [%+d, %+d]'
          % (t, (dist < 0).sum(), (dist >= 0).sum(), np.median(dist), dist.min(), dist.max()))

# ---------------- 判决 ----------------
print()
print('=' * 116)
print('§6 事前判据核对（逐笔独立口径为准）')
print('=' * 116)
q_shift = (meds <= real['med']).mean()
q_perm = (pr <= real['med']).mean()
strong = (q_perm >= 0.95) and (q_shift >= 0.80)
print('  平移网格分位 %.1f%%  随机置换分位 %.1f%%' % (q_shift*100, q_perm*100))
print('  [时点有信息] 随机 ≥95%% 且 平移 ≥80%% : %s' % strong)
print('  [时点零信息] 否则 ⇒ G1 成立、G0 废、出场是「暂缺」不是「消解」: %s' % (not strong))
