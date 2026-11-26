# -*- coding: utf-8 -*-
"""d184 R53 决定性对照：时点置换（Hermes R53-1 第 5/6 条）

问题：分 3 批 300/375/450 日的 +6.28%，是「三个时点挑得好」，还是
      「把暴露从 100% 降到 0」这件事本身？

两条零自由度检验（都不引入任何待选参数）：
  A 平移网格：保持三批间距 75/75 严格不变，只把起点 s 从 20 扫到 500。
     ⚠ 事后发现：骨架「持仓中不接新信号」使 Σw 并不恒定（卖点越晚吞掉越多
     后续买入，11 笔→5 笔），时点与暴露路径耦合。故年化分位被暴露时长污染，
     判决一律以 EWR 为主；真正零自由度的检验见 d185（逐笔独立口径）。
  B 随机置换：三个时点在 [20,500] 内随机有序抽（保序、最小间距 20），
     500 次，看真实组合在分布中的分位。用 EWR 消掉暴露水平差异。

判据（事前写死）：
  - 真实时点在平移网格里若不是显著靠前（未进前 20% 分位），或
    在随机置换里分位 < 95%，⇒ 时间维零信息，G1 成立、G0 废（出场不是"补上"）
  - 真实时点显著优于随机分布 ≥95 分位 ⇒ 时间维确有信息，才谈"补上"
  - ⚠ 买入笔数必须同报：随机时点变大会吞掉后续信号，笔数变化会污染年化，
    故判决以 EWR（暴露加权收益）为主、年化为辅
"""
import numpy as np
import d183_common as K

rng = np.random.default_rng(20261113)


def one(times):
    nav, w, tr, occ, wavg, wsum = K.run_ladder(times)
    ann, dd, cal = K.metrics(nav)
    nb = sum(1 for t in tr if t[0] == '买')
    return dict(ann=ann, dd=dd, cal=cal, ewr=K.ewr(w), wsum=wsum,
                occ=occ, wavg=wavg, nb=nb, seg=K.seg_ann(nav))


print('=' * 116)
print('§0 真实时点 [300,375,450] 基准')
print('=' * 116)
real = one([300, 375, 450])
print('  年化 %.2f%%  回撤 %.1f%%  Calmar %.3f  EWR %.4f%%  Σw %.0f  在场 %.1f%%  均仓 %.1f%%  买入 %d 笔'
      % (real['ann']*100, real['dd']*100, real['cal'], real['ewr']*100,
         real['wsum'], real['occ']*100, real['wavg']*100, real['nb']))
print('  三段 ' + ' / '.join('%.2f%%' % (x*100) for x in real['seg']))

# ---------------- A 平移网格 ----------------
print()
print('=' * 116)
print('§A 平移网格：三批间距固定 75/75，只平移起点 s（⚠ Σw 因吞信号不恒定，以 EWR 为准）')
print('=' * 116)
print(f"{'s':>6}{'时点':>22}{'年化':>9}{'回撤':>9}{'Calmar':>8}{'EWR':>10}{'Σw':>8}{'买入':>6}")
shift = []
for s in range(20, 501, 10):
    t = [s, s + 75, s + 150]
    r = one(t)
    shift.append((s, t, r))
for s, t, r in shift:
    mark = '  ★真实' if s == 300 else ''
    print(f'{s:>6}{str(t):>22}{r["ann"]*100:>8.2f}%{r["dd"]*100:>8.1f}%{r["cal"]:>8.3f}'
          f'{r["ewr"]*100:>9.4f}%{r["wsum"]:>8.0f}{r["nb"]:>6d}{mark}')

anns = np.array([r['ann'] for _, _, r in shift])
ewrs = np.array([r['ewr'] for _, _, r in shift])
cals = np.array([r['cal'] for _, _, r in shift])
pr_a = (anns <= real['ann']).mean()
pr_e = (ewrs <= real['ewr']).mean()
pr_c = (cals <= real['cal']).mean()
print('-' * 116)
print('  网格 %d 格：年化 [%.2f%%, %.2f%%] 中位 %.2f%%  | EWR [%.4f%%, %.4f%%] 中位 %.4f%%'
      % (len(shift), anns.min()*100, anns.max()*100, np.median(anns)*100,
         ewrs.min()*100, ewrs.max()*100, np.median(ewrs)*100))
print('  真实 s=300 分位：年化 %.1f%%  |  EWR %.1f%%  |  Calmar %.1f%%'
      % (pr_a*100, pr_e*100, pr_c*100))
print('  Σw 极差 %.0f（若>0 说明末端截断仍在，需以 EWR 为准）'
      % (np.array([r['wsum'] for _, _, r in shift]).max() - np.array([r['wsum'] for _, _, r in shift]).min()))

# ---------------- B 随机置换 ----------------
print()
print('=' * 116)
print('§B 随机置换：三时点在 [20,500] 内随机有序抽，保序、最小间距 20，500 次')
print('=' * 116)
perm = []
for _ in range(500):
    t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    while np.min(np.diff(t)) < 20:
        t = np.sort(rng.choice(np.arange(20, 501), size=3, replace=False))
    r = one([int(x) for x in t])
    perm.append((list(map(int, t)), r))

pa = np.array([r['ann'] for _, r in perm])
pe = np.array([r['ewr'] for _, r in perm])
pc = np.array([r['cal'] for _, r in perm])
print('  随机 500 次：年化 中位 %.2f%%  P5 %.2f%%  P95 %.2f%%  | EWR 中位 %.4f%%  P5 %.4f%%  P95 %.4f%%'
      % (np.median(pa)*100, np.percentile(pa, 5)*100, np.percentile(pa, 95)*100,
         np.median(pe)*100, np.percentile(pe, 5)*100, np.percentile(pe, 95)*100))
print('  真实 [300,375,450] 分位：年化 %.1f%%  |  EWR %.1f%%  |  Calmar %.1f%%'
      % ((pa <= real['ann']).mean()*100, (pe <= real['ewr']).mean()*100, (pc <= real['cal']).mean()*100))
print('  随机组合中优于真实年化的比例 %.1f%%（若接近 50%% ⇒ 时点零信息）'
      % ((pa > real['ann']).mean()*100))

# ---------------- C 逐笔顶日分布（Hermes 第 9 条）----------------
print()
print('=' * 116)
print('§C 11 笔各自顶日 vs 三个卖点的相对位置（不是「去最大一轮」）')
print('=' * 116)
P = K.PEAKS
print(f"{'买日':<12}{'顶日':>6}{'300前':>8}{'300-375':>9}{'375-450':>9}{'450后':>8}")
for i, a in enumerate(K.SIG):
    p = P[i]
    bucket = ('300前' if p < 300 else '300-375' if p < 375 else '375-450' if p < 450 else '450后')
    print(f'{str(K.d["d"].iloc[a].date()):<12}{p:>6}' + ''.join(
        f'{("  ●" if b == bucket else ""):>8}' if j == 0 else f'{("  ●" if b == bucket else ""):>9}'
        for j, b in enumerate(['300前', '300-375', '375-450', '450后'])))
print('  顶日 %s' % sorted(P.tolist()))
print('  中位 %.0f  |  <300: %d/11  |  300-450: %d/11  |  ≥450: %d/11'
      % (np.median(P), (P < 300).sum(), ((P >= 300) & (P < 450)).sum(), (P >= 450).sum()))
print('  ⚠ 中位 341 恰在 300 与 375 之间 → 「第一批贴近顶部」可能是两个中位数偶然撞上')

# ---------------- 判决 ----------------
print()
print('=' * 116)
print('§D 事前判据核对')
print('=' * 116)
ok_strong = ((pe <= real['ewr']).mean() >= 0.95) and (pr_e >= 0.80)
print('  [时点有信息] 随机置换 EWR 分位 ≥95%% 且 平移网格 EWR 分位 ≥80%% : %s' % ok_strong)
print('  [时点零信息] 否则 ⇒ G1 成立（纯暴露效应）、G0 废（出场是暂缺，不是消解）: %s' % (not ok_strong))
print('  判据提醒：随机时点会吞掉后续买入信号，年化受笔数污染；判决以 EWR 为主。')
