# -*- coding: utf-8 -*-
"""d183_common R53 共享数据层 —— 出场规则重做（消解还是补上）

统一口径（所有 R53 脚本必须 import 本模块，杜绝两套实现不一致）：
  - 长样本骨架：1997-01-02 → 2026-11-11，7235 根，标的＝上证 close
  - 信号沿用 d141.load() + d155.build()：rA≤1.10、冷却 180 日 → 11 笔买入
    ⚠ rA 由 CPI 定基算出，CPI 只有 2008-01 起 → 2008-02 前 rA 为 NaN，
      故 11 笔全部落在 2008 年后；「1997-2010 分段」的前 11 年零交易，只计息。
  - 空仓 2.5%/年日复利（RF/252）；单边成本 0.1%，按【卖出份额】计
  - 三段：1997-2010 / 2011-2016 / 2017-2026（骨架体例，仅作对照，不进判决）
  - 连续仓位口径（2008-05 起、PB 扩张窗口、月末调仓）走 d177_common，
    ★ 与本层不同源，只比 w_t 形状，不比数值，不混报 ★

Hermes R53-1 裁决（事前写死）：
  - G0 降弱命题：出场是「暂缺」，不是「消解」；只有减仓时点与分批时点重叠才谈消解
  - 决定性对照＝时点置换：暴露阶梯形状固定，三个卖点随机抽，看真实时点的分位
  - 比较一律用暴露加权收益 EWR=Σ(w·r)/Σ(w)（零参数、零自由度），不「匹配平均暴露」
  - G2 改：报 11 笔各自顶日分布与三个窗口的相对位置，不用「去最大一轮」
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d155_exit2 import build as build_sig

RF = 0.025
COST = 0.001
DAILY_RF = (1 + RF) ** (1 / 252) - 1
SEGS = [('97-10', '1997-01-01', '2010-12-31'),
        ('11-16', '2011-01-01', '2016-12-31'),
        ('17-26', '2017-01-01', '2026-12-31')]

d = load()
D = d['d'].values
C = d['close'].values
RA = d['rA'].values
n = len(C)
yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
B = build_sig(d)
SIG = B['sig']                      # 11 笔买入日索引
SIGSET = set(SIG)

# 每笔买入的持有窗口末尾＝下一笔买入日（骨架：持仓中不接新信号）
def trade_spans():
    out = []
    for k, a in enumerate(SIG):
        b = SIG[k + 1] if k + 1 < len(SIG) else n - 1
        out.append((a, min(b, n - 1)))
    return out

SPANS = trade_spans()


PEAK_WIN = 500      # 与 28 号图 c4 一致：买入后固定 500 日窗口内取最高点


def peak_days():
    """每笔买入后固定 500 日窗口内，最高点出现在买入后第几个交易日（末笔截断到样本末）。"""
    out = []
    for a in SIG:
        b = min(a + PEAK_WIN, n - 1)
        out.append(int(np.argmax(C[a:b + 1])))
    return np.array(out)

PEAKS = peak_days()


def run_ladder(times, cost=COST, rf=DAILY_RF):
    """按 11 笔骨架买入、每笔在【相对持有日】times 各时点等分减仓，跑日频资金曲线。

    ★ 严格沿用 d155.equity 的状态机：空仓遇信号满仓；持仓中不接新信号；
      卖点按 t=i−entry 触发；卖完（w=0）后回到空仓，等下一个信号。
      times 可为任意长度 k（k 批各卖 1/k）；3 批 300/375/450 可复现 28 号
      +6.28%/−26.9%。
    返回 nav(长度 n), wpath(长度 n), trades, 在场率, 平均仓位, 累计暴露 Σw
    """
    times = sorted(int(t) for t in times)
    k = len(times)
    nav = np.ones(n)
    wpath = np.zeros(n)
    cash = 1.0          # 现金（仅空仓日计息，与 d155 一致）
    sh = 0.0            # 持有份额
    w = 1.0             # 剩余比例（1 → 2/3 → 1/3 → 0）
    entry = -1
    sold = 0            # 已触发的批次数
    trades = []
    for i in range(n):
        # --- 当日净值（动作前，与 d155 顺序一致）---
        if sh > 0:
            nav[i] = sh * C[i] + cash
            wpath[i] = sh * C[i] / nav[i] if nav[i] > 0 else 0.0
        else:
            nav[i] = cash

        # --- 当日动作 ---
        if sh > 0:
            t = i - entry
            while sold < k and t >= times[sold]:
                target = (k - sold - 1) / k      # 第 sold+1 批卖完后的剩余比例
                if w > target + 1e-9:
                    q = sh * ((w - target) / w)
                    cash += q * C[i] * (1 - cost)
                    sh -= q
                    w = target
                    trades.append(('卖', i, C[i]))
                sold += 1
            if w <= 1e-9 and sh > 0:             # 兜底清仓
                cash += sh * C[i] * (1 - cost)
                sh = 0.0
                trades.append(('卖', i, C[i]))
        elif i in SIGSET and i > 0:
            sh = cash * (1 - cost) / C[i]
            cash = 0.0
            entry = i
            w = 1.0
            sold = 0
            trades.append(('买', i, C[i]))

        # 空仓计息（下一日体现）
        if sh == 0 and i < n - 1:
            cash *= (1 + rf)

    occ = (wpath > 0.01).mean()
    return pd.Series(nav, index=d['d']), wpath, trades, occ, wpath.mean(), wpath.sum()


def metrics(nav):
    v = np.asarray(nav, dtype=float)
    ann = v[-1] ** (1 / yrs) - 1
    dd = (v / np.maximum.accumulate(v) - 1).min()
    cal = ann / abs(dd) if dd < 0 else np.nan
    return ann, dd, cal


def seg_ann(nav):
    s = pd.Series(np.asarray(nav, dtype=float), index=d['d'])
    out = []
    for _, a, b in SEGS:
        t = s[a:b]
        out.append((t.iloc[-1] / t.iloc[0]) ** (365.25 / (t.index[-1] - t.index[0]).days) - 1)
    return out


def ewr(wpath, rets=None):
    """暴露加权收益 Σ(w·r)/Σ(w)，零参数消掉平均暴露差异（Hermes R53-1 第 7 条）。

    wpath: 日度实际仓位；rets: 日度标的收益（默认用上证 close）
    """
    if rets is None:
        rets = np.zeros(n)
        rets[1:] = C[1:] / C[:-1] - 1
    w = np.asarray(wpath[:-1], dtype=float)
    r = np.asarray(rets[1:], dtype=float)
    return float(np.sum(w * r) / np.sum(w)) if np.sum(w) > 0 else np.nan


def describe():
    print('R53 共享层 d183_common')
    print('  样本 %s → %s  n=%d  %.1f 年' % (d['d'].iloc[0].date(), d['d'].iloc[-1].date(), n, yrs))
    print('  rA 非空首日 %s（CPI 起点）→ 11 笔全在 2008 年后' % d['d'][d.rA.notna()].iloc[0].date())
    print('  骨架买入 %d 笔: %s' % (len(SIG), ', '.join(str(d['d'].iloc[i].date()) for i in SIG)))
    print('  各笔最高点天数', PEAKS.tolist(), ' 中位 %.0f' % np.median(PEAKS))
    print('  三段', [x[0] for x in SEGS])


if __name__ == '__main__':
    describe()
    nav, w, tr, occ, wavg, wsum = run_ladder([300, 375, 450])
    a, dd, cal = metrics(nav)
    print('\n【复现 28 号 分3批 300/375/450】年化 %.2f%%  回撤 %.1f%%  Calmar %.3f' % (a*100, dd*100, cal))
    print('  三段', ['%.2f%%' % (x*100) for x in seg_ann(nav)])
    print('  交易数', len(tr), ' 在场 %.1f%%  均仓 %.1f%%  Σw=%.0f  EWR=%.4f%%'
          % (occ*100, wavg*100, wsum, ewr(w)*100))
