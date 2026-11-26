# -*- coding: utf-8 -*-
"""
d40 供给曲线形状检验（Hermes R34 Q2 处方①：对称性/形状，最省最强）

H1（用户的参考点供给曲线）:
   S(P) 不是单调递增，而是「深水下冻结 -> 成本附近峰值 -> 高于成本再升」
   => S(+d)/S(-d) 在 |d| 小处最大，随 |d| 增大而下降（局部峰在 0 附近）

竞争解释（趋势/动量）:
   d = P/anchor - 1，d 越大 = 趋势越强 = 换手越高
   => 比值随 |d| 单调上升

机械 artifact（Hermes 预警）:
   C 由成交量加权，S 又用成交量度量 => 单峰分布下 S(+d)≈S(-d)，比值≈1 且平坦

对照锚（volume-blind）:
   250 日均线（等权）。若成本锚的比值 ≈ 均线锚的比值，则「成本」不携带信息。

预注册判决规则（跑之前写死）:
  1) 两个锚的比值在所有 |d| 档都 ≈1（0.95~1.05）       -> 无参考点效应，H1 垮
  2) 比值随 |d| 单调上升，且成本锚 ≈ 均线锚             -> 是趋势不是成本，H1 垮
  3) 比值在最小 |d| 档最高、随 |d| 下降，且成本锚 > 均线锚 -> H1 成立方向
  4) 比值在最小 |d| 档最高、但两个锚一样                -> 形状对但归因不成立（只到「一般参考点」，不是「筹码成本」）
"""
import os, glob, json, random, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
MAXN = int(os.environ.get("MAXN", "1500"))
EDGES = [0.005, 0.015, 0.03, 0.06, 0.12, 0.25, 1e9]
LAB = ["0.5-1.5%", "1.5-3%", "3-6%", "6-12%", "12-25%", ">25%"]
EPS = 1e-6


def ybin(a):
    return np.digitize(a, EDGES) - 1


def cost_anchor(p, alpha):
    """换手衰减筹码成本 C_t = (1-a)C_{t-1} + a P_t"""
    C = np.empty_like(p)
    C[0] = p[0]
    for i in range(1, len(p)):
        a = alpha[i]
        C[i] = (1.0 - a) * C[i - 1] + a * p[i]
    return C


def main():
    fs = sorted(glob.glob(R + "/*.parquet"))
    random.seed(20261112)
    if MAXN and MAXN < len(fs):
        fs = random.sample(fs, MAXN)

    # cell[(anchor, yr, yb, q5, side)] = [sum_log_rt, n]
    cell = {}
    used = 0
    skipped = 0

    for k, f in enumerate(fs):
        try:
            d = pq.read_table(f, columns=["date", "close", "turnover"]).to_pandas()
        except Exception:
            skipped += 1
            continue
        if len(d) < 400:
            skipped += 1
            continue
        d = d.sort_values("date").reset_index(drop=True)
        p = d["close"].astype("float64").values
        tr = d["turnover"].astype("float64").values          # 比例(0.0026 = 0.26%)
        if not np.isfinite(p).all() or not np.isfinite(tr).all():
            m = np.isfinite(p) & np.isfinite(tr)
            p, tr = p[m], tr[m]
            d = d[m].reset_index(drop=True)
            if len(p) < 400:
                skipped += 1
                continue
        if (p <= 0).any():
            skipped += 1
            continue

        dt = pd.to_datetime(d["date"])
        yr = dt.dt.year.values
        ret60 = pd.Series(p).pct_change(60).values

        alpha = np.clip(tr, 0.0, 0.5)
        C = cost_anchor(p, alpha)
        MA = pd.Series(p).rolling(250, min_periods=120).mean().values

        with np.errstate(divide="ignore", invalid="ignore"):
            dC = p / C - 1.0
            dM = p / MA - 1.0

        # 归一化换手：除以该股该年中位换手（无量纲，去掉水平漂移）
        s = pd.Series(tr)
        med = s.groupby(pd.Series(yr)).transform("median").values
        rt = tr / np.where(med > 0, med, np.nan)
        rt = np.clip(rt, 1e-4, 50.0)
        lrt = np.log(rt + EPS)

        # 过去 60 日收益的股票内五分位（控制动量）
        try:
            q5 = pd.qcut(pd.Series(ret60), 5, labels=False, duplicates="drop").values
        except Exception:
            q5 = np.zeros(len(p))

        ok_base = np.isfinite(lrt) & np.isfinite(q5) & (yr >= 2011)

        for anchor, dd, warm in (("cost", dC, 120), ("ma", dM, 250)):
            ok = ok_base & np.isfinite(dd) & (np.abs(dd) >= EDGES[0])
            ok[:warm] = False
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            yb = ybin(np.abs(dd[idx]))
            side = (dd[idx] > 0).astype(np.int8)
            for j, i in enumerate(idx):
                key = (anchor, int(yr[i]), int(yb[j]), int(q5[i]), int(side[j]))
                e = cell.get(key)
                if e is None:
                    cell[key] = [float(lrt[i]), 1]
                else:
                    e[0] += float(lrt[i])
                    e[1] += 1
        used += 1
        if (k + 1) % 200 == 0:
            print("  ... %d/%d  used=%d" % (k + 1, len(fs), used), flush=True)

    print("\n样本: %d 只可用 / %d 只读取失败或过短" % (used, skipped))

    # 聚合：每个 (anchor, yb) 下，跨 (yr, q5) 加权合并正负两侧
    out = {}
    for anchor in ("cost", "ma"):
        rows = []
        for bi, lab in enumerate(LAB):
            sp = sn = 0.0
            np_ = nn = 0
            cells = 0
            cells_gt1 = 0
            for (a, y, b, q, sd), (sl, n) in cell.items():
                if a != anchor or b != bi:
                    continue
                if sd == 1:
                    sp += sl; np_ += n
                else:
                    sn += sl; nn += n
            if np_ >= 200 and nn >= 200:
                mp, mn = sp / np_, sn / nn
                ratio = float(np.exp(mp - mn))
                # 逐 cell 的比值方向一致性
                byk = {}
                for (a, y, b, q, sd), (sl, n) in cell.items():
                    if a != anchor or b != bi or n < 20:
                        continue
                    kk = (y, q)
                    byk.setdefault(kk, {1: [0.0, 0], 0: [0.0, 0]})[sd][0] += sl
                    byk[kk][sd][1] += n
                for kk, v in byk.items():
                    if v[1][1] >= 20 and v[0][1] >= 20:
                        r = (v[1][0] / v[1][1]) - (v[0][0] / v[0][1])
                        cells += 1
                        if r > 0:
                            cells_gt1 += 1
                rows.append(dict(bin=lab, ratio=round(ratio, 4),
                                 n_up=int(np_), n_dn=int(nn),
                                 cells=cells, cells_gt1=cells_gt1))
        out[anchor] = rows

    print("\n=== S(+d)/S(-d)：正偏离侧换手 ÷ 负偏离侧换手（几何均值比）===")
    print("%-10s | %-28s | %-28s" % ("|d| 档", "成本锚（换手加权）", "均线锚（等权 250 日）"))
    print("%-10s | %8s %9s %8s | %8s %9s %8s" % ("", "比值", "n上/n下", "cell正%", "比值", "n上/n下", "cell正%"))
    for i, lab in enumerate(LAB):
        a = out["cost"][i] if i < len(out["cost"]) else None
        b = out["ma"][i] if i < len(out["ma"]) else None
        fa = ("%8.3f %9s %7.0f%%" % (a["ratio"], "%d/%d" % (a["n_up"], a["n_dn"]),
                                      100.0 * a["cells_gt1"] / max(1, a["cells"]))) if a else "%8s" % "-"
        fb = ("%8.3f %9s %7.0f%%" % (b["ratio"], "%d/%d" % (b["n_up"], b["n_dn"]),
                                      100.0 * b["cells_gt1"] / max(1, b["cells"]))) if b else "%8s" % "-"
        print("%-10s | %s | %s" % (lab, fa, fb))

    json.dump(out, open("d40_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写 d40_result.json")


if __name__ == "__main__":
    main()
