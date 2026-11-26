# -*- coding: utf-8 -*-
"""d34 两个决定性补检

① 块 bootstrap VR：保留短滞后结构（块长 10/20 日），若 VR60 仍 <1，排除「微观结构伪影」
   （Hermes 指出的：shuffle 把买卖价差跳动/非同步交易全打掉了，对照太"干净"）
② 锚的比较：成本锚 vs 最高价锚 vs 均线锚 vs 纯反转
   —— 他的理论说锚是「持仓成本」。若三个锚的解释力一样，就只是笼统反转，归因不到成本
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = "G:/探讨集/_人性理论"
COLS = ["date", "close", "turnover"]
rng = np.random.default_rng(7)


def load(f):
    try:
        d = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        return None
    if len(d) < 900: return None
    d = d.sort_values("date")
    c = d["close"].to_numpy(float); tr = d["turnover"].to_numpy(float)
    dt = pd.to_datetime(d["date"])
    ok = np.isfinite(c) & (c > 0) & np.isfinite(tr) & (tr >= 0)
    if ok.sum() < 900: return None
    # 剔除上市前 250 日（Hermes Q3-①）
    first = np.where(ok)[0]
    if len(first) == 0: return None
    ok2 = ok.copy(); ok2[first[0]:first[0] + 250] = False
    c = np.where(ok2, c, np.nan); tr = np.where(ok2, tr, np.nan)
    return dict(code=os.path.basename(f)[:6], close=c, turn=tr,
                yr=dt.dt.year.to_numpy(), ok=ok2)


def vr_calc(r, q):
    r = r[np.isfinite(r)]; T = len(r)
    if T < max(200, 12 * q): return None
    dev = r - r.mean(); var1 = (dev @ dev) / (T - 1)
    if var1 <= 0: return None
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    rq = cs[q:] - cs[:-q]
    return ((rq @ rq) / (len(rq) - 1) / q) / var1


def block_boot(r, L, n=200):
    T = len(r); nb = T // L
    if nb < 4: return []
    base = np.arange(nb * L).reshape(nb, L)
    out = []
    for _ in range(n):
        o = rng.permutation(nb)
        idx = base[o].ravel()
        out.append(vr_calc(r[idx], 60))
    return [x for x in out if x is not None]


def partA_block(stocks, ns=400):
    """抽 ns 只做块 bootstrap（每个样本要重算 200 次）"""
    res = {"L10": [], "L20": [], "real": [], "iid": [], "iid_shuf": []}
    cand = [s for s in stocks if np.isfinite(s["close"]).sum() > 1200]
    pick = list(rng.choice(len(cand), min(ns, len(cand)), replace=False))
    for k in pick:
        s = cand[k]
        lp = np.log(s["close"]); r = np.diff(lp)
        r = r[np.isfinite(r)]
        if len(r) < 1000: continue
        v = vr_calc(r, 60)
        if v is None: continue
        res["real"].append(v)
        b10 = block_boot(r, 10, 120)
        b20 = block_boot(r, 20, 120)
        rs = rng.permutation(r)
        res["iid_shuf"].append(vr_calc(rs, 60))
        if b10: res["L10"].append(np.median(b10))
        if b20: res["L20"].append(np.median(b20))
        # iid 理论值：VR 应≈1
        res["iid"].append(1.0)
    out = {k: (float(np.median(v)), len(v)) for k, v in res.items() if len(v)}
    return out


def partB_anchor(stocks, win=250, h=60):
    """四个锚对未来 h 日收益的解释力（每票内部去均值后算相关）"""
    keys = ["cost", "high", "ma", "rev"]
    corr = {k: [] for k in keys}
    for s in stocks:
        c = s["close"]; tr = s["turn"]; n = len(c)
        if np.isfinite(c).sum() < 700: continue
        w = np.where(np.isfinite(tr), tr, 0.0) * np.where(np.isfinite(c), c, 0.0)
        ww = pd.Series(w).rolling(win, min_periods=120).sum().to_numpy()
        wt = pd.Series(np.where(np.isfinite(tr), tr, 0.0)).rolling(win, min_periods=120).sum().to_numpy()
        RP = np.full(n, np.nan); m = wt > 0; RP[m] = ww[m] / wt[m]
        HIP = pd.Series(c).rolling(win, min_periods=120).max().to_numpy()
        MA = pd.Series(c).rolling(win, min_periods=120).mean().to_numpy()
        f = np.full(n, np.nan); f[:-h] = c[h:] / c[:-h] - 1
        r5 = np.full(n, np.nan); r5[:-5] = c[5:] / c[:-5] - 1
        anchors = {"cost": c / RP - 1, "high": c / HIP - 1, "ma": c / MA - 1,
                   "rev": -np.log(pd.Series(c).pct_change(20).to_numpy() + 1)}
        for k in keys:
            x = anchors[k]
            mm = np.isfinite(x) & np.isfinite(f)
            if mm.sum() < 400: corr[k].append(np.nan); continue
            xx = x[mm] - x[mm].mean(); yy = f[mm] - f[mm].mean()
            sx = np.sqrt((xx @ xx) * (yy @ yy))
            corr[k].append((xx @ yy) / sx if sx > 0 else np.nan)
    out = {}
    for k in keys:
        v = np.array([z for z in corr[k] if np.isfinite(z)])
        if len(v) == 0: continue
        out[k] = dict(corr_med=float(np.median(v)), n=int(len(v)),
                      share_neg=float((v < 0).mean()),
                      r2_med=float(np.median(v) ** 2))
    return out


def main():
    t0 = time.time()
    fs = sorted(glob.glob(os.path.join(R, "*.parquet")))
    st = [s for s in (load(f) for f in fs) if s]
    print("loaded", len(st), "%.0fs" % (time.time() - t0), flush=True)
    print("\n===== A 块 bootstrap（保留短滞后） =====", flush=True)
    A = partA_block(st); print(json.dumps(A, ensure_ascii=False, indent=1), flush=True)
    print("\n===== B 四个锚的解释力 =====", flush=True)
    B = partB_anchor(st); print(json.dumps(B, ensure_ascii=False, indent=1), flush=True)
    json.dump({"A": A, "B": B}, open(os.path.join(OUT, "d34_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\ndone %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
