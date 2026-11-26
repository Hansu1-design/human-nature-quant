# -*- coding: utf-8 -*-
"""d31 人性理论三条结构命题（修掉 d30 的三处设计缺陷）

A 价值锚：方差比检验 VR(q)（Lo-MacKinlay），shuffle 随机游走对照
B 解套抛压：上穿 250 日换手加权成本线，**按当日涨幅分箱配对**，排除短期反转混杂
C 时长规律：每票只取首段下跌期（去相关）+ 随机游走对照
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = "G:/探讨集/_人性理论"
COLS = ["date", "high", "low", "close", "volume", "turnover"]
MAXN = int(os.environ.get("MAXN", "0"))
NSEED = 7
rng = np.random.default_rng(NSEED)


def load(f):
    try:
        d = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        return None
    if len(d) < 700:
        return None
    d = d.sort_values("date")
    c = d["close"].to_numpy(float); hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float); v = d["volume"].to_numpy(float)
    tr = d["turnover"].to_numpy(float)
    ok = np.isfinite(c) & (c > 0) & np.isfinite(tr) & (tr >= 0)
    if ok.sum() < 700:
        return None
    return dict(code=os.path.basename(f)[:6], close=np.where(ok, c, np.nan),
                high=np.where(ok, hi, np.nan), low=np.where(ok, lo, np.nan),
                vol=np.where(ok, v, np.nan), turn=np.where(ok, tr, np.nan), ok=ok)


def longest_finite(a):
    m = np.isfinite(a)
    if not m.any():
        return np.array([])
    idx = np.where(m)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, brk + 1)
    seg = max(segs, key=len)
    return a[seg]


# ---------------- A 方差比检验 ----------------
def vr(lp, q):
    r = np.diff(lp)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < max(120, 12 * q):
        return None
    mu = r.mean()
    dev = r - mu
    var1 = (dev @ dev) / (T - 1)
    if var1 <= 0:
        return None
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    rq = cs[q:] - cs[:-q]           # 重叠 q 期累计偏离
    varq = (rq @ rq) / (len(rq) - 1) / q
    VR = varq / var1
    z = (VR - 1) / np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * T))
    return VR, z, T


def part_A(stocks, qs=(5, 20, 60)):
    real = {q: [] for q in qs}; shuf = {q: [] for q in qs}
    for s in stocks:
        lp = longest_finite(np.log(s["close"]))
        if len(lp) < 800:
            continue
        r = np.diff(lp)
        for q in qs:
            a = vr(lp, q)
            if a: real[q].append(a[0])
            # 随机游走对照：打乱收益率顺序
            rs = rng.permutation(r)
            a2 = vr(np.concatenate([[0.0], np.cumsum(rs)]), q)
            if a2: shuf[q].append(a2[0])
    out = {"n_stock": len(real[qs[0]])}
    for q in qs:
        v = np.array(real[q]); w = np.array(shuf[q])
        if len(v) == 0: continue
        # 单边：VR<1 为均值回复
        zz = (v.mean() - 1) / (v.std(ddof=1) / np.sqrt(len(v)))
        out[f"q{q}"] = dict(VR_med=float(np.median(v)), VR_mean=float(v.mean()),
                            share_lt1=float((v < 1).mean()), z_crossstock=float(zz),
                            shuf_VR_med=float(np.median(w)) if len(w) else None,
                            edge=float(np.median(v) - np.median(w)) if len(w) else None,
                            n=int(len(v)))
    return out


# ---------------- B 解套抛压（涨幅分箱配对） ----------------
def part_B(stocks, win=250, pre_dn=0.98):
    recs = []
    for s in stocks:
        c = s["close"]; tr = s["turn"]; n = len(c)
        if np.isfinite(c).sum() < 500: continue
        # 250 日换手加权均价（固定窗口，不含当日）
        w = np.where(np.isfinite(tr), tr, 0.0) * np.where(np.isfinite(c), c, 0.0)
        ww = pd.Series(w).rolling(win, min_periods=120).sum().to_numpy()
        wc = pd.Series(np.where(np.isfinite(tr), tr, 0.0)).rolling(win, min_periods=120).sum().to_numpy()
        RP = np.full(n, np.nan)
        okk = wc > 0
        RP[okk] = ww[okk] / wc[okk]
        RP = np.concatenate([[np.nan], RP[:-1]])   # 用 t-1 的锚
        r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
        f20 = np.full(n, np.nan); f20[:-20] = c[20:] / c[:-20] - 1
        f60 = np.full(n, np.nan); f60[:-60] = c[60:] / c[:-60] - 1
        for i in range(win + 2, n - 61):
            if not (np.isfinite(RP[i]) and np.isfinite(c[i]) and np.isfinite(c[i - 1]) and np.isfinite(r[i])):
                continue
            if r[i] <= 0.005 or r[i] >= 0.20:   # 只看有效上涨，剔涨停级异常
                continue
            unlock = 1 if (c[i - 1] < RP[i] * pre_dn and c[i] >= RP[i]) else 0
            recs.append((s["code"], i, unlock, r[i], tr[i] * 100 if np.isfinite(tr[i]) else np.nan,
                         f20[i] * 100 if np.isfinite(f20[i]) else np.nan,
                         f60[i] * 100 if np.isfinite(f60[i]) else np.nan))
    df = pd.DataFrame(recs, columns=["code", "i", "unlock", "ret", "turn", "f20", "f60"])
    df["bin"] = pd.cut(df["ret"], [0.005, 0.02, 0.04, 0.07, 0.20],
                       labels=["0.5-2%", "2-4%", "4-7%", "7-20%"])
    out = {"n_unlock": int(df["unlock"].sum()), "n_ctrl": int((1 - df["unlock"]).sum())}
    tab = []
    for b in df["bin"].cat.categories:
        g = df[df["bin"] == b]
        u = g[g["unlock"] == 1]; k = g[g["unlock"] == 0]
        if len(u) < 30 or len(k) < 30: continue
        tab.append(dict(bin=str(b), n_u=int(len(u)), n_c=int(len(k)),
                        turn_u=float(u["turn"].median()), turn_c=float(k["turn"].median()),
                        f20_u=float(u["f20"].median()), f20_c=float(k["f20"].median()),
                        f20_up_u=float((u["f20"] > 0).mean()), f20_up_c=float((k["f20"] > 0).mean()),
                        f60_u=float(u["f60"].median()), f60_c=float(k["f60"].median()),
                        diff20=float(u["f20"].median() - k["f20"].median())))
    out["by_bin"] = tab
    # 总体（分箱加权差）
    if tab:
        wsum = sum(t["n_u"] for t in tab)
        out["pooled_diff20"] = sum(t["diff20"] * t["n_u"] for t in tab) / wsum if wsum else None
        out["pooled_turn_ratio"] = (sum(t["turn_u"] * t["n_u"] for t in tab) /
                                   max(1e-9, sum(t["turn_c"] * t["n_u"] for t in tab)))
    df.to_parquet(os.path.join(OUT, "d31_B.parquet"), index=False)
    return out


# ---------------- C 下跌时长（每票首段 + 随机游走对照） ----------------
def episodes_from_close(c, dd=0.20, win=250, maxlen=500, first_only=True):
    n = len(c)
    hh = pd.Series(c).rolling(win, min_periods=60).max().to_numpy()
    draw = c / hh - 1
    lens = []; cens = []
    inep = False; start = 0
    for i in range(n):
        if not np.isfinite(draw[i]): continue
        if not inep:
            if draw[i] <= -dd:
                inep = True; start = i
        else:
            if draw[i] >= -dd / 2:
                lens.append(i - start); cens.append(0); inep = False
                if first_only: break
            elif i - start >= maxlen:
                lens.append(maxlen); cens.append(1); inep = False
                if first_only: break
    return lens, cens


def haz_curve(L, C, maxlen=500, bs=30):
    nb = int(np.ceil(maxlen / bs)); hz = []
    for k in range(nb):
        lo = k * bs
        at_risk = int(np.sum(L > lo))
        if at_risk < 40: hz.append(None); continue
        hz.append(int(np.sum((L > lo) & (L <= lo + bs) & (C == 0))) / at_risk)
    return hz


def part_C(stocks, dd=0.20, maxlen=500):
    Lr, Cr = [], []
    Ls, Cs = [], []
    for s in stocks:
        c = s["close"]
        if np.isfinite(c).sum() < 700: continue
        a, b = episodes_from_close(c, dd, maxlen=maxlen, first_only=True)
        Lr += a; Cr += b
        # 随机游走对照：同漂移同波动，打乱后重建
        lp = longest_finite(np.log(c))
        if len(lp) > 700:
            r = np.diff(lp); rs = rng.permutation(r)
            cs2 = np.exp(lp[0] + np.concatenate([[0.0], np.cumsum(rs)]))
            a2, b2 = episodes_from_close(cs2, dd, maxlen=maxlen, first_only=True)
            Ls += a2; Cs += b2
    L = np.array(Lr, float); Cc = np.array(Cr); L2 = np.array(Ls, float); C2 = np.array(Cs)
    out = {"n_episode": int(len(L)), "n_censored": int(Cc.sum()) if len(Cc) else 0}
    if len(L):
        out.update(len_mean=float(L.mean()), len_med=float(np.median(L)),
                   len_p25=float(np.percentile(L, 25)), len_p75=float(np.percentile(L, 75)),
                   cv=float(L.std(ddof=1) / L.mean()),
                   share_le60=float((L <= 60).mean()), share_ge250=float((L >= 250).mean()),
                   hazard=haz_curve(L, Cc, maxlen))
        h = np.array([x for x in out["hazard"] if x is not None])
        if len(h) >= 4:
            k = max(2, len(h) // 3)
            out["haz_early"] = float(h[:k].mean()); out["haz_late"] = float(h[-k:].mean())
            out["haz_ratio"] = out["haz_late"] / out["haz_early"] if h[:k].mean() > 0 else None
    if len(L2):
        out["rw"] = dict(n=int(len(L2)), len_med=float(np.median(L2)),
                         cv=float(L2.std(ddof=1) / L2.mean()),
                         hazard=haz_curve(L2, C2, maxlen)[:10])
    return out


def main():
    t0 = time.time()
    fs = sorted(glob.glob(os.path.join(R, "*.parquet")))
    if MAXN:
        fs = list(rng.choice(fs, min(MAXN, len(fs)), replace=False))
    print("files", len(fs), flush=True)
    st = [s for s in (load(f) for f in fs) if s]
    print("loaded", len(st), "%.0fs" % (time.time() - t0), flush=True)

    print("\n===== A 方差比检验 =====", flush=True)
    A = part_A(st); print(json.dumps(A, ensure_ascii=False, indent=1), flush=True)
    print("\n===== B 解套抛压（涨幅分箱） =====", flush=True)
    B = part_B(st); print(json.dumps(B, ensure_ascii=False, indent=1), flush=True)
    print("\n===== C 下跌时长 =====", flush=True)
    C = part_C(st); print(json.dumps(C, ensure_ascii=False, indent=1), flush=True)

    json.dump({"A": A, "B": B, "C": C, "n_stock": len(st),
               "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(OUT, "d31_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\ndone %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
