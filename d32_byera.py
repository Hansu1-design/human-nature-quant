# -*- coding: utf-8 -*-
"""d32 把 d31 的三条结果做成「能不能叫人性常数」的判决

1) A 分三个时代算 VR(q)：均值回复的强度本身变不变
2) C 分年算下跌期中位与 CV：那个「123 天」是常数还是会漂
3) B 补「拉扯」检验：解套后的路径效率（他说的「涨了就卖、来回拉扯」）
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = "G:/探讨集/_人性理论"
COLS = ["date", "close", "high", "low", "volume", "turnover"]
rng = np.random.default_rng(7)


def load(f):
    try:
        d = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        return None
    if len(d) < 700: return None
    d = d.sort_values("date")
    c = d["close"].to_numpy(float); tr = d["turnover"].to_numpy(float)
    dt = pd.to_datetime(d["date"]).to_numpy()
    ok = np.isfinite(c) & (c > 0) & np.isfinite(tr) & (tr >= 0)
    if ok.sum() < 700: return None
    return dict(code=os.path.basename(f)[:6], close=np.where(ok, c, np.nan),
                turn=np.where(ok, tr, np.nan), date=dt, ok=ok)


def vr_calc(r, q):
    r = r[np.isfinite(r)]
    T = len(r)
    if T < max(160, 12 * q): return None
    dev = r - r.mean()
    var1 = (dev @ dev) / (T - 1)
    if var1 <= 0: return None
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    rq = cs[q:] - cs[:-q]
    VR = ((rq @ rq) / (len(rq) - 1) / q) / var1
    z = (VR - 1) / np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * T))
    return VR, z


ERAS = [("2011-2015", 2011, 2015), ("2016-2020", 2016, 2020), ("2021-2025", 2021, 2025)]


def A_by_era(stocks):
    res = {e[0]: {5: [], 20: [], 60: []} for e in ERAS}
    for s in stocks:
        lp = np.log(s["close"]); yr = pd.to_datetime(s["date"]).year
        r = np.full(len(lp), np.nan); r[1:] = np.diff(lp)
        for name, y0, y1 in ERAS:
            m = (yr >= y0) & (yr <= y1) & np.isfinite(r)
            if m.sum() < 700: continue
            rr = r[m]
            for q in (5, 20, 60):
                a = vr_calc(rr, q)
                if a: res[name][q].append(a[0])
    out = {}
    for name, _, _ in ERAS:
        out[name] = {f"q{q}": dict(med=float(np.median(v)), n=len(v),
                                   share_lt1=float((np.array(v) < 1).mean()))
                     for q, v in res[name].items() if len(v) > 50}
    return out


def C_by_year(stocks, dd=0.20, win=250, maxlen=500):
    rows = []
    for s in stocks:
        c = s["close"]; dt = s["date"]
        if np.isfinite(c).sum() < 700: continue
        hh = pd.Series(c).rolling(win, min_periods=60).max().to_numpy()
        draw = c / hh - 1
        inep = False; start = 0; start_i = 0
        for i in range(len(c)):
            if not np.isfinite(draw[i]): continue
            if not inep:
                if draw[i] <= -dd: inep = True; start_i = i
            else:
                if draw[i] >= -dd / 2:
                    rows.append((str(pd.Timestamp(dt[start_i]).year), i - start_i, 0,
                                 c[start_i] / hh[start_i] - 1))
                    inep = False; break
                elif i - start_i >= maxlen:
                    rows.append((str(pd.Timestamp(dt[start_i]).year), maxlen, 1,
                                 c[start_i] / hh[start_i] - 1))
                    inep = False; break
    df = pd.DataFrame(rows, columns=["yr", "len", "cens", "dd0"])
    out = {"n": int(len(df))}
    per = {}
    for y, g in df.groupby("yr"):
        g2 = g[g["cens"] == 0]
        if len(g2) < 25: continue
        per[y] = dict(n=int(len(g2)), med=float(g2["len"].median()),
                      cv=float(g2["len"].std(ddof=1) / g2["len"].mean()),
                      p75=float(g2["len"].quantile(.75)),
                      dd_med=float((g2["dd0"] * 100).median()))
    out["by_year"] = dict(sorted(per.items()))
    # 时代合并
    era = {}
    for name, y0, y1 in ERAS:
        g = df[(df["yr"].astype(int) >= y0) & (df["yr"].astype(int) <= y1) & (df["cens"] == 0)]
        if len(g) < 60: continue
        era[name] = dict(n=int(len(g)), med=float(g["len"].median()),
                         cv=float(g["len"].std(ddof=1) / g["len"].mean()))
    out["by_era"] = era
    df.to_parquet(os.path.join(OUT, "d32_C.parquet"), index=False)
    return out


def B_path_eff(stocks, win=250, pre_dn=0.98, hz=20):
    """解套后 hz 日的路径效率：|净变动| / 累计绝对变动。低 = 拉扯"""
    rec = []
    for s in stocks:
        c = s["close"]; tr = s["turn"]; n = len(c)
        if np.isfinite(c).sum() < 500: continue
        w = np.where(np.isfinite(tr), tr, 0.0) * np.where(np.isfinite(c), c, 0.0)
        ww = pd.Series(w).rolling(win, min_periods=120).sum().to_numpy()
        wt = pd.Series(np.where(np.isfinite(tr), tr, 0.0)).rolling(win, min_periods=120).sum().to_numpy()
        RP = np.full(n, np.nan); m = wt > 0; RP[m] = ww[m] / wt[m]
        RP = np.concatenate([[np.nan], RP[:-1]])
        r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
        for i in range(win + 2, n - hz - 2):
            if not (np.isfinite(RP[i]) and np.isfinite(c[i]) and np.isfinite(c[i - 1]) and np.isfinite(r[i])):
                continue
            if r[i] <= 0.005 or r[i] >= 0.20: continue
            seg = c[i:i + hz + 1]
            if not np.all(np.isfinite(seg)): continue
            path = np.abs(np.diff(seg)).sum()
            if path <= 0: continue
            eff = abs(seg[-1] - seg[0]) / path
            unlock = 1 if (c[i - 1] < RP[i] * pre_dn and c[i] >= RP[i]) else 0
            rec.append((unlock, r[i], eff, abs(seg[-1] / seg[0] - 1) * 100))
    df = pd.DataFrame(rec, columns=["unlock", "ret", "eff", "netmove"])
    df["bin"] = pd.cut(df["ret"], [0.005, 0.02, 0.04, 0.07, 0.20],
                       labels=["0.5-2%", "2-4%", "4-7%", "7-20%"])
    out = []
    for b in df["bin"].cat.categories:
        g = df[df["bin"] == b]
        u = g[g["unlock"] == 1]; k = g[g["unlock"] == 0]
        if len(u) < 30 or len(k) < 30: continue
        out.append(dict(bin=str(b), n_u=int(len(u)),
                        eff_u=float(u["eff"].median()), eff_c=float(k["eff"].median()),
                        eff_diff=float(u["eff"].median() - k["eff"].median()),
                        move_u=float(u["netmove"].median()), move_c=float(k["netmove"].median())))
    return {"by_bin": out,
            "pooled_eff_diff": (sum(x["eff_diff"] * x["n_u"] for x in out) /
                                max(1, sum(x["n_u"] for x in out))) if out else None}


def main():
    t0 = time.time()
    fs = sorted(glob.glob(os.path.join(R, "*.parquet")))
    st = [s for s in (load(f) for f in fs) if s]
    print("loaded", len(st), "%.0fs" % (time.time() - t0), flush=True)

    print("\n===== A 分时代 VR =====", flush=True)
    A = A_by_era(st); print(json.dumps(A, ensure_ascii=False, indent=1), flush=True)
    print("\n===== C 分年下跌时长 =====", flush=True)
    C = C_by_year(st); print(json.dumps(C, ensure_ascii=False, indent=1), flush=True)
    print("\n===== B 拉扯（路径效率） =====", flush=True)
    B = B_path_eff(st); print(json.dumps(B, ensure_ascii=False, indent=1), flush=True)

    json.dump({"A": A, "B": B, "C": C, "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(OUT, "d32_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\ndone %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
