# -*- coding: utf-8 -*-
"""
R1 v2：把「盈亏状态」滞后一天，切断同期内生性。

d20 的缺陷：g_t = close_t/RP250_t - 1，当天状态含当天收盘价 -> 「浮盈日」≈「上涨日」，
换手与当日涨跌同向，测出的是同期共振，不是处置效应（决策在状态之后）。

v2 预注册（跑前写死）：
  主检验 P：状态取 t-1 日收盘，换手取 t 日；rt 按「个股-年度中位」归一；
             R_y = median_日(rt|浮盈, t-1) / median_日(rt|浮亏, t-1)
  稳健 R1：rt 再除以「当日全市场横截面中位」——剔除大盘放量/缩量的共同因子
  稳健 R2：固定队列——只保留 2010-03-01 前已上市的股票（剔除新股//板块结构变化）
  稳健 R3：用均值代替中位数
  判决规则：主检验与三条稳健必须同号且 |t|>2 才算「有趋势」，否则「不可判」。
  预期方向（README §5.1 写死）：随年份下降。
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = os.path.dirname(os.path.abspath(__file__))
WIN = 250
REPAIR = -0.22
GAIN, LOSS = 0.01, -0.01
NEAR, FAR = -0.05, -0.30

def one(f):
    code = os.path.basename(f).split(".")[0]
    d = pq.read_table(f, columns=["date", "close", "turnover"]).to_pandas()
    if len(d) < WIN + 90:
        return None
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    c = d["close"].astype("float64").values
    t = d["turnover"].astype("float64").values
    if (np.isfinite(c) & (c > 0)).sum() < WIN + 90:
        return None
    ret = np.ones(len(c)); ret[1:] = c[1:] / c[:-1]
    bad = np.where(ret < (1 + REPAIR))[0]; bad = bad[bad > 0]
    k = np.ones(len(c))
    for i in bad:
        k[:i] *= ret[i]
    p = c * k
    s = pd.Series(p)
    hi = s.rolling(WIN, min_periods=WIN).max().values
    tw = pd.Series(t).clip(lower=0).fillna(0).values
    num = pd.Series(p * tw).rolling(WIN, min_periods=WIN).sum().values
    den = pd.Series(tw).rolling(WIN, min_periods=WIN).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = np.where(den > 0, num / den, np.nan)
        g = p / rp - 1.0
        dd = p / hi - 1.0
    df = pd.DataFrame({"date": d["date"].values, "turn": t, "g": g, "d": dd})
    df["g_l"] = df["g"].shift(1)      # 昨日收盘时的盈亏状态
    df["d_l"] = df["d"].shift(1)
    df = df[np.isfinite(df["g_l"]) & np.isfinite(df["d_l"]) & (df["turn"] > 0)]
    if df.empty:
        return None
    df["code"] = code
    df["first"] = d["date"].iloc[0]
    return df

def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(R, "*.parquet")))
    parts = []
    for i, f in enumerate(files):
        try:
            o = one(f)
        except Exception:
            continue
        if o is not None:
            parts.append(o)
    D = pd.concat(parts, ignore_index=True)
    del parts
    print("载入 %d 行 / %d 只 / %.0fs" % (len(D), D["code"].nunique(), time.time() - t0), flush=True)

    # 归一 1：个股-年度中位换手
    D["yr"] = D["date"].dt.year
    med_sy = D.groupby(["code", "yr"])["turn"].transform("median")
    D["rt_y"] = D["turn"] / med_sy
    # 归一 2：当日全市场横截面中位换手
    med_d = D.groupby("date")["turn"].transform("median")
    D["rt_d"] = D["turn"] / med_d
    D = D[np.isfinite(D["rt_y"]) & np.isfinite(D["rt_d"]) & (D["rt_y"] > 0)]
    print("归一完成，有效 %d 行 %.0fs" % (len(D), time.time() - t0), flush=True)

    old = D["first"] <= pd.Timestamp("2010-03-01")
    print("固定队列（2010-03 前上市）行数 %d / %d" % (old.sum(), len(D)), flush=True)

    def build(sub, cG, cL, col, agg="median"):
        a = sub.groupby([sub["date"], sub["yr"]])[col].agg(agg).unstack()  # 行=日期
        # 分档：在同一张表上按掩码筛
        return a

    def series(sub, gmask, lmask, col, agg="median"):
        G = sub.loc[gmask, ["date", "yr", col]]
        L = sub.loc[lmask, ["date", "yr", col]]
        f = "median" if agg == "median" else "mean"
        g = G.groupby("date")[col].agg(f)
        l = L.groupby("date")[col].agg(f)
        dd = pd.DataFrame({"G": g, "L": l}).dropna()
        if dd.empty:
            return pd.Series(dtype=float)
        dd["yr"] = dd.index.year
        r = dd.groupby("yr").apply(lambda x: x["G"].median() / x["L"].median())
        n = dd.groupby("yr").size()
        return r[n >= 40]

    def fit(s):
        y = np.array(s.index, float); x = np.array(s.values, float); n = len(x)
        if n < 6:
            return dict(slope=np.nan, n=n)
        A = np.vstack([y, np.ones(n)]).T
        b = np.linalg.lstsq(A, x, rcond=None)[0]
        pr = A @ b
        r2 = 1 - ((x - pr) ** 2).sum() / max(1e-12, ((x - x.mean()) ** 2).sum())
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1] if n > 3 else np.nan
        return dict(slope=float(b[0]), r2=float(r2), ac1=float(ac1), n=n,
                    ac1_t=float(ac1 * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ac1 ** 2))))

    # 日块安慰剂（年内随机分半日，估斜率噪声）
    def placebo(sub, gmask, lmask, col, nrep=300, seed=11):
        rng = np.random.default_rng(seed)
        G = sub.loc[gmask, ["date", col]]; L = sub.loc[lmask, ["date", col]]
        g = G.groupby("date")[col].median(); l = L.groupby("date")[col].median()
        dd = pd.DataFrame({"G": g, "L": l}).dropna(); dd["yr"] = dd.index.year
        out = []
        for _ in range(nrep):
            xs, v1, v2 = [], [], []
            for y, x in dd.groupby("yr"):
                if len(x) < 40:
                    continue
                m = rng.random(len(x)) < 0.5
                a, b = x[m], x[~m]
                if len(a) < 10 or len(b) < 10:
                    continue
                xs.append(y); v1.append(a["G"].median() / a["L"].median())
                v2.append(b["G"].median() / b["L"].median())
            if len(xs) < 6:
                continue
            A = np.vstack([np.array(xs, float), np.ones(len(xs))]).T
            out.append(np.linalg.lstsq(A, np.array(v1), rcond=None)[0][0]
                       - np.linalg.lstsq(A, np.array(v2), rcond=None)[0][0])
        v = np.array(out)
        return float(v.std(ddof=1) / 2)

    cfg = [
        ("P_A_主", D, D["g_l"] > GAIN, D["g_l"] < LOSS, "rt_y", "median"),
        ("P_B_主", D, D["d_l"] > NEAR, D["d_l"] < FAR, "rt_y", "median"),
        ("S1_A_日归", D, D["g_l"] > GAIN, D["g_l"] < LOSS, "rt_d", "median"),
        ("S1_B_日归", D, D["d_l"] > NEAR, D["d_l"] < FAR, "rt_d", "median"),
        ("S2_A_老队列", D[old], D.loc[old, "g_l"] > GAIN, D.loc[old, "g_l"] < LOSS, "rt_y", "median"),
        ("S2_B_老队列", D[old], D.loc[old, "d_l"] > NEAR, D.loc[old, "d_l"] < FAR, "rt_y", "median"),
        ("S3_A_均值", D, D["g_l"] > GAIN, D["g_l"] < LOSS, "rt_y", "mean"),
        ("S3_B_均值", D, D["d_l"] > NEAR, D.loc[:, "d_l"] < FAR, "rt_y", "mean"),
    ]

    out = {"meta": dict(n_row=int(len(D)), n_stock=int(D["code"].nunique()),
                        span=[str(D["date"].min())[:10], str(D["date"].max())[:10]],
                        n_old=int(old.sum()))}
    print("\n%-14s %10s %8s %8s %8s  %s" % ("口径", "slope/yr", "t", "r2", "ac1", "序列首末"))
    for name, sub, gm, lm, col, agg in cfg:
        s = series(sub, gm, lm, col, agg)
        if len(s) < 6:
            print("%-14s 样本不足 (%d 年)" % (name, len(s))); continue
        f = fit(s)
        se = placebo(sub, gm, lm, col)
        t = f["slope"] / se if se > 0 else np.nan
        out[name] = dict(series={int(k): float(v) for k, v in s.items()}, **f,
                         se=se, t=float(t))
        print("%-14s %+10.5f %+8.2f %8.3f %+8.3f  %.3f -> %.3f"
              % (name, f["slope"], t, f["r2"], f["ac1"], s.iloc[0], s.iloc[-1]), flush=True)

    json.dump(out, open(os.path.join(OUT, "d21_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n耗时 %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
