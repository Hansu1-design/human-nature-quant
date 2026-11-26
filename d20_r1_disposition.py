# -*- coding: utf-8 -*-
"""
R1 处置效应：浮盈档换手 / 浮亏档换手 的年度序列（预注册坐标 1，见 README §5.1）

预注册（跑之前写死，不许事后改口）：
  统计量  R_y = median_{日内}( rt | 浮盈档 ) / median_{日内}( rt | 浮亏档 )
          rt = 个股当日换手 / 该股该年中位换手   （无量纲，消除个股与年度水平差）
  预期方向 随年份下降（机构占比升 -> 散户式处置效应减弱）

两个锚（口径分歧，两者必须同号才算判决）：
  A 成本锚：g = 复权价 / RP250 - 1，RP250 = 250 日换手率加权均价（近似平均持仓成本）
  B 高点锚：d = 复权价 / HI250 - 1，HI250 = 250 日复权最高价（预注册字面口径）
  A: 浮盈 g>+1%，浮亏 g<-1%，中间剔除
  B: 近高点 d>-5%，深跌 d<-30%

数据：G:/AStockData/supplement/daily_tencent_full（3,179 只，前复权，含 turnover）
输出：d20_result.json + d20_daily.parquet（日级聚合，供 bootstrap 复算）
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = os.path.dirname(os.path.abspath(__file__))
WIN = 250
REPAIR = -0.22          # 单日跌幅超此值判为除权/复牌缺口，做前后拼接修复
GAIN, LOSS = 0.01, -0.01
NEAR, FAR = -0.05, -0.30

def one(f):
    d = pq.read_table(f, columns=["date", "close", "turnover"]).to_pandas()
    if len(d) < WIN + 60:
        return None
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    c = d["close"].astype("float64").values
    t = d["turnover"].astype("float64").values
    ok = np.isfinite(c) & (c > 0)
    if ok.sum() < WIN + 60:
        return None

    # 修复除权缺口：把缺口日之前的价格整体缩放，使该日收益归零
    ret = np.ones(len(c))
    ret[1:] = c[1:] / c[:-1]
    bad = np.where(ret < (1 + REPAIR))[0]
    bad = bad[bad > 0]
    nrep = len(bad)
    k = np.ones(len(c))
    for i in bad:
        k[:i] *= ret[i]          # ret[i] 是当日的跳空比，把之前的价格乘上去
    p = c * k                    # 拼接后的连续价（当前段不动）

    s = pd.Series(p)
    hi = s.rolling(WIN, min_periods=WIN).max().values
    tw = pd.Series(t).clip(lower=0).fillna(0).values
    num = pd.Series(p * tw).rolling(WIN, min_periods=WIN).sum().values
    den = pd.Series(tw).rolling(WIN, min_periods=WIN).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = np.where(den > 0, num / den, np.nan)

    g = p / rp - 1.0
    dd = p / hi - 1.0

    yr = d["date"].dt.year.values
    df = pd.DataFrame({"date": d["date"].values, "yr": yr, "turn": t,
                       "g": g, "d": dd, "nrep": nrep})
    df = df[np.isfinite(df["g"]) & np.isfinite(df["d"]) & (df["turn"] > 0)]
    if df.empty:
        return None
    # 个股-年度中位换手 -> 相对换手 rt
    med = df.groupby("yr")["turn"].transform("median")
    df["rt"] = df["turn"] / med
    df = df[np.isfinite(df["rt"])]
    return df[["date", "yr", "rt", "g", "d"]], nrep

def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(R, "*.parquet")))
    parts, nrep_all, nstock = [], 0, 0
    tot_rows = 0
    for i, f in enumerate(files):
        try:
            out = one(f)
        except Exception:
            continue
        if out is None:
            continue
        df, nrep = out
        nrep_all += nrep
        nstock += 1
        tot_rows += len(df)
        parts.append(df)
        if (i + 1) % 400 == 0:
            print("  ...%d/%d  stocks=%d rows=%d  %.0fs" % (i + 1, len(files), nstock, tot_rows, time.time() - t0), flush=True)
    D = pd.concat(parts, ignore_index=True)
    del parts
    print("载入完成: %d 只 / %d 行 / 修复除权缺口 %d 处 / %.1fs" % (nstock, len(D), nrep_all, time.time() - t0), flush=True)

    # 日级聚合（每天的 G/L 档相对换手中位）
    def dayagg(mask, tag):
        x = D.loc[mask, ["date", "yr", "rt"]]
        a = x.groupby("date")["rt"].agg(["median", "mean", "size"]).reset_index()
        a.columns = ["date", "med_" + tag, "mean_" + tag, "n_" + tag]
        return a.set_index("date")

    GA = D["g"] > GAIN
    LA = D["g"] < LOSS
    NB = D["d"] > NEAR
    FB = D["d"] < FAR
    print("档位样本: A盈 %d / A亏 %d | B近 %d / B深 %d" % (GA.sum(), LA.sum(), NB.sum(), FB.sum()), flush=True)

    dg, dl = dayagg(GA, "G"), dayagg(LA, "L")
    dn, df_ = dayagg(NB, "N"), dayagg(FB, "F")
    DD = dg.join(dl, how="outer").join(dn, how="outer").join(df_, how="outer").sort_index()
    DD["yr"] = DD.index.year
    DD.to_parquet(os.path.join(OUT, "d20_daily.parquet"))

    def series(colG, colL, minn=20):
        g = DD[[colG]].dropna()
        l = DD[[colL]].dropna()
        rg = g.groupby(g.index.year)[colG].median()
        rl = l.groupby(l.index.year)[colL].median()
        ng = g.groupby(g.index.year)[colG].size()
        nl = l.groupby(l.index.year)[colL].size()
        s = (rg / rl).dropna()
        s = s[(ng.reindex(s.index) >= 5) & (nl.reindex(s.index) >= 5)]
        return s, rg, rl, ng, nl

    RA, gA, lA, ngA, nlA = series("med_G", "med_L")
    RB, gB, lB, ngB, nlB = series("med_N", "med_F")
    RA_mean, _, _, _, _ = series("mean_G", "mean_L")
    RB_mean, _, _, _, _ = series("mean_N", "mean_F")

    def fit(s):
        y = s.index.values.astype(float)
        x = s.values.astype(float)
        n = len(x)
        A = np.vstack([y, np.ones(n)]).T
        beta, *_ = np.linalg.lstsq(A, x, rcond=None)
        pred = A @ beta
        ss = ((x - pred) ** 2).sum()
        sst = ((x - x.mean()) ** 2).sum()
        r2 = 1 - ss / sst if sst > 0 else np.nan
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1] if n > 3 else np.nan
        return dict(slope=float(beta[0]), intercept=float(beta[1]), r2=float(r2),
                    ac1=float(ac1), n=int(n),
                    ac1_t=float(ac1 * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ac1 ** 2))) if n > 3 else None)

    fA, fB = fit(RA), fit(RB)

    # 日块安慰剂：年内把交易日随机分两半，各算 R_y，取斜率差的分布 -> 噪声尺度
    def placebo(nrep=300, seed=7):
        rng = np.random.default_rng(seed)
        yrs = sorted(DD["yr"].unique())
        res = []
        for _ in range(nrep):
            slopes = []
            for tag, (cG, cL) in {"A": ("med_G", "med_L"), "B": ("med_N", "med_F")}.items():
                xs, ys = [], []
                for y in yrs:
                    sub = DD[DD["yr"] == y]
                    if len(sub) < 40:
                        continue
                    pick = rng.random(len(sub)) < 0.5
                    a, b = sub[pick], sub[~pick]
                    if len(a) < 10 or len(b) < 10:
                        continue
                    r1 = a[cG].median() / a[cL].median()
                    r2 = b[cG].median() / b[cL].median()
                    if np.isfinite(r1) and np.isfinite(r2):
                        xs.append(y); ys.append((r1, r2))
                if len(xs) > 5:
                    xs = np.array(xs, float)
                    ys = np.array(ys)
                    A2 = np.vstack([xs, np.ones(len(xs))]).T
                    b1 = np.linalg.lstsq(A2, ys[:, 0], rcond=None)[0][0]
                    b2 = np.linalg.lstsq(A2, ys[:, 1], rcond=None)[0][0]
                    slopes.append((tag, b1, b2))
            res.append({t: (b1 - b2) for t, b1, b2 in slopes})
        out = {}
        for tag in ("A", "B"):
            v = np.array([r[tag] for r in res if tag in r])
            out[tag] = dict(sd_diff=float(v.std(ddof=1)), se_slope=float(v.std(ddof=1) / 2), n=int(len(v)))
        return out

    plc = placebo()

    # 稳健性：剔除 2015；分年样本量
    def fit_ex(s, ex):
        return fit(s[[y for y in s.index if y not in ex]])

    robust = {
        "A_ex2015": fit_ex(RA, {2015}),
        "B_ex2015": fit_ex(RB, {2015}),
        "A_mean": fit(RA_mean),
        "B_mean": fit(RB_mean),
    }

    # 水平：浮盈档/浮亏档各自相对换手（是否都在 1 附近？）
    lvl = {
        "A_gain_med": {int(k): float(v) for k, v in gA.items()},
        "A_loss_med": {int(k): float(v) for k, v in lA.items()},
        "B_near_med": {int(k): float(v) for k, v in gB.items()},
        "B_far_med": {int(k): float(v) for k, v in lB.items()},
        "n_day_gain": {int(k): int(v) for k, v in ngA.items()},
        "n_day_loss": {int(k): int(v) for k, v in nlA.items()},
    }

    res = dict(
        meta=dict(n_stock=nstock, n_row=int(len(D)), n_repair=int(nrep_all),
                  window=WIN, span=[str(D["date"].min())[:10], str(D["date"].max())[:10]],
                  gain=GAIN, loss=LOSS, near=NEAR, far=FAR),
        R_A={int(k): float(v) for k, v in RA.items()},
        R_B={int(k): float(v) for k, v in RB.items()},
        fit_A=fA, fit_B=fB, placebo=plc, robust=robust, levels=lvl,
    )
    json.dump(res, open(os.path.join(OUT, "d20_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n=== 年度序列 ===")
    print("年    R_A(成本锚)  R_B(高点锚)")
    for y in sorted(set(RA.index) | set(RB.index)):
        print("%d    %s   %s" % (y, ("%.4f" % RA[y]) if y in RA.index else "  --  ",
                                 ("%.4f" % RB[y]) if y in RB.index else "  --  "))
    print("\n=== 拟合 ===")
    for tag, f in (("A", fA), ("B", fB)):
        se = plc[tag]["se_slope"]
        print("%s: slope=%+.6f/yr  t=%+.2f  r2=%.3f  ac1=%+.3f(t=%.2f)  15y累计=%+.4f"
              % (tag, f["slope"], f["slope"] / se if se > 0 else np.nan, f["r2"],
                 f["ac1"], f["ac1_t"], f["slope"] * 14))
    print("\n安慰剂 SE:", {k: round(v["se_slope"], 6) for k, v in plc.items()})
    print("稳健:", {k: round(v["slope"], 5) for k, v in robust.items()})
    print("\n水平: A盈 %s\n      A亏 %s" % ({k: round(v, 3) for k, v in list(lvl["A_gain_med"].items())[:5]},
                                          {k: round(v, 3) for k, v in list(lvl["A_loss_med"].items())[:5]}))

    # 判决
    def verdict(f, se, tag):
        t = f["slope"] / se if se > 0 else np.nan
        if abs(t) < 2:
            return "%s: 持平（|t|=%.2f<2）-> 损失厌恶未见趋势" % (tag, abs(t))
        return "%s: %s 显著（t=%+.2f）" % (tag, "下降" if f["slope"] < 0 else "上升", t)

    print("\n=== 判决 ===")
    print(verdict(fA, plc["A"]["se_slope"], "A"))
    print(verdict(fB, plc["B"]["se_slope"], "B"))
    print("耗时 %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
