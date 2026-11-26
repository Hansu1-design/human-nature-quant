# -*- coding: utf-8 -*-
"""d33 排掉「制度/构成变化」这个竞争解释

d32 发现两件事都在单调变化：
  VR60 0.974 → 0.726 → 0.688（越来越均值回复）
  下跌时长中位 84 → 130.5 → 220 天（越来越长）
但同期 A 股：股票数 2000→5400、注册制、创业板涨跌幅 10%→20%、量化占比上升。
必须证明这不是「换了一批股票」造成的。

做法：
 ① 固定队列：只留 2010 年底前上市、样本 ≥3300 日的票（构成恒定）
 ② 市值分层：每个时代内按流通市值三分位，看趋势是否只在某一层出现
 ③ 用「同票」配对：同一批票在三个时代各自的 VR60，看差值
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = "G:/探讨集/_人性理论"
COLS = ["date", "close", "turnover", "outstanding_share"]
ERAS = [("2011-2015", 2011, 2015), ("2016-2020", 2016, 2020), ("2021-2025", 2021, 2025)]


def load(f):
    try:
        d = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        return None
    if len(d) < 900: return None
    d = d.sort_values("date")
    c = d["close"].to_numpy(float); tr = d["turnover"].to_numpy(float)
    sh = d["outstanding_share"].to_numpy(float)
    dt = pd.to_datetime(d["date"])
    ok = np.isfinite(c) & (c > 0) & np.isfinite(tr) & (tr >= 0)
    if ok.sum() < 900: return None
    return dict(code=os.path.basename(f)[:6], close=np.where(ok, c, np.nan),
                turn=np.where(ok, tr, np.nan), share=sh, date=dt.to_numpy(),
                yr=dt.dt.year.to_numpy(), ok=ok, first=int(dt.iloc[0].year))


def vr_calc(r, q):
    r = r[np.isfinite(r)]; T = len(r)
    if T < max(160, 12 * q): return None
    dev = r - r.mean(); var1 = (dev @ dev) / (T - 1)
    if var1 <= 0: return None
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    rq = cs[q:] - cs[:-q]
    return ((rq @ rq) / (len(rq) - 1) / q) / var1


def ep_len(c, dd=0.20, win=250, maxlen=500):
    hh = pd.Series(c).rolling(win, min_periods=60).max().to_numpy()
    draw = c / hh - 1
    inep = False; s = 0
    for i in range(len(c)):
        if not np.isfinite(draw[i]): continue
        if not inep:
            if draw[i] <= -dd: inep = True; s = i
        else:
            if draw[i] >= -dd / 2: return i - s
            if i - s >= maxlen: return maxlen
    return None


def run(stocks, tag, size_split=False):
    res = {e[0]: {"vr60": [], "len": [], "vr20": []} for e in ERAS}
    for s in stocks:
        lp = np.log(s["close"])
        r = np.full(len(lp), np.nan); r[1:] = np.diff(lp)
        for name, y0, y1 in ERAS:
            m = (s["yr"] >= y0) & (s["yr"] <= y1)
            if m.sum() < 800: continue
            rr = r[m]
            v = vr_calc(rr, 60)
            if v is not None: res[name]["vr60"].append(v)
            v2 = vr_calc(rr, 20)
            if v2 is not None: res[name]["vr20"].append(v2)
            seg = np.where(m, s["close"], np.nan)
            L = ep_len(seg)
            if L: res[name]["len"].append(L)
    out = {}
    for name, _, _ in ERAS:
        d = res[name]
        out[name] = dict(
            n_vr=len(d["vr60"]),
            vr60_med=float(np.median(d["vr60"])) if d["vr60"] else None,
            vr20_med=float(np.median(d["vr20"])) if d["vr20"] else None,
            n_len=len(d["len"]),
            len_med=float(np.median(d["len"])) if d["len"] else None,
            len_cv=float(np.std(d["len"], ddof=1) / np.mean(d["len"])) if len(d["len"]) > 10 else None,
        )
    return {tag: out}


def main():
    t0 = time.time()
    fs = sorted(glob.glob(os.path.join(R, "*.parquet")))
    allst = [s for s in (load(f) for f in fs) if s]
    print("loaded", len(allst), "%.0fs" % (time.time() - t0), flush=True)

    # ① 全样本
    r_all = run(allst, "全样本")

    # ② 固定队列：2010 年前上市 且 有效天数 >= 3300
    fixed = [s for s in allst if s["first"] <= 2010 and s["ok"].sum() >= 3300]
    print("固定队列", len(fixed), flush=True)
    r_fix = run(fixed, "固定队列2010")

    # ③ 市值分层（用 2016 年中位流通市值三分位，队列固定在三层内）
    def mcap(s):
        m = (s["yr"] == 2016)
        if m.sum() < 20: return None
        v = (s["close"][m] * s["share"][m])
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.median(v)) if len(v) else None
    mc = [(s, mcap(s)) for s in fixed]
    mc = [(s, v) for s, v in mc if v]
    val = np.array([v for _, v in mc])
    q1, q2 = np.percentile(val, [33, 67])
    small = [s for s, v in mc if v <= q1]
    big = [s for s, v in mc if v > q2]
    print("分层 小%d 大%d" % (len(small), len(big)), flush=True)
    r_sm = run(small, "小市值")
    r_bg = run(big, "大市值")

    out = {}
    for d in (r_all, r_fix, r_sm, r_bg):
        out.update(d)
    print("\n===== 稳健性 =====", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    json.dump(out, open(os.path.join(OUT, "d33_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\ndone %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
