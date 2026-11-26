# -*- coding: utf-8 -*-
"""d30 验证「人性理论」的三条结构命题（不是预测检验）

命题 A（价值锚）：「回归他本来的价值」 —— 价格对长期中枢的偏离是否被修正，半衰期多长
命题 B（解套抛压）：「高位接盘的，涨了就卖」「冲进来他们就跑」—— 价格反弹到持仓成本线时的抛压
命题 C（时长众数）：「人性不变，只能撑这么长时间」—— 下跌持续期的 hazard 是否非常数

全部用本机 daily_tencent_full（前复权，含 turnover）。
"""
import os, glob, json, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = "G:/探讨集/_人性理论"
COLS = ["date", "high", "low", "close", "volume", "turnover"]
MAXN = int(os.environ.get("MAXN", "0"))   # 0 = 全量


def load(f):
    try:
        d = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        return None
    if len(d) < 700:
        return None
    d = d.sort_values("date")
    c = d["close"].to_numpy(float)
    hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    v = d["volume"].to_numpy(float)
    tr = d["turnover"].to_numpy(float)
    ok = np.isfinite(c) & (c > 0) & np.isfinite(tr) & (tr >= 0)
    if ok.sum() < 700:
        return None
    c2 = np.where(ok, c, np.nan)
    return dict(code=os.path.basename(f)[:6], close=c2,
                high=np.where(ok, hi, np.nan), low=np.where(ok, lo, np.nan),
                vol=np.where(ok, v, np.nan), turn=np.where(ok, tr, np.nan),
                ok=ok, n=int(ok.sum()))


def clean_log(c):
    """前复权价取对数；跳空 >25% 视为缺口，标记为 nan 不参与"""
    lp = np.log(c)
    d = np.abs(np.diff(lp))
    jump = np.zeros(len(lp), bool)
    jump[1:] = d > 0.25
    lp2 = lp.copy()
    lp2[jump] = np.nan
    return lp2


# ---------------- 命题 A：价值锚 / 均值回复 ----------------
def part_A(stocks, hs=(20, 60, 120, 250), win=250, nshuf=400, seed=7):
    rng = np.random.default_rng(seed)
    B = {h: [] for h in hs}
    Bs = {h: [] for h in hs}      # shuffle 对照
    cnt = 0
    for k, s in enumerate(stocks):
        lp = clean_log(s["close"])
        ser = pd.Series(lp)
        ma = ser.rolling(win, min_periods=200).mean().shift(1).to_numpy()  # 不含当期
        x = lp - ma
        if np.isfinite(x).sum() < 400:
            continue
        cnt += 1
        for h in hs:
            f = np.full(len(lp), np.nan)
            f[:-h] = lp[h:] - lp[:-h]
            m = np.isfinite(x) & np.isfinite(f)
            if m.sum() < 300:
                B[h].append(np.nan); Bs[h].append(np.nan); continue
            xx = x[m]; yy = f[m]
            xx = xx - xx.mean(); yy = yy - yy.mean()
            sxx = (xx * xx).sum()
            B[h].append((xx * yy).sum() / sxx if sxx > 0 else np.nan)
            # shuffle 对照：打乱日收益顺序重建价格路径
            if nshuf and k < nshuf:
                r = np.diff(lp[m])
                r = r[np.isfinite(r)]
                if len(r) > 300:
                    r2 = rng.permutation(r)
                    lps = np.concatenate([[0.0], np.cumsum(r2)])
                    xs = lps - pd.Series(lps).rolling(win, min_periods=200).mean().shift(1).to_numpy()
                    ff = np.full(len(lps), np.nan); ff[:-h] = lps[h:] - lps[:-h]
                    mm = np.isfinite(xs) & np.isfinite(ff)
                    if mm.sum() > 300:
                        a = xs[mm] - xs[mm].mean(); b = ff[mm] - ff[mm].mean()
                        sxx2 = (a * a).sum()
                        Bs[h].append((a * b).sum() / sxx2 if sxx2 > 0 else np.nan)
    out = {"n_stock": cnt, "win": win}
    for h in hs:
        v = np.array([z for z in B[h] if np.isfinite(z)])
        vs = np.array([z for z in Bs[h] if np.isfinite(z)])
        if len(v) == 0:
            continue
        # 半衰期：b_h = rho^h - 1
        rho = np.where(1 + v > 0, (1 + v) ** (1.0 / h), np.nan)
        hl = np.where((rho > 0) & (rho < 1), np.log(0.5) / np.log(rho), np.nan)
        # 符号二项检验
        neg = int((v < 0).sum()); n = len(v)
        z = (neg - n / 2) / np.sqrt(n * 0.25) if n else np.nan
        out[f"h{h}"] = dict(b_med=float(np.median(v)), b_mean=float(v.mean()),
                            neg_share=neg / n, z=float(z), n=n,
                            hl_med=float(np.nanmedian(hl)) if np.isfinite(hl).any() else None,
                            shuf_b_med=float(np.median(vs)) if len(vs) else None,
                            shuf_neg_share=float((vs < 0).mean()) if len(vs) else None)
    return out


# ---------------- 命题 B：解套抛压 ----------------
def part_B(stocks, min_hist=250, deep=0.95, profit=1.05):
    """成本线 RP = 换手衰减筹码均价。事件：从套牢区上穿成本线（解套）"""
    rows = []
    stat = dict(cross=0, hold=0, hold_up=0)
    for s in stocks:
        c = s["close"]; tr = s["turn"]; n = len(c)
        # 滚动换手加权成本
        RP = np.full(n, np.nan)
        p = None; acc = 0
        for i in range(n):
            if not np.isfinite(c[i]):
                if p is not None: RP[i] = p
                continue
            if p is None:
                p = c[i]; RP[i] = p; acc += 1; continue
            a = tr[i] if np.isfinite(tr[i]) else 0.0
            a = min(max(a, 0.0), 0.5)
            p = (1 - a) * p + a * c[i]
            RP[i] = p; acc += 1
        if acc < 600:
            continue
        r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
        for i in range(min_hist, n - 61):
            if not (np.isfinite(c[i]) and np.isfinite(c[i - 1]) and np.isfinite(RP[i - 1])):
                continue
            if not np.isfinite(r[i]):
                continue
            # 解套：昨日在成本线下方 5% 以外，今日收盘站上昨日成本线
            if c[i - 1] < RP[i - 1] * deep and c[i] >= RP[i - 1]:
                kind = 1   # 解套抛压组
            # 对照：已在浮盈区 5% 以上，今日继续上涨
            elif c[i - 1] > RP[i - 1] * profit and r[i] > 0:
                kind = 2
            else:
                continue
            fwd = {}
            for hz in (5, 20, 60):
                if i + hz < n and np.isfinite(c[i + hz]):
                    fwd[hz] = c[i + hz] / c[i] - 1
                else:
                    fwd[hz] = np.nan
            # 同向效率：|净涨| / 路径长度（低=拉扯）
            seg = c[i + 1:i + 21]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 15:
                path = np.sum(np.abs(np.diff(np.concatenate([[c[i]], seg]))))
                eff = abs(seg[-1] / c[i] - 1) / (path / c[i]) if path > 0 else np.nan
            else:
                eff = np.nan
            rows.append((s["code"], i, kind, tr[i] if np.isfinite(tr[i]) else np.nan,
                         r[i], fwd[5], fwd[20], fwd[60], eff))
    df = pd.DataFrame(rows, columns=["code", "i", "kind", "turn", "ret", "f5", "f20", "f60", "eff"])
    out = {"n_event": int((df["kind"] == 1).sum()), "n_ctrl": int((df["kind"] == 2).sum())}
    for k, lab in ((1, "unlock"), (2, "profit_up")):
        g = df[df["kind"] == k]
        if len(g) == 0: continue
        out[lab] = dict(
            n=int(len(g)),
            turn_med=float(g["turn"].median() * 100),
            ret_med=float(g["ret"].median() * 100),
            f5_med=float(g["f5"].median() * 100), f5_up=float((g["f5"] > 0).mean()),
            f20_med=float(g["f20"].median() * 100), f20_up=float((g["f20"] > 0).mean()),
            f60_med=float(g["f60"].median() * 100), f60_up=float((g["f60"] > 0).mean()),
            eff_med=float(g["eff"].median()),
        )
    # 解套后 5 日内跌回成本线下方（直接证据：卖了就跌）
    out["diff_20"] = (out.get("unlock", {}).get("f20_med", np.nan) -
                      out.get("profit_up", {}).get("f20_med", np.nan))
    df.to_parquet(os.path.join(OUT, "d30_B_events.parquet"), index=False)
    return out


# ---------------- 命题 C：下跌持续期 hazard ----------------
def part_C(stocks, dd=0.20, win=250, maxlen=500, buckets=10):
    lens = []; cens = []
    for s in stocks:
        c = s["close"]; n = len(c)
        hh = pd.Series(c).rolling(win, min_periods=60).max().to_numpy()
        draw = c / hh - 1
        inep = False; start = 0
        for i in range(n):
            if not np.isfinite(draw[i]):
                continue
            if not inep:
                if draw[i] <= -dd:
                    inep = True; start = i
            else:
                if draw[i] >= -dd / 2:
                    lens.append(i - start); cens.append(0); inep = False
                elif i - start >= maxlen:
                    lens.append(maxlen); cens.append(1); inep = False
    L = np.array(lens); Cc = np.array(cens)
    if len(L) == 0:
        return {}
    out = dict(n_episode=int(len(L)), n_censored=int(Cc.sum()),
               len_mean=float(L.mean()), len_med=float(np.median(L)),
               len_p25=float(np.percentile(L, 25)), len_p75=float(np.percentile(L, 75)),
               cv=float(L.std() / L.mean()))
    # 30 日桶 hazard（KM 风格：分母为活到桶起点的）
    bs = 30
    nb = int(np.ceil(maxlen / bs))
    haz = []
    for k in range(nb):
        lo, hi = k * bs, (k + 1) * bs
        at_risk = int(np.sum(L > lo))
        if at_risk < 30:
            haz.append(None); continue
        n_exit = int(np.sum((L > lo) & (L <= hi) & (Cc == 0)))
        haz.append(n_exit / at_risk)
    out["hazard_30d"] = haz
    h = np.array([x for x in haz if x is not None])
    if len(h) >= 4:
        # 前 1/3 与后 1/3 的 hazard 比
        k = max(2, len(h) // 3)
        out["haz_early"] = float(np.mean(h[:k]))
        out["haz_late"] = float(np.mean(h[-k:]))
        out["haz_ratio_late_early"] = out["haz_late"] / out["haz_early"] if out["haz_early"] > 0 else None
    # 几何分布对照：常数 hazard = 1/mean，CV 应 ≈1（截断前）
    out["geom_cv_expect"] = 1.0
    np.save(os.path.join(OUT, "d30_C_lens.npy"), np.vstack([L, Cc]))
    return out


def main():
    t0 = time.time()
    fs = sorted(glob.glob(os.path.join(R, "*.parquet")))
    if MAXN:
        rng = np.random.default_rng(11); fs = list(rng.choice(fs, min(MAXN, len(fs)), replace=False))
    print("files", len(fs), flush=True)
    stocks = []
    for f in fs:
        s = load(f)
        if s: stocks.append(s)
    print("loaded", len(stocks), "elapsed %.0fs" % (time.time() - t0), flush=True)

    print("\n===== A 价值锚：未来收益对『偏离长期中枢』的斜率 =====", flush=True)
    A = part_A(stocks)
    print(json.dumps(A, ensure_ascii=False, indent=1), flush=True)

    print("\n===== B 解套抛压 =====", flush=True)
    B = part_B(stocks)
    print(json.dumps(B, ensure_ascii=False, indent=1), flush=True)

    print("\n===== C 下跌持续期 hazard =====", flush=True)
    CC = part_C(stocks)
    print(json.dumps(CC, ensure_ascii=False, indent=1), flush=True)

    json.dump({"A": A, "B": B, "C": CC, "n_stock": len(stocks),
               "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(OUT, "d30_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\ndone %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
