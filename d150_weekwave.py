# -*- coding: utf-8 -*-
"""
d150 「1234 周线」波段机 —— 照先生 2026-11-12 21:31 原话实现

原话拆解（逐句对应）：
 ①「底价买入后一年内」            → 入场沿用骨架：离底 rA<=1.10、冷却180日；一年计时从首次入场起算
 ②「当连续两周，周线低于前第三周的周线…一共1234周，34周的周线低于2
    或者在1附近徘徊，或者低于1，就卖出」
     → 4 周窗口（1最早、4最新）：wc4<wc2 且 wc3<wc2，且 wc4<=wc1*tol（tol=1.01）
 ③「1.5到两个月后发现确实是一直在跌，那就两个月再买入」
     → 卖出后等 6~9 周（主档 8 周）；rebuy_cond=True 时要求价格低于卖价，False 时无条件回补
 ④「买入如果涨，就继续按照1234周线，大概2-3个月卖出」 → 回到 ② 循环
 ⑤「一年以后，发现大盘开始回落，就直接全部卖出」
     → 满 365 天后不再做波段；从持仓期高点回落 15% 则清仓
 ⑥「持续下跌两个月后，关注周线有支撑开始买入，买入一到两个月，无论是否有盈利直接卖出」
     → 熊市模式：60 日跌幅<=-10% 且 周线收阳且站上 5 周均线 → 买；持 8 周无条件卖
 ⑦「持续这样」→ 循环 ⑥

口径：
 - 信号在周线收盘确认，**次日收盘**成交，无未来函数
 - 成本单边 0.1%（另测 0.2/0.3/0.5）；现金 2.5%/年日复利
 - B0 对照：同一入场、同一最终出场，**不做中间波段**
"""
import json
import numpy as np
import pandas as pd

pd.set_option('display.width', 260)
RF = 0.025
DAILY_RF = (1 + RF) ** (1 / 243) - 1

from d141_eat_bite import load

FLAT, WAVE, LATE, BEAR, BEAR_WAIT = 0, 1, 2, 3, 4
EPS = 1e-9


def market_weeks(d):
    c = d['close'].values
    per = d['d'].dt.to_period('W-FRI')
    gb = d.groupby(per)
    widx = gb.apply(lambda x: int(x.index[-1]), include_groups=False).values.astype(int)
    keys = list(gb.groups.keys())
    kmap = {k: i for i, k in enumerate(keys)}
    wpos = np.array([kmap[p] for p in per], dtype=int)
    return widx, wpos, c[widx]


def run(d, wave=True, rebuy_wk=8, sell_frac=1.0, cost=0.001, rebuy_cond=True, cfg=None):
    cfg = cfg or {}
    tol = cfg.get('wk_ratio', 1.01)
    drop1y = cfg.get('drop_after1y', -0.15)
    bear_drop = cfg.get('bear_drop', -0.10)
    bear_hold = cfg.get('bear_hold_wk', 8)
    cool = cfg.get('cool', 180)
    hold_days = cfg.get('hold_days', 365)
    bear_on = cfg.get('bear', True)
    force_days = cfg.get('force_days', 0)
    exit_month = cfg.get('exit_month', False)

    c = d['close'].values
    n = len(c)
    ra = d['rA'].values
    widx, wpos, wc = market_weeks(d)
    # 上月是否收阴（对每个 i，查上一个完整月）
    _mon = d.d.dt.to_period('M')
    _mc = d.groupby(_mon)['close'].last()
    _mneg = (_mc < _mc.shift(1)).values
    _plist = list(_mc.index)
    _code = {p: k for k, p in enumerate(_plist)}
    _monc = np.array([_code[p] for p in _mon])
    prev_neg = np.zeros(n, dtype=bool)
    ok = _monc >= 1
    prev_neg[ok] = _mneg[_monc[ok] - 1]
    is_wkend = np.zeros(n, dtype=bool)
    is_wkend[widx] = True

    nav = np.zeros(n)
    cash, sh, pos = 1.0, 0.0, 0.0
    st = FLAT
    entry0 = -1
    leg = -1
    last_exit = -999
    sell_i, sell_px = -1, 0.0
    peak = 0.0
    armed = False
    trades = []
    pend = None
    occ, nwave_sell = 0, 0
    wv, cur = [], None

    def px(i, side):
        return c[i] * (1 + cost) if side == 'buy' else c[i] * (1 - cost)

    for i in range(n):
        if pend is not None:
            act, frac, why = pend
            pend = None
            if act == 'BUY' and cash > 1e-12:
                amt = cash * frac
                sh += amt / px(i, 'buy')
                cash -= amt
                pos = min(1.0, pos + frac * (1 - pos))
                if entry0 < 0:
                    entry0 = i
                leg = i
                peak = c[i]
                trades.append(dict(t='买', d=str(d.d.iloc[i].date()), px=float(c[i]),
                                   frac=float(frac), why=why))
                if why == '回补' and cur is not None:
                    cur['rb_i'], cur['rb_px'] = i, c[i]
                    wv.append(cur)
                    cur = None
            elif act == 'SELL' and sh > 1e-12:
                q = sh * frac
                cash += q * px(i, 'sell')
                sh -= q
                pos = max(0.0, pos * (1 - frac))
                sell_i, sell_px = i, c[i]
                last_exit = i
                if sh <= 1e-12:
                    sh = 0.0
                    pos = 0.0
                trades.append(dict(t='卖', d=str(d.d.iloc[i].date()), px=float(c[i]),
                                   frac=float(frac), why=why))
                if why == '周线转弱':
                    cur = dict(sell_i=i, sell_px=c[i], rb_i=None, rb_px=None)

        nav[i] = cash + sh * c[i]
        if pos <= EPS and i < n - 1:
            cash *= (1 + DAILY_RF)
        if pos > EPS:
            peak = max(peak, c[i])
            occ += 1

        if not is_wkend[i] or i + 1 >= n:
            continue

        k = wpos[i]
        age = (d.d.iloc[i] - d.d.iloc[entry0]).days if entry0 >= 0 else 0
        wave_sig = False
        if k >= 3:
            c1, c2, c3, c4 = wc[k - 3], wc[k - 2], wc[k - 1], wc[k]
            wave_sig = (c4 < c2) and (c3 < c2) and (c4 <= c1 * tol)

        if st == WAVE:
            # ---- 真离场扳机 ①满 N 天无条件卖 ----
            if force_days and age >= force_days and pos > EPS:
                pend = ('SELL', 1.0, '满期卖出')
                st = FLAT
                armed = True
            # ---- 真离场扳机 ②月线收阴即卖 ----
            elif exit_month and pos > EPS and prev_neg[i]:
                pend = ('SELL', 1.0, '上月收阴')
                st = FLAT
                armed = True
            elif age >= hold_days:
                if pos > EPS:
                    st = LATE
                else:
                    st = FLAT
                    armed = True
            elif pos < 1 - EPS and sell_i > 0:
                wks = (d.d.iloc[i] - d.d.iloc[sell_i]).days / 7.0
                if wks >= rebuy_wk and ((not rebuy_cond) or c[i] < sell_px):
                    pend = ('BUY', 1.0, '回补')
            elif wave and wave_sig and pos >= 1 - EPS:
                pend = ('SELL', sell_frac, '周线转弱')
                nwave_sell += 1

        elif st == LATE:
            if pos > EPS:
                if c[i] / peak - 1 <= drop1y:
                    pend = ('SELL', 1.0, '满一年回落')
                    st = FLAT
                    armed = True
            else:
                st = FLAT
                armed = True

        elif st == FLAT:
            if (not np.isnan(ra[i])) and ra[i] <= 1.10 and (i - last_exit) > cool:
                pend = ('BUY', 1.0, '骨架入场')
                st = WAVE
            elif bear_on and armed and i >= 60:
                dd60 = c[i] / c[i - 60] - 1
                sup = (k >= 5) and (wc[k] > wc[k - 1]) and (wc[k] > np.mean(wc[k - 5:k]))
                if dd60 <= bear_drop and sup:
                    pend = ('BUY', 1.0, '熊市支撑')
                    st = BEAR

        elif st == BEAR:
            if pos > EPS and (d.d.iloc[i] - d.d.iloc[leg]).days / 7.0 >= bear_hold:
                pend = ('SELL', 1.0, '熊市满期')
                st = BEAR_WAIT

        elif st == BEAR_WAIT:
            if (not np.isnan(ra[i])) and ra[i] <= 1.10 and (i - last_exit) > cool:
                pend = ('BUY', 1.0, '骨架入场')
                st = WAVE
            elif bear_on and armed and i >= 60:
                dd60 = c[i] / c[i - 60] - 1
                sup = (k >= 5) and (wc[k] > wc[k - 1]) and (wc[k] > np.mean(wc[k - 5:k]))
                if dd60 <= bear_drop and sup:
                    pend = ('BUY', 1.0, '熊市支撑')
                    st = BEAR

    nav[-1] = cash + sh * c[-1]
    return nav, trades, occ / n, nwave_sell, wv


def m(nav, yrs):
    return nav[-1] ** (1 / yrs) - 1, (nav / np.maximum.accumulate(nav) - 1).min()


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25

    nav0 = (1 + DAILY_RF) ** np.arange(n)
    nav1 = c / c[0]
    g0, d0 = m(nav0, yrs)
    g1, d1 = m(nav1, yrs)

    print('=' * 132)
    print(f'样本 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}   {yrs:.1f} 年   '
          f'现金 {RF*100:.1f}%/年日复利   成本单边 0.1%')
    print('=' * 132)
    print(f'{"H0 全程空仓生息":<34} {0:4d} 笔  年化 {g0*100:+7.2f}%  回撤 {d0*100:+7.1f}%  净值 {nav0[-1]:7.2f}')
    print(f'{"H1 买入持有":<34} {1:4d} 笔  年化 {g1*100:+7.2f}%  回撤 {d1*100:+7.1f}%  净值 {nav1[-1]:7.2f}')

    scen = [
        ('B0 对照：不波段，只等一年后回落', dict(wave=False)),
        ('B1 只做骨架入场（关熊市接刀）', dict(wave=False, cfg=dict(bear=False))),
        ('B2 B1+一年内波段（干净对照）', dict(wave=True, rebuy_wk=8, rebuy_cond=False, cfg=dict(bear=False))),
        ('B3 B1+全期波段（窗口放开·关接刀）', dict(wave=True, rebuy_wk=8, rebuy_cond=False, cfg=dict(bear=False, hold_days=99999))),
        ('B4 B1+全期波段·卖一半', dict(wave=True, rebuy_wk=8, sell_frac=0.5, rebuy_cond=False, cfg=dict(bear=False, hold_days=99999))),
        ('S1 原话版（全卖·须跌回才回补）', dict(wave=True, rebuy_wk=8, sell_frac=1.0)),
        ('S7 全卖·8周无条件回补', dict(wave=True, rebuy_wk=8, sell_frac=1.0, rebuy_cond=False)),
        ('S8 卖一半·8周无条件回补', dict(wave=True, rebuy_wk=8, sell_frac=0.5, rebuy_cond=False)),
        ('S9 卖1/3·8周无条件回补', dict(wave=True, rebuy_wk=8, sell_frac=0.34, rebuy_cond=False)),
        ('S10 全卖·4周无条件回补', dict(wave=True, rebuy_wk=4, sell_frac=1.0, rebuy_cond=False)),
        ('S11 全卖·12周无条件回补', dict(wave=True, rebuy_wk=12, sell_frac=1.0, rebuy_cond=False)),
        ('S12 全卖·8周无条件·宽1.02', dict(wave=True, rebuy_wk=8, sell_frac=1.0, rebuy_cond=False, cfg=dict(wk_ratio=1.02))),
        ('S13 全卖·8周无条件·严1.00', dict(wave=True, rebuy_wk=8, sell_frac=1.0, rebuy_cond=False, cfg=dict(wk_ratio=1.00))),
        ('S14 窗口放开到持有全期（全卖）', dict(wave=True, rebuy_wk=8, sell_frac=1.0, rebuy_cond=False, cfg=dict(hold_days=99999))),
        ('S15 窗口放开到持有全期（卖一半）', dict(wave=True, rebuy_wk=8, sell_frac=0.5, rebuy_cond=False, cfg=dict(hold_days=99999))),
    ]
    out = {}
    print()
    print(f'{"策略":<36} {"笔数":>4} {"波段卖":>6} {"年化":>9} {"最大回撤":>9} {"净值":>8} {"在场":>7} {"Calmar":>7}')
    for nm, kw in scen:
        nav, tr, occ, nw, wv = run(d, **kw)
        g, dd = m(nav, yrs)
        cal = abs(g / dd) if dd < 0 else np.nan
        out[nm] = dict(g=float(g), dd=float(dd), nav=float(nav[-1]), ntr=len(tr),
                       occ=float(occ), nwave=int(nw))
        print(f'{nm:<36} {len(tr):4d} {nw:6d} {g*100:+8.2f}% {dd*100:+8.1f}% {nav[-1]:8.2f} {occ*100:6.0f}% {cal:7.2f}')

    print()
    print('成本敏感性')
    for tag, kw0 in [('B0 不波段', dict(wave=False)),
                     ('S7 全卖·8周无条件', dict(wave=True, rebuy_wk=8, rebuy_cond=False))]:
        row = []
        for cc in (0.001, 0.002, 0.003, 0.005):
            nav, tr, occ, _, _ = run(d, cost=cc, **kw0)
            g, _ = m(nav, yrs)
            row.append(f'{cc*100:.1f}%→{g*100:+5.2f}%')
        print(f'  {tag:<24} ' + '  '.join(row))

    for tag, kw in [('B0 不波段', dict(wave=False)),
                    ('S7 全卖·8周无条件回补', dict(wave=True, rebuy_wk=8, rebuy_cond=False))]:
        nav, tr, occ, nw, wv = run(d, **kw)
        print()
        print('=' * 132)
        print(f'{tag} 逐笔明细（共 {len(tr)} 笔）  在场 {occ*100:.0f}%  波段卖 {nw} 次')
        print('=' * 132)
        for j, t in enumerate(tr, 1):
            print(f'{j:3d} {t["t"]}  {t["d"]}  {t["px"]:9.2f}  {t["frac"]*100:4.0f}%  {t["why"]}')

    # ---------- 波段回合逐个判 ----------
    print()
    print('=' * 132)
    print('每一次波段卖出的实际账单（卖出→回补；「不动」=同期一直拿着）')
    print('=' * 132)
    for tag, kw in [('S7 8周无条件', dict(wave=True, rebuy_wk=8, rebuy_cond=False)),
                    ('S14 窗口放开', dict(wave=True, rebuy_wk=8, rebuy_cond=False, cfg=dict(hold_days=99999)))]:
        nav, tr, occ, nw, wv = run(d, **kw)
        print(f'\n[{tag}] 波段 {len(wv)} 回合')
        if not wv:
            print('  （无）')
            continue
        print(f'{"序":>3} {"卖出日":<12} {"卖价":>9} {"回补日":<12} {"回补价":>9} {"不动持有":>9} {"波段份额":>9} {"波段净效果":>10}')
        tot = 0.0
        for j, r in enumerate(wv, 1):
            if r['rb_i'] is None:
                continue
            a = r['sell_px']; b = r['rb_px']
            hold = b / a - 1                 # 不卖，一直拿到回补日
            wave = a / b - 1                 # 卖了拿现金，回补时同样钱多买到的份额
            tot += wave
            print(f'{j:3d} {str(d.d.iloc[r["sell_i"]].date()):<12} {a:9.2f} '
                  f'{str(d.d.iloc[r["rb_i"]].date()):<12} {b:9.2f} '
                  f'{hold*100:+8.2f}% {wave*100:+8.2f}% {(wave-hold)*100:+9.2f}pp')
        print(f'  合计 {len(wv)} 回合：波段份额增益简单相加 {tot*100:+.2f}%（负＝卖低买高，倒做）')

    json.dump(out, open('d150_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n>> d150_result.json')


if __name__ == '__main__':
    main()
