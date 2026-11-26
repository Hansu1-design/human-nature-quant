# -*- coding: utf-8 -*-
"""d151 把先生那套规则拆成三条，逐条判生死"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d150_weekwave import run, m, DAILY_RF, RF

d = load()
c = d['close'].values
n = len(c)
yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25

BASE = dict(cost=0.001)
W = dict(cost=0.001, rebuy_wk=8, rebuy_cond=False)
OFF = dict(cfg=dict(bear=False))

cases = [
    ('① 骨架入场 + 买满 250 日卖（=现行 v1.5 口径，d145 H2）', None),
    ('② 骨架入场 + 一年后回落15%清仓（先生的⑤）', dict(wave=False, **OFF)),
    ('③ ② + 熊市支撑接刀（先生的⑥）', dict(wave=False)),
    ('④ ② + 一年内1234周线波段（先生的②③④）', dict(wave=True, **W, **OFF)),
    ('⑤ ② + 全期1234周线波段（窗口放开）', dict(wave=True, **W, cfg=dict(bear=False, hold_days=99999))),
    ('⑥ 全套（先生原话）', dict(wave=True, **W)),
]

print('=' * 118)
print(f'{"方案":<52} {"笔数":>4} {"年化":>9} {"最大回撤":>9} {"净值":>7} {"Calmar":>7}')
print('=' * 118)
rows = {}
for nm, kw in cases:
    if kw is None:
        nav = c / c[0]
        g, dd = m(nav, yrs)
        print(f'{nm:<52} {1:4d} {g*100:+8.2f}% {dd*100:+8.1f}% {nav[-1]:7.2f} {"—":>7}')
        rows[nm] = (g, dd, nav[-1])
        continue
    nav, tr, occ, nw, wv = run(d, **kw)
    g, dd = m(nav, yrs)
    cal = abs(g / dd) if dd < 0 else np.nan
    print(f'{nm:<52} {len(tr):4d} {g*100:+8.2f}% {dd*100:+8.1f}% {nav[-1]:7.2f} {cal:7.2f}')
    rows[nm] = (g, dd, nav[-1])

# ---------- 逐条增量 ----------
print()
print('=' * 118)
print('逐条增量（相对上一行）')
print('=' * 118)
keys = list(rows)
for i in range(1, len(keys)):
    a, b = rows[keys[i - 1]], rows[keys[i]]
    print(f'{keys[i][:46]:<48} 年化 {a[0]*100:+6.2f}% → {b[0]*100:+6.2f}%  '
          f'({(b[0]-a[0])*100:+5.2f}pp)   回撤 {a[1]*100:+6.1f}% → {b[1]*100:+6.1f}%  ({(b[1]-a[1])*100:+5.1f}pp)')

# ---------- 波段 71 回合分布 ----------
nav, tr, occ, nw, wv = run(d, wave=True, **W, cfg=dict(bear=False, hold_days=99999))
ws, hs = [], []
for r in wv:
    if r['rb_i'] is None:
        continue
    a, b = r['sell_px'], r['rb_px']
    ws.append(a / b - 1)
    hs.append(b / a - 1)
ws = np.array(ws); hs = np.array(hs)
print()
print('=' * 118)
print(f'全期波段 {len(ws)} 回合的分布（波段份额增益 = 卖价/回补价 − 1，正=卖高买低赚到）')
print('=' * 118)
print(f'  赚的次数 {int((ws>0).sum())} / {len(ws)}（胜率 {(ws>0).mean()*100:.1f}%）')
print(f'  中位 {np.median(ws)*100:+.2f}%   均值 {ws.mean()*100:+.2f}%')
print(f'  最好 {ws.max()*100:+.2f}%   最差 {ws.min()*100:+.2f}%')
print(f'  简单相加 {ws.sum()*100:+.2f}%')
print(f'  同期「不动」：中位 {np.median(hs)*100:+.2f}%  均值 {hs.mean()*100:+.2f}%  简单相加 {hs.sum()*100:+.2f}%')

# ---------- ② 的逐笔 ----------
nav, tr, occ, nw, wv = run(d, wave=False, **OFF)
print()
print('=' * 118)
print(f'② 骨架入场 + 一年后回落15%清仓 —— 逐笔（{len(tr)} 笔，在场 {occ*100:.0f}%）')
print('=' * 118)
prev = None
for t in tr:
    tag = ''
    if t['t'] == '卖' and prev is not None:
        r = t['px'] / prev - 1
        tag = f'  本笔 {r*100:+7.2f}%'
    print(f'  {t["t"]}  {t["d"]}  {t["px"]:9.2f}  {t["why"]:<10}{tag}')
    if t['t'] == '买':
        prev = t['px']
