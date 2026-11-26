# -*- coding: utf-8 -*-
"""d178 R52：CGO 的三问
Q1 慢变量三条基线（Hermes 指出：随机游走全满足，只能证"慢"不能证"有信息"）
Q2 水平 vs 变化分解（定生死）：谁在预测未来收益
Q3 换手敏感性 + 独立换手代理（验 CGO 是不是"成本分布"而非"价格平滑"）
Q4 崩盘归因：CGO 闸门加最小持有期把换手压到 PB 量级，看 Calmar 是否回升
   → 回升 ⇒ 主因是成本，"范畴错误"说作废；仍崩 ⇒ 才谈范畴
"""
import numpy as np, pandas as pd
import d177_common as K

c, cgo, V = K.c_all, K.CGO_all, K.V_all
idx_all = c.index

print('=' * 84)
print('Q1 慢变量三条基线（对照：同期的价格随机游走也满足，故不能当证据）')
print('=' * 84)
lag1 = float(cgo.autocorr(1))
dcgo = cgo.diff()
r = c.pct_change()
b = float(np.polyfit(cgo.values[:-1], cgo.values[1:], 1)[0])
half = -np.log(2)/np.log(b) if 0 < b < 1 else np.inf
print(f'  ACF(1) = {lag1:.5f}        （价格水平 ACF(1) = {float(c.autocorr(1)):.5f}）')
print(f'  std(ΔCGO) = {dcgo.std()*100:.3f}pp/日   std(日收益) = {r.std()*100:.3f}%/日')
print(f'  AR(1) 系数 b = {b:.5f}  → 半衰期 {half:.0f} 个交易日（≈{half/250:.2f} 年）')
print('  ⚠ 随机游走同样满足 ACF≈1 / 日变化小 / 半衰期长。此三条不能证"有信息"。')

print()
print('=' * 84)
print('Q2 水平 vs 变化分解 —— 把 CGO 拆开，看是谁在预测未来收益')
print('=' * 84)
D = pd.DataFrame({'c': c, 'cgo': cgo}).copy()
D['chg20'] = D['cgo'] - D['cgo'].shift(20)      # 变化项：近 20 日成本分布移动
D['chg60'] = D['cgo'] - D['cgo'].shift(60)
for h in [250, 750]:
    D[f'f{h}'] = D['c'].shift(-h)/D['c'] - 1
S = D.dropna()
print(f'样本 n={len(S)}  {S.index[0].date()} → {S.index[-1].date()}')
print()
print(f"{'自变量':<12}{'未来1年中位差(高-低)':>22}{'未来3年中位差':>18}{'corr(1年)':>11}{'corr(3年)':>11}")
for name in ['cgo', 'chg20', 'chg60']:
    row = [name]
    for h in [250, 750]:
        q = S[name].quantile([.2, .8]).values
        lo = S[S[name] <= q[0]][f'f{h}'].median()
        hi = S[S[name] >= q[1]][f'f{h}'].median()
        row.append((hi-lo)*100)
    r1 = float(S[name].corr(S['f250'])); r3 = float(S[name].corr(S['f750']))
    print(f'{name:<12}{(row[1]-0):>21.1f}pp{row[2]:>17.1f}pp{r1:>11.3f}{r3:>11.3f}')

# 偏相关：同时放水平项与变化项
print()
print('--- 多元回归（标准化系数，控制彼此）：f250 ~ level + chg20 ---')
X = S[['cgo', 'chg20']].copy()
y = S['f250'].values
Xs = (X - X.mean())/X.std()
A = np.column_stack([np.ones(len(Xs)), Xs.values])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
resid = y - A @ coef
se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (resid @ resid)/(len(y)-3))
print(f'  水平项 cgo    β={coef[1]*100:+.3f}pp/1sd   t={coef[1]/se[1]:+.2f}')
print(f'  变化项 chg20  β={coef[2]*100:+.3f}pp/1sd   t={coef[2]/se[2]:+.2f}')
print('  判读：只有变化项显著 ⇒ CGO 是短期反转指标的伪装（与 C5 共线）；')
print('        只有水平项显著 ⇒ 真慢变量，可进背景层；两者都无 ⇒ 只是描述。')

print()
print('=' * 84)
print('Q3 换手敏感性：高换手期 CGO 是否回归更快（GH 递推的数学含义）')
print('=' * 84)
# 独立换手代理 V2：用成交量而非成交额
_raw2 = K._d['volume'].astype(float)
V2 = (_raw2 / _raw2.rolling(250, min_periods=60).median() * 0.015).clip(0.0005, 0.12).fillna(0.015)
print(f'两代理 corr = {float(pd.concat([V, V2], axis=1).dropna().corr().iloc[0,1]):.3f}（低 ⇒ 代理选择会影响结果）')

def halflife_by_turn(vser, label):
    A = pd.DataFrame({'cgo': cgo, 'v': vser}).dropna()
    med = A['v'].median()
    out = []
    for lab, sub in [('低换手', A[A.v <= med]), ('高换手', A[A.v > med])]:
        x = sub['cgo'].values[:-1]; y2 = sub['cgo'].values[1:]
        bb = float(np.polyfit(x, y2, 1)[0])
        hl = -np.log(2)/np.log(bb) if 0 < bb < 1 else np.inf
        out.append((lab, bb, hl, len(sub)))
    print(f'--- {label}')
    for lab, bb, hl, nn in out:
        print(f'    {lab}  n={nn:5d}  AR(1) b={bb:.5f}  半衰期 {hl:6.1f} 日')
    ratio = out[0][2]/out[1][2]
    print(f'    低/高 半衰期比 = {ratio:.2f}×  （>1 表示高换手回归更快）')
    return ratio

r1 = halflife_by_turn(V, '代理① 成交额/点位（d169 口径）')
r2 = halflife_by_turn(V2, '代理② 成交量（独立构造）')
print()
print('  ⚠ 自省：CGO 本身是按 V 递推构造的，"高换手回归快"部分是构造使然，不是实证发现。')
print('     两个代理若给出同向结论，只能说明"代理确实携带换手信息"，不能证明成本分布真实存在。')

print()
print('=' * 84)
print('Q4 崩盘归因：CGO≤−12% 闸门 + 最小持有期，把换手压到 PB 量级')
print('=' * 84)
cond = (K.cgo <= -0.12).fillna(False).values
print(f"{'最小持有(月)':>12}{'换手次数':>9}{'累计换手额':>11}{'年化':>9}{'回撤':>9}{'Calmar':>9}{'在场%':>8}")
for mh in [0, 1, 2, 3, 6, 9, 12, 18]:
    nv, tr, turn, on, wavg = K.curve_binary(cond, min_hold_months=mh)
    ann, dd, cal, seg = K.metrics(nv)
    print(f'{mh:>12}{tr:>9}{turn:>11.2f}{ann*100:>8.2f}%{dd*100:>8.1f}%{cal:>9.3f}{on*100:>7.1f}%')
print()
print('对照：PB≤15% 点估计在同口径下')
condpb = (K.pb <= 0.15).fillna(False).values
nv, tr, turn, on, wavg = K.curve_binary(condpb)
ann, dd, cal, seg = K.metrics(nv)
print(f'{"PB≤15%":>12}{tr:>9}{turn:>11.2f}{ann*100:>8.2f}%{dd*100:>8.1f}%{cal:>9.3f}{on*100:>7.1f}%')
print()
print('判据：若 CGO 换手压到 ≈32 次时 Calmar 回升到 PB 同量级 ⇒ 主因是成本，"范畴错误"说作废。')
