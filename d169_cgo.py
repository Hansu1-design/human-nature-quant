# -*- coding: utf-8 -*-
"""d169 用 Grinblatt-Han(2005) 的 CGO 重算成本结构，替代我们的"250日额加权成本"
GH 参考价格递推：RP_t = (1 - V_t) * RP_{t-1} + V_t * P_t   （V=换手率，权重大量衰减）
CGO_t = (P_t - RP_t) / P_t
换手率代理：上证无流通市值序列，用 成交额/点位 的相对值校准到日均 1.5%
"""
import pandas as pd, numpy as np
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
for c_ in ['close','amount','volume']: d[c_]=pd.to_numeric(d[c_],errors='coerce')
d=d.dropna(subset=['close']).set_index('date').sort_index()
d=d[d.index>='1996-01-01']
c=d['close']; amt=d['amount']

# 换手率代理
raw=amt/c
v=(raw/raw.rolling(250,min_periods=60).median()*0.015).clip(0.0005,0.12).fillna(0.015)
print('换手率代理 中位',round(float(v.median())*100,2),'%  P10',round(float(v.quantile(.1))*100,2),'%  P90',round(float(v.quantile(.9))*100,2),'%')

P=c.values; V=v.values; n=len(P)
RP=np.zeros(n); RP[0]=P[0]
for i in range(1,n):
    RP[i]=(1-V[i])*RP[i-1]+V[i]*P[i]
cgo=(P-RP)/P
S=pd.DataFrame({'c':P,'rp':RP,'cgo':cgo},index=c.index)
print('CGO 分布:', S['cgo'].describe(percentiles=[.1,.5,.9]).round(3).to_dict())
print('当前: 点位',round(float(P[-1])),' 参考成本',round(float(RP[-1])),' CGO',round(float(cgo[-1])*100,1),'%')
print()

# 未来 250 日
n=len(S); v_=S['c'].values
fr=np.full(n,np.nan); fm=np.full(n,np.nan)
for i in range(n-250):
    seg=v_[i+1:i+251]; fr[i]=seg[-1]/v_[i]-1; fm[i]=seg.min()/v_[i]-1
S['fret']=fr; S['fmdd']=fm
S2=S.dropna()
print('='*72)
print('【CGO 时序检验】上证 1996-2026，CGO 分档 → 未来 250 日')
print('='*72)
q=S2['cgo'].quantile([0,.2,.4,.6,.8,1.0]).values
S2=S2.copy(); S2['g']=pd.cut(S2['cgo'],bins=q,labels=['最低','低','中','高','最高'],include_lowest=True)
g=S2.groupby('g',observed=True).agg(n=('fret','size'),cgo=('cgo','median'),ret=('fret','median'),
    win=('fret',lambda x:(x>0).mean()),mdd=('fmdd','median'))
g['cgo']=(g.cgo*100).round(1); g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0); g['mdd']=(g.mdd*100).round(1)
print(g.to_string())
print()
print('corr(CGO, 未来250日收益) =',round(float(S2['cgo'].corr(S2['fret'])),3))
print('corr(CGO, 未来250日最大回撤) =',round(float(S2['cgo'].corr(S2['fmdd'])),3))

# 三段
print()
print('=== 分段（CGO 最高档 vs 最低档 的未来250日中位收益）===')
for a,b in [('1997','2010'),('2011','2016'),('2017','2026')]:
    s=S2[f'{a}-01-01':f'{b}-12-31']
    if len(s)<200: continue
    lo=s[s.g=='最低']['fret'].median(); hi=s[s.g=='最高']['fret'].median()
    print(f'{a}-{b} n={len(s):4d}  最低档 {lo*100:+6.1f}%   最高档 {hi*100:+6.1f}%   差 {(lo-hi)*100:+6.1f}pp')
