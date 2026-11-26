# -*- coding: utf-8 -*-
"""d175 底锚前视检验：五个底的实际点位 & 扩展窗口锚的单调性"""
import pandas as pd, numpy as np
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna(subset=['close']).set_index('date').sort_index()
c=d['close']
cpi=pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
cpi['m']=cpi['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]),int(str(s)[5:7]),1))
cpi['yoy']=pd.to_numeric(cpi['全国-当月'],errors='coerce')
cpi=cpi.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
cpi['mom']=(cpi['yoy']/100.)**(1/12.); i0=cpi.index[cpi['m']==pd.Timestamp('2008-01-01')][0]
v=[np.nan]*len(cpi); v[i0]=100.
for i in range(i0+1,len(cpi)): v[i]=v[i-1]*cpi.loc[i,'mom']
for i in range(i0-1,-1,-1): v[i]=v[i+1]/cpi.loc[i+1,'mom']
ci=pd.Series(v,index=cpi['m'])
real=(c/ci.reindex(c.index,method='ffill')*100).dropna()

BOTS=[('2008-11-04',1664),('2013-06-25',1849),('2016-01-28',2638),('2019-01-03',2440),('2024-02-05',2635)]
print('=== 五个熊底的实际点位（CPI 2008-01=100）===')
vals=[]
for dt,nom in BOTS:
    x=float(real.asof(pd.Timestamp(dt)))
    vals.append(x)
    print(f'  {dt}  名义 {nom:5.0f}  实际 {x:6.0f}')
print('中位',round(np.median(vals),0),' 均值',round(np.mean(vals),0))
print('→ 仓里用的 1880 与中位的差',round(1880-np.median(vals),0))
print()

# 扩展窗口锚
print('='*72)
print('【扩展窗口锚】每个时点只用"已过去250日以上"的底')
print('='*72)
anchor=pd.Series(index=real.index,dtype=float)
bl=[(pd.Timestamp(dt),float(real.asof(pd.Timestamp(dt)))) for dt,n in BOTS]
for i,dt in enumerate(real.index):
    av=[x for (bd,x) in bl if bd<=dt-pd.Timedelta(days=365)]
    if len(av)>=1: anchor.iloc[i]=np.median(av)
    if len(av)==1: anchor.iloc[i]=np.nan   # 只有一个底时中位=它自己，仍属"已知"，保留
for i,dt in enumerate(real.index):
    av=[x for (bd,x) in bl if bd<=dt-pd.Timedelta(days=365)]
    anchor.iloc[i]=np.median(av) if len(av)>=1 else np.nan
print('锚的演变（每次变化时）:')
prev=None
for i,dt in enumerate(real.index):
    a=anchor.iloc[i]
    if not np.isnan(a) and a!=prev:
        print(f'  {dt.date()}  锚 → {a:.0f}')
        prev=a
rA_ext=real/anchor
print('当前扩展锚',round(float(anchor.iloc[-1]),0),' rA_ext',round(float(rA_ext.iloc[-1]),3),' vs 固定锚 rA',round(float(real.iloc[-1]/1880),3))
print()

def tab(rA,name):
    S=pd.DataFrame({'c':c.reindex(rA.index),'rA':rA}).dropna()
    S['f1']=S['c'].shift(-250)/S['c']-1
    S2=S.dropna()
    q=S2['rA'].quantile([0,.2,.4,.6,.8,1.]).unique()
    if len(q)<6: print(name,'分位重复，跳过'); return
    S2=S2.copy(); S2['g']=pd.cut(S2['rA'],bins=q,labels=['1最低','2低','3中','4高','5最高'],include_lowest=True)
    g=S2.groupby('g',observed=True).agg(n=('f1','size'),rA=('rA','median'),ret=('f1','median'),win=('f1',lambda x:(x>0).mean()))
    g['rA']=g.rA.round(2); g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0)
    print(f'--- {name}  n={len(S2)}')
    print(g.to_string())
    dif=g['ret'].diff().dropna()
    print('  单调递减:',bool((dif<=0).all()),' 步长:',[round(x,1) for x in dif])
    print()

tab(real/1880,'固定锚 1880（带前视）')
tab(rA_ext,'扩展锚（无前视）')

print('='*72)
print('【CGO 与短期反转共线】')
print('='*72)
amt=pd.to_numeric(d['amount'],errors='coerce')
raw=amt/c; V=(raw/raw.rolling(250,min_periods=60).median()*0.015).clip(0.0005,0.12).fillna(0.015).values
P=c.values; n=len(P); RP=np.zeros(n); RP[0]=P[0]
for i in range(1,n): RP[i]=(1-V[i])*RP[i-1]+V[i]*P[i]
cgo=pd.Series((P-RP)/P,index=c.index)
X=pd.DataFrame({'cgo':cgo,'r20':c.pct_change(20),'r60':c.pct_change(60),'r250':c.pct_change(250)}).dropna()
for k in ['r20','r60','r250']:
    print(f'  corr(CGO, 过去{k[1:]}日收益) = {float(X.cgo.corr(X[k])):+.3f}')
