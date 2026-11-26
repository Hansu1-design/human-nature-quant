# -*- coding: utf-8 -*-
"""d174 三条证伪：(1)扩展窗口重算1880 (2)CGO与C5共线 (3)PB绝对水平分档"""
import pandas as pd, numpy as np
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
for x in ['close','amount']: d[x]=pd.to_numeric(d[x],errors='coerce')
d=d.dropna(subset=['close']).set_index('date').sort_index(); c=d['close']; amt=d['amount']
cpi=pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
cpi['m']=cpi['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]),int(str(s)[5:7]),1))
cpi['yoy']=pd.to_numeric(cpi['全国-当月'],errors='coerce')
cpi=cpi.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
cpi['mom']=(cpi['yoy']/100.)**(1/12.); i0=cpi.index[cpi['m']==pd.Timestamp('2008-01-01')][0]
v=[np.nan]*len(cpi); v[i0]=100.
for i in range(i0+1,len(cpi)): v[i]=v[i-1]*cpi.loc[i,'mom']
for i in range(i0-1,-1,-1): v[i]=v[i+1]/cpi.loc[i+1,'mom']
real=(c/pd.Series(v,index=cpi['m']).reindex(c.index,method='ffill')*100)

print('#'*78)
print('# 证伪一：扩展窗口重算"底锚"（每个时点只用当时已知的熊底）')
print('#'*78)
# 已知熊底（日期, 实际点位）——按时间顺序，扩展窗口取已出现底的中位
bots=[('1996-01-19',None),('1997-09-23',None),('2001-02-22',None),('2002-01-22',None),
      ('2003-11-13',None),('2005-06-06',None),('2005-07-11',None),('2008-11-04',None),
      ('2013-06-25',None),('2014-03-12',None),('2016-01-28',None),('2019-01-03',None),
      ('2024-02-05',None)]
# 用局部极小值自动识别（回撤>=30% 后的最低点）
cv=c.values; ix=c.index
mins=[]
for i in range(60,len(cv)-60):
    w=cv[i-60:i+61]
    if cv[i]==w.min() and cv[i]<=cv[max(0,i-500):i].max()*0.75:
        if not mins or (i-mins[-1])>120: mins.append(i)
print('自动识别局部深底 n=',len(mins),' 日期:',[ix[i].date() for i in mins][:15])
realv=real.values
known=[]   # (位置i, 实际点位)
anchor=np.full(len(cv),np.nan)
for i in range(len(cv)):
    # 只纳入 i-250 之前出现的底（避免用当天的底）
    avail=[r for (j,r) in known if j<=i-250]
    if len(avail)>=2:
        anchor[i]=np.median(avail)
    if i in mins: known.append((i,realv[i]))
rA_ext=realv/anchor
print('扩展锚 有效点',np.sum(~np.isnan(rA_ext)))
print('当前扩展锚',round(float(anchor[-1]),0) if not np.isnan(anchor[-1]) else 'NA',
      ' rA_ext',round(float(rA_ext[-1]),3) if not np.isnan(rA_ext[-1]) else 'NA')
print('（固定锚 1880 的 rA = %.3f）'%float(realv[-1]/1880))

# 检验扩展锚 rA 的单调性
S=pd.DataFrame({'c':cv,'rA':rA_ext},index=ix).dropna()
S['f1']=S['c'].shift(-250)/S['c']-1
S2=S.dropna()
q=S2['rA'].quantile([0,.2,.4,.6,.8,1.]).values
S2=S2.copy(); S2['g']=pd.cut(S2['rA'],bins=q,labels=['最低','低','中','高','最高'],include_lowest=True)
g=S2.groupby('g',observed=True).agg(n=('f1','size'),rA=('rA','median'),ret=('f1','median'),win=('f1',lambda x:(x>0).mean()))
g['rA']=g.rA.round(2); g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0)
print('\n【扩展锚 rA】分档 → 未来250日（若无前视则仍应单调）')
print(g.to_string())
mono=all(g['ret'].diff().dropna()<=0)
print('是否单调递减（低→高 收益递减）:', mono)

print()
print('#'*78)
print('# 证伪二：CGO 与 C5（短期反转）是否共线')
print('#'*78)
raw=amt/c; V=(raw/raw.rolling(250,min_periods=60).median()*0.015).clip(0.0005,0.12).fillna(0.015).values
P=c.values; n=len(P); RP=np.zeros(n); RP[0]=P[0]
for i in range(1,n): RP[i]=(1-V[i])*RP[i-1]+V[i]*P[i]
cgo=pd.Series((P-RP)/P,index=c.index)
r60=c.pct_change(60); r250=c.pct_change(250); r20=c.pct_change(20)
X=pd.DataFrame({'cgo':cgo,'r20':r20,'r60':r60,'r250':r250}).dropna()
print('corr(CGO, 过去20日收益) =',round(float(X.cgo.corr(X.r20)),3))
print('corr(CGO, 过去60日收益) =',round(float(X.cgo.corr(X.r60)),3))
print('corr(CGO, 过去250日收益)=',round(float(X.cgo.corr(X.r250)),3))
print('→ 判据 >0.7 即并入 C5')

print()
print('#'*78)
print('# 证伪三：PB 绝对水平分档（绕开分位算子）')
print('#'*78)
pb=pd.read_parquet('G:/AStockData/supplement/valuation/index_pb_沪深300.parquet')
pb['日期']=pd.to_datetime(pb['日期']); pb=pb.set_index('日期').sort_index()
pbs=pd.to_numeric(pb['市净率'],errors='coerce').dropna()
idx2=pd.to_numeric(pb['沪深300'] if '沪深300' in pb.columns else pb.iloc[:,0],errors='coerce')
cols=pb.columns.tolist()
print('PB 文件列:',cols)
lv=pb['市净率'].dropna()
# 用沪深300 指数自身收益作为未来收益（标的一���）
px=pd.to_numeric(pb[cols[0]],errors='coerce').dropna()
print('指数列名',cols[0],' 最后',round(float(px.iloc[-1]),1))
A=pd.DataFrame({'pb':lv,'px':px}).dropna()
A['f1']=A['px'].shift(-12)/A['px']-1     # 月频，12 期≈1年
A2=A.dropna()
bins=[0,1.0,1.3,1.6,2.0,2.5,10]
A2=A2.copy(); A2['g']=pd.cut(A2['pb'],bins=bins,labels=['<1.0','1.0-1.3','1.3-1.6','1.6-2.0','2.0-2.5','>2.5'])
g2=A2.groupby('g',observed=True).agg(n=('f1','size'),pb=('pb','median'),ret=('f1','median'),win=('f1',lambda x:(x>0).mean()))
g2['pb']=g2.pb.round(2); g2['ret']=(g2.ret*100).round(1); g2['win']=(g2.win*100).round(0)
print('\n【PB 绝对水平分档】沪深300，未来 12 个月')
print(g2.to_string())
print('当前 PB',round(float(lv.iloc[-1]),3),'→ 落入档位:', '1.3-1.6' if 1.3<=float(lv.iloc[-1])<1.6 else '?')
print()
# 只用 2012 之后（窗口>=45期）重跑分位六档
print('#'*78); print('# 补充：只用 2012-01 后（扩张窗口 >=45 期）重跑 PB 分位六档'); print('#'*78)
arr=lv.values; ixl=lv.index
pct=np.array([np.nan]*len(arr))
for i in range(len(arr)):
    if i>=36: pct[i]=(arr[:i+1]<arr[i]).mean()
B=pd.DataFrame({'pb':pct,'px':px.reindex(ixl).ffill()},index=ixl).dropna()
B=B[B.index>='2012-01-01']
B['f1']=B['px'].shift(-12)/B['px']-1
B2=B.dropna()
bins2=[0,0.10,0.15,0.20,0.25,0.35,1.0]
B2=B2.copy(); B2['g']=pd.cut(B2['pb'],bins=bins2,labels=['≤10%','10-15%','15-20%','20-25%','25-35%','>35%'])
g3=B2.groupby('g',observed=True).agg(n=('f1','size'),ret=('f1','median'),win=('f1',lambda x:(x>0).mean()))
g3['ret']=(g3.ret*100).round(1); g3['win']=(g3.win*100).round(0)
print(g3.to_string())
