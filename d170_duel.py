# -*- coding: utf-8 -*-
"""d170 四坐标闸门对决：rA / PB分位 / CGO / 组合，同一套三段检验
统一口径：日频资金曲线、空仓 2.5%/年、单边 0.1%、只用 T 日及以前数据
"""
import pandas as pd, numpy as np
RF=0.025
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
for x in ['close','amount']: d[x]=pd.to_numeric(d[x],errors='coerce')
d=d.dropna(subset=['close']).set_index('date').sort_index(); d=d[d.index>='1996-01-01']
c=d['close']; amt=d['amount']
n=len(c); idx=c.index
rf_d=(1+RF)**(1/250)-1

def curve(cond,cost=0.001):
    nav=np.ones(n); pos=0.0; cash=1.0; on=0; tr=0
    for i in range(n-1):
        want=bool(cond[i])
        if want and pos==0: pos=cash*(1-cost)/c.iloc[i+1]; cash=0.0; tr+=1
        elif (not want) and pos>0: cash=pos*c.iloc[i+1]*(1-cost); pos=0.0; tr+=1
        if pos>0: nav[i+1]=pos*c.iloc[i+1]; on+=1
        else: cash*=(1+rf_d); nav[i+1]=cash
    return pd.Series(nav,index=idx),on,tr

def stats(nv,seg):
    s=nv[seg]; r=s.iloc[-1]/s.iloc[0]-1; yrs=(s.index[-1]-s.index[0]).days/365.25
    ann=(s.iloc[-1]/s.iloc[0])**(1/yrs)-1
    dd=(s/s.cummax()-1).min()
    return ann,dd,(ann/RF if False else (ann/abs(dd) if dd<0 else np.nan))

# --- rA ---
def cpi_real():
    cpi=pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
    cpi['m']=cpi['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]),int(str(s)[5:7]),1))
    cpi['yoy']=pd.to_numeric(cpi['全国-当月'],errors='coerce')
    cpi=cpi.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
    cpi['mom']=(cpi['yoy']/100.0)**(1/12.0)
    i0=cpi.index[cpi['m']==pd.Timestamp('2008-01-01')][0]
    v=[np.nan]*len(cpi); v[i0]=100.0
    for i in range(i0+1,len(cpi)): v[i]=v[i-1]*cpi.loc[i,'mom']
    for i in range(i0-1,-1,-1): v[i]=v[i+1]/cpi.loc[i+1,'mom']
    return pd.Series(v,index=cpi['m'])
ci=cpi_real()
real=(c/ci.reindex(c.index,method='ffill')*100)
rA=real/1880.0
print('rA 当前',round(float(rA.iloc[-1]),3))

# --- CGO ---
raw=amt/c; V=(raw/raw.rolling(250,min_periods=60).median()*0.015).clip(0.0005,0.12).fillna(0.015).values
P=c.values; RP=np.zeros(n); RP[0]=P[0]
for i in range(1,n): RP[i]=(1-V[i])*RP[i-1]+V[i]*P[i]
cgo=pd.Series((P-RP)/P,index=idx)

# --- PB 分位（沪深300，月频，2005起）---
pb=pd.read_parquet('G:/AStockData/supplement/valuation/index_pb_沪深300.parquet')
pb['日期']=pd.to_datetime(pb['日期']); pb=pb.set_index('日期').sort_index()
pb['市净率']=pd.to_numeric(pb['市净率'],errors='coerce')
pbs=pb['市净率'].dropna()
# 扩张窗口分位（只用历史，避免前视），最少 36 期
pct=pd.Series(index=pbs.index,dtype=float)
arr=pbs.values
for i in range(len(arr)):
    if i>=36: pct.iloc[i]=(arr[:i+1]<arr[i]).mean()
pct=pct.reindex(idx,method='ffill')
print('PB 当前',round(float(pbs.iloc[-1]),2),' 分位',round(float(pct.dropna().iloc[-1]),3))

CFG=[('rA',rA),('CGO',cgo),('PBpct',pct)]
for nm,s in CFG:
    print(f'{nm}: 覆盖 {s.first_valid_index().date()} → 有效 {s.notna().sum()}')

print()
print('='*96)
print('【闸门对决】同一套三段检验；Calmar=年化/|最大回撤|')
print('='*96)
hdr=f"{'闸门':<22}{'全期年化':>9}{'回撤':>9}{'Calmar':>8}  |{'97-10':>8}{'11-16':>8}{'17-26':>8}{'持仓%':>7}{'换手':>6}"
print(hdr); print('-'*96)
SEGS=[('全期','1997-01-01','2026-12-31'),('97-10','1997-01-01','2010-12-31'),
      ('11-16','2011-01-01','2016-12-31'),('17-26','2017-01-01','2026-12-31')]
def run(label,cond):
    nv,on,tr=curve(cond)
    out=[]
    for nm,a,b in SEGS:
        s=nv[a:b]
        if len(s)<250: out.append(np.nan); continue
        ann=(s.iloc[-1]/s.iloc[0])**(365.25/max(1,(s.index[-1]-s.index[0]).days))-1
        dd=(s/s.cummax()-1).min()
        out.append(ann)
    full=nv['1997-01-01':]
    ann=(full.iloc[-1]/full.iloc[0])**(365.25/(full.index[-1]-full.index[0]).days)-1
    dd=(full/full.cummax()-1).min()
    print(f"{label:<22}{ann*100:>8.2f}%{dd*100:>8.1f}%{ann/abs(dd):>8.3f}  |"
          +''.join(f'{x*100:>7.1f}%' if not np.isnan(x) else '      -' for x in out)
          +f"{on/n*100:>6.0f}%{tr:>6d}")
    return ann,dd

cond_rA=(rA<=1.20).fillna(False).values
run('A: rA<=1.20',cond_rA)
cond_pb=(pct<=0.30).fillna(False).values
run('B: PB分位<=30%',cond_pb)
cond_pb2=(pct<=0.20).fillna(False).values
run('B2: PB分位<=20%',cond_pb2)
cond_cgo=(cgo<=-0.12).fillna(False).values
run('C: CGO<=-12%',cond_cgo)
cond_cgo2=(cgo<=-0.08).fillna(False).values
run('C2: CGO<=-8%',cond_cgo2)
# 组合：两者都满足
both=((rA<=1.20)&(cgo<=-0.08)).fillna(False).values
run('D: rA<=1.20 & CGO<=-8%',both)
both2=((rA<=1.35)&(cgo<=-0.05)).fillna(False).values
run('D2: rA<=1.35 & CGO<=-5%',both2)
run('基准: 全程持有',np.ones(n,dtype=bool))
