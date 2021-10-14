# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:34:28 2021

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
import glob
import pickle
filelist=glob.glob( 'D:/mooring/*.sv.csv',recursive=True)

#%%
with open(filelist[0]) as myfile:
    head = [next(myfile) for x in range(2)]
samplecount=int(head[1].split(',')[5])
a=['Ping_date',
 'Ping_time',
 'Ping_milliseconds',
 'Range_start',
 'Range_stop',
 'Sample_count',] 
b=np.arange(0,samplecount,1)
c = b.astype(str)
nam=np.append(np.array(a),c   )

m=pd.DataFrame()

for file in filelist[0:13]:
    df = pd.read_csv(file,sep=',',header=0,index_col=False,names=nam)
    
    # filter out corrupted data
    a=df.iloc[:,0].str.split('-',expand=True)
    b=df.iloc[:,1].str.split(':',expand=True)
    ix_delete= np.where((a.iloc[:,0].astype('int')<2000) | (b.iloc[:,0].astype('int')>23)   )[0] 
    df=df.drop(ix_delete)
    
    mooring_time=pd.to_datetime(df.iloc[:,0]+' '+df.iloc[:,1])
    sv=pd.DataFrame(df.iloc[:,6:-1].values,index=mooring_time)
    sv_downsampled=sv.resample('1min').mean()
    
    m=pd.concat([ m , sv_downsampled] )
    
# 
pickle.dump( m, open( "mooring_sv_2014_1min_avg.pkl", "wb" ) )
# import pickle
# m = pickle.load( open( "mooring_sv_2014_1min_avg.pkl", "rb" ) )
#%%

with open(filelist[13]) as myfile:
    head = [next(myfile) for x in range(2)]
samplecount=int(head[1].split(',')[5])
a=['Ping_date',
 'Ping_time',
 'Ping_milliseconds',
 'Range_start',
 'Range_stop',
 'Sample_count',] 
b=np.arange(0,samplecount,1)
c = b.astype(str)
nam=np.append(np.array(a),c   )

m=pd.DataFrame()

for file in filelist[13:26]:
    df = pd.read_csv(file,sep=',',header=0,index_col=False,names=nam)
    
    # filter out corrupted data
    a=df.iloc[:,0].str.split('-',expand=True)
    b=df.iloc[:,1].str.split(':',expand=True)
    ix_delete= np.where((a.iloc[:,0].astype('int')<2000) | (a.iloc[:,1].astype('int')>12) | (a.iloc[:,1].astype('int')<1) | (b.iloc[:,0].astype('int')>23)   )[0] 
    df=df.drop(ix_delete)
    
    mooring_time=pd.to_datetime(df.iloc[:,0]+' '+df.iloc[:,1])
    sv=pd.DataFrame(df.iloc[:,6:-1].values,index=mooring_time)
    sv_downsampled=sv.resample('1min').mean()
    
    m=pd.concat([ m , sv_downsampled] )
    
# 
pickle.dump( m, open( "mooring_sv_2015_1min_avg.pkl", "wb" ) )

#%%

with open(filelist[39]) as myfile:
    head = [next(myfile) for x in range(2)]
samplecount=int(head[1].split(',')[5])
a=['Ping_date',
 'Ping_time',
 'Ping_milliseconds',
 'Range_start',
 'Range_stop',
 'Sample_count',] 
b=np.arange(0,samplecount,1)
c = b.astype(str)
nam=np.append(np.array(a),c   )

m=pd.DataFrame()

for file in filelist[39:]:
    df = pd.read_csv(file,sep=',',header=0,index_col=False,names=nam)
    
    # filter out corrupted data
    a=df.iloc[:,0].str.split('-',expand=True)
    b=df.iloc[:,1].str.split(':',expand=True)
    ix_delete= np.where((a.iloc[:,0].astype('int')<2000) | (a.iloc[:,1].astype('int')>12) | (a.iloc[:,1].astype('int')<1) | (b.iloc[:,0].astype('int')>23)   )[0] 
    df=df.drop(ix_delete)
    
    mooring_time=pd.to_datetime(df.iloc[:,0]+' '+df.iloc[:,1])
    sv=pd.DataFrame(df.iloc[:,6:-1].values,index=mooring_time)
    sv_downsampled=sv.resample('1min').mean()
    
    m=pd.concat([ m , sv_downsampled] )
    
# 
pickle.dump( m, open( "mooring_sv_2019_1min_avg.pkl", "wb" ) )