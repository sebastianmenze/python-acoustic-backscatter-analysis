# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:06:26 2021

@author: Administrator
"""
from matplotlib import pyplot as plt

import numpy as np
import scipy.ndimage as nd
import pandas as pd
from scipy.signal import convolve2d
import pickle

m = pickle.load( open( "mooring_sv_2015_1min_avg.pkl", "rb" ) )

time_step=1.0 # min
depth_step=0.40022425469168904 #m

depthvec=np.arange(0,298.567294, depth_step )


transducer_depth=260


#%%

m_sv=m.iloc[:,10:-150]
m_depth= transducer_depth -  depthvec[9:-150]


plt.figure(num=0)
plt.clf()
plt.imshow(m_sv.iloc[0:5000,:], aspect='auto',vmin=-90,vmax=-60,extent=[m_depth[0],m_depth[-1],0,1])
plt.colorbar()
#%%


def swarm_detector(m_sv,thr,mincan,maxlink,minsho):
    
    m_meta=pd.DataFrame({'id': pd.Series(dtype='int'),
                       'startindex': pd.Series(dtype='int'),  
                       'startdepth': pd.Series(dtype='int'),
                       'pixel_length': pd.Series(dtype='int'),
                       'pixel_height': pd.Series(dtype='int'),
                       'pixel_area': pd.Series(dtype='int'),
                       'com_timeindex': pd.Series(dtype='int'),
                       'com_depthindex': pd.Series(dtype='int'),
                       'integrated_backscatter': pd.Series(dtype='float'),
                       'average_backscatter': pd.Series(dtype='float')})
    
    interval_length=10000
    interval_start=0
    
    # thr=-70
    # mincan=(3,10)
    # maxlink=(3,15)
    # minsho=(3,15)
        
    swarm_counter=0
    
    m_mask=np.zeros(m_sv.shape)
    
    while interval_start < m_sv.shape[0]:
            
        sv=np.transpose( m_sv.values[interval_start:interval_start+interval_length,:] )
        
        # smoothen the echogram
        k = np.ones((3, 3))/3**2
        sv_smooth = 10*np.log10( convolve2d( np.power(10,sv/10), k,'same',boundary='symm'))   
          
        
        idim=np.arange(sv_smooth.shape[0])
        jdim=np.arange(sv_smooth.shape[1])
            
        # get mask with candidate shoals by masking Sv above threshold
        mask = np.ma.masked_greater(sv_smooth, thr).mask
        if isinstance(mask, np.bool_):
            mask = np.zeros_like(sv_smooth, dtype=bool)
        
        
        # iterate through shoal candidates
        candidateslabeled= nd.label(mask, np.ones((3,3)))[0]
        candidateslabels = pd.factorize(candidateslabeled[candidateslabeled!=0])[1]
        for cl in candidateslabels:
           
            #measure candidate's height and width
            candidate       = candidateslabeled==cl 
            idx             = np.where(candidate)[0]
            jdx             = np.where(candidate)[1]
            candidateheight = idim[max(idx)] - idim[min(idx)]+1
            candidatewidth  = jdim[max(jdx)] - jdim[min(jdx)]+1
           
            # remove candidate from mask if smaller than min candidate size
            if (candidateheight<mincan[0]) | (candidatewidth<mincan[1]):
                mask[idx, jdx] = False
        
        # declare linked-shoals array
        linked    = np.zeros(mask.shape, dtype=int)
        
        # iterate through shoals
        shoalslabeled = nd.label(mask, np.ones((3,3)))[0]
        shoalslabels  = pd.factorize(shoalslabeled[shoalslabeled!=0])[1]
        for fl in shoalslabels:
            shoal = shoalslabeled==fl
        
            # get i/j frame coordinates for the shoal
            i0 = min(np.where(shoal)[0])
            i1 = max(np.where(shoal)[0])
            j0 = min(np.where(shoal)[1])
            j1 = max(np.where(shoal)[1])
           
            # get i/j frame coordinates including linking distance around the shoal
            i00 = np.nanargmin(abs(idim-(idim[i0]-(maxlink[0]+1))))
            i11 = np.nanargmin(abs(idim-(idim[i1]+(maxlink[0]+1))))+1
            j00 = np.nanargmin(abs(jdim-(jdim[j0]-(maxlink[1]+1))))
            j11 = np.nanargmin(abs(jdim-(jdim[j1]+(maxlink[1]+1))))+1
           
            # find neighbours around shoal
            around                  = np.zeros_like(mask, dtype=bool)
            around[i00:i11,j00:j11] = True       
            neighbours              = around & mask # & ~feature      
            neighbourlabels         = pd.factorize(shoalslabeled[neighbours])[1]
            neighbourlabels         = neighbourlabels[neighbourlabels!=0]
            neighbours              = np.isin(shoalslabeled, neighbourlabels)
           
            # link neighbours by naming them with the same label number
            if (pd.factorize(linked[neighbours])[1]==0).all():
                linked[neighbours] = np.max(linked)+1
           
            # if some are already labeled, rename all with the minimum label number
            else:
                formerlabels        = pd.factorize(linked[neighbours])[1]
                minlabel            = np.min(formerlabels[formerlabels!=0])
                linked[neighbours] = minlabel
                for fl in formerlabels[formerlabels!=0]:
                    linked[linked==fl] = minlabel
        
        # iterate through linked shoals
        linkedlabels   = pd.factorize(linked[linked!=0])[1]
        
        labelmask=np.zeros(linked.shape,dtype='int')
        
        for ll in linkedlabels: 
            # measure linked shoal's height and width
            linkedshoal       = linked==ll
            idx               = np.where(linkedshoal)[0]
            jdx               = np.where(linkedshoal)[1]
            linkedshoalheight = idim[max(idx)] - idim[min(idx)]+1
            linkedshoalwidth  = jdim[max(jdx)] - jdim[min(jdx)]+1   
            
            # patch=sv[idim[min(idx)]:idim[max(idx)] , jdim[min(jdx)] :  jdim[max(jdx)]]
            # remove linked shoal from mask if larger than min linked shoal size
            if (linkedshoalheight<minsho[0]) | (linkedshoalwidth<minsho[1]):
                mask[idx, jdx] = False
                
            if (linkedshoalheight>minsho[0]) | (linkedshoalwidth>minsho[1]):
                # add
                mask[idx, jdx] = True
                            
                integrated_backscatter=np.sum(np.power(10,sv[idx,jdx] /10))  
                average_backscatter=np.mean(np.power(10,sv[idx,jdx] /10))  
                startindex=interval_start + jdim[min(jdx)]
                startdepth=idim[min(idx)]
                
                a=nd.center_of_mass(linkedshoal)
                com_x=interval_start +int(a[1])
                com_y=int(a[0])
                
                a=pd.DataFrame({'id': pd.Series(swarm_counter,dtype='int'),
                           'startindex': pd.Series(startindex,dtype='int'), 
                           'startdepth': pd.Series(startdepth,dtype='int'),     
                           'pixel_length': pd.Series(linkedshoalwidth,dtype='int'),
                           'pixel_height': pd.Series(linkedshoalheight,dtype='int'),
                           'pixel_area': pd.Series(idx.size,dtype='int'),
                           'com_timeindex': pd.Series(com_x,dtype='int'),
                           'com_depthindex': pd.Series(com_y,dtype='int'),
                           'integrated_backscatter': pd.Series(integrated_backscatter,dtype='float') ,
                           'average_backscatter': pd.Series(average_backscatter,dtype='float') })
                m_meta=pd.concat([ m_meta ,a ] , ignore_index = True)
                
                labelmask[idx, jdx] = swarm_counter
                swarm_counter=swarm_counter+1
        
                m_mask[interval_start:interval_start+interval_length,:]=np.transpose(labelmask)
        
        # make sure the intervals dos not cut off shoals
        x=np.sum(mask,axis=0 )                             
        if  x[-1]==0:
            interval_start=interval_start+interval_length
        else:
            xx=np.where(x==0)[0]
            interval_start=interval_start+ xx[-1]
        print(interval_start)    
        
    return m_meta,m_mask

#%%


m_sv=m.iloc[:,10:-150].copy()
m_time=m.index.copy()

thr=-70
mincan=(3,10)
maxlink=(3,15)
minsho=(3,15)

m_meta,m_mask = swarm_detector(m_sv,thr,mincan,maxlink,minsho)

m_meta['com_depth'] = m_depth[m_meta['com_depthindex']] 

#%% pre selection

ix_sorted=np.flip( np.argsort(m_meta['average_backscatter'] ) )       
ix_sorted=   m_meta['id'][ix_sorted]

plt.figure(num=0)
plt.clf()

k=1
for ix in ix_sorted[:(4*4)]:
    plt.subplot(4,4,k)
    
    ix1=m_meta['startindex'][ix] 
    ix2=m_meta['startindex'][ix]  + m_meta['pixel_length'][ix] 
    ix3=m_meta['startdepth'][ix] 
    ix4=m_meta['startdepth'][ix]  + m_meta['pixel_height'][ix]     
    
    sv_patch= m_sv.iloc[ ix1:ix2 , ix3:ix4  ]
    plt.imshow(sv_patch, aspect='auto',vmin=-70,vmax=-30,extent=[0,sv_patch.shape[1]*time_step,0,sv_patch.shape[1]*depth_step])
    plt.colorbar()
    plt.title(str( m_meta['average_backscatter'][ix]  ) )
    k=k+1
#%%

ixdel= m_meta['average_backscatter'].values>2e-5
ixdel=np.where( ixdel)[0]

swarm_df=m_meta.copy()
for ix in ixdel:
    swarm_df=swarm_df.drop(ix,index=None)
    ix_m= m_mask == m_meta['id'][ix]
    m_mask[ix_m]=0
    
swarm_df=swarm_df.reset_index(drop=True)

m_krill=m.iloc[:,10:-150]
m_krill[m_mask==0]=np.nan;

# np.sort(swarm_df['average_backscatter'].values )


ix_sorted=np.flip( np.argsort(swarm_df['average_backscatter'].values ) )       

plt.figure(num=0)
plt.clf()

k=1
for ix in ix_sorted[:(4*4)]:
    plt.subplot(4,4,k)
    
    ix1=swarm_df['startindex'][ix] 
    ix2=swarm_df['startindex'][ix]  + swarm_df['pixel_length'][ix] 
    ix3=swarm_df['startdepth'][ix] 
    ix4=swarm_df['startdepth'][ix]  + swarm_df['pixel_height'][ix]     
    
    sv_patch= m_sv.iloc[ ix1:ix2 , ix3:ix4  ]
    plt.imshow(sv_patch, aspect='auto',vmin=-70,vmax=-30,extent=[0,sv_patch.shape[1]*time_step,0,sv_patch.shape[1]*depth_step])
    plt.colorbar()
    plt.title(str( swarm_df['average_backscatter'][ix]  ) )
    k=k+1
    

#%%


       
       
timevec = pd.Series( pd.date_range(start=m.index[0],end=m.index[-1],freq='1d') )

s_time=m.index.copy()
swarm_time=s_time[swarm_df['startindex'].values ]

timediff_max= 60 * 60* 12# in sec
difmat= np.squeeze( timevec.values - swarm_time.values[:, None] )

ix_mat=np.abs( difmat.astype(float)/1e9 ) < timediff_max

integrated_krill_backscatter=np.zeros(timevec.shape)
averaged_krill_backscatter=np.zeros(timevec.shape)
area_krill_backscatter=np.zeros(timevec.shape)
depth_krill_backscatter=np.zeros(timevec.shape) *np.nan


for i in range(timevec.shape[0] ):
    if np.sum(ix_mat[:,i])>0:
        integrated_krill_backscatter[i]=np.sum( swarm_df['integrated_backscatter'][ ix_mat[:,i]] )
        averaged_krill_backscatter[i]=np.mean( swarm_df['average_backscatter'][ ix_mat[:,i]] )
        area_krill_backscatter[i]=np.sum( swarm_df['pixel_area'][ ix_mat[:,i]] ) *depth_step *60
        depth_krill_backscatter[i]=np.mean( swarm_df['com_depth'][ ix_mat[:,i]] )
        
        # ix1=swarm_df['startindex'][ ix_mat[:,i]]
        # ix2=swarm_df['startindex'][ ix_mat[:,i]] + swarm_df['pixel_length'][ ix_mat[:,i]]
        # breakpoint()
        # sv_daily[i,:]=np.mean(m_krill[ ix1:ix2,:])
        
m_krill[m_mask==0]=np.nan;
        
sv_daily=np.zeros([len(timevec),m_krill.shape[1]])
for i in range(timevec.shape[0] ):
    ixx=np.where( (m_krill.index>=timevec[i]) & (m_krill.index < timevec[i] + pd.Timedelta('1d') ) )[0]
    
    sv_daily[i,:]= np.mean( m_krill.iloc[ixx,:] ,axis=0)    

        
plt.figure(num=6)
plt.clf()
plt.subplot(412)
plt.plot(timevec,integrated_krill_backscatter,'-k')
plt.ylabel('Swarm $\int s_v$ [$m^{2} m^{-2}$]')
plt.grid()
plt.title('b) Daily integrated krill backscatter (acoustic biomass proxy)')

plt.xlim([timevec.min(),timevec.max()])
plt.subplot(413)
plt.plot(timevec,averaged_krill_backscatter,'-k')
plt.ylabel('Swarm $\overline{s_v}$ [$m^{2} m^{-3}$]')
plt.grid()
plt.xlim([timevec.min(),timevec.max()])
plt.title('c) Avg. swarm volume backscatter (swarm density proxy)') 

ax1=plt.subplot(414)
plt.plot(timevec,area_krill_backscatter,'-k')
plt.ylabel('Swarm beam area [m s]')
plt.grid()
plt.xlim([timevec.min(),timevec.max()])
plt.title('d) Avg. swarm beam area (swarm size proxy)') 


ax2=plt.subplot(411)
plt.imshow(np.transpose(sv_daily),origin='lower', extent=[0,1,m_depth[0],m_depth[-1]] ,aspect='auto',vmin=-70,vmax=-50)
plt.ylabel('Depth [m]')
plt.grid()
plt.title('a) Daily avg. krill volume backscatter')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins1 = inset_axes(ax2,
                    width="10%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='lower left', borderpad=3)
# fig.colorbar(im1, cax=axins1, orientation="horizontal", ticks=[1, 2, 3])

plt.colorbar(label='$s_v$' ,cax=axins1,orientation="horizontal")
plt.tight_layout()

locs= ax1.get_xticks()
xl = ax1.get_xlim()
locs = (locs-xl[0])/ (xl[1]-xl[0])
labels = ax1.get_xticklabels().copy()
lab=[]
for i in range(len(labels)):
    lab.append(labels[i]._text)
ax2.set_xticks(locs)
ax2.set_xticklabels(lab)

# plt.savefig('2014_azfp_krill_mooring_timeseries.jpg',dpi=200)

#%%

timevec = pd.Series( pd.date_range(start=m.index[0],end=m.index[-1],freq='1w') )
hours=np.arange(0,24)

com_sv=np.ones([len(timevec),len(hours)]) * np.nan
for i1 in range(timevec.shape[0] ):
   for i2 in range(hours.shape[0] ):
 
       ix_day=(m_krill.index>=timevec[i1]) & (m_krill.index < timevec[i1] + pd.Timedelta('1d') ) 
       ix_hour= m_krill.index.hour == hours[i2]
       ix=np.where( ix_day & ix_hour)[0]
       
       a=m_krill.iloc[ix,:].copy()
       ixna=a.isna().values
       
       if np.sum(ixna)<len(a.values.flatten()):
           b= np.power(10, a.values / 10 )
           b[ixna]=0     
           com=list(nd.center_of_mass( b))
           if int(com[1])>=len(m_depth):
               com[1]=len(m_depth)-1
           com_sv[i1,i2]= m_depth[int(com[1])]
           

hours=np.arange(0,24)
avg_krill_backscatter_per_hor=np.empty([24,3])
depth_per_hor=np.empty([24,3])

i=0
for hour in hours:
    ix=swarm_time.hour==hour
    avg_krill_backscatter_per_hor[i,:]= np.percentile( swarm_df['average_backscatter'][ix],[25,50,75])  
    depth_per_hor[i,:]=np.percentile( swarm_df['com_depth'][ix],[25,50,75])  
    i=i+1

plt.figure(12)
plt.clf()
plt.subplot(311)
plt.fill_between(hours, avg_krill_backscatter_per_hor[:,0],avg_krill_backscatter_per_hor[:,2], alpha=0.2,label='25th & 75th percentile')
plt.plot(hours, avg_krill_backscatter_per_hor[:,1],label='Median')
plt.grid()
plt.xlabel('Hour')
plt.ylabel('Swarm $\overline{s_v}$ [$m^{2} m^{-3}$]')
plt.legend()
plt.title('a) Acoustic density over hour of day (UTC)') 
# plt.yscale('log')
# plt.subplot(311)
# plt.fill_between(hours, 10*np.log10(avg_krill_backscatter_per_hor[:,0]),10*np.log10(avg_krill_backscatter_per_hor[:,2]), alpha=0.2)
# plt.plot(hours, 10*np.log10(avg_krill_backscatter_per_hor[:,1]))
# plt.grid()
# plt.xlabel('Hour')
# plt.ylabel('Swarm $\overline{s_v}$ [$m^{2} m^{-3}$]')
# # plt.yscale('lin')

plt.subplot(312)
plt.fill_between(hours, depth_per_hor[:,0],depth_per_hor[:,2], alpha=0.2,label='25th & 75th percentile')
plt.plot(hours, depth_per_hor[:,1],label='Median')
plt.grid()
plt.xlabel('Hour')
plt.ylabel('Swarm COG depth [m]')
plt.legend()
plt.title('b) Swarm depth over hour of day (UTC)') 


plt.subplot(313)

plt.title('c) Swarm depth per week and hour of day (UTC)') 

plt.imshow(np.transpose(com_sv),aspect='auto',vmin=25,vmax=150)  

plt.colorbar(label='Swarm COG Depth [m]')
plt.ylabel('Hour')
plt.grid()


x_label =  pd.date_range(start=m.index[0],end=m.index[-1],freq='4w').strftime('%Y-%m')
x_tick = np.arange(0,4*len(x_label),4)
plt.xticks(x_tick,x_label,rotation=20)


plt.tight_layout()

# plt.savefig('2014_azfp_krill_mooring_swarm_DVM_statistics.jpg',dpi=200)



#%%


plt.figure(13)
plt.clf()

plt.subplot(311)
x=swarm_df['integrated_backscatter']

# binedges=np.linspace(0, x.max(), num=100)
binedges=np.logspace(-7, 0, num=50)

counts,bins= np.histogram(x ,bins=binedges)
counts=counts/np.sum(counts)
plt.plot(bins[0:-1],counts,'.-k')
plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.title('a) Integrated swarm backscatter (acoustic biomass)')
plt.xlabel('Swarm $\int s_v$ [$m^{2} m^{-2}$]')

# plt.subplot(222)
# x=swarm_df['average_backscatter']

# # binedges=np.linspace(0, x.max(), num=50)
# binedges=np.logspace(-8, -4, num=100)

# counts,bins= np.histogram(x ,bins=binedges)
# plt.plot(bins[0:-1],counts,'.-k')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()

plt.subplot(312)
x=10*np.log10( swarm_df['average_backscatter'] )

binedges=np.linspace(x.min(), x.max(), num=50)
# binedges=np.logspace(-8, -4, num=100)

counts,bins= np.histogram(x ,bins=binedges)
counts=counts/np.sum(counts)
plt.plot(bins[0:-1],counts,'.-k')
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.xlabel('Swarm $\overline{s_v}$ [dB re 1 $m^{2} m^{-3}$]')
plt.title('b) Avg. swarm backscatter (acoustic density)')

plt.subplot(313)
x=swarm_df['pixel_area']*depth_step *60

# binedges=np.linspace(0, x.max(), num=100)
binedges=np.logspace(2, 7, num=50)

counts,bins= np.histogram(x ,bins=binedges)
plt.plot(bins[0:-1],counts,'.-k')
plt.xscale('log')
plt.grid()
plt.xlabel('Swarm beam area [m s]')
plt.title('c) Swarm beam area (swarm size proxy)') 



plt.tight_layout()

# plt.savefig('2014_azfp_krill_mooring_histogram.jpg',dpi=200)

#%% 

# import pickle

# pickle.dump( [swarm_df,m_krill,m_mask,m_time,m_depth], open( "mooring_krill_swarm_data_2014.pkl", "wb" ) )