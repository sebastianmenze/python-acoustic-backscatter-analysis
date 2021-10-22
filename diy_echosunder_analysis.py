# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:24:26 2021

@author: Administrator
"""


from matplotlib import pyplot as plt

import numpy as np
import scipy.ndimage as nd
import pandas as pd
from scipy.signal import convolve2d
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

from scipy.signal import find_peaks

from scipy.ndimage.filters import uniform_filter1d
from echolab2.instruments import EK80, EK60

import numpy as np
# from echopy.utils.transform import lin, log
from skimage.morphology import remove_small_objects
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.measure import label
from scipy.signal import convolve2d
import scipy.ndimage as nd

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
        # sv=m_sv[:,interval_start:interval_start+interval_length] 
       
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
                # s_patch[swarm_counter]=patch
                
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

# m_meta['com_depth'] = m_depth[m_meta['com_depthindex']] 


def bottom_detection(Sv, r, r0=10, r1=1000, roff=0,
          thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7)):



    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): Sv threshold above which seabed might occur (dB).
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                  of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                  of the dilation kernel.
           
    Returns:
        bool: 2D array with seabed mask.
    """
    
     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        return np.zeros_like(Sv, dtype=bool)
    
    # get indexes for range offset and range limits
    r0   = np.nanargmin(abs(r - r0))
    r1   = np.nanargmin(abs(r - r1))
    roff = np.nanargmin(abs(r - roff))
    
    # set to -999 shallow and deep waters (prevents seabed detection)
    Sv_ = Sv.copy()
    Sv_[ 0:r0, :] = -999
    Sv_[r1:  , :] = -999
    
    # return empty mask if there is nothing above threshold
    if not (Sv_>thr).any():
        
        mask = np.zeros_like(Sv_, dtype=bool)
        return mask
    
    # search for seabed otherwise    
    else:
        
        # potential seabed will be everything above the threshold, the rest
        # will be set as -999
        seabed          = Sv_.copy()
        seabed[Sv_<thr] = -999
        
        # run erosion cycles to remove fake seabeds (e.g: spikes, small shoals)
        for i in range(ec):
            seabed = erosion(seabed, np.ones(ek))
        
        # run dilation cycles to fill seabed breaches   
        for i in range(dc):
            seabed = dilation(seabed, np.ones(dk))
        
        # mask as seabed everything greater than -999 
        mask = seabed>-999        
        
        # if seabed occur in a ping...
        idx = np.argmax(mask, axis=0)
        for j, i in enumerate(idx):
            if i != 0:
                
                # ...apply range offset & mask all the way down 
                i -= roff
                if i<0:
                    i = 0
                mask[i:, j] = True 
                
    return mask           
           
#%% process data ccmlar
# raw_files_folder=r'D:\krill_cruises\Krill_Orkneys_2020\LSSS\S2020001_PSaga_Sea[9567]\Raw'
raw_files_folder=r'I:\postdoc_krill\krill_cruises\2016001 SAGA SEA\ACOUSTIC_DATA\EK60\SURVEY'
os.chdir(raw_files_folder)
rawfiles= np.sort( glob.glob(  '*.raw'  ) )    

#%%


# from echopy import transform as tf
# from echopy import resample as rs
# from echopy import mask_impulse as mIN
# from echopy import mask_seabed as mSB
# from echopy import get_background as gBN
# from echopy import mask_signal2noise as mSN
# from echopy import mask_range as mRG
# from echopy import mask_shoals as mSH

from echopy.processing import mask_seabed, get_background, correct_absorption,mask_impulse,mask_shoals,mask_signal2noise
# from echopy.processing import mask_seabed 

dir(echopy)

mask_seabed.ariza( np.transpose(sv.values), r, r0=10, r1=1000, roff=0, thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7) )

#%%
# rawfiles= np.sort( glob.glob(  os.path.join(raw_files_folder, '*.raw')  ) )    



rawfile=rawfiles[10]

rawfile=r"I:\postdoc_krill\krill_cruises\2016001 SAGA SEA\ACOUSTIC_DATA\EK60\SURVEY\L0022-D20160212-T213223-ES60.raw"

for rawfile in rawfiles:  

           raw_obj = EK60.EK60()
           raw_obj.read_raw(rawfile)
            
           print(raw_obj)
           
           k=list(raw_obj.raw_data.keys())
                        
           raw_data = raw_obj.raw_data[k[1]][0]
            
           cal_obj = raw_data.get_calibration()
            # Get sv values
           sv_obj = raw_data.get_sv(calibration = cal_obj)
            # Get sv as depth
            #sv_obj_as_depth = raw_data.get_sv(calibration = cal_obj,
            #    return_depth=True)
            
           positions = raw_obj.nmea_data.interpolate(sv_obj, 'GLL')[1]
            # positions['latitude']
                    
            # meter_dif=geopy.distance.distance( (lsss_sv['Latitude'].iloc[0],lsss_sv['Longitude'].iloc[0]), (lsss_sv['Latitude'].iloc[-1],lsss_sv['Longitude'].iloc[-1]) ).km * 1000     
            
        
            # Get frequency label
           freq = sv_obj.frequency
           
           np.shape(sv_obj.data)
            # Expand sv values into a 3d object
           # data3d = np.expand_dims(sv_obj.data, axis=0)
           
           sv= pd.DataFrame( 10*np.log10( sv_obj.data ) , index=positions['ping_time'])
           # sv_downsampled=sv.resample('1min').mean()
  
           r=sv_obj.range
           x=np.arange(sv.shape[0])
           
           sv_clean, m120in_ = mask_impulse.wang( np.transpose(sv.values), thr=(-90,-30), erode=[(5,5)],dilate=[(7,7)], median=[(7,7)])                     
         
           m=mask_seabed.ariza( sv_clean, r, r0=10, r1=1000, roff=0, thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7) )
           sv_clean[ m]=-999
           
           attenuation=cal_obj.absorption_coefficient
           rr           = np.arange(len(r))                
           xx           = np.arange(len(x))                
           # bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20, r120, alpha120)
           
           # delete spiky pings
           sv_clean[sv_clean==-999]=np.nan

           a=np.nanmean(sv_clean,axis=0)      
           ap=find_peaks(a,width=1,prominence=5)
           ix_del=ap[0]
           sv_clean_spikefree=sv_clean.copy()
           sv_clean_spikefree[: ,ix_del]=np.nan
          
           a=pd.Series(np.nanmean(sv_clean_spikefree,axis=0)    )
           b=a.interpolate()
           
           ap=find_peaks(b,width=1,prominence=5)
           ix_del=ap[0]
           sv_clean_spikefree[: ,ix_del]=np.nan
           
           sv_clean=sv_clean_spikefree.copy()
            # for i in range( ap[1]['left_bases'].size ):
          #       i1= ap[1]['left_bases'][i]+1 
          #       i2=ap[1]['right_bases'][i]-1             
          #       sv_clean_spikefree[: ,i1:i2]=np.nan
          

    
           # a=np.nanmean(sv_clean_spikefree,axis=0)      
           # ap=find_peaks(a,width=1,prominence=3)
           # ix_del=ap[0]
           # sv_clean_spikefree[: ,ix_del]=np.nan
           
           # plt.figure(0)
           # plt.clf()  
           # plt.subplot(211)
           # a=np.nanmean(sv_clean,axis=0)            
           # plt.plot(x[ap[0]],a[ap[0]],'or')
       
           # plt.plot(x,a)
           # plt.subplot(212)
           # a=np.nanmean(sv_clean_spikefree,axis=0)             
           # plt.plot(x,a)

            
           

           bg, m_bg = get_background.derobertis(  np.transpose(sv.values), rr, xx, 5, 10, r,attenuation)          
           sv_clean     = 10*np.log10( np.power(10,sv_clean/10) - np.power(10,bg/10 ))
           
             # mask low signal-to-noise 
           mask             = mask_signal2noise.derobertis(sv_clean, bg, thr=5)
           sv_clean[mask] = -999
           sv_clean[np.isnan(sv_clean)]=-999
           
           sv_clean[r<10,:]=-999
           
           
           plt.figure(0)
           plt.clf()    
           plt.subplot(211)
           plt.imshow( np.transpose(sv.values),aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-30])          
           plt.subplot(212)
           plt.imshow( sv_clean,aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-30])
           
           plt.figure(1)
           plt.clf()    
           plt.subplot(211)
           plt.imshow( np.transpose(sv.values),aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-30])          
           plt.subplot(212)
           plt.imshow( bg,aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-30])

                     #         # filter out background
           # svm=np.mean( sv ,axis=1 )
           # background=np.transpose(np.broadcast_to(svm,np.transpose(sv).shape))
           # sv_bgrm=sv-background
    
           # plt.figure(1)
           # plt.clf()      
           # plt.imshow( np.transpose(sv_bgrm),aspect='auto'  )
           # plt.colorbar()
           # plt.clim([00,40])
    
         # -------------------------------------------------------------------------

           sv_swarms= pd.DataFrame( np.transpose(sv_clean), index=positions['ping_time'])
           sv_swarms=sv_swarms.resample('5s').mean()
           
           thr=-70
           mincan=(3,10)
           maxlink=(3,15)
           minsho=(3,15)
           m_meta,m_mask= swarm_detector(sv_swarms,thr,mincan,maxlink,minsho)
           
           plt.figure(0)
           plt.clf()    
           plt.subplot(211)
           plt.imshow( np.transpose(sv_swarms),aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-50])          
           plt.contour(np.transpose(m_mask),[0,1],edgecolor='w')
           plt.subplot(212)
           plt.imshow( sv_clean,aspect='auto'  )
           plt.colorbar()
           plt.clim([-90,-50])
