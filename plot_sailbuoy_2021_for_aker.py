# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:29:46 2021

@author: Administrator
"""



from echolab2.instruments import EK80, EK60

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
from scipy.interpolate import interp1d

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_signal2noise as pip
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH


##### cmap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

color_ek80=np.array([[0.55294118, 0.49019608, 0.58823529, 1.        ],
       [0.49411765, 0.44313725, 0.51764706, 1.        ],
       [0.43921569, 0.39215686, 0.44705882, 1.        ],
       [0.38039216, 0.34509804, 0.37647059, 1.        ],
       [0.32156863, 0.29803922, 0.30588235, 1.        ],
       [0.26666667, 0.29803922, 0.36862745, 1.        ],
       [0.20784314, 0.3254902 , 0.50588235, 1.        ],
       [0.15294118, 0.35294118, 0.63921569, 1.        ],
       [0.09411765, 0.37647059, 0.77254902, 1.        ],
       [0.03529412, 0.40392157, 0.90980392, 1.        ],
       [0.03529412, 0.4       , 0.97647059, 1.        ],
       [0.03529412, 0.32941176, 0.91764706, 1.        ],
       [0.05882353, 0.25882353, 0.85882353, 1.        ],
       [0.08627451, 0.18823529, 0.8       , 1.        ],
       [0.11372549, 0.11764706, 0.74117647, 1.        ],
       [0.14117647, 0.04705882, 0.68235294, 1.        ],
       [0.14509804, 0.19215686, 0.64705882, 1.        ],
       [0.14901961, 0.3372549 , 0.61176471, 1.        ],
       [0.15294118, 0.48235294, 0.57647059, 1.        ],
       [0.15686275, 0.62745098, 0.54117647, 1.        ],
       [0.16078431, 0.77254902, 0.50588235, 1.        ],
       [0.14509804, 0.78431373, 0.47843137, 1.        ],
       [0.11764706, 0.7254902 , 0.45490196, 1.        ],
       [0.09411765, 0.67058824, 0.43529412, 1.        ],
       [0.06666667, 0.61176471, 0.41176471, 1.        ],
       [0.03921569, 0.55294118, 0.38823529, 1.        ],
       [0.08235294, 0.54509804, 0.36078431, 1.        ],
       [0.26666667, 0.63529412, 0.32156863, 1.        ],
       [0.44705882, 0.7254902 , 0.28235294, 1.        ],
       [0.63137255, 0.81568627, 0.24313725, 1.        ],
       [0.81568627, 0.90588235, 0.20392157, 1.        ],
       [1.        , 1.        , 0.16470588, 1.        ],
       [0.99607843, 0.89803922, 0.16862745, 1.        ],
       [0.99215686, 0.8       , 0.17254902, 1.        ],
       [0.99215686, 0.70196078, 0.17647059, 1.        ],
       [0.98823529, 0.6       , 0.18039216, 1.        ],
       [0.98823529, 0.50196078, 0.18431373, 1.        ],
       [0.98823529, 0.45490196, 0.24705882, 1.        ],
       [0.98823529, 0.43137255, 0.33333333, 1.        ],
       [0.98823529, 0.41176471, 0.42352941, 1.        ],
       [0.98823529, 0.38823529, 0.50980392, 1.        ],
       [0.98823529, 0.36470588, 0.6       , 1.        ],
       [0.98823529, 0.33333333, 0.62745098, 1.        ],
       [0.98823529, 0.28627451, 0.54509804, 1.        ],
       [0.99215686, 0.23921569, 0.4627451 , 1.        ],
       [0.99215686, 0.18823529, 0.37647059, 1.        ],
       [0.99607843, 0.14117647, 0.29411765, 1.        ],
       [1.        , 0.09411765, 0.21176471, 1.        ],
       [0.94117647, 0.11764706, 0.20392157, 1.        ],
       [0.88627451, 0.14509804, 0.2       , 1.        ],
       [0.83137255, 0.17254902, 0.19607843, 1.        ],
       [0.77647059, 0.2       , 0.19215686, 1.        ],
       [0.72156863, 0.22352941, 0.18823529, 1.        ],
       [0.69019608, 0.22352941, 0.19215686, 1.        ],
       [0.66666667, 0.21176471, 0.2       , 1.        ],
       [0.64705882, 0.2       , 0.21176471, 1.        ],
       [0.62352941, 0.18431373, 0.21960784, 1.        ],
       [0.6       , 0.17254902, 0.22745098, 1.        ],
       [0.58823529, 0.15294118, 0.21960784, 1.        ],
       [0.59215686, 0.12156863, 0.17647059, 1.        ]])

cmap_ek80 = ListedColormap(color_ek80)

#####

raw_files_folder=r'D:\sailbuoy_2021\transect_lines'

os.chdir(raw_files_folder)
rawfiles= np.sort( glob.glob(  '*.raw'  ) )    
# rawfiles= np.sort( glob.glob(  os.path.join(raw_files_folder, '*.raw')  ) )    


krill_time=np.array([],dtype='datetime64[ms]')
krill_nasc=np.array([])
krill_lat=np.array([])
krill_lon=np.array([])
krill_duration=np.array([],dtype='timedelta64[ms]')
# rawfile=rawfiles[0]

sv_mat=np.zeros([rawfiles.shape[0],1175])
i=0
#%%
for rawfile in rawfiles:  
    # if not os.path.isfile( rawfile[:-4]+'.npy' ):
        try:
            
            
           raw_obj = EK80.EK80()
           raw_obj.read_raw(rawfile)
            
           print(raw_obj)
            
           raw_data = raw_obj.raw_data['EKA 268629-07 ES200-7CDK-Split'][0]
            
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
            
            # Expand sv values into a 3d object
           data3d = np.expand_dims(sv_obj.data, axis=0)
           sv= np.flip( np.rot90( 10*np.log10( data3d[0,:,:] ) ) )
            
           # plt.figure(0)
           # plt.clf()
            
           # plt.imshow( sv  )
           # plt.clim([-90,-30])
           # plt.colorbar()
           
           Sv120=sv
           r120=sv_obj.range
           
           distancecovered_guess=(2*0.514444 * (60*10)) /1000
           km120=np.linspace(0,distancecovered_guess,sv.shape[1]) # guess 10min rec 2knot speed (2*0.514444 * (60*10)) /1000
           t120=sv_obj.ping_time
           
         # alpha120=40*np.ones(r120.shape)  # absoprtion coef
         
           #--------------------------------------------------------------------------       
           # Clean impulse noise      
           # Sv120in, m120in_ = mIN.wang(Sv120, thr=(-70,-30), erode=[(3,3)],
           #                          dilate=[(7,7)], median=[(7,7)])
           # Sv120in, m120in_ = mIN.wang(Sv120, thr=(-70,-40), erode=[(3,3)],
           #                    dilate=[(7,7)], median=[(7,7)])
           Sv120in=sv.copy()  
        # -------------------------------------------------------------------------
         # estimate and correct background noise       
           p120           = np.arange(len(t120))                
           s120           = np.arange(len(r120))  
         
                       
           bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20, r120, 0.044926) # whats correct absoprtion?
           Sv120clean     = tf.log(tf.lin(Sv120in) - tf.lin(bn120))
          
           # plt.figure(num=1)
           # plt.clf()
           # plt.subplot(311)
           # plt.imshow((Sv120),aspect='auto')
           # plt.clim([-82,-40])
           # plt.colorbar()
           # plt.subplot(312)
           # plt.imshow((Sv120in),aspect='auto')
           # plt.clim([-82,-40])
           # plt.colorbar()
           # plt.subplot(313)
           # plt.imshow((Sv120clean),aspect='auto')
           # plt.clim([-82,-40])
           # plt.colorbar()          
         
         # -------------------------------------------------------------------------
         # mask low signal-to-noise 
           m120sn             = mSN.derobertis(Sv120clean, bn120, thr=12)
           Sv120clean[m120sn] = np.nan
           
           # Sv120clean=sv.copy()
           # Sv120clean[Sv120clean<-55]=-999
           
        # get mask for seabed
           m120sb = mSB.ariza(Sv120, r120, r0=20, r1=1000, roff=0,
                              thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
          # m120sb = mSB.ariza(Sv120, r120, r0=20, r1=1000, roff=0,
          #                   thr=-38, ec=1, ek=(3,3), dc=10, dk=(3,7))
           # m120sb = mSB.ariza(Sv120, r120, r0=20, r1=1000, roff=0,
           #                 thr=-38, ec=1, ek=(3,5), dc=10, dk=(3,10))           
           
           Sv120clean[m120sb]=-999
                
         
           # -------------------------------------------------------------------------
           # get swarms mask
           k = np.ones((3, 3))/3**2
           Sv120cvv = tf.log(convolve2d(tf.lin(Sv120clean), k,'same',boundary='symm'))   
         
     
         
           # plt.figure(num=1)
           # plt.clf()
           # plt.subplot(211)
           # plt.imshow((sv),aspect='auto')
           # plt.clim([-82,-50])
           # plt.colorbar()
           # plt.subplot(212)
           # plt.imshow((Sv120cvv),aspect='auto')
           # plt.clim([-82,-50])
           # plt.colorbar()
          
           # p120           = np.arange(np.shape(Sv120cvv)[1])                
           # s120           = np.arange(np.shape(Sv120cvv)[0])           
                 
           m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
                                     mincan=(3,10), maxlink=(3,15), minsho=(3,15))
          
         
    
           
         # -------------------------------------------------------------------------
         # get Sv with only swarms
           Sv120sw                    = Sv120clean.copy()
           Sv120sw[~m120sh] = np.nan
          
           ixdepthvalid=(r120<250) & (r120>0)
           Sv120sw[~ixdepthvalid,:]=np.nan
           ###########
           
           datestr=str(t120[0])
           latstr="{:.6f}".format(positions['latitude'][0])
           lonstr="{:.6f}".format(positions['longitude'][0])
               
           fig=plt.figure(num=1)
           fig.set_size_inches(8, 8)
           plt.clf()
           plt.subplot(211)
           plt.pcolor(t120,-r120,Sv120,cmap=cmap_ek80)
           # plt.imshow( Sv120,aspect='auto',extent=[t120[0],t120[-1],r120[0],-r120[-1]],cmap=cmap_ek80 )
           
           plt.clim([-82,-30])
           plt.title(datestr+' lat:'+latstr+' lon:'+lonstr )
           plt.colorbar(label='Volume backscatter')
           plt.ylabel('Depth in m')
            
           plt.subplot(212)         
           plt.pcolor(t120,-r120,Sv120sw,cmap=cmap_ek80)
           plt.clim([-82,-30])
           plt.title('CCAMLR/rapidkrill swarm detection backscatter' )
           plt.colorbar(label='Volume backscatter')
           plt.ylabel('Depth in m')
           
           
           plt.savefig(datestr.replace(':','_').replace('.','_')[:-4]+'_lat_'+latstr+'_lon_'+lonstr +'_ek80colors.jpg' ,dpi=150)
           
               #############
           # nasc_time=t120
           # cellthickness=np.abs(np.mean(np.diff( r120) )) 
           # nasc=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cellthickness ,axis=0)    
           
           # sv_profile=np.nanmean(Sv120sw,1) 
           # sv_mat[i,:]=sv_profile 
           # i=i+1        
                         
           
           # krill_time=np.append(krill_time,nasc_time[0])
           # krill_nasc=np.append(krill_nasc,np.mean(nasc)    )
           # krill_duration=np.append(krill_duration, nasc_time[-1]-nasc_time[0] )
           
           # krill_lat=np.append(krill_lat,positions['latitude'][0])
           # krill_lon=np.append(krill_lon,positions['longitude'][0])     
           
           del raw_obj
        except:
            print('error')
            