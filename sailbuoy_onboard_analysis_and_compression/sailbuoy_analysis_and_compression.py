# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:07:25 2021

@author: Administrator
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import glob 
import os

from echolab2.instruments import EK80, EK60

from echopy.processing import mask_seabed, get_background, correct_absorption,mask_impulse,mask_shoals,mask_signal2noise
from echopy.utils.transform import lin, log

from scipy.ndimage import gaussian_filter
from matplotlib.path import Path
import matplotlib

# pip install opencv-python
import cv2

# #%% code if you wanna use the simrad colorscale
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# color_ek80=np.array([[0.55294118, 0.49019608, 0.58823529, 1.        ],
#        [0.49411765, 0.44313725, 0.51764706, 1.        ],
#        [0.43921569, 0.39215686, 0.44705882, 1.        ],
#        [0.38039216, 0.34509804, 0.37647059, 1.        ],
#        [0.32156863, 0.29803922, 0.30588235, 1.        ],
#        [0.26666667, 0.29803922, 0.36862745, 1.        ],
#        [0.20784314, 0.3254902 , 0.50588235, 1.        ],
#        [0.15294118, 0.35294118, 0.63921569, 1.        ],
#        [0.09411765, 0.37647059, 0.77254902, 1.        ],
#        [0.03529412, 0.40392157, 0.90980392, 1.        ],
#        [0.03529412, 0.4       , 0.97647059, 1.        ],
#        [0.03529412, 0.32941176, 0.91764706, 1.        ],
#        [0.05882353, 0.25882353, 0.85882353, 1.        ],
#        [0.08627451, 0.18823529, 0.8       , 1.        ],
#        [0.11372549, 0.11764706, 0.74117647, 1.        ],
#        [0.14117647, 0.04705882, 0.68235294, 1.        ],
#        [0.14509804, 0.19215686, 0.64705882, 1.        ],
#        [0.14901961, 0.3372549 , 0.61176471, 1.        ],
#        [0.15294118, 0.48235294, 0.57647059, 1.        ],
#        [0.15686275, 0.62745098, 0.54117647, 1.        ],
#        [0.16078431, 0.77254902, 0.50588235, 1.        ],
#        [0.14509804, 0.78431373, 0.47843137, 1.        ],
#        [0.11764706, 0.7254902 , 0.45490196, 1.        ],
#        [0.09411765, 0.67058824, 0.43529412, 1.        ],
#        [0.06666667, 0.61176471, 0.41176471, 1.        ],
#        [0.03921569, 0.55294118, 0.38823529, 1.        ],
#        [0.08235294, 0.54509804, 0.36078431, 1.        ],
#        [0.26666667, 0.63529412, 0.32156863, 1.        ],
#        [0.44705882, 0.7254902 , 0.28235294, 1.        ],
#        [0.63137255, 0.81568627, 0.24313725, 1.        ],
#        [0.81568627, 0.90588235, 0.20392157, 1.        ],
#        [1.        , 1.        , 0.16470588, 1.        ],
#        [0.99607843, 0.89803922, 0.16862745, 1.        ],
#        [0.99215686, 0.8       , 0.17254902, 1.        ],
#        [0.99215686, 0.70196078, 0.17647059, 1.        ],
#        [0.98823529, 0.6       , 0.18039216, 1.        ],
#        [0.98823529, 0.50196078, 0.18431373, 1.        ],
#        [0.98823529, 0.45490196, 0.24705882, 1.        ],
#        [0.98823529, 0.43137255, 0.33333333, 1.        ],
#        [0.98823529, 0.41176471, 0.42352941, 1.        ],
#        [0.98823529, 0.38823529, 0.50980392, 1.        ],
#        [0.98823529, 0.36470588, 0.6       , 1.        ],
#        [0.98823529, 0.33333333, 0.62745098, 1.        ],
#        [0.98823529, 0.28627451, 0.54509804, 1.        ],
#        [0.99215686, 0.23921569, 0.4627451 , 1.        ],
#        [0.99215686, 0.18823529, 0.37647059, 1.        ],
#        [0.99607843, 0.14117647, 0.29411765, 1.        ],
#        [1.        , 0.09411765, 0.21176471, 1.        ],
#        [0.94117647, 0.11764706, 0.20392157, 1.        ],
#        [0.88627451, 0.14509804, 0.2       , 1.        ],
#        [0.83137255, 0.17254902, 0.19607843, 1.        ],
#        [0.77647059, 0.2       , 0.19215686, 1.        ],
#        [0.72156863, 0.22352941, 0.18823529, 1.        ],
#        [0.69019608, 0.22352941, 0.19215686, 1.        ],
#        [0.66666667, 0.21176471, 0.2       , 1.        ],
#        [0.64705882, 0.2       , 0.21176471, 1.        ],
#        [0.62352941, 0.18431373, 0.21960784, 1.        ],
#        [0.6       , 0.17254902, 0.22745098, 1.        ],
#        [0.58823529, 0.15294118, 0.21960784, 1.        ],
#        [0.59215686, 0.12156863, 0.17647059, 1.        ]])

# cmap_ek80 = ListedColormap(color_ek80)


#%%
# raw_files_folder=r'C:\Users\a5278\Documents\postdoc_krill\sailbuoydata\raw\rawwithgps'
raw_files_folder=r'D:\sailbuoy_2021\transect_lines'

rawfiles= np.sort( glob.glob( raw_files_folder+ '/*.raw'  ) )    

# rawfile=r"C:\Users\a5278\Documents\postdoc_krill\sailbuoydata\raw\rawwithgps\SB_KRILL_2019-Phase0-D20200130-T160057-0.raw"

bytevar=[]

for rawfile in rawfiles:  
 
 try:
   raw_obj = EK80.EK80()
   raw_obj.read_raw(rawfile)
    
   # print(raw_obj)
    
   k=list(raw_obj.raw_data.keys())          
   raw_data = raw_obj.raw_data[k[0]][0]       
   cal_obj = raw_data.get_calibration()
   sv_obj = raw_data.get_sv(calibration = cal_obj)
   # positions = raw_obj.nmea_data.interpolate(sv_obj, 'GLL')[1]
   freq = sv_obj.frequency
   # sv= pd.DataFrame( 10*np.log10( sv_obj.data ) , index=positions['ping_time'])

   r=sv_obj.range
   time_seconds= (sv_obj.ping_time-sv_obj.ping_time[0]).astype('float')/1000 
   time= sv_obj.ping_time-sv_obj.ping_time[0]
   sv= pd.DataFrame( 10*np.log10( sv_obj.data ) ,index=time )


#% clean up

   
   sv_clean, m120in_ = mask_impulse.wang( np.transpose(sv.values), thr=(-90,-30), erode=[(5,5)],dilate=[(7,7)], median=[(7,7)])                     
 
   m=mask_seabed.ariza( sv_clean, r, r0=10, r1=1000, roff=0, thr=-30, ec=1, ek=(1,3), dc=10, dk=(3,7) )
   sv_clean[ m]=-999
   
   attenuation=cal_obj.absorption_coefficient
   # rr           = np.arange(len(r))                
   x_vec=np.arange(sv.shape[0])
   bg, m_bg = get_background.derobertis(  np.transpose(sv.values), r, x_vec, 5, 10, r,np.mean(attenuation))          
   sv_clean     = 10*np.log10( np.power(10,sv_clean/10) - np.power(10,bg/10 ))
   
   # mask low signal-to-noise 
   mask             = mask_signal2noise.derobertis(sv_clean, bg, thr=3)
   sv_clean[mask] = -999
   sv_clean[np.isnan(sv_clean)]=-999
   
   sv_clean[r<10,:]=-999
   sv_clean=pd.DataFrame( np.transpose(sv_clean) ,index=time )

   sv_clean=sv_clean.resample('10s').mean()
   time_resampled=np.linspace(0,time_seconds[-1],sv_clean.shape[0])
   sv_smooth = gaussian_filter(sv_clean, sigma=[0,5]) # (vert horz)
   
   
   
   # _, binary = cv2.threshold(sv_clean.values.astype('float32'), -70, 255, cv2.THRESH_BINARY)
   z=sv_smooth>=-70
   image_8bit = np.uint8(z * 255)
   contours, hierarchy = cv2.findContours(image_8bit, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

           # find largest area contour
   c_area=[]
   for i in range(len(contours)):
        c_area.append( cv2.contourArea(contours[i]))  
   c_area=np.array(c_area)    
   good_c=np.where(c_area>10)[0]

   countour_timespan_s=time_seconds[-1].astype('float16')
   countour_depthspan_m=r[-1].astype('float16')

   countour_id=[]
   countour_time=[]
   countour_f=[]
   countour_value=[]
   x=[]
   y=[] 
   patchcounter=0
   for cid in  good_c:
        v=  np.squeeze(  contours[cid]   )
        x=time_resampled[v[:,1]]
        y=-r[v[:,0]]
               # plt.figure(5)
               # plt.clf()    
               # plt.imshow(np.transpose(sv_clean),aspect='auto')
               # plt.plot(v[:,1],v[:,0],'.-k')    
                   
                   
       
        kk_1,kk_2=np.meshgrid(np.arange( sv_clean.shape[1] ),np.arange( sv_clean.shape[0] ))   
        xx, yy = kk_1.flatten(), kk_2.flatten()
        points = np.vstack((xx,yy)).T
        pp = Path(v) # make a polygon
        grid = pp.contains_points(points)
        # mask = grid.reshape(kk_1.shape) # now you have a mask with points inside a polygon  
        meansv=np.mean( sv_clean.values.flatten()[grid] )
        print(meansv)
        
        if meansv>=-70:
        
            countour_id.append(patchcounter)
            patchcounter=patchcounter+1
            countour_time.append([x.astype('float16')])
            countour_f.append([y.astype('float16')])           
            countour_value.append(meansv.astype('float16'))
   
   clist=[countour_timespan_s,countour_depthspan_m,countour_id,countour_value,countour_time,countour_f]
   
   filesize=0
   if len(countour_id)>0:
       txt=rawfile.split('\\')       
       pickle.dump(clist , open( txt[-1][0:-3]+'pkl', "wb" ) )
       filesize=os.path.getsize( txt[-1][0:-3]+'pkl') 
   bytevar.append(filesize)   
   
   plt.figure(0)
   plt.clf()    
   plt.subplot(311)
   plt.imshow( np.transpose(sv.values),aspect='auto',extent=[x_vec[0],x_vec[-1],r[-1],0] ,cmap='viridis')
   plt.colorbar(label='$s_v$ in dB')
   plt.clim([-90,-30])        
   plt.title('Raw echogram - '+str(float(sv.values.nbytes/1000)) + ' kbytes')
   plt.xlabel('Ping number')
   plt.ylabel('Depth in m')

   
   plt.subplot(312)
   plt.imshow( np.transpose(sv_smooth),aspect='auto',extent=[x_vec[0],x_vec[-1],r[-1],0] ,cmap='viridis')
   plt.colorbar(label='$s_v$ in dB')
   plt.clim([-90,-30])
   plt.title('Processed echogram - '+str(float(sv_smooth.nbytes/1000)) + ' kbytes')
   plt.xlabel('Ping number')
   plt.ylabel('Depth in m')
   
   ax=plt.subplot(313)

   cmap = matplotlib.cm.get_cmap('viridis')
   # cmap=cmap_ek80
   for i_patch in clist[2]:
       time_s=clist[4][i_patch][0]
       depth_m=clist[5][i_patch][0]
       
       svval=clist[3][i_patch]
       svval=(svval + 90)/ (-30+90)
       rgba = cmap(svval)      
       plt.fill(time_s,depth_m,color=rgba)
   plt.ylim([-clist[1],0])
   plt.xlim([0,clist[0]])   
   ax.set_facecolor(cmap(0) )
   plt.clim([-90,-30])

   plt.colorbar(label='$s_v$ in dB')
   plt.title('Transmitted echogram - '+str(float(filesize/1000)) + ' kbytes')
   plt.xlabel('Time')
   plt.ylabel('Depth in m')
   
   plt.tight_layout()
   txt=rawfile.split('\\')
   plt.savefig(txt[-1][0:-3]+'jpg' )

 except:
   print('--> corrupted file!')  

#%%

bytevar=np.array(bytevar)

plt.figure(1)
plt.clf()  
plt.subplot(211)
plt.plot(bytevar/1000)
plt.ylabel('kB')
plt.title('Time series of transitted kB - total: '+str(np.sum(bytevar/1000)))

plt.subplot(212)
plt.hist(bytevar/1000,50)
plt.xlabel('kB')
plt.title('Histogram of transitted kB - total: '+str(np.sum(bytevar/1000)))
plt.yscale('log')
plt.ylabel('counts')
plt.tight_layout()

# plt.savefig('2021_compressed_echogram_statistics.jpg' )

#%% read contour frm pickle

file=r"C:\Users\a5278\Documents\postdoc_krill\krillswarm_contour_algorithm\SB_KRILL_2019-Phase0-D20210305-T025029-0.pkl"

clist = pickle.load( open( file, "rb" ) )

plt.figure(0)
plt.clf()    
  
ax=plt.subplot(111)
plt.imshow([[0,0]])
cmap = matplotlib.cm.get_cmap('viridis')
for i_patch in clist[2]:
    time_s=clist[4][i_patch][0]
    depth_m=clist[5][i_patch][0]
    
    svval=clist[3][i_patch]
    svval=(svval + 90)/ (-30+90)
    rgba = cmap(svval)      
    plt.fill(time_s,depth_m,color=rgba)
plt.ylim([-clist[1],0])
plt.xlim([0,clist[0]])   
ax.set_facecolor(cmap(0) )
plt.clim([-90,-30])

plt.colorbar(label='$s_v$ in dB')
plt.xlabel('Time')
plt.ylabel('Depth in m')

plt.tight_layout()
