# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:53:49 2021

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
from echopy import mask_signal2noise as mSN
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH


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
          
           ixdepthvalid=(r120<250) & (r120>20)
           Sv120sw[~ixdepthvalid,:]=np.nan
           
    
           # plt.figure(num=1)
           # plt.clf()
           # plt.subplot(211)
           # plt.imshow((Sv120),aspect='auto')
           # plt.clim([-82,-50])
           # plt.colorbar()
           # plt.subplot(212)
           # plt.imshow((Sv120sw),aspect='auto')
           # plt.clim([-82,-50])
           # plt.colorbar()
           # plt.savefig(rawfile[0:-3]+'png' )
               
           nasc_time=t120
           cellthickness=np.abs(np.mean(np.diff( r120) )) 
           nasc=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cellthickness ,axis=0)    
           
           sv_profile=np.nanmean(Sv120sw,1) 
           sv_mat[i,:]=sv_profile 
           i=i+1        
                         
           
           krill_time=np.append(krill_time,nasc_time[0])
           krill_nasc=np.append(krill_nasc,np.mean(nasc)    )
           krill_duration=np.append(krill_duration, nasc_time[-1]-nasc_time[0] )
           
           krill_lat=np.append(krill_lat,positions['latitude'][0])
           krill_lon=np.append(krill_lon,positions['longitude'][0])     
           
           del raw_obj
        except:
            print('error')
            
a=np.sum( sv_mat  ,1      ) !=0     
sv_mat=sv_mat[a,:]
sv_mat[sv_mat<-100]=np.nan
sv_mat[sv_mat>-30]=np.nan


surveydata=pd.DataFrame()
surveydata['time']=krill_time
surveydata['nasc_obs']=krill_nasc
surveydata['lat']=krill_lat
surveydata['lon']=krill_lon
# surveydata.to_pickle("./surveydata_sailbuoy_2021_rapidkrilll_mean.pkl")    

#%%


plt.figure(num=0)
plt.clf()

plt.pcolor(krill_time,-r120,np.flipud(np.rot90( sv_mat) ))
plt.colorbar()
plt.clim([-80,-30])
plt.grid()

plt.figure(num=1)
plt.clf()

plt.hist(sv_mat.ravel() ,100 )

np.nanmean( sv_mat.ravel()   )

4*np.pi*1852**2 * np.power(10, np.nanmean( sv_mat.ravel()   ) /10 )

#%%

sv_clean=sv_mat.copy()
sv_clean[:,700:-1]=np.nan

plt.figure(num=0)
plt.clf()

plt.pcolor(np.rot90( sv_clean) )
plt.colorbar()
plt.clim([-80,-40])

# plt.savefig(r'D:\sailbuoy_2021\krill backscatter echogram.jpg',dpi=200)



n=sv_mat.shape[0]
m=sv_mat.shape[1]

sv_depth=np.tile(r120,[n,1])
sv_lat=np.repeat(krill_lat[:,np.newaxis], m, 1)
sv_lon=np.repeat(krill_lon[:,np.newaxis], m, 1)
sv_time=np.repeat(krill_time[:,np.newaxis], m, 1)

#%%
fig = plt.figure(10)
plt.clf()
ax = plt.axes(projection='3d')

x=-sv_clean.copy().ravel()
sizevec= (x - np.nanmin(x)) / ( np.nanmax(x) - np.nanmin(x))
sizevec=1+sizevec*50
np.nanmin(sizevec)
np.nanmax(sizevec)

ax.scatter3D(sv_lon.ravel(),sv_lat.ravel(),-sv_depth.ravel(),s=sizevec, c=sv_clean.ravel(), cmap='plasma_r');
# plt.clim([-90,-40])
# fig.colorbar(ax=ax)

# plt.savefig(r'D:\sailbuoy_2021\3d krill backscatter.jpg',dpi=200)

fig = plt.figure(11)
plt.clf()
ax = plt.axes(projection='3d')

x=-sv_clean.copy().ravel()
sizevec= (x - np.nanmin(x)) / ( np.nanmax(x) - np.nanmin(x))
sizevec=1+sizevec*50
np.nanmin(sizevec)
np.nanmax(sizevec)

ax.scatter3D(sv_lon.ravel(),sv_lat.ravel(),-sv_depth.ravel(),s=sizevec, c=sv_time.ravel(), cmap='plasma');
# plt.clim([-90,-40])
# fig.colorbar(ax=ax)

# plt.savefig(r'D:\sailbuoy_2021\3d krill backscatter.jpg',dpi=200)

#%% 

cellthickness=np.abs(np.mean(np.diff( r120) )) 
krill_nasc=4*np.pi*1852**2 * np.nansum( np.power(10, sv_clean /10)*cellthickness ,axis=1)    


surveydata=pd.DataFrame()
surveydata['time']=krill_time
surveydata['nasc_obs']=krill_nasc
surveydata['lat']=krill_lat
surveydata['lon']=krill_lon

np.mean(krill_nasc)
np.median(krill_nasc)

#%%
kd=pd.Series(krill_duration)
kd.dt.seconds

nasc_per_s=  krill_nasc / kd.dt.seconds

plt.figure(num=0)
plt.clf()
plt.plot( krill_time, nasc_per_s ,'.b')

nasc_filt = uniform_filter1d(nasc_per_s, size=50)
plt.plot(krill_time,nasc_filt,'-r')

#%% load maps

from netCDF4 import Dataset
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import utm

# load data and slice out region of interest

# read mapdata

latlim=[-60.7,-60]
lonlim=[-47.5,-45.5]

spacer=1
gebcofile=r"C:\Users\a5278\Documents\gebco_2020_netcdf\GEBCO_2020.nc"
gebco = Dataset(gebcofile, mode='r')
g_lons = gebco.variables['lon'][:]
g_lon_inds = np.where((g_lons>=lonlim[0]) & (g_lons<=lonlim[1]))[0]
# jump over entries to reduce data
g_lon_inds=g_lon_inds[::spacer]

g_lons = g_lons[g_lon_inds].data
g_lats = gebco.variables['lat'][:]
g_lat_inds = np.where((g_lats>=latlim[0]) & (g_lats<=latlim[1]))[0]
# jump over entries to reduce data
g_lat_inds=g_lat_inds[::spacer]

g_lats = g_lats[g_lat_inds].data
d = gebco.variables['elevation'][g_lat_inds, g_lon_inds].data
gebco.close()

#%%

# plt.savefig(r'D:\sailbuoy_2021\2021_sailbuoy_withgps\map_1.jpg',dpi=200)

#%%


fig=plt.figure(num=3)
plt.clf()
fig.set_size_inches(10,5)

central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
central_lat = latlim[0]+(latlim[1]-latlim[0])/2
extent = [lonlim[0],lonlim[1], latlim[0],latlim[1]]
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))

ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))

ax.set_extent(extent)
  
ax.gridlines(draw_labels=True)
#ax.coastlines(resolution='50m')
#ax.add_feature(cartopy.feature.LAND)

d_plot=d
d_plot[d<-4000]=-4000

plt.contourf(g_lons, g_lats, d_plot, np.arange(-4000,0,100),cmap='Blues_r',
                  linestyles=None, transform=ccrs.PlateCarree())
CS=plt.contour(g_lons, g_lats, d, [-2000,-1000,-500],colors='k',linewidth=.1,
                  linestyles='-', transform=ccrs.PlateCarree())
plt.clabel(CS, inline=True, fontsize=10, fmt='%i')

CS=plt.contourf(g_lons, g_lats, d, [0,8000],colors='silver',linewidth=1,
                  linestyles='-', transform=ccrs.PlateCarree())

CS=plt.contour(g_lons, g_lats, d, [0],colors='k',linewidth=1,
                  linestyles='-', transform=ccrs.PlateCarree())


# m_loc=[-(45+58.331/60) , - (60+24.281/60)]
# plt.plot(m_loc[0],m_loc[1],'.r',markersize=20,transform=ccrs.PlateCarree() )
ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )



plt.scatter(surveydata['lon'][ix_date],surveydata['lat'][ix_date],10,surveydata['nasc_obs'][ix_date],transform=ccrs.PlateCarree() )
plt.clim([0,500])

plt.colorbar(label='NASC')

#%%


# fig=plt.figure(num=3)
# plt.clf()
# fig.set_size_inches(10,5)

# central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
# central_lat = latlim[0]+(latlim[1]-latlim[0])/2
# extent = [lonlim[0],lonlim[1], latlim[0],latlim[1]]
# #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))

# ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))

# ax.set_extent(extent)
  
# ax.gridlines(draw_labels=True)
# #ax.coastlines(resolution='50m')
# #ax.add_feature(cartopy.feature.LAND)

# d_plot=d
# d_plot[d<-4000]=-4000

# plt.contourf(g_lons, g_lats, d_plot, np.arange(-4000,0,100),cmap='Blues_r',
#                   linestyles=None, transform=ccrs.PlateCarree())
# CS=plt.contour(g_lons, g_lats, d, [-2000,-1000,-500],colors='k',linewidth=.1,
#                   linestyles='-', transform=ccrs.PlateCarree())
# plt.clabel(CS, inline=True, fontsize=10, fmt='%i')

# CS=plt.contourf(g_lons, g_lats, d, [0,8000],colors='silver',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())

# CS=plt.contour(g_lons, g_lats, d, [0],colors='k',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())


# # m_loc=[-(45+58.331/60) , - (60+24.281/60)]
# # plt.plot(m_loc[0],m_loc[1],'.r',markersize=20,transform=ccrs.PlateCarree() )
# ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )



# plt.scatter(surveydata['lon'][ix_date],surveydata['lat'][ix_date],10+surveydata['nasc_obs'][ix_date] / 100 ,c=surveydata['nasc_obs'][ix_date] ,edgecolor='k',transform=ccrs.PlateCarree() )
# plt.clim([0,5000])

# plt.colorbar(label='NASC')


#%%

# fig=plt.figure(num=2)
# plt.clf()
# fig.set_size_inches(10,5)

# central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
# central_lat = latlim[0]+(latlim[1]-latlim[0])/2
# extent = [lonlim[0],lonlim[1], latlim[0],latlim[1]]
# #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))

# ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))

# ax.set_extent(extent)
  
# ax.gridlines(draw_labels=True)
# #ax.coastlines(resolution='50m')
# #ax.add_feature(cartopy.feature.LAND)

# d_plot=d
# d_plot[d<-4000]=-4000

# plt.contourf(g_lons, g_lats, d_plot, np.arange(-4000,0,100),cmap='Blues_r',
#                   linestyles=None, transform=ccrs.PlateCarree())
# CS=plt.contour(g_lons, g_lats, d, [-2000,-1000,-500],colors='k',linewidth=.1,
#                   linestyles='-', transform=ccrs.PlateCarree())
# plt.clabel(CS, inline=True, fontsize=10, fmt='%i')

# CS=plt.contourf(g_lons, g_lats, d, [0,8000],colors='silver',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())

# CS=plt.contour(g_lons, g_lats, d, [0],colors='k',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())


# # m_loc=[-(45+58.331/60) , - (60+24.281/60)]
# # plt.plot(m_loc[0],m_loc[1],'.r',markersize=20,transform=ccrs.PlateCarree() )


# plt.scatter(surveydata['lon'],surveydata['lat'],10,10*np.log10(surveydata['nasc_obs']),transform=ccrs.PlateCarree() )
# plt.clim([0,50])

# plt.colorbar(label='sv')

#%%


# fig=plt.figure(num=4)
# plt.clf()
# fig.set_size_inches(10,5)

# central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
# central_lat = latlim[0]+(latlim[1]-latlim[0])/2
# extent = [lonlim[0],lonlim[1], latlim[0],latlim[1]]
# #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))

# ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))

# ax.set_extent(extent)
  
# ax.gridlines(draw_labels=True)
# #ax.coastlines(resolution='50m')
# #ax.add_feature(cartopy.feature.LAND)

# d_plot=d
# d_plot[d<-4000]=-4000

# plt.contourf(g_lons, g_lats, d_plot, np.arange(-4000,0,100),cmap='Blues_r',
#                   linestyles=None, transform=ccrs.PlateCarree())
# CS=plt.contour(g_lons, g_lats, d, [-2000,-1000,-500],colors='k',linewidth=.1,
#                   linestyles='-', transform=ccrs.PlateCarree())
# plt.clabel(CS, inline=True, fontsize=10, fmt='%i')

# CS=plt.contourf(g_lons, g_lats, d, [0,8000],colors='silver',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())

# CS=plt.contour(g_lons, g_lats, d, [0],colors='k',linewidth=1,
#                   linestyles='-', transform=ccrs.PlateCarree())



# ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )

# plt.scatter(surveydata['lon'][ix_date],surveydata['lat'][ix_date],10,surveydata['time'][ix_date],transform=ccrs.PlateCarree() )

# plt.colorbar(label='time')

#%%

from scipy.interpolate import Rbf

ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )   & (surveydata['lat'].notnull() )  

np.sum(surveydata['lat']==np.nan )

x=surveydata['lon'][ix_date.values]
y=surveydata['lat'][ix_date]
# z=surveydata['nasc_obs'][ix_date]
z=nasc_per_s[ix_date]

latlim=[ x.min() ,x.max() ]
lonlim=[y.min() ,y.max() ]
nx=100
ny=100
xvec=np.linspace(latlim[0],latlim[1],nx)
yvec=np.linspace(lonlim[0],lonlim[1],ny)
x_m, y_m=np.meshgrid(xvec,yvec)



rbf = Rbf(x, y, z, epsilon=.02,smooth=1)
z_m = rbf(x_m, y_m)

thr=10
z_m[z_m>thr]=thr
z_m[z_m<0]=0

plt.figure(5)
plt.clf()
plt.contourf(x_m,y_m,z_m,50)
plt.plot(x,y,'.r')
plt.clim([0,thr])
plt.colorbar()

#%%

  # conversion factor
length_in_mm=40
w_in_g= 2.236e-6 *  np.power( length_in_mm,3.314)   
# ts=-73.6 # 120khz
ts=-77.46 # 200khz
crosssec=4*np.pi *np.power(10, ts/10)
c= (w_in_g/crosssec) / 1852**2

krill_density= np.mean( surveydata['nasc_obs'] ) * c

krill_density= np.median( surveydata['nasc_obs'] ) * c

plt.figure(9)
plt.clf()
cutoff=np.arange(1000,100000,100)
m=np.empty(np.shape(cutoff))
i=0
for co  in cutoff:   
    ix=surveydata['nasc_obs']<co
    m[i]=np.mean( surveydata['nasc_obs'][ix] )
    i=i+1
plt.plot(cutoff,m,'.-k')

plt.figure(8)
plt.clf()

plt.subplot(211)
plt.hist(surveydata['nasc_obs'],bins=np.arange(0,5000,50) )
plt.grid()


plt.subplot(212)
plt.hist(surveydata['nasc_obs'],bins=np.arange(0,5000,50) )
plt.grid()
plt.xscale('log')
plt.yscale('log')

#%%

from scipy import stats
fig=plt.figure(num=1)      
plt.clf()


binedges=np.logspace(1, 4, num=50, endpoint=True, base=10.0)

counts,bins= np.histogram(surveydata['nasc_obs'] ,bins=binedges,density=True)


plt.plot(bins[0:-1],counts,'.-k')
plt.xlabel('NASC')
plt.ylabel('Counts')
plt.xscale('log')
plt.yscale('log')
plt.grid()


fl=stats.lognorm.fit(surveydata['nasc_obs'])
plt.plot(bins,stats.lognorm.pdf(bins,fl[0],fl[1],fl[2]),'-b')

fl=stats.norm.fit(counts)
plt.plot(bins,stats.norm.pdf(bins,fl[0],fl[1]),'-g')


a=np.log(bins[0:-1],)
b=np.log(counts)
ixdel=np.isinf(b)
a=np.delete(a,np.where(ixdel))
b=np.delete(b,np.where(ixdel))
coef = np.polyfit(a,b,1)

x=np.logspace(1,4,50)
y=np.exp(coef[1]+coef[0]*np.log(x))
plt.plot(x,y,'-r')



import powerlaw
results = powerlaw.Fit(surveydata['nasc_obs'])
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal_positive')
print(R, p)

print(results.lognormal.mu)
print(results.lognormal.sigma)

print(results.lognormal_positive.mu)
print(results.lognormal_positive.sigma)

plt.figure(8)
plt.clf()
figCCDF = results.plot_pdf(color='b', linewidth=2)
results.power_law.plot_pdf(color='b', linestyle='--', ax=figCCDF)
results.lognormal_positive.plot_pdf(color='r', linestyle='--', ax=figCCDF)


#%%

# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

dmax=1000

d_plot=d.copy()
d_plot[d_plot<-dmax]=-dmax
d_plot[d_plot>10]=10

fig = plt.figure(3)
plt.clf()
ax = plt.axes(projection='3d')

g_lons_mat,g_lat_mat=np.meshgrid(g_lons, g_lats)
# ax.plot_wireframe(g_lons_mat,g_lat_mat, np.zeros(np.shape(g_lons_mat)), color='black')

# ax.contour3D(g_lons_mat,g_lat_mat, d_plot,500)
# ax.plot_surface(g_lons_mat,g_lat_mat, d_plot,rstride=3, cstride=3,
#                 cmap='gray', edgecolor='none',alpha=1)
# ax.contour3D(g_lons_mat,g_lat_mat, d_plot,500)

# ax.plot_trisurf(g_lons_mat.ravel(),g_lat_mat.ravel(), d_plot.ravel(),cmap='gray',shade=False)

ax.contourf(g_lons_mat,g_lat_mat, d_plot,np.arange(-1000,0,1),cmap='YlGnBu_r',edgecolor='k',vmin=-dmax,vmax=0)
ax.contourf(g_lons_mat,g_lat_mat, d_plot,np.arange(0,10,1),cmap='gist_earth',vmin=0,vmax=500)

# ax.contour3D(g_lons_mat,g_lat_mat, d_plot,np.arange(-4000,0,100),colors='k',linewidths=1,linestyles='solid')

# ax.contour3D(g_lons_mat,g_lat_mat, d_plot,np.arange(-4000,0,200),colors='k',linewidths=1,linestyles='solid',extend3d=True)

# ax.contour3D(g_lons_mat,g_lat_mat, d_plot,np.arange(-4000,0,100),colors='k',linewidths=1,linestyles='solid',extend3d=True)

ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )

ax.scatter3D(surveydata['lon'][ix_date],surveydata['lat'][ix_date],10,s=10+ surveydata['nasc_obs'][ix_date]/100, c=surveydata['nasc_obs'][ix_date], cmap='inferno');

ax.view_init(elev=50., azim=60)

# plt.savefig(r'D:\sailbuoy_2021\2021_sailbuoy_withgps\map_3d_1.jpg',dpi=200)

#%%



plt.figure(8)
plt.clf()

x=  np.diff( surveydata['time'] ) /(60e9)

plt.plot(x)
#%% objective mapping


def barnes_objective(xs, ys, zs, XI, YI, XR, YR, RUNS=3):
    #-- remove singleton dimensions
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    zs = np.squeeze(zs)
    XI = np.squeeze(XI)
    YI = np.squeeze(YI)
    #-- size of new matrix
    if (np.ndim(XI) == 1):
        nx = len(XI)
    else:
        nx,ny = np.shape(XI)

    #-- Check to make sure sizes of input arguments are correct and consistent
    if (len(zs) != len(xs)) | (len(zs) != len(ys)):
        raise Exception('Length of X, Y, and Z must be equal')
    if (np.shape(XI) != np.shape(YI)):
        raise Exception('Size of XI and YI must be equal')

    #-- square of Barnes smoothing lengths scale
    xr2 = XR**2
    yr2 = YR**2
    #-- allocate for output zp array
    zp = np.zeros_like(XI.flatten())
    #-- first analysis
    for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
        dx = np.abs(xs - XY[0])
        dy = np.abs(ys - XY[1])
        #-- calculate weights
        w = np.exp(-dx**2/xr2 - dy**2/yr2)
        zp[i] = np.sum(zs*w)/sum(w)

    #-- allocate for even and odd zp arrays if iterating
    if (RUNS > 0):
        zpEven = np.zeros_like(zs)
        zpOdd = np.zeros_like(zs)
    #-- for each run
    for n in range(RUNS):
        #-- calculate even and odd zp arrays
        for j,xy in enumerate(zip(xs,ys)):
            dx = np.abs(xs - xy[0])
            dy = np.abs(ys - xy[1])
            #-- calculate weights
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zpEven[j] = zpOdd[j] + np.sum((zs - zpOdd)*w)/np.sum(w)
            else:#-- odd
                zpOdd[j] = zpEven[j] + np.sum((zs - zpEven)*w)/np.sum(w)
        #-- calculate zp for run n
        for i,XY in enumerate(zip(XI.flatten(),YI.flatten())):
            dx = np.abs(xs - XY[0])
            dy = np.abs(ys - XY[1])
            w = np.exp(-dx**2/xr2 - dy**2/yr2)
            if ((n % 2) == 0):#-- even (% = modulus)
                zp[i] = zp[i] + np.sum((zs - zpEven)*w)/np.sum(w)
            else:#-- odd
                zp[i] = zp[i] + np.sum((zs - zpOdd)*w)/np.sum(w)

    #-- reshape to original dimensions
    if (np.ndim(XI) != 1):
        ZI = zp.reshape(nx,ny)
    else:
        ZI = zp.copy()

    #-- return output matrix/array
    return ZI

ix_date=(surveydata['time']>pd.Timestamp(2021,3,3,20,0,0)) & (surveydata['time']<pd.Timestamp(2021,3,18,0,0,0) )   & (surveydata['lat'].notnull() )  

np.sum(surveydata['lat']==np.nan )

x=surveydata['lon'][ix_date.values]
y=surveydata['lat'][ix_date]
z=surveydata['nasc_obs'][ix_date]
# z=nasc_per_s[ix_date]

latlim=[ x.min() ,x.max() ]
lonlim=[y.min() ,y.max() ]
nx=100
ny=100
xvec=np.linspace(latlim[0],latlim[1],nx)
yvec=np.linspace(lonlim[0],lonlim[1],ny)
x_m, y_m=np.meshgrid(xvec,yvec)

z_m = barnes_objective(x, y, z, x_m, y_m, .04, .03)
z_m[z_m<0]=0

thr=1000

plt.figure(5)
plt.clf()
plt.contourf(x_m,y_m,z_m,50)
plt.plot(x,y,'.r')
plt.clim([0,thr])
plt.colorbar()

#%% autocorrelation xy
