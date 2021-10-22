# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:59:10 2021

@author: Administrator
"""


#%%

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

fig=plt.figure(num=2)
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


plt.scatter(surveydata['lon'],surveydata['lat'],10,surveydata['nasc_obs'],transform=ccrs.PlateCarree() )
plt.clim([0,100])


# plt.savefig(r'C:\Users\a5278\Documents\passive_acoustics\mooringmap.jpg',dpi=200)

#%%


fig=plt.figure(num=2)
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


plt.scatter(surveydata['lon'],surveydata['lat'],10,10*np.log10(surveydata['nasc_obs']),transform=ccrs.PlateCarree() )
plt.clim([0,50])

plt.colorbar(label='NASC')



