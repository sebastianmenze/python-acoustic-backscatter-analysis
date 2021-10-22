# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:47:59 2021

@author: Administrator
"""



# make_nmea_GLL

import pandas as pd
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def interpolate(t,lat,lon):
    ts = pd.date_range(start=t[0],end=t[1], freq='5S')
#    lat = np.array([-63.15684,-63.1572])
#    lon = np.array([-59.83428, -59.83581])
    
    # Interpolate
    k = np.arange(len(ts))
    lon_new = lon[0]+(lon[1]-lon[0])*k/100
    lat_new = lat[0]+(lat[1]-lat[0])*k/100
    
    #plt.plot(lon,lat,'k',lon_new,lat_new,'bo')
    
    data = {'Lat':lat_new,'Lon':lon_new}
    
    A_interp = pd.DataFrame(data,columns=['Lat','Lon'],index=ts)
    
    return A_interp

def lat2latdm(lat):
    latdeg = math.floor(abs(lat))
    return '{}{:05.2f}'.format(latdeg,60*(abs(lat)-latdeg))

def lon2londm(lon):
    londeg = math.floor(abs(lon))
    return '{:03}{:05.2f}'.format(londeg,60*(abs(lon)-londeg))

iridiumtable=r"D:\sailbuoy_2021\SB AKBM Data Jan March 2021\sailbuoy_2021_lat_lon.csv"

A = pd.read_csv(iridiumtable, sep=',',dtype={'time':'str'})
# A['time']= pd.to_datetime( A['time'] , '%Y-%m-%d %H:%M:%S UTC')
A['time']= pd.to_datetime( A['time'])
A['time']=A['time'].dt.tz_localize(None)

# Divide the messages into days.
# naive = dt.replace(tzinfo=None)

# startdate=dt.datetime(2021,1,1,tzinfo=dt.timezone.utc)
startdate=dt.datetime(2021,1,1)

datelist=pd.to_datetime(A['time'])
maxdate=max(datelist).to_pydatetime()
timed=maxdate-startdate
timed.days

for i_day in range(0,timed.days):
  
  currentday=startdate+dt.timedelta(days=i_day)
# filename = 'nmea_datagrams.csv'
# f = open(filename,'w')
  
  ix=np.where( (datelist>currentday) & (datelist<(currentday+dt.timedelta(days=1))) )
  
  if len(ix[0])>0:
    
    filename = 'nmea_sailbuoy_'+currentday.strftime('%Y-%m-%d')+'.csv'
    f = open(filename,'w')
  
    for i_k in range(0,ix[0].shape[0]):
      k=ix[0][i_k]
      
      lat = A.iloc[k]['latitude']
      lon = A.iloc[k]['longitude']
      lat_next = A.iloc[k+1]['latitude']
      lon_next = A.iloc[k+1]['longitude']
      
      date =  A.iloc[k]['time'].strftime('%Y-%m-%d')  
      
      timestamp = A.iloc[k]['time']
      timestamp_next = A.iloc[k+1]['time']
      td = timestamp_next-timestamp
      
      interTime = [timestamp+dt.timedelta(seconds=5),timestamp_next-dt.timedelta(seconds=5)]
      
      A_interp = interpolate(interTime,[lat,lat_next],[lon,lon_next])
      
      timestamp_write =  A.iloc[k]['time'].strftime('%Y-%m-%d %H:%M:%S' )  
      
      lat_dm = lat2latdm(lat)
      lon_dm = lon2londm(lon)
      
      # $GPGLL,4916.45,N,12311.12,W,225444,A,*1D
      # Write the original timestamp from the file
      f.write('{},$GPGLL,{},S,{},W,{},A,*1D\n'.format(A.iloc[k]['time'],
              lat_dm,lon_dm,timestamp_write))
      
      # Write the interpolated timestamps
      for index,row in A_interp.iterrows():
          f.write('{},$GPGLL,{},S,{},W,{},A,*1D\n'.format(str(index),\
                  lat2latdm(row['Lat']),\
                  lon2londm(row['Lon']),\
                  "".join(str(index).split()[1].split(':'))))
          
    f.close()
        
    
    


#%%
import pandas as pd
from datetime import datetime
import glob
import datetime as dt

# Import the tools for doing EK stuff
import EKtools
    
def main():
    # Input path
#    in_path = r't:\cmr-st-data\SB\Antarctic 2019 data'
    #in_path = r'C:\Users\wurst\Documents\postdoc_krill\sailbuoydata\raw'
    
    # breakpoint()
    
    in_path = r'D:\sailbuoy_2021\SB AKBM Data Jan March 2021'
    # Output path
    #out_path = r'C:\Users\wurst\Documents\postdoc_krill\sailbuoydata\raw\rawwithgps'
    out_path = r'D:\sailbuoy_2021\2021_sailbuoy_withgps'

    # Get all files in the input folder.
    files = glob.glob('{}\*.raw'.format(in_path))
    outfiles = glob.glob('{}\*.raw'.format(out_path))
    myList = [i.split('\\')[-1] for i in outfiles] 
    
    # file=files[0]
    for file in files:
        print('\n'+'='*80)
        print('Working on file: {}'.format(file))
        print('\n'+'='*80)
        filename = file.split('\\')[-1]
        
        if not filename in myList:
    
          # Open the input and output files to stream
          stream = open(r'{}\{}'.format(in_path,filename), 'rb')
          output = open(r'{}\{}'.format(out_path,filename),'wb')
          
          currentday=dt.datetime.strptime(filename.split('-')[2],'D%Y%m%d')
          
          # Open the file containing the positions
          nmeaname='nmea_sailbuoy_'+currentday.strftime('%Y-%m-%d')+'.csv'
          gpgll = pd.read_csv(r'D:\sailbuoy_2021/'+nmeaname,header=None,dtype=str)
          
          end = False
          # Make a 
          lastDGtime = datetime(1900,1,1)
          count = 0
          while not end:
              # Read the datagram and get the time
              try:
                  datagram = EKtools.readDatagram(stream)
                  
                  header = EKtools.parseDatagramHeader(datagram)
          
                  thistime = header['dgTime']
                  #print(thistime)
                  
                  #-----------------------------------------
                  # Insert the GPGLL datagram if supposed to
                  #-----------------------------------------
                  # Check whether it is time to insert new GPGLL
                  dts = pd.to_datetime(gpgll.iloc[:][0])
                  inds = (dts > lastDGtime) & (dts < thistime)
                  print('The last datagram time: {}'.format(lastDGtime))
                  print('The current datagram time: {}'.format(thistime))
                  if not gpgll.loc[inds].empty:
                      gpgll_msg = gpgll.loc[inds].iloc[0][1:].map(str).str.cat(sep=',')
                      gpgll_time = dts[inds]
                      # Remove the checksum
                      # Insert carriage return and linefeed
                      gpgll_msg = '{}\r\n'.format(gpgll_msg[:-4])
                      print('='*30)
                      print('Will insert current message: {}'.format(gpgll_msg[:-2]))
                      print('='*30)
                      if count > 1: # Wait to at least 5 datagrams are read, to avoid interferring with the start of the file
                          EKtools.writeGPGLL(output,gpgll_msg,gpgll_time)
                  else:
                      print('Found no message to insert.')
                  
                  lastDGtime = thistime
                  
                  # Write the current datagram
                  EKtools.writeDatagram(output,datagram)
                  
                  count += 1
              except:
                  print('Reached end of file - closing the file.')
                  end = True
                  stream.close()
                  output.close()
            
if __name__ == "__main__":
    main()
