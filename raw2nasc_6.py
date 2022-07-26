# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:45:43 2022

@author: Administrator
"""

from skimage.transform import  resize

from echolab2.instruments import EK80, EK60

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

from scipy.ndimage.filters import uniform_filter1d
from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from skimage.transform import  resize

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_signal2noise as pip
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH
from echopy import mask_signal2noise as mSN

from skimage.transform import rescale, resize 

from pyproj import Geod
geod = Geod(ellps="WGS84")

import sys
import matplotlib
# matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5.QtWidgets import QShortcut
# from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import qdarktheme
import time

from pathlib import Path


from pykrige.ok import OrdinaryKriging



class MplCanvas(FigureCanvasQTAgg ):

    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(figsize=None, dpi=dpi,facecolor='gray')
        # self.axes = self.fig.add_subplot(111)
        # self.axes.set_facecolor('gray')

        super(MplCanvas, self).__init__(self.fig)


class Worker(QtCore.QThread):
      
    def scan_folder(self):
        
            new_df_files = pd.DataFrame([])           
            new_df_files['path'] = glob.glob( os.path.join( self.folder_source,'*.raw') )  
            new_df_files['date'] = pd.to_datetime( new_df_files['path'].str.split('\\').str[-1].values,format='D%Y%m%d-T%H%M%S.raw' )
            new_df_files['to_do']=True 
            
            self.df_files=pd.concat([self.df_files,new_df_files])
            self.df_files.drop_duplicates(inplace=True)
            
            self.df_files =  self.df_files.sort_values('date',ascending=False)
            self.df_files=self.df_files.reset_index(drop=True)
            
            print('found '+str(len(self.df_files)) + ' raw files')
         
            
            # look for already processed data
            self.df_files['to_do']=True            
            nasc_done = glob.glob( '*_nasctable.h5' )
            nasc_done= list(map(lambda x: x.replace('_nasctable.h5','') , nasc_done))        
            names = self.df_files['path'].apply(lambda x: Path(x).stem)              
            # print(names)
            # print(nasc_done)
            ix_done= names.isin( nasc_done  )  

            # print(ix_done)
            self.df_files.loc[ix_done,'to_do'] = False        
            self.n_todo=np.sum(self.df_files['to_do'])
            print('To do: ' + str(self.n_todo))
            
            
    def pass_folder(self,folder_source):
        self.folder_source=folder_source

    def scan_and_process(self):
        if self.not_processing:
            self.not_processing=False
            self.scan_folder()         
            self.process()
            self.not_processing=True

    def process(self):
        ix_todo=self.df_files['to_do']==True
        files=self.df_files.loc[ix_todo,'path'].values
        for rawfile in files:
                print('working on '+rawfile)
                self.read_raw(rawfile)
                self.detect_krill_swarms()                
                self.df_nasc_file.to_hdf( Path(rawfile).stem + '_nasctable.h5', key='df', mode='w'  )
                self.df_sv_swarm.resample('1min').mean().to_hdf( Path(rawfile).stem + '_sv_swarm.h5', key='df', mode='w'  )
                # self.df_files.loc[i,'to_do'] = False
          
    
    def start(self):
        print('go')
        
        self.not_processing=True        
        self.df_files = pd.DataFrame(columns=['path','date','to_do'])
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.scan_and_process)       
        self.timer.start(1000) 

        

    def stop(self):
        # self.keepRunning = False
        self.terminate()

    def read_raw(self,rawfile):       
   
        raw_obj = EK80.EK80()
        raw_obj.read_raw(rawfile)
         
        print(raw_obj)
        
        raw_freq= list(raw_obj.frequency_map.keys())
        
        self.ekdata=dict()
        
        for f in raw_freq:
        
            raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]                   
            cal_obj = raw_data.get_calibration()
            sv_obj = raw_data.get_sv(calibration = cal_obj)              
            self.positions = raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1]
            
            svr = np.transpose( 10*np.log10( sv_obj.data ) )
            
            r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
            t=sv_obj.ping_time

            sv=  resize(svr,[ len(r) , len(t) ] )

            # print(sv.shape)
            t=sv_obj.ping_time
            
             # estimate and correct background noise       
            p         = np.arange(len(t))                
            s         = np.arange(len(r))          
            bn, m120bn_ = gBN.derobertis(sv, s, p, 5, 20, r, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
            b=pd.DataFrame(bn)
            bn=  b.interpolate(axis=1).interpolate(axis=0).values                        
            sv_clean     = tf.log(tf.lin(sv) - tf.lin(bn))

          # -------------------------------------------------------------------------
          # mask low signal-to-noise 
            msn             = mSN.derobertis(sv_clean, bn, thr=12)
            sv_clean[msn] = np.nan
 
         # get mask for seabed
            mb = mSB.ariza(sv, r, r0=20, r1=1000, roff=0,
                               thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
            sv_clean[mb]=-999
                                                
            df_sv=pd.DataFrame( np.transpose(sv_clean) )
            df_sv.index=t
            df_sv.columns=r
            
            self.ekdata[f]=df_sv

            
    def detect_krill_swarms(self):
         # sv= self.echodata[rawfile][ 120000.0] 
         sv= self.ekdata[ 120000.0]                    
         t120 =sv.index
         r120 =sv.columns.values

         Sv120=  np.transpose( sv.values )
         # get swarms mask
         k = np.ones((3, 3))/3**2
         Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
         p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
         s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
         m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
                                    mincan=(3,10), maxlink=(3,15), minsho=(3,15))

        # -------------------------------------------------------------------------
        # get Sv with only swarms
         Sv120sw =  Sv120.copy()
         Sv120sw[~m120sh] = np.nan
  
         ixdepthvalid= (r120>=20) & (r120<=500)
         Sv120sw[~ixdepthvalid,:]=np.nan
  
         
         cell_thickness=np.abs(np.mean(np.diff( r120) ))               
         nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cell_thickness ,axis=0)   
         
         # nasc_swarm[nasc_swarm>20000]=np.nan
                         
          
         df_sv_swarm=pd.DataFrame( np.transpose(Sv120sw) )
         df_sv_swarm.index=t120
         df_sv_swarm.columns=r120
          # print('df_sv')
         
         df_nasc_file=pd.DataFrame([])
         df_nasc_file['time']=self.positions['ping_time']
         df_nasc_file['lat']=self.positions['latitude']
         df_nasc_file['lon']=self.positions['longitude']
         df_nasc_file['distance_m']=np.append(np.array([0]),geod.line_lengths(lons=self.positions['longitude'],lats=self.positions['latitude']) )
         
         
         df_nasc_file['nasc']=nasc_swarm
         df_nasc_file.index=self.positions['ping_time']
         
         # df_nasc_file=df_nasc_file.resample('5s').mean()
         
         
         # print(df_nasc_file)
         self.df_nasc_file = df_nasc_file
         self.df_sv_swarm = df_sv_swarm
         # self.df_sv = sv

         print('Krill detection complete: '+str(np.sum(nasc_swarm)) ) 

               
####################################################################################    
        
class MainWindow(QtWidgets.QMainWindow):
    

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas =  MplCanvas(self, dpi=150)
                
        self.echodata=dict()
        self.echodata_swarm=dict()
        self.df_nasc=pd.DataFrame([])

        
        self.filecounter=-1
        self.filenames = None
        self.df_files = pd.DataFrame([])

        self.folder_source=''
        self.statusBar().setStyleSheet("background-color : k")
        self.label_folders = QtWidgets.QLabel("Source: "+self.folder_source)
        self.statusBar().addPermanentWidget(self.label_folders)          
        
        
       # Thread:
           
        self.thread = QtCore.QThread()         
        self.worker = Worker()
        self.worker.moveToThread(self.thread)


                   
        menuBar = self.menuBar()

        # Creating menus using a title
        openMenu = menuBar.addAction("Select folders")
        openMenu.triggered.connect(self.openfolderfunc)
        
        # autoMenu = menuBar.addMenu("Automatic processing")
        # m_swarm = autoMenu.addAction("Swarm detection")
        # m_swarm.triggered.connect(automatic_processing)

        self.startautoMenu = menuBar.addAction("Start processing")
        self.startautoMenu.triggered.connect(self.startClicked)
        
        self.exitautoMenu = menuBar.addAction("Stop processing")
        self.exitautoMenu.triggered.connect(self.stopClicked)     

        # self.exitautoMenu = menuBar.addAction("Update plots")
        # self.exitautoMenu.triggered.connect(self.update_plots)     
        
  
        quitMenu = menuBar.addAction("Quit")
        quitMenu.triggered.connect(QtWidgets.QApplication.instance().quit)     
    

        toolbar = QtWidgets.QToolBar()
        
        # button_previous=QtWidgets.QPushButton('<--Previous')
        # button_previous.clicked.connect(self.previous_file)
        # toolbar.addWidget(button_previous)
        # button_next=QtWidgets.QPushButton('Next-->')
        # button_next.clicked.connect(self.next_file)
        # toolbar.addWidget(button_next)
        
        self.checkbox_log=QtWidgets.QCheckBox('Update plots')
        self.checkbox_log.setChecked(False)            
        toolbar.addWidget(self.checkbox_log)
        self.checkbox_log.toggled.connect(self.update_plots)            
        
        toolbar.addWidget(QtWidgets.QLabel('Start:'))
        self.startdate = QtWidgets.QDateEdit(calendarPopup=True)
        self.startdate.setDateTime(QtCore.QDateTime.fromString('1970-01-01', "yyyy-MM-dd") )
        toolbar.addWidget( self.startdate)

        toolbar.addWidget(QtWidgets.QLabel('End:'))
        self.enddate = QtWidgets.QDateEdit(calendarPopup=True)
        self.enddate.setDateTime(QtCore.QDateTime.fromString('2100-01-01', "yyyy-MM-dd") )
        toolbar.addWidget( self.enddate)
         
        tnav = NavigationToolbar( self.canvas, self)       
        toolbar.addWidget(tnav)
       
        outer_layout = QtWidgets.QVBoxLayout()
        outer_layout.addWidget(toolbar)
        outer_layout.addWidget(self.canvas)
    
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
        
        
        self.show()  
        
        homepath =str(os.path.expanduser("~"))
        workpath=  os.path.join(homepath,'krilldata')
        if not os.path.exists(workpath):
            os.mkdir( workpath )
        
        os.chdir(workpath)
        
    


        

    def openfolderfunc(self):
        self.folder_source = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Source folder with raw files')
        self.df_files['path'] = glob.glob(self.folder_source+'\\*.raw')  
        
        # print(self.df_files['path'].str.split('\\').str[-1])
        self.df_files['date'] = pd.to_datetime( self.df_files['path'].str.split('\\').str[-1],format='D%Y%m%d-T%H%M%S.raw' )
        ix_time= (self.df_files['date'] >= self.startdate.dateTime().toPyDateTime() ) & (self.df_files['date'] <= self.enddate.dateTime().toPyDateTime() )
        
        self.df_files['status'] = 0
        self.df_files.loc[ix_time,'status'] = 1
        
        self.df_files =  self.df_files.sort_values('date',ascending=False)
        
        # look for already processed data
        nasc_done = glob.glob( '*_nasctable.h5' )
        nasc_done= list(map(lambda x: x.replace('_nasctable.h5','') , nasc_done))        
        names = self.df_files['path'].apply(lambda x: Path(x).stem)              
        ix_done= names.isin( nasc_done  )  
         
        self.filecounter=-1   
        self.df_nasc=pd.DataFrame([])

        print( self.df_files  )   
        
        # if self.checkbox_log.isChecked():
        #     self.folder_target = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Target folder for saving processed data')

        # self.statusBar().setStyleSheet("background-color : k")
        self.statusBar().removeWidget(self.label_folders) 
        self.label_folders = QtWidgets.QLabel("Source: "+self.folder_source )
        self.statusBar().addPermanentWidget(self.label_folders)                
    

            
    # def next_file(self):
    #      if len(self.df_files)>0:
    #         print('old filecounter is: '+str(self.filecounter))
    #         self.filecounter=self.filecounter+1
            
    #         if self.filecounter>len(self.df_files)-1:
    #                 self.filecounter=len(self.df_files)-1
    #                 print('That was it')
    #         # rawfile = self.df_files.loc[self.filecounter,'path']            
    #         self.read_raw()
    #         self.detect_krill_swarms()                 
         
 
    # def previous_file(self):
    #      if len(self.df_files)>0:
    #         print('old filecounter is: '+str(self.filecounter))
    #         self.filecounter=self.filecounter-1
            
    #         if self.filecounter<0:
    #                 self.filecounter=0
    #         self.read_raw()
    #         self.detect_krill_swarms()            
        
    # def read_raw(self):
    #     # if len(self.filenames)>0:
            
    #     #    for rawfile in  self.filenames:
    #            rawfile = self.df_files.loc[self.filecounter,'path']            
          
    #            raw_obj = EK80.EK80()
    #            raw_obj.read_raw(rawfile)
                
    #            print(raw_obj)
               
    #            raw_freq= list(raw_obj.frequency_map.keys())
               
    #            self.ekdata=dict()
               
    #            for f in raw_freq:
               
    #                raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]                   
    #                cal_obj = raw_data.get_calibration()
    #                sv_obj = raw_data.get_sv(calibration = cal_obj)              
    #                self.positions = raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1]
                   
    #                svr = np.transpose( 10*np.log10( sv_obj.data ) )
                   
    #                r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
    #                t=sv_obj.ping_time
       
    #                sv=  resize(svr,[ len(r) , len(t) ] )
       
    #                print(sv.shape)
    #                t=sv_obj.ping_time
                   
    #                 # estimate and correct background noise       
    #                p         = np.arange(len(t))                
    #                s         = np.arange(len(r))          
    #                bn, m120bn_ = gBN.derobertis(sv, s, p, 5, 20, r, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
    #                b=pd.DataFrame(bn)
    #                bn=  b.interpolate(axis=1).interpolate(axis=0).values                        
    #                sv_clean     = tf.log(tf.lin(sv) - tf.lin(bn))

    #              # -------------------------------------------------------------------------
    #              # mask low signal-to-noise 
    #                msn             = mSN.derobertis(sv_clean, bn, thr=12)
    #                sv_clean[msn] = np.nan
        
    #             # get mask for seabed
    #                mb = mSB.ariza(sv, r, r0=20, r1=1000, roff=0,
    #                                   thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
    #                sv_clean[mb]=-999
                                                       
    #                df_sv=pd.DataFrame( np.transpose(sv_clean) )
    #                df_sv.index=t
    #                df_sv.columns=r
                   
    #                self.ekdata[f]=df_sv
    #                # self.echodata[rawfile] =   self.ekdata

    #                # print( self.echodata )    
                   
    #            # raw_sv=np.empty([len(raw_freq)])

            
    # def detect_krill_swarms(self):
    #   # if len(self.filenames)>0:                
    #   #      for rawfile in  self.filenames:       
    #            rawfile = self.df_files.loc[self.filecounter,'path']            
               
    #            # sv= self.echodata[rawfile][ 120000.0] 
    #            sv= self.ekdata[ 120000.0]                    
    #            t120 =sv.index
    #            r120 =sv.columns.values

    #            Sv120=  np.transpose( sv.values )
    #            # get swarms mask
    #            k = np.ones((3, 3))/3**2
    #            Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
    #            p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
    #            s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
    #            m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
    #                                       mincan=(3,10), maxlink=(3,15), minsho=(3,15))

    #           # -------------------------------------------------------------------------
    #           # get Sv with only swarms
    #            Sv120sw =  Sv120.copy()
    #            Sv120sw[~m120sh] = np.nan
    
    #            ixdepthvalid= (r120>=20) & (r120<=500)
    #            Sv120sw[~ixdepthvalid,:]=np.nan
        
               
    #            cell_thickness=np.abs(np.mean(np.diff( r120) ))               
    #            nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cell_thickness ,axis=0)   
               
    #            nasc_swarm[nasc_swarm>20000]=np.nan
                               
                
    #            df_sv=pd.DataFrame( np.transpose(Sv120sw) )
    #            df_sv.index=t120
    #            df_sv.columns=r120
    #            # print('df_sv')
               
    #            df_nasc_file=pd.DataFrame([])
    #            df_nasc_file['time']=self.positions['ping_time']
    #            df_nasc_file['lat']=self.positions['latitude']
    #            df_nasc_file['lon']=self.positions['longitude']
    #            df_nasc_file['nasc']=nasc_swarm
    #            df_nasc_file.index=self.positions['ping_time']
    #            df_nasc_file=df_nasc_file.resample('5s').mean()
               
    #            self.df_nasc = pd.concat([ self.df_nasc,df_nasc_file ])
    #            print(self.df_nasc)

    #            self.df_swarm=df_sv
               
    #            # self.echodata_swarm[rawfile] = df_sv     
               
    #            self.plot_echogram()
         
    def update_plots(self):
        if self.checkbox_log.isChecked():
            print('Update plots')
            self.plottimer = QtCore.QTimer(self)
            self.plottimer.timeout.connect(self.scan_and_vizualize)
            self.plottimer.start(5000)  
        else:
            print('STOP  plots')
            self.plottimer.stop()  
            
               
    def scan_and_vizualize(self):
        
        nasc_done = glob.glob( '*_nasctable.h5' )
        df_nasc=pd.DataFrame([])
        for file in nasc_done:
            df=pd.read_hdf(file,key='df')
            df_nasc=pd.concat([df_nasc,df])
        self.df_nasc=df_nasc   

        sv_done = glob.glob( '*_sv_swarm.h5' )
        df_sv=pd.DataFrame([])
        for file in sv_done:
            df=pd.read_hdf(file,key='df')
            # df=df.resample('1min').mean()
            df_sv=pd.concat([df_sv,df])
            
        df_plot=df_sv.values
        
        if len(nasc_done)>0:    
            
            self.canvas.fig.clf() 
            self.canvas.fig.set_facecolor('gray')
            
            self.canvas.axes1 = self.canvas.fig.add_subplot(211)
            self.canvas.axes1.set_facecolor('k')   
            
            xt = [ 0, (df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60) ,  df_sv.columns[-1].astype(float) , df_sv.columns[0].astype(float)]
            # print(xt)
            img=self.canvas.axes1.imshow( np.rot90( df_plot ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
            self.canvas.fig.colorbar(img)
            
            self.canvas.axes1.set_ylabel('Depth [m]')
            self.canvas.axes1.set_xlabel('Time [min]')
            self.canvas.axes1.grid()
            
            
    
            self.canvas.axes3 = self.canvas.fig.add_subplot(223)
            self.canvas.axes3.set_facecolor('gray')   
            self.canvas.axes3.plot( self.df_nasc['nasc'],'.r' )   
            self.canvas.axes3.plot( self.df_nasc['nasc'].resample('10min').mean() ,'-k' )   
            self.canvas.axes3.grid()
            self.canvas.axes3.set_title('Krill NASC')
     
            self.canvas.axes4 = self.canvas.fig.add_subplot(224)
            self.canvas.axes4.set_facecolor('k')   
                          
            nasc_cutoff=self.df_nasc['nasc'].resample('10min').mean().max()
            if nasc_cutoff>20000:
                nasc_cutoff=20000
    
           
            try:
                
                df_krig=self.df_nasc.copy()
                df_krig=df_krig.resample('10min').mean()
                
                ix= (df_krig['lat'].notna()) 
                
                
                            
                a =df_krig.loc[ix,'nasc'].values.copy()
                a[a>nasc_cutoff]=nasc_cutoff
                
                latlim=[ self.df_nasc['lat'].min() , self.df_nasc['lat'].max()  ]
                lonlim=[ self.df_nasc['lon'].min() , self.df_nasc['lon'].max()  ]
                # print(latlim)
                # print(lonlim)
    
                OK = OrdinaryKriging(
                    360+df_krig.loc[ix,'lon'].values,
                    df_krig.loc[ix,'lat'].values,
                    a,
                    coordinates_type='geographic',
                    variogram_model="spherical",
                    # variogram_parameters = { 'range' :  np.rad2deg( 50 / 6378  ) ,'sill':700000,'nugget':600000}
                    )
                 
                d_lats=np.linspace(latlim[0],latlim[1],100)
                d_lons=np.linspace(lonlim[0],lonlim[1],100)
                
                # lat,lon=np.meshgrid( g_lats,g_lons )
                
                z1, ss1 = OK.execute("grid", d_lons, d_lats)                                  
                z_grid=z1.data
                z_grid[ ss1.data>np.percentile( ss1.data,75) ] =np.nan
                # print(z_grid)
                sc=self.canvas.axes4.imshow( z_grid ,vmin=0,vmax=nasc_cutoff, origin='lower',aspect='auto',extent=[lonlim[0],lonlim[1],latlim[0],latlim[1]] )   
                
            except:
                print('kriging error')       
                 
            sc=self.canvas.axes4.scatter( df_krig['lon'],df_krig['lat'],20,df_krig['nasc'],vmin=0,vmax=nasc_cutoff,edgecolor='k' )   
                        
            # sc=self.canvas.axes4.scatter( self.df_nasc['lon'],self.df_nasc['lat'],20,self.df_nasc['nasc'],vmin=0,vmax=nasc_cutoff,edgecolor='k' )   
            self.canvas.axes4.grid()
            self.canvas.fig.colorbar(sc)
    
    
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()                              
   

       


    def startClicked(self):
        if not self.thread.isRunning():

            
            self.worker.pass_folder(self.folder_source)

            self.thread.started.connect(self.worker.start)
            self.thread.start()
            
            self.startautoMenu.setEnabled(False)
            self.statusBar().setStyleSheet("background-color : rgb(115, 6, 6)")
            self.label_1 = QtWidgets.QLabel("Automatic processing activated")
            self.statusBar().addPermanentWidget(self.label_1)
            
    def stopClicked(self):
        self.thread.terminate()
        self.statusBar().setStyleSheet("background-color : k")
        self.statusBar().removeWidget(self.label_1)   
        self.startautoMenu.setEnabled(True)
      
        
        print(result)  
   


app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Krill3000 ")    
app.setStyleSheet(qdarktheme.load_stylesheet())
  


w = MainWindow()

# timer = QtCore.QTimer()
# timer.timeout.connect(w.scan_and_vizualize)
# timer.start(5000)  


sys.exit(app.exec_())
     