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


from pykrige.ok import OrdinaryKriging



class MplCanvas(FigureCanvasQTAgg ):

    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(figsize=None, dpi=dpi,facecolor='gray')
        # self.axes = self.fig.add_subplot(111)
        # self.axes.set_facecolor('gray')

        super(MplCanvas, self).__init__(self.fig)



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
        
  
        
        def scan_folder():
            if len(self.df_files)>0:      
                self.df_files['path'] = glob.glob(self.folder_source+'\\*.raw')  
                self.df_files['date'] = pd.to_datetime( self.df_files['path'].str.split('\\').str[-1].values,format='D%Y%m%d-T%H%M%S.raw' )
                self.df_files =  self.df_files.sort_values('date',ascending=False)

                print('found '+str(len(self.df_files)) + ' files')
            else:
                print('No folder')

        def openfolderfunc():
            self.folder_source = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Source folder with raw files')
            self.df_files['path'] = glob.glob(self.folder_source+'\\*.raw')  
            
            # print(self.df_files['path'].str.split('\\').str[-1])
            self.df_files['date'] = pd.to_datetime( self.df_files['path'].str.split('\\').str[-1],format='D%Y%m%d-T%H%M%S.raw' )
            ix_time= (self.df_files['date'] >= self.startdate.dateTime().toPyDateTime() ) & (self.df_files['date'] <= self.enddate.dateTime().toPyDateTime() )
            
            self.df_files['status'] = 0
            self.df_files.loc[ix_time,'status'] = 1
            
            self.df_files =  self.df_files.sort_values('date',ascending=False)
            
            self.filecounter=-1   
            self.df_nasc=pd.DataFrame([])

            print( self.df_files  )   
            
            if self.checkbox_log.isChecked():
                self.folder_target = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Target folder for saving processed data')
                
            
            
            # fname_canidates, ok = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()",'',"Raw Files (*.raw)")
            # if len( fname_canidates ) >0:                   
            #    self.filenames = np.array( fname_canidates )   
            #    self.filecounter=-1   

            #    print( self.filenames  )    

                
        def next_file():
             if len(self.df_files)>0:
                print('old filecounter is: '+str(self.filecounter))
                self.filecounter=self.filecounter+1
                
                if self.filecounter>len(self.df_files)-1:
                        self.filecounter=len(self.df_files)-1
                        print('That was it')
                rawfile = self.df_files.loc[self.filecounter,'path']            
                read_raw(rawfile)
                detect_krill_swarms(rawfile)                 
             
 
        def previous_file():
             if len(self.df_files)>0:
                print('old filecounter is: '+str(self.filecounter))
                self.filecounter=self.filecounter-1
                
                if self.filecounter<0:
                        self.filecounter=0
                rawfile = self.df_files.loc[self.filecounter,'path']            
                read_raw(rawfile)
                detect_krill_swarms(rawfile)            
            
        def read_raw(rawfile):
            # if len(self.filenames)>0:
                
            #    for rawfile in  self.filenames:
                
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
           
                       print(sv.shape)
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
                       # self.echodata[rawfile] =   self.ekdata

                       # print( self.echodata )    
                       
                   # raw_sv=np.empty([len(raw_freq)])

                
        def detect_krill_swarms(rawfile):
          # if len(self.filenames)>0:                
          #      for rawfile in  self.filenames:       
                   
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
                   
                   nasc_swarm[nasc_swarm>20000]=np.nan
                                   
                    
                   df_sv=pd.DataFrame( np.transpose(Sv120sw) )
                   df_sv.index=t120
                   df_sv.columns=r120
                   # print('df_sv')
                   
                   df_nasc_file=pd.DataFrame([])
                   df_nasc_file['time']=self.positions['ping_time']
                   df_nasc_file['lat']=self.positions['latitude']
                   df_nasc_file['lon']=self.positions['longitude']
                   df_nasc_file['nasc']=nasc_swarm
                   df_nasc_file.index=self.positions['ping_time']
                   df_nasc_file=df_nasc_file.resample('5s').mean()
                   
                   self.df_nasc = pd.concat([ self.df_nasc,df_nasc_file ])
                   print(self.df_nasc)

                   
                   # self.echodata_swarm[rawfile] = df_sv     
                   
                   plot_echogram(sv,df_sv,rawfile)
             
               
                   
                   
        def detect_krill_dbdiff():
             print('d')
               
        # def plot_echogram(df_sv,df_swarm,rawfile):
        #     print('d')
        #     self.canvas.fig.clf() 
        #     self.canvas.fig.set_facecolor('gray')
    
        #     self.canvas.axes1 = self.canvas.fig.add_subplot(221)
        #     self.canvas.axes1.set_facecolor('gray')   
            
        #     img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40)        
        #     self.canvas.fig.colorbar(img)
            
        #     self.canvas.axes1.set_ylabel('Depth')
        #     self.canvas.axes1.set_xlabel('Time')
        #     self.canvas.axes1.set_title(rawfile.split('\\')[-1])
        #     self.canvas.axes1.grid()
                   
    
    
        #     self.canvas.axes2 = self.canvas.fig.add_subplot(223, sharex=self.canvas.axes1, sharey=self.canvas.axes1)
        #     self.canvas.axes2.set_facecolor('gray')            
    
        #     self.canvas.axes2.set_ylabel('Depth')
        #     self.canvas.axes2.set_xlabel('Time')
           
        #     img=self.canvas.axes2.imshow( np.rot90( df_swarm.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40)        
        #     self.canvas.fig.colorbar(img)
            
        #     self.canvas.axes2.grid()
            
            
        #     self.canvas.axes3 = self.canvas.fig.add_subplot(222)
        #     self.canvas.axes3.set_facecolor('gray')   
        #     self.canvas.axes3.plot( self.df_nasc['nasc'],'-r' )   
        #     self.canvas.axes3.grid()
   
    
        #     self.canvas.axes4 = self.canvas.fig.add_subplot(224)
        #     self.canvas.axes4.set_facecolor('gray')   
        #     sc=self.canvas.axes4.scatter( self.df_nasc['lon'],self.df_nasc['lat'],20,self.df_nasc['nasc'],vmin=0,vmax=20000 )   
        #     self.canvas.axes4.grid()
        #     self.canvas.fig.colorbar(sc)

        #     self.canvas.fig.tight_layout()
        #     self.canvas.draw()
        #     self.canvas.flush_events()
      
        def plot_echogram(df_sv,df_swarm,rawfile):
            self.canvas.fig.clf() 
            self.canvas.fig.set_facecolor('gray')
    
            self.canvas.axes1 = self.canvas.fig.add_subplot(221)
            self.canvas.axes1.set_facecolor('k')   
            
            xt = [ 0, (df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60) ,  df_sv.columns[-1].astype(float) , df_sv.columns[0].astype(float)]
            # print(xt)
            img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
            self.canvas.fig.colorbar(img)
            
            self.canvas.axes1.set_ylabel('Depth [m]')
            self.canvas.axes1.set_xlabel('Time [min]')
            self.canvas.axes1.set_title(rawfile.split('\\')[-1])
            self.canvas.axes1.grid()
                   
    
    
            self.canvas.axes2 = self.canvas.fig.add_subplot(223, sharex=self.canvas.axes1, sharey=self.canvas.axes1)
            self.canvas.axes2.set_facecolor('k')            
    
            self.canvas.axes2.set_ylabel('Depth [m]')
            self.canvas.axes2.set_xlabel('Time [min]')
           
            img=self.canvas.axes2.imshow( np.rot90( df_swarm.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
            self.canvas.fig.colorbar(img)
            
            self.canvas.axes2.grid()
            
            
            self.canvas.axes3 = self.canvas.fig.add_subplot(222)
            self.canvas.axes3.set_facecolor('gray')   
            self.canvas.axes3.plot( self.df_nasc['nasc'],'.r' )   
            self.canvas.axes3.plot( self.df_nasc['nasc'].resample('10min').mean() ,'-k' )   
            self.canvas.axes3.grid()
            self.canvas.axes3.set_title('Krill NASC')
  
    
            self.canvas.axes4 = self.canvas.fig.add_subplot(224)
            self.canvas.axes4.set_facecolor('k')   
            
            # nasc_cutoff=1500
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
 

           
            
        # def plot_echogram(df_sv,df_swarm,rawfile):
        #     print('d')
        #     self.canvas.fig.clf() 
        #     self.canvas.fig.set_facecolor('gray')

        #     self.canvas.axes1 = self.canvas.fig.add_subplot(211)
        #     self.canvas.axes1.set_facecolor('gray')   
            
        #     img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40)        
        #     self.canvas.fig.colorbar(img)
            
        #     self.canvas.axes1.set_ylabel('Depth')
        #     self.canvas.axes1.set_xlabel('Time')
        #     self.canvas.axes1.set_title(rawfile)
        #     self.canvas.axes1.grid()
            
          


        #     self.canvas.axes2 = self.canvas.fig.add_subplot(212, sharex=self.canvas.axes1, sharey=self.canvas.axes1)
        #     self.canvas.axes2.set_facecolor('gray')            

        #     self.canvas.axes2.set_ylabel('Depth')
        #     self.canvas.axes2.set_xlabel('Time')
           
        #     img=self.canvas.axes2.imshow( np.rot90( df_swarm.values ) , aspect='auto',cmap='viridis',origin = 'lower',vmin=-80,vmax=-40)        
        #     self.canvas.fig.colorbar(img)
            
        #     self.canvas.axes2.grid()
            
        #     self.canvas.fig.tight_layout()
        #     self.canvas.draw()
        #     self.canvas.flush_events()
     
        def automatic_processing():
            if len(self.df_files)>0:     
                # et=[]
                # for rawfile in  self.filenames:            
                #    raw_obj = EK80.EK80()
                #    raw_obj.read_raw(rawfile)
                #    et.append( raw_obj.end_time)
                # et=np.array(et)
                # ix_sort= np.flip( np.argsort(et) )
                # # print(np.argsort(et))
                self.statusBar().setStyleSheet("background-color : rgb(115, 6, 6)")
                self.label_1 = QtWidgets.QLabel("Automatic processing activated")
                self.statusBar().addPermanentWidget(self.label_1)
                
                for rawfile in  self.df_files['path']:
                  try:  


                      self.statusBar().showMessage("Processing "+rawfile)

                      read_raw(rawfile)
                      detect_krill_swarms(rawfile)    
                  except Exception as e:
                    print(e)                 
                self.statusBar().setStyleSheet("background-color : k")
                self.statusBar().removeWidget(self.label_1)
                   
        menuBar = self.menuBar()

        # Creating menus using a title
        openMenu = menuBar.addAction("Select folders")
        openMenu.triggered.connect(openfolderfunc)
        
        # autoMenu = menuBar.addMenu("Automatic processing")
        # m_swarm = autoMenu.addAction("Swarm detection")
        # m_swarm.triggered.connect(automatic_processing)

        startautoMenu = menuBar.addAction("Start processing")
        startautoMenu.triggered.connect(automatic_processing)
        
        exitautoMenu = menuBar.addAction("Stop processing")
        # exitautoMenu.triggered.connect(stop_processing)     
    
  
        quitMenu = menuBar.addAction("Quit")
        quitMenu.triggered.connect(QtWidgets.QApplication.instance().quit)     
    

        toolbar = QtWidgets.QToolBar()
        button_previous=QtWidgets.QPushButton('<--Previous')
        button_previous.clicked.connect(previous_file)
        toolbar.addWidget(button_previous)
        button_next=QtWidgets.QPushButton('Next-->')
        button_next.clicked.connect(next_file)
        toolbar.addWidget(button_next)
        
        self.checkbox_log=QtWidgets.QCheckBox('Logging')
        self.checkbox_log.setChecked(False)            
        toolbar.addWidget(self.checkbox_log)
        
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
        
        
        timer = QtCore.QTimer()
        timer.timeout.connect(scan_folder)
        timer.start(1000)  


        
app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Krill3000 ")    
app.setStyleSheet(qdarktheme.load_stylesheet())
  


w = MainWindow()

# timer = QtCore.QTimer()
# timer.timeout.connect(scan_folder(w))
# timer.start(1000)  

sys.exit(app.exec_())
     