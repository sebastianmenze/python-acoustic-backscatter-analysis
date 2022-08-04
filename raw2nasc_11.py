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
# from skimage.transform import  resize

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
import cartopy.crs as ccrs
import cartopy as cart

from matplotlib.colors import ListedColormap
import re
import traceback



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
        
            dates=[]
            for fname in new_df_files['path']:
                
                datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                dates.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
            new_df_files['date'] = dates
        

            new_df_files['to_do']=True 
            
            
            self.df_files=pd.concat([self.df_files,new_df_files])
            self.df_files.drop_duplicates(inplace=True)
            
            self.df_files =  self.df_files.sort_values('date')
            self.df_files=self.df_files.reset_index(drop=True)
            
            print('found '+str(len(self.df_files)) + ' raw files')
         
            
            # look for already processed data
            self.df_files['to_do']=True    
            
            if os.path.isfile('list_of_rawfiles.csv'):
                df_files_done =  pd.read_csv('list_of_rawfiles.csv',index_col=0)
                df_files_done=df_files_done.loc[ df_files_done['to_do']==False,: ]
            
                names = self.df_files['path'].apply(lambda x: Path(x).stem)       
                names_done = df_files_done['path'].apply(lambda x: Path(x).stem)       
                
            # print(names)
            # print(nasc_done)
                ix_done= names.isin( names_done  )  

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
        
        

        echogram=pd.DataFrame([])    
        positions=pd.DataFrame([])    
        
        unit_length_min=pd.to_timedelta(10,'min')

        for index, row in self.df_files.iterrows():
            if self.keepworking & (row['to_do']==True):
                rawfile=row['path']
                print('working on '+rawfile)
                try:
                    
                    # breakpoint()
                    
                    echogram_file, positions_file = self.read_raw(rawfile)
                    
                    
                    echogram = pd.concat([ echogram,echogram_file ])
                    positions = pd.concat([ positions,positions_file ])
                    t=echogram.index
                    
                    # print(echogram)
                    
                    # print( [ t.max() , t.min() ])
                    
                    while (t.max() - t.min()) > unit_length_min:
                        
                        # print(  (t.min() + unit_length_min) > t)
                        ix_end = np.where( (t.min() + unit_length_min) > t )[0][-1]
                        ix_start=t.argmin()
                        # print([ix_start,ix_end])
                        
                    # accumulate 10 min snippet  
                        new_echogram = echogram.iloc[ix_start:ix_end,:]
                        new_positions = positions.iloc[ix_start:ix_end,:]
                        echogram = echogram.iloc[ix_end::,:]
                        positions = positions.iloc[ix_end::,:]
                        t=echogram.index

                        # try:
                        df_nasc_file, df_sv_swarm = self.detect_krill_swarms(new_echogram,new_positions)   
                        name = t.min().strftime('D%Y%m%d-T%H%M%S' )         
                        
                        df_sv_swarm[ new_echogram==-999 ] =-999
                        
                        df_nasc_file.to_hdf( name + '_nasctable.h5', key='df', mode='w'  )
                        df_nasc_file.to_csv( name + '_nasctable.csv'  )
                        df_sv_swarm.to_hdf( name + '_sv_swarm.h5', key='df', mode='w'  )
                        # self.df_files.loc[i,'to_do'] = False
                        # except Exception as e:
                        #   print(e)                      
                    self.df_files.loc[index,'to_do']=False            
                    self.df_files.to_csv('list_of_rawfiles.csv')
                   
                except Exception as e:
                    print(e)               
                    print(traceback.format_exc())

    def start(self):
        print('go')
        self.keepworking=True       
        self.not_processing=True        
        self.df_files = pd.DataFrame(columns=['path','date','to_do'])
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.scan_and_process)       
        self.timer.start(1000) 

        

    def stop(self):
        # self.keepRunning = False
        self.keepworking=False       
        # self.terminate()
        self.quit()

    def read_raw(self,rawfile):       
        df_sv=pd.DataFrame( [] )
        positions=pd.DataFrame( []  )
        
        print('Echsounder frequencies are: ')

   
        try:     
            raw_obj = EK80.EK80()
            raw_obj.read_raw(rawfile)
            print(raw_obj)
        except Exception as e:            
            print(e)       
            try:     
                raw_obj = EK60.EK60()
                raw_obj.read_raw(rawfile)
                print(raw_obj)
            except Exception as e:
                print(e)       
                
                                           
        
        raw_freq= list(raw_obj.frequency_map.keys())
        
        # self.ekdata=dict()
        
        # for f in raw_freq:
        f=120000.0    
        print(raw_freq)
     
        raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]                   
        cal_obj = raw_data.get_calibration()
        sv_obj = raw_data.get_sv(calibration = cal_obj)    
          
        positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )
       
        svr = np.transpose( 10*np.log10( sv_obj.data ) )
        
        # print(svr)

       
        # r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
        r=np.arange( 0 , sv_obj.range.max() , 0.5 )

        t=sv_obj.ping_time

        sv=  resize(svr,[ len(r) , len(t) ] )

       # print(sv.shape)
       
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
        
        # print(df_sv)
        # print(positions)
           
        return df_sv, positions

            
    def detect_krill_swarms(self,sv,positions):
         # sv= self.echodata[rawfile][ 120000.0] 
         # sv= self.ekdata[ 120000.0]          
         # breakpoint()
              
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
         # df_nasc_file['time']=positions['ping_time']
         df_nasc_file['lat']=positions['latitude']
         df_nasc_file['lon']=positions['longitude']
         df_nasc_file['distance_m']=np.append(np.array([0]),geod.line_lengths(lons=positions['longitude'],lats=positions['latitude']) )
         
         bottomdepth=[]
         for index, row in sv.iterrows():
             if np.sum(row==-999)>0:
                 bottomdepth.append( np.min(r120[row==-999]) )
             else:
                 bottomdepth.append( r120.max() )
            
         df_nasc_file['bottomdepth_m']=bottomdepth
            
           
         df_nasc_file['nasc']=nasc_swarm
         df_nasc_file.index=positions['ping_time']
         
         # df_nasc_file=df_nasc_file.resample('5s').mean()
         print('Krill detection complete: '+str(np.sum(nasc_swarm)) ) 
        
         return df_nasc_file, df_sv_swarm
         # print(df_nasc_file)
         # self.df_nasc_file = df_nasc_file
         # self.df_sv_swarm = df_sv_swarm
         # self.df_sv = sv


               
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
        openMenu = menuBar.addAction("Select source folder")
        openMenu.triggered.connect(self.openfolderfunc)
        
        # autoMenu = menuBar.addMenu("Automatic processing")
        # m_swarm = autoMenu.addAction("Swarm detection")
        # m_swarm.triggered.connect(automatic_processing)

        self.startautoMenu = menuBar.addAction("Start processing")
        self.startautoMenu.triggered.connect(self.startClicked)
        self.startautoMenu.setEnabled(False)
        
        self.exitautoMenu = menuBar.addAction("Stop processing")
        self.exitautoMenu.triggered.connect(self.stopClicked)     
        self.exitautoMenu.setEnabled(False)

        self.settingsMenu = menuBar.addAction("Settings")
        self.settingsMenu.triggered.connect(self.settings_func)     
        
        self.showfolderbutton =  menuBar.addAction('Show data folder')
        self.showfolderbutton.setEnabled(False)
        self.showfolderbutton.triggered.connect(self.showfoldefunc)     
       
        
        self.exportmenue = menuBar.addMenu("Export")
        # self.emailsetupbutton =  self.exportmenue.addAction('Setup Email')
        # self.emailsetupbutton =  self.exportmenue.addAction('Send Email')
        self.emailsetupbutton =  self.exportmenue.addAction('Export interpolated grid')
        self.emailsetupbutton =  self.exportmenue.addAction('Export echogram')
       
        # self.exitautoMenu = menuBar.addAction("Send emails")
        # self.exitautoMenu.triggered.connect(self.update_plots)     
        
  
        quitMenu = menuBar.addAction("Quit")
        quitMenu.triggered.connect(self.func_quit)     
    

        toolbar = QtWidgets.QToolBar()
        

        self.checkbox_emails=QtWidgets.QCheckBox('Send mails ')
        self.checkbox_emails.setChecked(False)            
        toolbar.addWidget(self.checkbox_emails)
        
        self.checkbox_kriging=QtWidgets.QCheckBox('Interpolate map')
        self.checkbox_kriging.setChecked(True)            
        toolbar.addWidget(self.checkbox_kriging)
        
        toolbar.addSeparator()
        
        self.checkbox_updateplot=QtWidgets.QCheckBox('Plot realtime ')
        self.checkbox_updateplot.setChecked(False)            
        toolbar.addWidget(self.checkbox_updateplot)
        self.checkbox_updateplot.toggled.connect(self.update_plots)            
        self.checkbox_updateplot.setEnabled(False)            

        toolbar.addWidget(QtWidgets.QLabel('Echogram length:'))        
        self.echogram_time_length = QtWidgets.QSpinBox()
        self.echogram_time_length.setValue(10)
        self.echogram_time_length.setMinimum(0)
        toolbar.addWidget(  self.echogram_time_length)
        toolbar.addWidget(QtWidgets.QLabel('min.'))        


        toolbar.addWidget(QtWidgets.QLabel('Map of last:'))        
        self.map_time_length = QtWidgets.QSpinBox()
        self.map_time_length.setValue(5)
        self.map_time_length.setMinimum(0)
        toolbar.addWidget(  self.map_time_length)
        toolbar.addWidget(QtWidgets.QLabel('days'))        


        toolbar.addSeparator()

        self.checkbox_manual=QtWidgets.QPushButton('Plot timespan ')
        toolbar.addWidget(self.checkbox_manual)
        self.checkbox_manual.clicked.connect(self.exploration_mode)            
        self.checkbox_manual.setEnabled(False)           
        
        # toolbar.addSeparator()


        # button_previous=QtWidgets.QPushButton('<--Previous')
        # button_previous.clicked.connect(self.previous_file)
        # toolbar.addWidget(button_previous)
        # button_next=QtWidgets.QPushButton('Next-->')
        # button_next.clicked.connect(self.next_file)
        # toolbar.addWidget(button_next)
        
       
        toolbar.addWidget(QtWidgets.QLabel('Min. date:'))
        self.startdate = QtWidgets.QDateTimeEdit(calendarPopup=True)
        self.startdate.setDateTime(QtCore.QDateTime.fromString('1970-01-01', "yyyy-MM-dd") )
        toolbar.addWidget( self.startdate)

        toolbar.addWidget(QtWidgets.QLabel('Max. date::'))
        self.enddate = QtWidgets.QDateTimeEdit(calendarPopup=True)
        self.enddate.setDateTime(QtCore.QDateTime.fromString('2100-01-01', "yyyy-MM-dd") )
        toolbar.addWidget( self.enddate)
        
        toolbar.addSeparator()
         
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
        
        color_ek80=np.array([[0.55294118, 0.49019608, 0.58823529, 1. ],
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
        self.cmap_ek80 = ListedColormap(color_ek80)

        # homepath =str(os.path.expanduser("~"))
        # workpath=  os.path.join(homepath,'krilldata')
        # if not os.path.exists(workpath):
        #     os.mkdir( workpath )
        
        # os.chdir(workpath)
    def settings_func(self):
        print('s')
    
    def showfoldefunc(self):    
         os.startfile(self.workpath)
        
    
    def exploration_mode(self):
        
        self.checkbox_updateplot.setChecked(False)           
        self.update_plots

        os.chdir(self.workpath)
        nasc_done = np.array( glob.glob( '*_nasctable.h5' ) )
        
        if len(nasc_done)>0:
                        
            nascfile_times=pd.to_datetime( nasc_done,format='D%Y%m%d-T%H%M%S_nasctable.h5' )     
            ix_time= (nascfile_times >= self.startdate.dateTime().toPyDateTime() ) & (nascfile_times <= self.enddate.dateTime().toPyDateTime() )
            nasc_load= nasc_done[ ix_time ]
            
            
            df_nasc=pd.DataFrame([])
            for file in nasc_load:
                df=pd.read_hdf(file,key='df')
                df_nasc=pd.concat([df_nasc,df  ])
            self.df_nasc=df_nasc.sort_values('ping_time')   
            
            
            # geod = Geod("+ellps=WGS84")
            # ll=geod.line_lengths(self.df_nasc['lon'], self.df_nasc['lat'])
            # ll= np.concatenate([np.array([0]), ll ])
            timedelta =np.concatenate([np.array([0]), np.diff( self.df_nasc.index.values ) /1e9 ])           
            self.df_nasc['speed_ms'] = self.df_nasc['distance_m'] / timedelta.astype(float)
            self.df_nasc['speed_knots']=self.df_nasc['speed_ms']*1.94384449
            ix_too_slow = self.df_nasc['speed_knots'] < 5           
            self.df_nasc.loc[ix_too_slow,'nasc']=np.nan           

            df_krig=self.df_nasc.resample('10min').mean() 
            # df_krig=self.df_nasc
            # print(  self.df_nasc.index.min().strftime('%Y-%m-%d %X') ) 
            
            # self.startdate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.min().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )
            # self.enddate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.max().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )

            
            latlim=[ self.df_nasc['lat'].min() , self.df_nasc['lat'].max()  ]
            lonlim=[ self.df_nasc['lon'].min() , self.df_nasc['lon'].max()  ]
    
    
    
            # get onl the latest echograms
            sv_done = np.array( glob.glob( '*_sv_swarm.h5' ) )
            
            sv_times=pd.to_datetime( sv_done,format='D%Y%m%d-T%H%M%S_sv_swarm.h5' )
            ix_time= (sv_times >= self.startdate.dateTime().toPyDateTime() ) & (sv_times <= self.enddate.dateTime().toPyDateTime() )
            sv_load= sv_done[ ix_time ]
            # distance_covered= geod.line_length(lons=df_nasc_load['lon'],lats=df_nasc_load['lat']) / 1000  
            distance_covered=self.df_nasc['distance_m'].sum() /1000         
            
            
            df_sv=pd.DataFrame([])
            for file in sv_load:
                df=pd.read_hdf(file,key='df')
                df=df.resample('1min').mean()
                df_sv=pd.concat([df_sv,df])
            # df_sv_plot=pd.DataFrame( resize(df_sv.values,[len(df_sv.columns,1000)] ))
            # df_sv_plot.index= np.linspace( df_sv.index.min(), df_sv.index.max() ,1000)
            self.df_sv=df_sv
            
            self.canvas.fig.clf() 
            self.canvas.fig.set_facecolor('gray')
            
            self.canvas.axes1 = self.canvas.fig.add_subplot(121)
            self.canvas.axes1.set_facecolor('k')   
            
            self.minut = np.linspace(-(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,len(df_sv))
            
            xt = [ self.minut[0], 0,  df_sv.columns[-1].astype(float) , df_sv.columns[0].astype(float)]
            # print(xt)
            img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap=self.cmap_ek80,origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
                
            # r=   df_sv.columns.astype(float)         
            # bottomdepth=[]
            # for index, row in df_sv.iterrows():
            #      bottomdepth.append( np.min(r[row==-999]) )
            # t=np.linspace(-(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,len(df_sv))  
            
            # self.canvas.axes1.plot( t,bottomdepth,'--w')        
            
            def on_xlims_change_echogram(event_ax):
                # print("updated xlims: ", event_ax.get_xlim())
                ix_dates = np.where( (self.minut > event_ax.get_xlim()[0]) & (self.minut < event_ax.get_xlim()[1]) )[0]
                print( self.df_sv.index[ix_dates[0]] )
                print( self.df_sv.index[ix_dates[-1]] )
                ix = (self.df_nasc.index > self.df_sv.index[ix_dates[0]] )   & (self.df_nasc.index < self.df_sv.index[ix_dates[-1]]     )         
                # print(self.df_nasc.loc[ix,:])
                try:
                    line = self.newline.pop(0)
                    line.remove()
            

                except:
                    print('noline')
                self.newline=self.canvas.axes4.plot( self.df_nasc.loc[ix,'lon'],self.df_nasc.loc[ix,'lat'],'-r',transform=ccrs.PlateCarree())
                self.canvas.draw()
                # self.canvas.flush_events()        
                    
                
            # def on_ylims_change(event_ax):
            #     print("updated ylims: ", event_ax.get_ylim())
            
            self.canvas.axes1.callbacks.connect('xlim_changed', on_xlims_change_echogram)
            # self.canvas.axes1.callbacks.connect('ylim_changed', on_ylims_change)
            
                        
         
            self.canvas.axes1.set_ylabel('Depth [m]')
            self.canvas.axes1.set_xlabel('Time [min]')
            self.canvas.axes1.grid( alpha=0.5, linestyle='--')

            self.canvas.axes1_2=self.canvas.axes1.twiny()
            self.canvas.axes1_2.set_xlim([-distance_covered,0])            
            self.canvas.axes1_2.set_xlabel('Distance [km]')          

            self.canvas.axes1_3=self.canvas.axes1.twinx()
            self.canvas.axes1_3.set_ylabel('1 min avg. Krill NASC')          
            self.canvas.fig.colorbar(img,label='$s_V$ [dB]',pad=0.15)
            
            df_sv[df_sv==-999]=np.nan
            r=df_sv.columns.astype(float)
            cell_thickness=np.abs(np.mean(np.diff( r) ))               
            nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, np.transpose(df_sv.values) /10)*cell_thickness ,axis=0)   
            t=np.linspace(-(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,len(nasc_swarm)) 
            self.canvas.axes1_3.plot(t,nasc_swarm,'-b')          
            self.canvas.axes1_3.set_xlim([t[0],t[-1]])          
            # self.canvas.axes1_3.set_xticks([])          
            if np.max(nasc_swarm)==0:
                self.canvas.axes1_3.set_ylim([0,1])                        
            else:
                self.canvas.axes1_3.set_ylim([0,2*np.max(nasc_swarm)])          
      
    
            # self.canvas.axes3 = self.canvas.fig.add_subplot(223)
            # self.canvas.axes3.set_facecolor('gray')   
            # self.canvas.axes3.plot( self.df_nasc['nasc'],'.r' )   
            # self.canvas.axes3.plot( df_krig['nasc'] ,'-k' )   
            # self.canvas.axes3.grid()
            # self.canvas.axes3.set_title('Krill NASC')
     

     
            central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
            central_lat = latlim[0]+(latlim[1]-latlim[0])/2
            # extent = [lonlim[0]+0.5,lonlim[1]-0.5, latlim[0]+0.5,latlim[1]-0.5]
            #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))
            
            self.canvas.axes4 = self.canvas.fig.add_subplot(122,projection=ccrs.Orthographic(central_lon, central_lat))
            self.canvas.axes4.set_facecolor('k')  
            
            # self.canvas.axes4.set_extent(extent)
              

            ixnan=df_krig['nasc'] > 10000     
            df_krig.loc[ixnan,'nasc']=np.nan
            nasc_cutoff=df_krig['nasc'].max()
            # if nasc_cutoff>10000:
            #     nasc_cutoff=10000
        
            if self.checkbox_kriging.isChecked():            

                try:
                    # a =df_krig['nasc'].values.copy()
                    # a[a>nasc_cutoff]=np.nan
                    
                    ix= (df_krig['lat'].notna()) & (df_krig['nasc'].notna() )                        
    
                    
                    OK = OrdinaryKriging(
                        360+df_krig.loc[ix,'lon'].values,
                        df_krig.loc[ix,'lat'].values,
                        df_krig.loc[ix,'nasc'].values,
                        coordinates_type='geographic',
                        variogram_model="spherical",
                        # variogram_parameters = { 'range' :  np.rad2deg( 50 / 6378  ) ,'sill':700000,'nugget':600000}
                        )
                     
                    d_lats=np.linspace(latlim[0],latlim[1],100)
                    d_lons=np.linspace(lonlim[0],lonlim[1],100)
                    
                    lat,lon=np.meshgrid( d_lats,d_lons )
                    
                    z1, ss1 = OK.execute("grid", d_lons, d_lats)                                  
                    z_grid=z1.data
                    z_grid[ ss1.data>np.percentile( ss1.data,75) ] =np.nan
                    # print(z_grid)
                    # sc=self.canvas.axes4.imshow( z_grid ,vmin=0,vmax=nasc_cutoff, origin='lower',aspect='auto',extent=[lonlim[0],lonlim[1],latlim[0],latlim[1]] )   
    
                    self.interpolationplot=self.canvas.axes4.contourf(lon, lat, np.transpose(z_grid), 30,cmap='plasma',
                                      linestyles=None, transform=ccrs.PlateCarree())                
                except:
                    print('kriging error')       
                 
            sc=self.canvas.axes4.scatter( df_krig['lon'],df_krig['lat'],20,df_krig['nasc'],cmap='plasma',vmin=0,vmax=nasc_cutoff,edgecolor=None,transform=ccrs.PlateCarree() )   
           


            self.canvas.axes4.set_adjustable('datalim')
           
            # self.canvas.axes4.coastlines(resolution='110m', zorder=200, color='white')
            self.canvas.axes4.add_feature(cart.feature.LAND, zorder=100, color='white')
            # self.canvas.axes4.show()
            # self.canvas.axes4.add_feature(cart.feature.GSHHSFeature(edgecolor='w'))



            gl=self.canvas.axes4.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
            # self.canvas.axes4.tick_params(axis="y",direction="in", pad=-22)
            # self.canvas.axes4.tick_params(axis="x",direction="in", pad=-15)
            gl.xlabels_top = False
            gl.ylabels_right = False          
                   
            # sc=self.canvas.axes4.scatter( self.df_nasc['lon'],self.df_nasc['lat'],20,self.df_nasc['nasc'],vmin=0,vmax=nasc_cutoff,edgecolor='k' )   
            # self.canvas.axes4.grid()
            self.canvas.fig.colorbar(sc,label='10 min avg. Krill NASC')
            # self.canvas.show()
   
            self.canvas.fig.tight_layout()
            self.canvas.fig.subplots_adjust(wspace = .2)
            
            
            # def on_xlims_change_map(event_ax):
            #     print("updated xlims: ", event_ax.get_xlim())
            #     print("updated ylims: ", event_ax.get_ylim())

            #     # ix_latlim = np.where( (self.minut > event_ax.get_xlim()[0]) & (self.minut < event_ax.get_xlim()[1]) )[0]
            #     # print( self.df_sv.index[ix_dates[0]] )
            #     # print( self.df_sv.index[ix_dates[-1]] )
            #     # ix = (self.df_nasc.index > self.df_sv.index[ix_dates[0]] )   & (self.df_nasc.index < self.df_sv.index[ix_dates[-1]]     )         
            #     # # print(self.df_nasc.loc[ix,:])
            #     # try:
            #     #     line = self.interpolationplot.pop(0)
            #     #     line.remove()
            #     # except:
            #     #     print('noline')
            #     # self.newline=self.canvas.axes4.plot( self.df_nasc.loc[ix,'lon'],self.df_nasc.loc[ix,'lat'],'-r',transform=ccrs.PlateCarree())
            #     # self.canvas.draw()
            #     # # self.canvas.flush_events()        
                    
                
            # # def on_ylims_change(event_ax):
            # #     print("updated ylims: ", event_ax.get_ylim())
            
            # self.canvas.axes4.callbacks.connect('xlim_changed', on_xlims_change_map)
            
            
            
            self.canvas.draw()
            self.canvas.flush_events()                              
   

        
        # if len(nasc_done)>0:
        #     df_nasc=pd.DataFrame([])
        #     for file in nasc_done:
        #         df=pd.read_hdf(file,key='df')
        #         df_nasc=pd.concat([df_nasc,df])
        #     self.df_nasc=df_nasc.sort_values('time')   

        #     df_krig=self.df_nasc.resample('10min').mean() 
            
        #     latlim=[ self.df_nasc['lat'].min() , self.df_nasc['lat'].max()  ]
        #     lonlim=[ self.df_nasc['lon'].min() , self.df_nasc['lon'].max()  ]
    
    
    
        #     # get onl the latest echograms
        #     sv_done = np.array( glob.glob( '*_sv_swarm.h5' ) )
            
        #     sv_times=pd.to_datetime( sv_done,format='D%Y%m%d-T%H%M%S_sv_swarm.h5' )
        #     time_threshold=sv_times.max() - pd.to_timedelta( self.echogram_time_length.value() , 'min' )
        #     sv_load= sv_done[ sv_times>time_threshold ]
        #     df_nasc_load= df_nasc[ df_nasc.index>time_threshold ]
        #     # distance_covered= geod.line_length(lons=df_nasc_load['lon'],lats=df_nasc_load['lat']) / 1000  
        #     distance_covered=df_nasc_load['distance_m'].sum() /1000         
            
            
        #     df_sv=pd.DataFrame([])
        #     for file in sv_load:
        #         df=pd.read_hdf(file,key='df')
        #         # df=df.resample('1min').mean()
        #         df_sv=pd.concat([df_sv,df])
  
        #     self.canvas.fig.clf() 
        #     self.canvas.fig.set_facecolor('gray')
            
        #     self.canvas.axes1 = self.canvas.fig.add_subplot(211)
        #     self.canvas.axes1.set_facecolor('k')   
            
        #     xt = [ -(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,  df_sv.columns[-1].astype(float) , df_sv.columns[0].astype(float)]
        #     # print(xt)
        #     img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap=self.cmap_ek80,origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
        #     self.canvas.fig.colorbar(img,label='$s_V$ [dB]')
            
        #     self.canvas.axes1.set_ylabel('Depth [m]')
        #     self.canvas.axes1.set_xlabel('Time [min]')
        #     self.canvas.axes1.grid( alpha=0.5, linestyle='--')

        #     self.canvas.axes1_2=self.canvas.axes1.twiny()
        #     self.canvas.axes1_2.set_xlim([-distance_covered,0])            
        #     self.canvas.axes1_2.set_xlabel('Distance [km]')          
            
    
        #     self.canvas.axes3 = self.canvas.fig.add_subplot(223)
        #     self.canvas.axes3.set_facecolor('gray')   
        #     self.canvas.axes3.plot( self.df_nasc['nasc'],'.r' )   
        #     self.canvas.axes3.plot( df_krig['nasc'] ,'-k' )   
        #     self.canvas.axes3.grid()
        #     self.canvas.axes3.set_title('Krill NASC')
     

     
        #     central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
        #     central_lat = latlim[0]+(latlim[1]-latlim[0])/2
        #     # extent = [lonlim[0]+0.5,lonlim[1]-0.5, latlim[0]+0.5,latlim[1]-0.5]
        #     #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))
            
        #     self.canvas.axes4 = self.canvas.fig.add_subplot(224,projection=ccrs.Orthographic(central_lon, central_lat))
        #     self.canvas.axes4.set_facecolor('k')  
            
        #     # self.canvas.axes4.set_extent(extent)
              

        #     ixnan=df_krig['nasc'] > 10000     
        #     df_krig.loc[ixnan,'nasc']=np.nan
        #     nasc_cutoff=df_krig['nasc'].max()
        #     # if nasc_cutoff>10000:
        #     #     nasc_cutoff=10000
        

        #     try:
        #         # a =df_krig['nasc'].values.copy()
        #         # a[a>nasc_cutoff]=np.nan
                
        #         ix= (df_krig['lat'].notna()) & (df_krig['nasc'].notna() )                        

                
        #         OK = OrdinaryKriging(
        #             360+df_krig.loc[ix,'lon'].values,
        #             df_krig.loc[ix,'lat'].values,
        #             df_krig.loc[ix,'nasc'].values,
        #             coordinates_type='geographic',
        #             variogram_model="spherical",
        #             # variogram_parameters = { 'range' :  np.rad2deg( 50 / 6378  ) ,'sill':700000,'nugget':600000}
        #             )
                 
        #         d_lats=np.linspace(latlim[0],latlim[1],100)
        #         d_lons=np.linspace(lonlim[0],lonlim[1],100)
                
        #         lat,lon=np.meshgrid( d_lats,d_lons )
                
        #         z1, ss1 = OK.execute("grid", d_lons, d_lats)                                  
        #         z_grid=z1.data
        #         z_grid[ ss1.data>np.percentile( ss1.data,75) ] =np.nan
        #         # print(z_grid)
        #         # sc=self.canvas.axes4.imshow( z_grid ,vmin=0,vmax=nasc_cutoff, origin='lower',aspect='auto',extent=[lonlim[0],lonlim[1],latlim[0],latlim[1]] )   

        #         sc=self.canvas.axes4.contourf(lon, lat, np.transpose(z_grid), 30,cmap='plasma',
        #                           linestyles=None, transform=ccrs.PlateCarree())                
        #     except:
        #         print('kriging error')       
                 
        #     sc=self.canvas.axes4.scatter( df_krig['lon'],df_krig['lat'],20,df_krig['nasc'],cmap='plasma',vmin=0,vmax=nasc_cutoff,edgecolor=None,transform=ccrs.PlateCarree() )   
           
        #     ix_latest=df_krig.index.max()
        #     self.canvas.axes4.plot( df_krig.loc[ix_latest,'lon'],df_krig.loc[ix_latest,'lat'],'xr',markersize=8 ,transform=ccrs.PlateCarree())
        #     self.canvas.axes4.set_adjustable('datalim')
           
        #     # self.canvas.axes4.coastlines(resolution='110m', zorder=200, color='white')
        #     self.canvas.axes4.add_feature(cart.feature.LAND, zorder=100, color='white')
        #     # self.canvas.axes4.show()
        #     # self.canvas.axes4.add_feature(cart.feature.GSHHSFeature(edgecolor='w'))



        #     gl=self.canvas.axes4.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
        #     # self.canvas.axes4.tick_params(axis="y",direction="in", pad=-22)
        #     # self.canvas.axes4.tick_params(axis="x",direction="in", pad=-15)
        #     gl.xlabels_top = False
        #     gl.ylabels_right = False          
                   
        #     # sc=self.canvas.axes4.scatter( self.df_nasc['lon'],self.df_nasc['lat'],20,self.df_nasc['nasc'],vmin=0,vmax=nasc_cutoff,edgecolor='k' )   
        #     # self.canvas.axes4.grid()
        #     self.canvas.fig.colorbar(sc,label='Krill NASC')
        #     # self.canvas.show()
   
        #     self.canvas.fig.tight_layout()
        #     self.canvas.fig.subplots_adjust(wspace = .2)
            
            
        #     self.canvas.draw()
        #     self.canvas.flush_events()                              
      


        

    def openfolderfunc(self):
        self.df_files = pd.DataFrame([])
        
        self.folder_source = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Source folder with raw files')
        if len(self.folder_source)>0:
            self.df_files['path'] = glob.glob(self.folder_source+'\\*.raw')  
            
            if len(self.df_files['path'])>0:
                
                # fname=self.df_files['path'].str.split('\\').str[-1]
                # datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                # self.df_files['date'] = pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' )
                
                dates=[]
                for fname in self.df_files['path']:
                    
                    datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                    dates.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
                dates=np.array(dates)
                self.df_files['date'] = dates
                
                self.startdate.setDateTime(QtCore.QDateTime.fromString( dates.min().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )
                self.enddate.setDateTime(QtCore.QDateTime.fromString( dates.max().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )

            
                # ix_time= (self.df_files['date'] >= self.startdate.dateTime().toPyDateTime() ) & (self.df_files['date'] <= self.enddate.dateTime().toPyDateTime() )
                
                # self.df_files['status'] = 0
                # self.df_files.loc[ix_time,'status'] = 1
                
                self.df_files =  self.df_files.sort_values('date',ascending=True)
                
                # look for already processed data
                # nasc_done = glob.glob( '*_nasctable.h5' )
                # nasc_done= list(map(lambda x: x.replace('_nasctable.h5','') , nasc_done))        
                # names = self.df_files['path'].apply(lambda x: Path(x).stem)              
                # ix_done= names.isin( nasc_done  )  
                 
                self.filecounter=-1   
                self.df_nasc=pd.DataFrame([])
        
                print( self.df_files  )   
                
                # if self.checkbox_updateplot.isChecked():
                #     self.folder_target = QtWidgets.QFileDialog.getExistingDirectory(self,caption='Target folder for saving processed data')
        
                # self.statusBar().setStyleSheet("background-color : k")
                self.statusBar().removeWidget(self.label_folders) 
                self.label_folders = QtWidgets.QLabel("Source: "+self.folder_source )
                self.statusBar().addPermanentWidget(self.label_folders)                
                self.startautoMenu.setEnabled(True)
                self.checkbox_updateplot.setEnabled(True)            
                self.checkbox_manual.setEnabled(True)            
                self.showfolderbutton.setEnabled(True)

                self.workpath=  os.path.join(self.folder_source,'krill_data')
                if not os.path.exists(self.workpath):
                    os.mkdir( self.workpath )
                
                os.chdir(self.workpath)    

            
    # def next_file(self):
    #       if len(self.df_files)>0:
    #         print('old filecounter is: '+str(self.filecounter))
    #         self.filecounter=self.filecounter+1
            
    #         if self.filecounter>len(self.df_files)-1:
    #                 self.filecounter=len(self.df_files)-1
    #                 print('That was it')
    #         # rawfile = self.df_files.loc[self.filecounter,'path']            
    #         # self.read_raw()
    #         # self.detect_krill_swarms()                 
         
 
    # def previous_file(self):
    #       if len(self.df_files)>0:
    #         print('old filecounter is: '+str(self.filecounter))
    #         self.filecounter=self.filecounter-1
            
    #         if self.filecounter<0:
    #                 self.filecounter=0
    #         # self.read_raw()
    #         # self.detect_krill_swarms()            
        
         
    def update_plots(self):
        if self.checkbox_updateplot.isChecked():
            print('Update plots')
            self.plottimer = QtCore.QTimer(self)
            self.plottimer.timeout.connect(self.scan_and_vizualize)
            self.plottimer.start(10000)  
            # self.checkbox_manual.setChecked(False)           
        else:
            print('STOP plots')
            self.plottimer.stop()  
            
               
    def scan_and_vizualize(self):
        
        os.chdir(self.workpath)
        nasc_done =  np.array( glob.glob( '*_nasctable.h5' ) )
        
        if len(nasc_done)>0:
                        
            nascfile_times=pd.to_datetime( nasc_done,format='D%Y%m%d-T%H%M%S_nasctable.h5' )
            time_threshold=nascfile_times.max() +pd.to_timedelta(10,'min') - pd.to_timedelta( self.map_time_length.value() , 'days' )
            nasc_load= nasc_done[ nascfile_times>time_threshold ]
            
            
            df_nasc=pd.DataFrame([])
            for file in nasc_load:
                df=pd.read_hdf(file,key='df')
                df_nasc=pd.concat([df_nasc,df  ])
            self.df_nasc=df_nasc.sort_values('ping_time')   
            
            
            # geod = Geod("+ellps=WGS84")
            # ll=geod.line_lengths(self.df_nasc['lon'], self.df_nasc['lat'])
            # ll= np.concatenate([np.array([0]), ll ])
            timedelta =np.concatenate([np.array([0]), np.diff( self.df_nasc.index.values ) /1e9 ])           
            self.df_nasc['speed_ms'] = self.df_nasc['distance_m'] / timedelta.astype(float)
            self.df_nasc['speed_knots']=self.df_nasc['speed_ms']*1.94384449
            ix_too_slow = self.df_nasc['speed_knots'] < 5           
            self.df_nasc.loc[ix_too_slow,'nasc']=np.nan           

            df_krig=self.df_nasc.resample('10min').mean() 
            # df_krig=self.df_nasc
            # print(  self.df_nasc.index.min().strftime('%Y-%m-%d %X') ) 
            
            self.startdate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.min().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )
            self.enddate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.max().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )

            
            latlim=[ self.df_nasc['lat'].min() , self.df_nasc['lat'].max()  ]
            lonlim=[ self.df_nasc['lon'].min() , self.df_nasc['lon'].max()  ]
    
    
    
            # get onl the latest echograms
            sv_done = np.array( glob.glob( '*_sv_swarm.h5' ) )
            
            sv_times=pd.to_datetime( sv_done,format='D%Y%m%d-T%H%M%S_sv_swarm.h5' )
            time_threshold=sv_times.max()  - pd.to_timedelta( self.echogram_time_length.value() , 'min' )
            sv_load= sv_done[ sv_times>time_threshold ]
            df_nasc_load= df_nasc[ df_nasc.index>time_threshold ]
            # distance_covered= geod.line_length(lons=df_nasc_load['lon'],lats=df_nasc_load['lat']) / 1000  
            distance_covered=df_nasc_load['distance_m'].sum() /1000         
            
            
            df_sv=pd.DataFrame([])
            for file in sv_load:
                df=pd.read_hdf(file,key='df')
                # df=df.resample('1min').mean()
                df_sv=pd.concat([df_sv,df])
                
            ix_latest = (df_sv.index.max() -  df_sv.index) <  pd.to_timedelta( self.echogram_time_length.value() , 'min' )
            df_sv= df_sv.iloc[ix_latest,:]   
                
            self.canvas.fig.clf() 
            self.canvas.fig.set_facecolor('gray')
            
            self.canvas.axes1 = self.canvas.fig.add_subplot(121)
            self.canvas.axes1.set_facecolor('k')   
            
            xt = [ -(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,  df_sv.columns[-1].astype(float) , df_sv.columns[0].astype(float)]
            # print(xt)
            img=self.canvas.axes1.imshow( np.rot90( df_sv.values ) , aspect='auto',cmap=self.cmap_ek80,origin = 'lower',vmin=-80,vmax=-40,extent=xt)        
                
            # r=   df_sv.columns.astype(float)         
            # bottomdepth=[]
            # for index, row in df_sv.iterrows():
            #      bottomdepth.append( np.min(r[row==-999]) )
            # t=np.linspace(-(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,len(df_sv))  
            
            # self.canvas.axes1.plot( t,bottomdepth,'--w')        
            
         
            self.canvas.axes1.set_ylabel('Depth [m]')
            self.canvas.axes1.set_xlabel('Time [min]')
            self.canvas.axes1.grid( alpha=0.5, linestyle='--')

            self.canvas.axes1_2=self.canvas.axes1.twiny()
            self.canvas.axes1_2.set_xlim([-distance_covered,0])            
            self.canvas.axes1_2.set_xlabel('Distance [km]')          

            self.canvas.axes1_3=self.canvas.axes1.twinx()
            self.canvas.axes1_3.set_ylabel('Krill NASC')          
            self.canvas.fig.colorbar(img,label='$s_V$ [dB]',pad=0.15)
            
            df_sv[df_sv==-999]=np.nan
            r=df_sv.columns.astype(float)
            cell_thickness=np.abs(np.mean(np.diff( r) ))               
            nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, np.transpose(df_sv.values) /10)*cell_thickness ,axis=0)   
            t=np.linspace(-(df_sv.index.max()-df_sv.index.min() ) / (np.timedelta64(1, 's')*60), 0,len(nasc_swarm)) 
            self.canvas.axes1_3.plot(t,nasc_swarm,'-b')          
            self.canvas.axes1_3.set_xlim([t[0],t[-1]])          
            # self.canvas.axes1_3.set_xticks([])          
            if np.max(nasc_swarm)==0:
                self.canvas.axes1_3.set_ylim([0,1])                        
            else:
                self.canvas.axes1_3.set_ylim([0,2*np.max(nasc_swarm)])          
      
    
            # self.canvas.axes3 = self.canvas.fig.add_subplot(223)
            # self.canvas.axes3.set_facecolor('gray')   
            # self.canvas.axes3.plot( self.df_nasc['nasc'],'.r' )   
            # self.canvas.axes3.plot( df_krig['nasc'] ,'-k' )   
            # self.canvas.axes3.grid()
            # self.canvas.axes3.set_title('Krill NASC')
     

     
            central_lon= lonlim[0]+(lonlim[1]-lonlim[0])/2
            central_lat = latlim[0]+(latlim[1]-latlim[0])/2
            # extent = [lonlim[0]+0.5,lonlim[1]-0.5, latlim[0]+0.5,latlim[1]-0.5]
            #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= lonlim[0]+(lonlim[1]-lonlim[0])/2 ))
            
            self.canvas.axes4 = self.canvas.fig.add_subplot(122,projection=ccrs.Orthographic(central_lon, central_lat))
            self.canvas.axes4.set_facecolor('k')  
            
            # self.canvas.axes4.set_extent(extent)
              

            ixnan=df_krig['nasc'] > 10000     
            df_krig.loc[ixnan,'nasc']=np.nan
            nasc_cutoff=df_krig['nasc'].max()
            # if nasc_cutoff>10000:
            #     nasc_cutoff=10000
        
            if self.checkbox_kriging.isChecked():            

                try:
                    # a =df_krig['nasc'].values.copy()
                    # a[a>nasc_cutoff]=np.nan
                    
                    ix= (df_krig['lat'].notna()) & (df_krig['nasc'].notna() )                        
    
                    
                    OK = OrdinaryKriging(
                        360+df_krig.loc[ix,'lon'].values,
                        df_krig.loc[ix,'lat'].values,
                        df_krig.loc[ix,'nasc'].values,
                        coordinates_type='geographic',
                        variogram_model="spherical",
                        # variogram_parameters = { 'range' :  np.rad2deg( 50 / 6378  ) ,'sill':700000,'nugget':600000}
                        )
                     
                    d_lats=np.linspace(latlim[0],latlim[1],100)
                    d_lons=np.linspace(lonlim[0],lonlim[1],100)
                    
                    lat,lon=np.meshgrid( d_lats,d_lons )
                    
                    z1, ss1 = OK.execute("grid", d_lons, d_lats)                                  
                    z_grid=z1.data
                    z_grid[ ss1.data>np.percentile( ss1.data,75) ] =np.nan
                    # print(z_grid)
                    # sc=self.canvas.axes4.imshow( z_grid ,vmin=0,vmax=nasc_cutoff, origin='lower',aspect='auto',extent=[lonlim[0],lonlim[1],latlim[0],latlim[1]] )   
    
                    sc=self.canvas.axes4.contourf(lon, lat, np.transpose(z_grid), 30,cmap='plasma',
                                      linestyles=None, transform=ccrs.PlateCarree())                
                except:
                    print('kriging error')       
                 
            sc=self.canvas.axes4.scatter( df_krig['lon'],df_krig['lat'],20,df_krig['nasc'],cmap='plasma',vmin=0,vmax=nasc_cutoff,edgecolor=None,transform=ccrs.PlateCarree() )   
           
            # ix_latest=df_krig.index.max()
            # self.canvas.axes4.plot( df_krig.loc[ix_latest,'lon'],df_krig.loc[ix_latest,'lat'],'xr',markersize=8 ,transform=ccrs.PlateCarree())
          
            ix= self.df_nasc.index > time_threshold
            self.canvas.axes4.plot( self.df_nasc.loc[ix,'lon'],self.df_nasc.loc[ix,'lat'],'-r',transform=ccrs.PlateCarree())
            # ix_latest=self.df_nasc.index.max()
            # print(  self.df_nasc.loc[ix_latest,'lon'] )
            # self.canvas.axes4.plot( self.df_nasc.loc[ix_latest,'lon'],self.df_nasc.loc[ix_latest,'lat'],'^r',markersize=8,transform=ccrs.PlateCarree())

            self.canvas.axes4.set_adjustable('datalim')
           
            # self.canvas.axes4.coastlines(resolution='110m', zorder=200, color='white')
            self.canvas.axes4.add_feature(cart.feature.LAND, zorder=100, color='white')
            # self.canvas.axes4.show()
            # self.canvas.axes4.add_feature(cart.feature.GSHHSFeature(edgecolor='w'))



            gl=self.canvas.axes4.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
            # self.canvas.axes4.tick_params(axis="y",direction="in", pad=-22)
            # self.canvas.axes4.tick_params(axis="x",direction="in", pad=-15)
            gl.xlabels_top = False
            gl.ylabels_right = False          
                   
            # sc=self.canvas.axes4.scatter( self.df_nasc['lon'],self.df_nasc['lat'],20,self.df_nasc['nasc'],vmin=0,vmax=nasc_cutoff,edgecolor='k' )   
            # self.canvas.axes4.grid()
            self.canvas.fig.colorbar(sc,label='10 min avg. Krill NASC')
            # self.canvas.show()
   
            self.canvas.fig.tight_layout()
            self.canvas.fig.subplots_adjust(wspace = .2)
            
            
            self.canvas.draw()
            self.canvas.flush_events()                              
   

       


    def startClicked(self):
        if not self.thread.isRunning():

            
            self.worker.pass_folder(self.folder_source)

            self.thread.started.connect(self.worker.start)
            self.thread.start()
            
            self.exitautoMenu.setEnabled(True)           
            self.startautoMenu.setEnabled(False)
            self.statusBar().setStyleSheet("background-color : rgb(115, 6, 6)")
            self.label_1 = QtWidgets.QLabel("Automatic processing activated")
            self.statusBar().addPermanentWidget(self.label_1)
            
    def stopClicked(self):
        self.worker.stop()    
        # self.thread.terminate()
        
        self.thread.quit()
        self.thread.wait()
        
        self.statusBar().setStyleSheet("background-color : k")
        self.statusBar().removeWidget(self.label_1)   
        self.startautoMenu.setEnabled(True)
        self.exitautoMenu.setEnabled(False)           

    def func_quit(self):
        self.worker.stop()            
        self.thread.quit()
        self.thread.wait()    
        self.statusBar().setStyleSheet("background-color : k")
        # self.statusBar().removeWidget(self.label_1)   
        self.startautoMenu.setEnabled(True)
        self.exitautoMenu.setEnabled(False)     
        QtWidgets.QApplication.instance().quit()     
        # QCoreApplication.quit()
        self.close()    
        
app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Krill3000 ")    
app.setStyleSheet(qdarktheme.load_stylesheet())
  


w = MainWindow()

# timer = QtCore.QTimer()
# timer.timeout.connect(w.scan_and_vizualize)
# timer.start(5000)  


sys.exit(app.exec_())
     