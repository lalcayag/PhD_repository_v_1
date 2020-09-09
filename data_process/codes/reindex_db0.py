# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:37:25 2020

@author: lalc
"""
import numpy as np
import scipy as sp
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize, abspath
#import spectralfitting.spectralfitting as sf
from sqlalchemy import create_engine
from datetime import datetime, timedelta
date = datetime(1904, 1, 1) 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

file_out_path = '/mnt/mimer/lalc/db/scans/phase_1/east'

# In[Reindexing database filtered]
csv_database_0_ind = create_engine('sqlite:///'+file_out_path+'/raw_filt_0_east_phase1.db')
#csv_database_1_ind = create_engine('sqlite:///'+file_out_path+'/raw_filt_1_east_phase1.db')    
csv_database_0 = create_engine('sqlite:///'+file_out_path+'/raw_filt_0_phase1_east.db')
#csv_database_1 = create_engine('sqlite:///'+file_out_path+'/raw_filt_1_phase1_east.db')  
chunk=5000*45
off = 0
for i in range(300): 
    print(off/45)
    df0 = pd.read_sql_query('select * from "table_fil" ' +' LIMIT ' + str(int(chunk)) +
                               ' OFFSET ' + str(int(off)), csv_database_0)  
    # df1 = pd.read_sql_query('select * from "table_fil" ' +' LIMIT ' + str(int(chunk)) +
    #                            ' OFFSET ' + str(int(off)), csv_database_1)  
    
    s0 = df0.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    # s1 = df1.scan.unique()
    # t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    
    timestp = pd.to_datetime(t0).strftime("%Y%m%d").values
    df0['name'] = np.repeat(timestp,45)
    
    # timestp = pd.to_datetime(t1).strftime("%Y%m%d").values
    # df1['name'] = np.repeat(timestp,45)
    
    df0.set_index(['name','scan'], inplace=True)
    df0.to_sql('table_fil', csv_database_0_ind, if_exists='append',index_label = ['name','scan'])
    
    # df1.set_index(['name','scan'], inplace=True)
    # df1.to_sql('table_fil', csv_database_1_ind, if_exists='append',index_label = ['name','scan'])
    
    df0 = pd.read_sql_query('select * from "table_raw" ' +' LIMIT ' + str(int(chunk)) +
                               ' OFFSET ' + str(int(off)), csv_database_0)  
    # df1 = pd.read_sql_query('select * from "table_raw" ' +' LIMIT ' + str(int(chunk)) +
    #                            ' OFFSET ' + str(int(off)), csv_database_1)  
    
    s0 = df0.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    # s1 = df1.scan.unique()
    # t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    
    timestp = pd.to_datetime(t0).strftime("%Y%m%d").values
    df0['name'] = np.repeat(timestp,45)
    
    # timestp = pd.to_datetime(t1).strftime("%Y%m%d").values
    # df1['name'] = np.repeat(timestp,45)
    
    df0.set_index(['name','scan'], inplace=True)
    df0.to_sql('table_raw', csv_database_0_ind, if_exists='append',index_label = ['name','scan'])
    
    # df1.set_index(['name','scan'], inplace=True)
    # df1.to_sql('table_raw', csv_database_1_ind, if_exists='append',index_label = ['name','scan'])
    
    off = off+chunk   