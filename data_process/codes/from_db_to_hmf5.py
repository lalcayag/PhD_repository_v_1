# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:40:07 2020

@author: lalc
"""

import numpy as np
import scipy as sp
import pandas as pd
import os
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
date = datetime(1904, 1, 1) 

# In[Files paths]    
file_in_path_0_db = '/mnt/mimer/lalc/db/scans/phase_2/raw_filt_0_phase2.db'
file_in_path_1_db = '/mnt/mimer/lalc/db/scans/phase_2/raw_filt_1_phase2.db'

# In[Sql engines]

csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0_db)
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1_db)

# In[]
iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
labels_CNR = []
labels_Sb = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
    labels_Sb = np.concatenate((labels_Sb,np.array(['Sb_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan'])))     


lim = [-8,-24]


days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ind).values
days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ind).values
days0 = np.squeeze(days0)
days1 = np.squeeze(days1)
days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])


for dy in days:

    print('querying df0 for '+dy)
    df0 = pd.read_sql_query('select * from "table_fil" where name = ' + dy, csv_database_0_ind)
    
    print('querying df0r for '+dy)
    df0r = pd.read_sql_query('select * from "table_raw" where name = ' + dy, csv_database_0_ind)
    
    # Retrieving good CNR values from un-filtered scans
    for i in range(198):
        ind = (df0['CNR_'+str(i)]<lim[0])&(df0['CNR_'+str(i)]>lim[1])
        df0['ws_'+str(i)].loc[ind] = df0r['ws_'+str(i)].loc[ind]
    df0r = None 
    
 
    df0.drop(columns = labels_Sb, inplace=True)
    df0.drop(columns = labels_CNR, inplace=True)

    print('writing df0 for '+dy)
    df0.fillna(value=pd.np.nan, inplace=True)
    df0.to_hdf('/mnt/mimer/lalc/db/scans/phase_2/raw_filt_0_phase2.h5','df',mode='w',
                data_columns = df0.columns,format='table', append = True)
    # store0.append('/mnt/mimer/lalc/db/scans/phase_2/raw_filt_0_phase2.h5','df',mode='w',
    #            data_columns = df0.columns,format='table', append = True)
        
    df0 = None    
    print('querying df1 for '+dy)
    df1 = pd.read_sql_query('select * from "table_fil"', csv_database_1_ind)
    print('querying df1r for '+dy)
    df1r = pd.read_sql_query('select * from "table_raw"', csv_database_1_ind)
    # Retrieving good CNR values from un-filtered scans
    for i in range(198):
        ind = (df1['CNR_'+str(i)]<lim[0])&(df1['CNR_'+str(i)]>lim[1])
        df1['ws_'+str(i)].loc[ind] = df1r['ws_'+str(i)].loc[ind]
    df1r = None  
    
    # aux = list(labels_new[3:])
    # aux.remove('elev')    
    # df0.columns = aux    
    df1.drop(columns = labels_Sb, inplace=True)
    df1.drop(columns = labels_CNR, inplace=True)

    print('writing df1 for '+dy)
    df1.fillna(value=pd.np.nan, inplace=True)
    df1.to_hdf('/mnt/mimer/lalc/db/scans/phase_2/raw_filt_1_phase2.h5','df',mode='w',
               data_columns = df1.columns,format='table', append = True)
    df1 = None

# test

dftest = df_0.copy()
aux = list(labels_new[3:])
aux.remove('elev')
dftest.columns = aux

dftest.reset_index(inplace=True)
# dftest.index = dftest.index.values+3330

dftest.drop(columns = labels_Sb, inplace=True)
dftest.drop(columns = labels_CNR, inplace=True)
dftest.drop(columns = 'index', inplace=True)

dftest.to_hdf('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/test.h5','df0', mode='a', append = True,
                data_columns = dftest.columns,format='table')  