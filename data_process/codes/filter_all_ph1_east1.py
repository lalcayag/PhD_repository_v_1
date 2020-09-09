# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize
from os import listdir
# import tkinter as tkint
# import tkinter.filedialog
import matplotlib.pyplot as plt

import ppiscanprocess.filtering as fl
import ppiscanprocess.windfieldrec as wr

import pickle

import numpy as np
import pandas as pd
import scipy as sp

import importlib

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

  
# In[paths]    
# root = tkint.Tk()
# file_in_path_0 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_in_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
# root.destroy()


file_in_path_0 = '/mnt/mimer/lalc/Data/phase_1/east/SiroccoEast'
file_in_path_1 = '/mnt/mimer/lalc/Data/phase_1/east/VaraEast'
file_out_path = '/mnt/mimer/lalc/db/scans/phase_1/east'


# In[column labels]
iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed

labels_mask = []
labels_ws = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    
labels_new = np.concatenate((labels_new,np.array(['scan'])))  
# In[]
filelist_0 = [(filename,getsize(join(file_in_path_0,filename)))
             for filename in listdir(file_in_path_0) if getsize(join(file_in_path_0,filename))>1000]
size_0 = list(list(zip(*filelist_0))[1])
filelist_0 = list(list(zip(*filelist_0))[0])

filelist_1 = [(filename,getsize(join(file_in_path_1,filename)))
             for filename in listdir(file_in_path_1) if getsize(join(file_in_path_1,filename))>1000]
size_1 = list(list(zip(*filelist_1))[1])
filelist_1 = list(list(zip(*filelist_1))[0])

filelist_out = [filename for filename in listdir(file_out_path)]

# In[featres of the DBSCAN filter]
feat = ['ws','range_gate','CNR','azim','dvdr']    

# In[Actual filtering]
#West scanning
from sqlalchemy import create_engine
from datetime import datetime, timedelta
date = datetime(1904,1,1)
csv_database0 = create_engine('sqlite:///'+file_out_path+'/raw_filt_0_phase1_east.db')
csv_database1 = create_engine('sqlite:///'+file_out_path+'/raw_filt_1_phase1_east.db')

n=0
j0 = 1
j1 = 1
scan_0 = 0
scan_1 = 0
for counter, file_0 in enumerate(filelist_0,0):
    print(counter,file_0)
#    file_out_name= file_0[:14]+'_mask_0.pkl'
    if (file_0 in filelist_1):# & (~(file_out_name in filelist_out)):
        t_step = 3
        
        
        
        
        
        # file_0_path = join(file_in_path_0,file_0)
        # for chunk0 in pd.read_csv(file_0_path, sep=";", header=None, chunksize=int(t_step*45*100)):
        #     mask0 = pd.DataFrame()
        #     chunk0.columns = labels
        #     chunk0['scan'] = chunk0.groupby('azim').cumcount()
        #     chunk0.index += j0
        #     j0 = chunk0.index[-1] + 1
        #     ind=np.unique(chunk0.scan.values)%t_step==0
        #     times= np.unique(np.append(np.unique(chunk0.scan.values)[ind],
        #                                 chunk0.scan.values[-1]))  
        #     chunk0.scan+=scan_0
        #     times+=scan_0
        #     scan_0 = chunk0.scan.iloc[-1] + 1        
        #     for i in range(len(times)-1):
        #         print(file_0,size_0[counter],times[i],times[i+1])
        #         if i == len(times)-2:
        #             loc = (chunk0.scan>=times[i]) & (chunk0.scan<=times[i+1])
        #             print('here')
        #         else:
        #             loc = (chunk0.scan>=times[i]) & (chunk0.scan<times[i+1])
        #         mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(chunk0.loc[loc],feat)])
        #     mask0.columns = chunk0.ws.columns
        #     mask0.index = chunk0.index
        #     chunk = chunk0.copy()
        #     chunk.ws = chunk.ws.mask(mask0.ws)
        #     chunk.columns = labels_new
        #     chunk0.columns = labels_new
        #     chunk0.to_sql('table_raw', csv_database0, if_exists='append')
        #     chunk.to_sql('table_fil', csv_database0, if_exists='append')
            
            
            
#                d
#                mask = pd.concat([mask,fl.data_filt_DBSCAN(chunk0.loc[loc],feat)])  
#                azim = pd.concat([azim,chunk0.loc[loc]['azim']]) 
#        pd.concat([UV,pd.concat([floats,ints,strings],axis=1)])
#        mask['azim'] = azim
#        mask['scan'] = mask.groupby('azim').cumcount()
#        with open(join(file_out_path,file_0[:14])+'_mask_0.pkl', 'wb') as clust0:
#             pickle.dump(mask, clust0)
              
#        mask = pd.DataFrame()
#        azim = pd.DataFrame()   



  
        file_1_path = join(file_in_path_1,file_0)     
        for chunk1 in pd.read_csv(file_1_path, sep=";", header=None, chunksize=int(t_step*45*100)):
            mask1 = pd.DataFrame()
            chunk1.columns = labels
            chunk1['scan'] = chunk1.groupby('azim').cumcount()
            chunk1.index += j1
            j1 = chunk1.index[-1] + 1
            ind=np.unique(chunk1.scan.values)%t_step==0
            times= np.unique(np.append(np.unique(chunk1.scan.values)[ind],
                                        chunk1.scan.values[-1]))  
            chunk1.scan+=scan_1
            times+=scan_1
            scan_1 = chunk1.scan.iloc[-1] + 1
          
            for i in range(len(times)-1):
                print(file_0,size_0[counter],times[i],times[i+1])
                
                if i == len(times)-2:
                    loc = (chunk1.scan>=times[i]) & (chunk1.scan<=times[i+1])
                else:
                    loc = (chunk1.scan>=times[i]) & (chunk1.scan<times[i+1])
                mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(chunk1.loc[loc],feat)])
            mask1.columns = chunk1.ws.columns
            mask1.index = chunk1.index
            chunk = chunk1.copy()
            chunk.ws = chunk.ws.mask(mask1.ws)
            chunk.columns = labels_new
            chunk1.columns = labels_new
            chunk1.to_sql('table_raw', csv_database1, if_exists='append')
            chunk.to_sql('table_fil', csv_database1, if_exists='append')
            
            
            
# In[Reindexing database filtered]
# csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0_phase2.db')
# csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1_phase2.db')    
# csv_database_0 = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0.db')
# csv_database_1 = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1.db')  
# chunk=300*45
# off = 42000*45
# for i in range(300): 
#     print(off/45)
#     df0 = pd.read_sql_query('select * from "table_fil" ' +' LIMIT ' + str(int(chunk)) +
#                                ' OFFSET ' + str(int(off)), csv_database_0)  
#     df1 = pd.read_sql_query('select * from "table_fil" ' +' LIMIT ' + str(int(chunk)) +
#                                ' OFFSET ' + str(int(off)), csv_database_1)  
    
#     s0 = df0.scan.unique()
#     t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
#     s1 = df1.scan.unique()
#     t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    
#     timestp = pd.to_datetime(t0).strftime("%Y%m%d").values
#     df0['name'] = np.repeat(timestp,45)
    
#     timestp = pd.to_datetime(t1).strftime("%Y%m%d").values
#     df1['name'] = np.repeat(timestp,45)
    
#     df0.set_index(['name','scan'], inplace=True)
#     df0.to_sql('table_fil', csv_database_0_ind, if_exists='append',index_label = ['name','scan'])
    
#     df1.set_index(['name','scan'], inplace=True)
#     df1.to_sql('table_fil', csv_database_1_ind, if_exists='append',index_label = ['name','scan'])
    
#     df0 = pd.read_sql_query('select * from "table_raw" ' +' LIMIT ' + str(int(chunk)) +
#                                ' OFFSET ' + str(int(off)), csv_database_0)  
#     df1 = pd.read_sql_query('select * from "table_raw" ' +' LIMIT ' + str(int(chunk)) +
#                                ' OFFSET ' + str(int(off)), csv_database_1)  
    
#     s0 = df0.scan.unique()
#     t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
#     s1 = df1.scan.unique()
#     t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    
#     timestp = pd.to_datetime(t0).strftime("%Y%m%d").values
#     df0['name'] = np.repeat(timestp,45)
    
#     timestp = pd.to_datetime(t1).strftime("%Y%m%d").values
#     df1['name'] = np.repeat(timestp,45)
    
#     df0.set_index(['name','scan'], inplace=True)
#     df0.to_sql('table_raw', csv_database_0_ind, if_exists='append',index_label = ['name','scan'])
    
#     df1.set_index(['name','scan'], inplace=True)
#     df1.to_sql('table_raw', csv_database_1_ind, if_exists='append',index_label = ['name','scan'])
    
#     off = off+chunk   

      
            