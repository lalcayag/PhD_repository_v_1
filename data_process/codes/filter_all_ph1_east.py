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
        file_0_path = join(file_in_path_0,file_0)
        for chunk0 in pd.read_csv(file_0_path, sep=";", header=None, chunksize=int(t_step*45*100)):
            mask0 = pd.DataFrame()
            chunk0.columns = labels
            chunk0['scan'] = chunk0.groupby('azim').cumcount()
            chunk0.index += j0
            j0 = chunk0.index[-1] + 1
            ind=np.unique(chunk0.scan.values)%t_step==0
            times= np.unique(np.append(np.unique(chunk0.scan.values)[ind],
                                       chunk0.scan.values[-1]))  
            chunk0.scan+=scan_0
            times+=scan_0
            scan_0 = chunk0.scan.iloc[-1] + 1        
            for i in range(len(times)-1):
                print(file_0,size_0[counter],times[i],times[i+1])
                if i == len(times)-2:
                    loc = (chunk0.scan>=times[i]) & (chunk0.scan<=times[i+1])
                    print('here')
                else:
                    loc = (chunk0.scan>=times[i]) & (chunk0.scan<times[i+1])
                mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(chunk0.loc[loc],feat)])
            mask0.columns = chunk0.ws.columns
            mask0.index = chunk0.index
            chunk = chunk0.copy()
            chunk.ws = chunk.ws.mask(mask0.ws)
            chunk.columns = labels_new
            chunk0.columns = labels_new
            chunk0.to_sql('table_raw', csv_database0, if_exists='append')
            chunk.to_sql('table_fil', csv_database0, if_exists='append')
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
#                d
#                mask = pd.concat([mask,fl.data_filt_DBSCAN(chunk0.loc[loc],feat)])  
#                azim = pd.concat([azim,chunk0.loc[loc]['azim']]) 
#        pd.concat([UV,pd.concat([floats,ints,strings],axis=1)])
#        mask['azim'] = azim
#        mask['scan'] = mask.groupby('azim').cumcount()
#        with open(join(file_out_path,file_0[:14])+'_mask_0.pkl', 'wb') as clust0:
#             pickle.dump(mask, clust0)
#        df0.to_sql('table', csv_database0, if_exists='append') 
#                
#                
#                loc = (chunk1.scan>=times[i]) & (chunk1.scan<times[i+1])
#                mask = pd.concat([mask,fl.data_filt_DBSCAN(chunk1.loc[loc],feat)])  
#                azim = pd.concat([azim,chunk1.loc[loc]['azim']]) 
#        mask['azim'] = azim
#        mask['scan'] = mask.groupby('azim').cumcount()
#        with open(join(file_out_path,file_0[:14])+'_mask_1.pkl', 'wb') as clust1:
#             pickle.dump(mask, clust1)
#    

#df = pd.read_sql_query('SELECT * FROM "table" WHERE scan = 10 AND scan < 210', csv_database1)


# # In[]
# root = tkint.Tk()
# file_df = tkint.filedialog.askopenfilenames(parent=root,title='Choose a data file')
# root.destroy()             
# root = tkint.Tk()
# file_in_mask = tkint.filedialog.askdirectory(parent=root,title='Choose an masks dir')
# root.destroy()
# filelist_mask = [(filename,getsize(join(file_in_mask,filename)))
#              for filename in listdir(file_in_mask) if getsize(join(file_in_mask,filename))>1000]
# size_mask = list(list(zip(*filelist_mask))[1])
# filelist_mask = list(list(zip(*filelist_mask))[0])
# file_mask_0 = filelist_mask[int(np.where(np.array(filelist_mask) == file_df[0][-36:-22]+'_mask_0.pkl')[0])]             
             
# # In[]             
           
# t_step = 3
# df = pd.DataFrame()
# for chunk0 in pd.read_csv(file_df[0], sep=";", header=None, chunksize=int(t_step*45*10)):
#     chunk0.columns = labels
#     chunk0['scan'] = chunk0.groupby('azim').cumcount()
#     ind=np.unique(chunk0.scan.values)%t_step==0
#     times= np.unique(np.append(np.unique(chunk0.scan.values)[ind],
#                                chunk0.scan.values[-1]))          
#     for i in range(len(times)-1):
#         print(times[i],times[i+1])
#         loc = (chunk0.scan>=times[i]) & (chunk0.scan<times[i+1])
#         df = pd.concat([df,chunk0.loc[loc]])            
# df['scan'] = df.groupby('azim').cumcount()   


# with open(join(file_in_mask,file_mask_0), 'rb') as reader:
#         mask0 = pickle.load(reader)

# df_mask = df.copy()
# df_mask.ws=df_mask.ws.mask(mask0.ws)

# df_mask_CNR = df.copy()
# mask_CNR = ~((df_mask_CNR.CNR>-27) & (df_mask_CNR.CNR<-8))
# mask_CNR['scan'] =  mask0.scan
# mask_CNR['azim'] =  mask0.azim
# mask_CNR.columns =  mask0.columns
# df_mask_CNR.ws=df_mask_CNR.ws.mask(mask_CNR.ws)

# df_mask_tot = df.copy()
# mask_finCNR = ~((df_mask_CNR.CNR>-24) & (df_mask_CNR.CNR<-8))
# mask_finCNR['scan'] =  mask0.scan
# mask_finCNR['azim'] =  mask0.azim
# mask_finCNR.columns =  mask0.columns
# mask_fin = mask_finCNR.copy()
# mask_fin.ws = mask0.ws & mask_finCNR.ws
# df_mask_tot.ws=df_mask_tot.ws.mask(mask_fin.ws)

# phi0 = df.azim.unique()
# r0 = np.array(df.iloc[(df.azim==min(phi0)).nonzero()[0][0]].range_gate)
# r_0, phi_0 = np.meshgrid(r0, np.radians(phi0)) # meshgrid

# loc0 = np.array([6322832.3,0])
# loc1 = np.array([6327082.4,0])
# d = loc1-loc0

# scan_n = 150
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.use_sticky_edges = False
# ax.margins(0.07)
# im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df.ws.loc[df_mask.scan==scan_n].values,np.linspace(-7,3,100),cmap='jet')
# fig.colorbar(im)

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.use_sticky_edges = False
# ax.margins(0.07)
# im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_mask.ws.loc[df_mask.scan==scan_n].values,100,cmap='jet')
# fig.colorbar(im)

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.use_sticky_edges = False
# ax.margins(0.07)
# im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_mask_CNR.ws.loc[df_mask_CNR.scan==scan_n].values,100,cmap='jet')
# fig.colorbar(im)

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.use_sticky_edges = False
# ax.margins(0.07)
# im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_mask_tot.ws.loc[df_mask_tot.scan==scan_n].values,100,cmap='jet')
# fig.colorbar(im)


      
            