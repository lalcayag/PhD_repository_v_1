#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:22:58 2019

@author: leonardo
"""
import numpy as np
import scipy as sp
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize, abspath
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
#import spectralfitting.spectralfitting as sf
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# In[]    
root = tkint.Tk()
file_in_path_0 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()
root = tkint.Tk()
file_in_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()
root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

# In[]
filelist_0 = [(filename,getsize(join(file_in_path_0,filename)))
             for filename in listdir(file_in_path_0) if getsize(join(file_in_path_0,filename))>1000000]
size_0 = list(list(zip(*filelist_0))[1])
filelist_0 = list(list(zip(*filelist_0))[0])

filelist_1 = [(filename,getsize(join(file_in_path_1,filename)))
             for filename in listdir(file_in_path_1) if getsize(join(file_in_path_1,filename))>1000000]
size_1 = list(list(zip(*filelist_1))[1])
filelist_1 = list(list(zip(*filelist_1))[0])

filelist_out = [filename for filename in listdir(file_out_path)]

# In[column labels]
iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
labels_CNR = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan']))) 

# In[]
def tot_L(r_u,r_v,tau,eta):
    tau = tau[0,:]
    eta = eta[:,0]
    Lu = np.array(sc.integral_lenght_scale(r_u,tau,eta))
    Lv = np.array(sc.integral_lenght_scale(r_v,tau,eta)) 
    return np.r_[Lu,Lv]
######

loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1
date = datetime(1904, 1, 1)

switch = 1

csv_database_0 = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0.db')
csv_database_1 = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1.db')  
#csv_databaseuv = create_engine('sqlite:///'+file_out_path+'/csv_database_UV.db')

csv_database_r = create_engine('sqlite:///'+file_out_path+'/correlations_uv.db')

labels_short = np.array([ 'stop_time', 'azim'])
for w,r in zip(labels_ws,labels_rg):
    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate','CNR'])))
labels_short = np.concatenate((labels_short,np.array(['scan'])))     

col = 'SELECT '
i = 0
for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
    else:
        col = col + ' ' + w + ', ' + r + ',' + c + ', '        
    i+=1
   
scann = int(240*60/45)
chunksize = 45*scann
tot_scans = 20000
tot_iter = int(tot_scans*45/chunksize)
off = 129600#25920*45
lim = [-10,-24]

selec_fil = col + ' FROM "table_fil"'
selec_raw = col + ' FROM "table_raw"'
switch = 0
for i in range(tot_iter):
    print(off)
    df_0 = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_0)
    df = pd.read_sql_query(selec_raw+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_0)
    for i in range(198):
        ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
        df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
    df_1 = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_1)
    df = pd.read_sql_query(selec_raw+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_1)
    for i in range(198):
        ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
        df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
    df = None
    df_0.drop(columns = labels_CNR,inplace=True) 
    df_1.drop(columns = labels_CNR,inplace=True) 
    df_0.columns = labels_short
    df_1.columns = labels_short
    
    if switch == 0:
        phi0 = df_0.azim.unique()
        phi1 = df_1.azim.unique()               
        r0 = np.array(df_0.iloc[(df_0.azim==min(phi0)).nonzero()[0][0]].range_gate)
        r1 = np.array(df_1.iloc[(df_1.azim==min(phi0)).nonzero()[0][0]].range_gate)                
        r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
        r_1, phi_1 = np.meshgrid(r1, np.pi-np.radians(phi1)) # meshgrid                
        tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1),-d)
        switch = 1 
    U_out, V_out, grd, _ = wr.direct_wf_rec(df_0.astype(np.float32), df_1.astype(np.float32), tri, d,N_grid = 512)   
    U_mean = np.nanmean(np.array(U_out))
    V_mean = np.nanmean(np.array(V_out))
    gamma = np.arctan2(V_mean,U_mean)
    tau_out = []
    eta_out = []    
    r_u_out = []
    r_v_out = []
    r_uv_out = []  
    if len(U_out)>0:
        for U,V in zip(U_out, V_out):
            tau,eta,r_u,r_v,r_uv,_,_,_ = sc.spatial_autocorr_sq(grd,U,V,
                                           gamma=gamma, transform = False,
                                           transform_r = True,e_lim=.1,refine=32)
            tau_out.append(tau.astype(np.float32))
            eta_out.append(eta.astype(np.float32)) 
            r_u_out.append(r_u.astype(np.float32))
            r_v_out.append(r_v.astype(np.float32))
            r_uv_out.append(r_uv.astype(np.float32))
        scan = df_0.scan.unique()
        time = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in df_0.scan.unique()])    
        df_0 = None
        df_1 = None  
        r = np.vstack([np.c_[tau.flatten(),eta.flatten(),
                             r_u.flatten(),r_v.flatten(),r_uv.flatten(),
                             np.repeat(s,len(tau.flatten()))] for tau,eta,r_u,r_v,r_uv,s in zip(tau_out,eta_out,r_u_out,r_v_out,r_uv_out,scan)])
        t_stamp = np.hstack([np.repeat(t,len(tau.flatten())) for u,t in zip(tau_out,time)])       
        l = np.vstack([np.r_[tot_L(r_u,r_v,tau,eta),s] for tau,eta,r_u,r_v,s in zip(tau_out,eta_out,r_u_out,r_v_out,scan)])  
        U_out = None
        V_out = None  
        floats = pd.DataFrame(r[:,:5].astype(np.float32))
        floats.columns = ['tau','eta','r_u','r_v','r_uv']
        ints =  pd.DataFrame(r[:,5].astype(np.int32))
        ints.columns = ['scan']
        strings = pd.DataFrame(t_stamp)
        strings.columns = ['time']
        L = pd.DataFrame(l)
        L.columns = ['Lu_x','Lu_y','Lv_x','Lv_y','scan']
        t =  pd.DataFrame(time)
        t.columns = ['time']
        pd.concat([floats,ints,strings],axis=1).to_sql('corr', csv_database_r, if_exists='append',index = False)
        pd.concat([L,t],axis=1).to_sql('L', csv_database_r, if_exists='append',index = False)
    off = off+chunksize
            
          