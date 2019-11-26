# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:58:58 2019

@author: lalc
"""
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize
from os import listdir
import tkinter as tkint
import tkinter.filedialog
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

root = tkint.Tk()
file_in_mask = tkint.filedialog.askdirectory(parent=root,title='Choose an masks dir')
root.destroy()

# In[column labels]
iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))

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

filelist_mask = [(filename,getsize(join(file_in_mask,filename)))
             for filename in listdir(file_in_mask) if getsize(join(file_in_mask,filename))>1000]
size_mask = list(list(zip(*filelist_mask))[1])
filelist_mask = list(list(zip(*filelist_mask))[0])

# In[]
from sqlalchemy import create_engine
from datetime import datetime, timedelta

date = datetime(1904,1,1)
######

loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1

switch = 1
	
csv_database = create_engine('sqlite:///csv_database_U_V.db')

for counter, file_0 in enumerate(filelist_0,0):
    scan_init_0 = 0
    scan_init_1 = 0
    print(counter,file_0)
    mask = pd.DataFrame()
#    azim0 = pd.DataFrame() 
#    azim1 = pd.DataFrame() 
    file_out_name= file_0[:14]+'_mask_0.pkl'
    if (file_0 in filelist_1) & (~(file_out_name in filelist_out)):
        t_step = 3        

        
        file_0_path = join(file_in_path_0,file_0)
        file_mask_0 = filelist_mask[int(np.where(np.array(filelist_mask) == file_0_path[-36:-22]+'_mask_0.pkl')[0])]
        file_1_path = join(file_in_path_1,file_0)
        file_mask_1 = filelist_mask[int(np.where(np.array(filelist_mask) == file_1_path[-36:-22]+'_mask_1.pkl')[0])]
        
        with open(join(file_in_mask,file_mask_0), 'rb') as reader:
            mask0 = pickle.load(reader)          
        with open(join(file_in_mask,file_mask_1), 'rb') as reader:
            mask1 = pickle.load(reader) 
            
        for chunk0,chunk1 in zip(pd.read_csv(file_0_path, sep=";",
                                 header=None, chunksize=int(t_step*45*10)),
                                 pd.read_csv(file_1_path, sep=";",
                                 header=None, chunksize=int(t_step*45*10))):
            
            chunk0.columns = labels
            chunk0['scan'] = chunk0.groupby('azim').cumcount()
#            azim0 = pd.concat([azim0,chunk0.loc[loc]['azim']]) 
            chunk1.columns = labels
            chunk1['scan'] = chunk1.groupby('azim').cumcount()
            
            if switch == 1:
                phi0 = chunk0.azim.unique()
                phi1 = chunk1.azim.unique()               
                r0 = np.array(chunk0.iloc[(chunk0.azim==min(phi0)).nonzero()[0][0]].range_gate)
                r1 = np.array(chunk1.iloc[(chunk1.azim==min(phi0)).nonzero()[0][0]].range_gate)                
                r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
                r_1, phi_1 = np.meshgrid(r1, np.pi-np.radians(phi1)) # meshgrid                
                tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1),-d)
                switch = 0
            
            scan_end_0 = scan_init_0 + chunk0.scan.max()
            scan_end_1 = scan_init_1 + chunk1.scan.max()
            
            loc_mask0 = (mask0.scan>=scan_init_0) & (mask0.scan<=scan_end_0)
            loc_mask1 = (mask1.scan>=scan_init_1) & (mask1.scan<=scan_end_1)
            if (loc_mask1.sum()>0) & (loc_mask0.sum()>0):
                m0 = mask0.loc[loc_mask0].ws
                m1 = mask1.loc[loc_mask1].ws
                m0.index = chunk0.index
                m1.index = chunk1.index              
                chunk0.ws = chunk0.ws.mask(m0.ws)
                chunk1.ws = chunk1.ws.mask(m1.ws)            

                            
                ind0 = np.unique(chunk0.scan.values)%t_step==0
                times0 = np.unique(np.append(np.unique(chunk0.scan.values)[ind0],
                                           chunk0.scan.values[-1])) 
                ind1 = np.unique(chunk1.scan.values)%t_step==0
                times1 = np.unique(np.append(np.unique(chunk1.scan.values)[ind1],
                                           chunk1.scan.values[-1]))            
                
                times_scan0 = scan_init_0 + times0
                times_scan1 = scan_init_1 + times1                        
                if len(times0) <= len(times0):
                    times = times0
                else:
                    times = times1
                UV = pd.DataFrame()
                for i in range(len(times)-1):
                    print(times[i],times[i+1])
                    loc0 = (chunk0.scan>=times[i]) & (chunk0.scan<times[i+1])
                    loc1 = (chunk1.scan>=times[i]) & (chunk1.scan<times[i+1])             
                    U_out, V_out, grd, _ = wr.direct_wf_rec(chunk0.loc[loc0].astype(np.float32), chunk1.loc[loc1].astype(np.float32),
                                                                tri, d,N_grid = 512)   
                    init = scan_init_0 + times[i]
                    end = scan_init_0 + times[i+1]
                    scan = np.arange(init,end)
                    time = np.array([str(date+timedelta(seconds = chunk0.loc[loc0].loc[chunk0.loc[loc0].scan==s].stop_time.min()))[:-7]
                                      for s in chunk0.loc[loc0].scan.unique()])
                    vel = np.vstack([np.c_[grd[0][~np.isnan(u)],grd[1][~np.isnan(u)],
                                                  u[~np.isnan(u)],v[~np.isnan(u)],
                                                  np.asarray(~np.isnan(u)).nonzero()[0],np.asarray(~np.isnan(u)).nonzero()[1],  
                                                  np.repeat(s,np.sum(~np.isnan(u)))] for u,v,t,s in zip(U_out,V_out,time,scan)])
                    t_stamp = np.hstack([np.repeat(t,np.sum(~np.isnan(u))) for u,t in zip(U_out,time)])                         
            #        vel = np.c_[grd[1][:,0:(end-init)].T.flatten(),np.vstack(U_out),np.vstack(V_out),time,scan] #grd[0][0:(end-init),:].flatten()
#                    labels_U = ['x','y','U','V','ind_0','ind_1','time','scan']
                    #['x']+'y']+[]['U_'+str(g) for g in range(len(grd[0][0,:]))]+['V_'+str(g) for g in range(len(grd[0][0,:]))]+['time','scan']
                    floats = pd.DataFrame(vel[:,:4].astype(np.float32))
                    floats.columns = ['x','y','U','V']
#                    UV['x','y','U','V'] = floats
                    ints =  pd.DataFrame(vel[:,4:].astype(np.int_))
                    ints.columns = ['ind_0','ind_1','scan']
#                    UV['ind_0','ind_1','scan'] = ints
                    strings = pd.DataFrame(t_stamp)
                    strings.columns = ['time']
#                    UV['time'] = strings
                    UV = pd.concat([UV,pd.concat([floats,ints,strings],axis=1)])
                UV.to_sql('table', csv_database, if_exists='append')  
                scan_init_0 = scan_end_0
                scan_init_1 = scan_end_1
#                scan_init_0+=times0[-1]-1
#                scan_init_1+=times1[-1]-1    

# In[]
scan_n = 1

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)                
im = ax.contourf(U_out[scan_n],cmap = 'jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)                
plt.contourf(V_out[scan_n],cmap = 'jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),chunk_0.ws.loc[chunk_0.scan==scan_n].values,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_1*np.cos(phi_1),r_1*np.sin(phi_1),chunk_1.ws.loc[chunk_1.scan==scan_n].values,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0*np.cos(phi_0)-d[0]/2,r_0*np.sin(phi_0),chunk0.ws.loc[chunk0.scan==scan_n].values)
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_1*np.cos(phi_1)+d[0]/2,r_1*np.sin(phi_1),chunk1.ws.loc[chunk1.scan==scan_n].values,cmap='jet')
fig.colorbar(im)





