# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:24:46 2020

@author: lalc
"""
import numpy as np
import scipy as sp
import pandas as pd
import scipy.ndimage as nd
import os
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
import spectralfitting.spectralfitting as sf

from scipy.signal import find_peaks
from os import listdir
from os.path import isfile, join, getsize, abspath
from sqlalchemy import create_engine
from scipy.spatial import Delaunay
from datetime import datetime, timedelta
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage import measure
from scipy import ndimage
import joblib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[]
filedir = 'E:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 1/east'
filename0 = filedir+'/raw_filt_0_east_phase1.db'
filename1 = filedir+'/raw_filt_1_east_phase1.db'


# In[Database]
csv_database_0_ind = create_engine('sqlite:///'+filename0)
csv_database_1_ind = create_engine('sqlite:///'+filename1)  

# In[Reconstruction and correlations]
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

labels_short = np.array([ 'stop_time', 'azim'])

for w,r in zip(labels_ws,labels_rg):
    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate', 'CNR', 'Sb'])))
labels_short = np.concatenate((labels_short,np.array(['scan'])))   
lim = [-8,-24]
i=0
col = 'SELECT '
col_raw = 'SELECT '
for w,r,c, s in zip(labels_ws,labels_rg,labels_CNR, labels_Sb):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', ' + s + ', '
        col_raw = col_raw  +  w  +  ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', scan'
        col_raw = col_raw + ' ' + w
    else:
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', ' 
        col_raw = col_raw + ' ' + w + ', '
    i+=1

selec_fil = col + ' FROM "table_fil"'
selec_raw = col_raw + ' FROM "table_raw"'

days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ind).values
days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ind).values
days0 = np.squeeze(days0)
days1 = np.squeeze(days1)
days = np.unique(np.r_[days0[np.isin(days0,days1)],days1[np.isin(days1,days0)]])


dy = days[0]

# Reconstruction of chuncks of 1 hour scans?
switch = 0

query_fil_0 = selec_fil+ ' where name = ' + dy 
query_raw_0 = selec_raw+ ' where name = ' + dy 

query_fil_1 = selec_fil+ ' where name = ' + dy 
query_raw_1 = selec_raw+ ' where name = ' + dy

# First database loading
print('reading df0')
df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ind)
df = pd.read_sql_query(query_raw_0, csv_database_0_ind)
# Retrieving good CNR values from un-filtered scans
for i in range(198):
    ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
    df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None
df_0.columns = labels_short
# Second database loading
print('reading df1')
df_1 = pd.read_sql_query(query_fil_1, csv_database_1_ind)
df = pd.read_sql_query(query_raw_1, csv_database_1_ind)
for i in range(198):
    ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
    df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None   
df_1.columns = labels_short

loc0 = np.array([0,6322832.3])
loc1 = np.array([0,6327082.4])
d = loc1-loc0  
switch = 0

ulist=[]
vlist=[]

s_syn,t_scan,_,_ = synch_df(df_0,df_1,dtscan=45/2)
tri_calc = True
chunk = 256
print(s_syn)
if len(s_syn)>0:      
    ind_df0 = df_0.scan.isin(s_syn[:,0])
    df_0 = df_0.loc[ind_df0]
    ind_df1 = df_1.scan.isin(s_syn[:,1])
    df_1 = df_1.loc[ind_df1] 
    # 1 hour itervals
    s0 = s_syn[:,0]
    s1 = s_syn[:,1]
    t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])
    t1 = np.array([date+timedelta(seconds = df_1.loc[df_1.scan==s].stop_time.min()) for s in s1])                                                  
    tmin, tmax = np.min(np.r_[t0,t1]), np.max(np.r_[t0,t1])+timedelta(hours = 1)
    t_1h = pd.date_range(tmin, tmax,freq='1H')#.strftime('%Y%m%d%H%M)
    indt0 = [df_0.scan.isin(s0[(t0>=t_1h[i]) & (t0<t_1h[i+1])]) for i in range(len(t_1h)-1)]  
    indt1 = [df_1.scan.isin(s1[(t1>=t_1h[i]) & (t1<t_1h[i+1])]) for i in range(len(t_1h)-1)]       
    indt0 = [x for x in indt0 if np.sum(x) > 0]
    indt1 = [x for x in indt1 if np.sum(x) > 0] 
    if (len(indt0)>0)&(len(indt1)>0):
        for i0,i1 in zip(indt0,indt1):
            df0 = df_0.loc[i0]
            df1 = df_1.loc[i1]
            if switch == 0:
                phi0 = df0.azim.unique()
                phi1 = df1.azim.unique()               
                r0 = df0.range_gate.iloc[0].values
                r1 = df1.range_gate.iloc[0].values                
                r_0, phi_0 = np.meshgrid(r0, np.pi/2-np.radians(phi0)) # meshgrid
                r_1, phi_1 = np.meshgrid(r0, np.pi/2-np.radians(phi1)) # meshgrid                
                tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
                switch = 1 
            uo, vo, grdu, so = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk) 
            ulist.append(uo)
            vlist.append(vo)
ulist = [item for sublist in ulist for item in sublist]
vlist = [item for sublist in vlist for item in sublist]            


dr0 = np.array(np.r_[-d/2,np.ones(1)]).T 
dr1 = np.array(np.r_[d/2,np.ones(1)]).T

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)

for j in range(len(ulist)):
    ax0.cla()    

    # ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20) 
    ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
    ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
    im0 = ax0.contourf(grdu[0], grdu[1], ulist[j], np.linspace(-10,-2,10), cmap='jet')
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
    ax0.text(6000, 2500,'(a)',fontsize=30,color='k')
    
    if len(fig0.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts = fig0.axes[-1].get_position().get_points()
        # and its label
        label = fig0.axes[-1].get_ylabel()
        # and then remove the axes
        fig0.axes[-1].remove()
        # then we draw a new axes a the extents of the old one
        divider = make_axes_locatable(ax0)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
        cb.ax.tick_params(labelsize=24)
        cb.ax.set_ylabel(r'$U\:[m/s]$', fontsize = 24)
        # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=24, )
        # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=24, weight='bold') 
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        divider = make_axes_locatable(ax0)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig0.colorbar(im0, cax=cax, format=ticker.FuncFormatter(fm))
        cb.ax.set_ylabel('$U\:[m/s]$', fontsize = 24)
        cb.ax.tick_params(labelsize=24)
        # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=12, weight='bold')
        # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=12, weight='bold') 

    fig0.tight_layout()
    plt.pause(.6)


            

            if len(uo)>0:
                x = grdu[0][0,:]
                y = grdu[1][:,0]
                U_arr = np.vstack([uo[i] for i in range(len(uo))])
                V_arr = np.vstack([vo[i] for i in range(len(vo))])
                scan = s_syn[:,0]
                U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
                V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
                u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc)
                uvlist = [U_rot(grdu, u, v, gamma = gamma, tri_calc = False, tri_del = tri_del, mask_int = mask_int, mask = mask) for u,v in zip(uo,vo)]
                ur, vr = [uv[0] for uv in uvlist], [uv[1] for uv in uvlist]



