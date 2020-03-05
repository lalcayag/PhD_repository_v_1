# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:39:46 2019

@author: lalc
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
import seaborn as sns
from scipy.signal import find_peaks
import pickle
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
#import spectralfitting.spectralfitting as sf
from sqlalchemy import create_engine
from datetime import datetime, timedelta
date = datetime(1904, 1, 1) 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

from scipy import ndimage

# In[]    
root = tkint.Tk()
file_in_path_0 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()
root = tkint.Tk()
file_in_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_in_path_corr = tkint.filedialog.askdirectory(parent=root,title='Choose an corr dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()
root = tkint.Tk()
file_out_path_u_field = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

# In[]
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
def synch_df(df0,df1,dtscan=45):
    s0 = df0.scan.unique()
    s1 = df1.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    dt = [(t0[i]-t1[j]).total_seconds() for i in range(len(t0)) for j in range(len(t1))]
    s = np.array([[s_0,s_1] for s_0 in s0 for s_1 in s1])
    ind_synch = np.abs(dt)<dtscan
    if np.sum(ind_synch)==0:
       sync = []
       off0 = s0[-1]
       off1 = s1[-1]
    else:
       sync = s[ind_synch,:]
       off0 = sync[-1,0]+1
       off1 = sync[-1,1]+1
    return (sync,off0*45,off1*45)

def gamma_10m(df_0,df_1,U,V,su):
    t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in np.array(su)[:,0]])
    t1 = np.array([date+timedelta(seconds = df_1.loc[df_1.scan==s].stop_time.min()) for s in np.array(su)[:,1]])  
    tmin, tmax = np.min(np.r_[t0,t1]), np.max(np.r_[t0,t1])+timedelta(minutes = 10)
    t_10m = pd.date_range(tmin, tmax,freq='10T')#.strftime('%Y%m%d%H%M)
    gamma = []
    indt = [np.where((t0>t_10m[i]) & (t0<t_10m[i+1]))[0] for i in range(len(t_10m)-1)]  
    U_mean = [np.nanmedian(np.array([U[ii0] for ii0 in i0])) for i0 in indt]   
    V_mean = [np.nanmedian(np.array([V[ii0] for ii0 in i0])) for i0 in indt]     
    gamma = [np.arctan2(vi,ui) for vi,ui in zip(V_mean,U_mean)]
    gamma_out = np.zeros(len(U))
    for i in range(len(gamma)):
        gamma_out[indt[i]] = gamma[i]
    return gamma_out 

# In[]

loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1  
  
csv_database_r2 = create_engine('sqlite:///'+file_out_path+'/corr_uv_west_phase2_ind.db')
csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0_phase2.db')
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1_phase2.db')     
labels_short = np.array([ 'stop_time', 'azim'])
for w,r in zip(labels_ws,labels_rg):
    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate'])))
labels_short = np.concatenate((labels_short,np.array(['scan'])))   
lim = [-8,-24]
i=0
col = 'SELECT '
col_raw = 'SELECT '
for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
        col_raw = col_raw + w + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
        col_raw = col_raw + ' ' + w
    else:
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' 
        col_raw = col_raw + ' ' + w + ', '
    i+=1

selec_fil = col + ' FROM "table_fil"'
selec_raw = col_raw + ' FROM "table_raw"'

#init = pd.read_sql_query('select name, hms, scan from "L" where scan = (select max(scan) from "L")', csv_database_r2)
#n_i, h_i, scan_i = init.name.values[0], init.hms.values[0], init.scan.values[0]
chunk_scan = int(13*6)

days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ind).values
days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ind).values
days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r2).values
days_old = np.squeeze(days_old)
days0 = np.squeeze(days0)
days1 = np.squeeze(days1)
days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])
days = days[~np.isin(days,days_old)]
switch = 0
dy = '20160804'
U_out, V_out, su = [], [], [] 

query_fil = selec_fil+ ' where name = '+dy
query_raw = selec_raw+ ' where name = '+dy

df_0 = pd.read_sql_query(query_fil, csv_database_0_ind)
df = pd.read_sql_query(query_raw, csv_database_0_ind)
for i in range(198):
    ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
    df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None
df_0.drop(columns = labels_CNR,inplace=True)
df_0.columns = labels_short
#day_r = pd.to_datetime(pd.read_sql_query('select time from L where name = '+
#                                          dy,csv_database_r2).time.values).strftime("%H%M%S").values  
                                      
#if len(day_r)>0: 
s0 = df_0.scan.unique()
t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])                                         
ind = ~np.isin(pd.to_datetime(t0).strftime("%H%M%S").values,day_r)
s0 = s0[ind]
ind_df0 = df_0.scan.isin(s0)
df_0 = df_0.loc[ind_df0]
if ~(np.sum(ind)==0):
    df_1 = pd.read_sql_query(query_fil, csv_database_1_ind)
    df = pd.read_sql_query(query_raw, csv_database_1_ind)
    for i in range(198):
        ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
        df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
    df = None   
    df_1.drop(columns = labels_CNR,inplace=True) 
    df_1.columns = labels_short
    
    #Synchronous?    
    s_syn,_,_ = synch_df(df_0,df_1,dtscan=45/2)
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
                    r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
                    r_1, phi_1 = np.meshgrid(r0, np.pi-np.radians(phi1)) # meshgrid                
                    tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
                    switch = 1 
                u, v, grd, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = 512)  
                U_out.append(u), V_out.append(v), su.append(s)
                
U_out = [item for sublist in U_out for item in sublist]
V_out = [item for sublist in V_out for item in sublist]
su    = [item for sublist in su for item in sublist]

#with open('U_rec_08_08_2016.pkl', 'wb') as field:
#     pickle.dump((U_out, V_out, grd, su), field)
     
# Vorticity  dVy/dx-dVx/dy
#############################################################
from scipy.spatial import Delaunay
def vort_cont(U,V,grd):
    #print(U.shape,grd[0].shape,grd[0][0,:].shape)
    du_dy, du_dx = np.gradient(U, grd[0][0,:], grd[1][:,0])#!!!!!!!!!!!!!!!!!!
    dv_dy, dv_dx = np.gradient(V, grd[0][0,:], grd[1][:,0])#!!!!!!!!!!!!!!!!!!  
    vort = dv_dx - du_dy
    cont = du_dx + dv_dy
    return (vort, cont)

def shrink(grid,U):
    patch = ~np.isnan(U)
    ind_patch_x = np.sum(patch,axis=1) != 0
    ind_patch_y = np.sum(patch,axis=0) != 0
#    if np.sum(ind_patch_x) > np.sum(ind_patch_y):
#        ind_patch_y = ind_patch_x
#    elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
#        ind_patch_x = ind_patch_y        
    n = np.sum(ind_patch_x)
    m = np.sum(ind_patch_y)          
    ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
    ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
    U = np.reshape(U[ind_patch_grd],(n,m))
    grid_x = np.reshape(grid[0][ind_patch_grd],(n,m))
    grid_y = np.reshape(grid[1][ind_patch_grd],(n,m))
    return (grid_x,grid_y,U) 
   
def field_rot(x, y, U, V, gamma = None, grid = None, tri_calc = False, tri_del = []):
    
        
    U_mean = np.nanmean(U.flatten())
    V_mean = np.nanmean(V.flatten())
    # Wind direction
    gamma = np.arctan2(V_mean,U_mean)
    # Components in matrix of coefficients
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
    
    vel = np.array(np.c_[U.flatten(),V.flatten()]).T
    vel = np.dot(R[:-1,:-1],vel)
    print(R)
    U = vel[0,:]
    V = vel[1,:]
    mask = ~np.isnan(U)
    mask_int = []
  
    if tri_calc:
        if not grid:
            grid = np.meshgrid(x,y)       
        xtrans = 0
        ytrans = y[0]/2
        T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
        T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
        T = np.dot(np.dot(T1,R),T2)
        Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
        Xx = np.dot(T,Xx)   
        tri_del = Delaunay(np.c_[Xx[0,:][mask],Xx[1,:][mask]])
        mask_int = ~(tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()]) == -1)       
    return (U, V, mask, mask_int, tri_del) 

def U_rot(grid,U,V, tri_calc = True):
    x = grid[0][0,:]
    y = grid[1][:,0]
    U, V, mask,mask_int, tri_del = field_rot(x, y, U, V, grid = grid,
                                    tri_calc = tri_calc)
    U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
    U[~mask_int] = np.nan
    V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
    V[~mask_int] = np.nan      
    
    U = np.reshape(U,grid[0].shape)
    V = np.reshape(V,grid[0].shape)
    return (U,V)

def filterfft(vorti, mask, sigma=20):
    vort = vorti.copy()
    vort[np.isnan(vort)] = np.nanmean(vort)
    
    input_ = np.fft.fft2(vort)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.real(np.fft.ifft2(result))
    result[mask] = np.nan
    return result

#############################################################
vort_m = []
cont_m = []
for i in range(len(U_out)):
    print(i)
    vort, cont =  vort_cont(U_out[i], V_out[i], grd)    
    vort_m.append(vort)
    cont_m.append(cont)
#with open(file_out_path_u_field+'/U_rec_08_04_2016.pkl', 'wb') as field:
#     pickle.dump((U_out, V_out, grd, su), field)
#with open(file_out_path_u_field+'/vort_rec_08_04_2016.pkl', 'wb') as field:
#     pickle.dump((vort_m, cont_m), field)
#############################################################

with open(file_out_path_u_field+'/U_rec_08_06_2016.pkl', 'rb') as field:
     U_out, V_out, grd, su = pickle.load(field)
     
with open(file_out_path_u_field+'/vort_rec_08_06_2016.pkl', 'rb') as field:
     vort_m, cont_m = pickle.load(field) 
             
#contours = ax0.contour(grdx, grdy, done0,20, colors='black') 
#ax0.clabel(contours, inline=True, fontsize=8)
##im = ax.contourf(grdx, grdy, V-np.nanmean(V),20,cmap='RdGy') 
#im0 = ax0.contourf(grdx, grdy, np.sqrt(U**2+V**2),20,cmap='RdGy') 
#
#contours = ax1.contour(grdx, grdy, done1,20, colors='black') 
#ax1.clabel(contours, inline=True, fontsize=8)
##im = ax.contourf(grdx, grdy, V-np.nanmean(V),20,cmap='RdGy') 
#im1 = ax1.contourf(grdx, grdy, np.sqrt(U**2+V**2),20,cmap='RdGy') 
#
#contours = ax1.contour(grdx, grdy, done2,20, colors='black') 
#ax1.clabel(contours, inline=True, fontsize=8)
##im = ax.contourf(grdx, grdy, V-np.nanmean(V),20,cmap='RdGy') 
#im1 = ax1.contourf(grdx, grdy, np.sqrt(U**2+V**2),20,cmap='RdGy') 
#
#fig.colorbar(im0)
#fig.colorbar(im1)
# time average
#########################################################
ch = 7
init = 900
r_j = range(init,init+ch*50)
U_out = [U_out[i] for i in r_j]
V_out = [V_out[i] for i in r_j]
#vort_m = [vort_m[i] for i in r_j]

vort_ave = []
u_ave = []
v_ave = []
Uave = []
Vave = []
ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
for j in range(len(U_out))[::ch]:
    print(j)
    r_i = range(j,j+ch)
    chunk = 512
    vort_smooth_list = []
    for i in r_i:
        s = (np.min(dfL_phase2.loc[(dfL_phase2.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)*1.1/ds)+1
        v = filterfft(v,np.isnan(v),sigma = s)
        vort_smooth_list.append(filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s))
    vort_array = np.vstack(vort_smooth_list)
    vort_ave.append(vort_array.reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/ch)
    
    U_ave = np.vstack([np.sqrt(U_out[i]**2+ V_out[i]**2) for i in r_i]).reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/ch
    
    u_ave.append(np.vstack([filterfft(U_out[i],np.isnan(u),sigma=.01) 
    for i,u in zip(r_i,[U_out[i] for i in r_i])]).reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/ch)
    v_ave.append(np.vstack([filterfft(V_out[i],np.isnan(u),sigma=.01)
    for i,u in zip(r_i,[U_out[i] for i in r_i])]).reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/ch)

    Uave.append(filterfft(U_out[r_i[0]],np.isnan(U_out[r_i[0]]),sigma = 15))
    Vave.append(filterfft(V_out[r_i[0]],np.isnan(V_out[r_i[0]]),sigma = 15))

#with open(file_out_path_u_field+'/U_vort_ave_08_04_2016.pkl', 'wb') as field:
#     pickle.dump((vort_ave, u_ave, v_ave, Uave, Vave), field)
     
#######################################################
# stability conditions   
from sqlalchemy import create_engine
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)   
csv_database_r2 = create_engine('sqlite:///'+file_in_path_corr+'/corr_uv_west_phase2_ind.db')
days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r2).values
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)
heights = [241, 175, 103, 37, 7]
L_list_2 = np.hstack([L_smooth(np.array(res_flux_2)[:,i,:]) for i in range(len(heights))])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_2)[:,2,0]])
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]
cols = [item for sublist in cols for item in sublist]
stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2
dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2.index = dfL_phase2.index
aux2 = dfL_phase2.time
dfL_phase2.time = pd.to_datetime(dfL_phase2.time)
for i in range(len(t2)-1):
    print(i,t2[i])
    ind = (dfL_phase2.time>=t2[i]) & (dfL_phase2.time<=t2[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values
dfL_phase2.time = aux2
dfL_phase2[cols] = Lph2
dfL_phase2.sort_values(by=['time'], inplace = True)

ax = dfL_phase2.loc[(dfL_phase2.name=='20160804')|(dfL_phase2.name=='20160805')|
        (dfL_phase2.name=='20160806')].plot(x = 'time', y='$L_{flux,103}$',style='o')
########################################################    
# Averages    

with open(file_out_path_u_field+'/U_rec_08_06_2016.pkl', 'rb') as field:
     U_out, V_out, grd, su = pickle.load(field)   
with open(file_out_path_u_field+'/vort_rec_08_06_2016.pkl', 'rb') as field:
     vort_m, cont_m = pickle.load(field)    
with open(file_out_path_u_field+'/U_vort_ave_08_06_2016.pkl', 'rb') as field:
     vort_ave, u_ave, v_ave, Uave, Vave = pickle.load(field)
    
fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
j = 28
#for j in range(len(u_ave)):    
#    ax.cla() 
   
grdx, grdy, u = shrink(grd, u_ave[j])
_, _, v = shrink(grd, v_ave[j])
#_, _, U = shrink(grd, Uave[j])
#_, _, V = shrink(grd, Vave[j])
grdxv, grdyv, vort = shrink(grd, vort_ave[j])
U_mean,V_mean = np.nanmean(u), np.nanmean(v) 
##########################

ax.cla()  
levels = np.linspace(np.nanmin(vort), np.nanmax(vort), 10)
levels = levels[np.abs(levels) > np.nanmax(vort)*.01]        
contours = ax.contour(grdyv, grdxv, np.fliplr(vort), levels, colors='black',alpha=.8,linewidths = 1) 
ax.clabel(contours, inline=True, fontsize=12) 
im = ax.contourf(grdy, grdx, np.fliplr(np.sqrt(u**2+v**2)),20,cmap='jet') 
contours = ax.contour(grdyv, grdxv, np.fliplr(vort), levels = [.00], colors='black',linewidths = 3) 
#ax.clabel(contours, inline=True, inline_spacing  = 0, fontsize=20,fmt = '%1.0f') 
divider = make_axes_locatable(ax)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax)
cb.ax.set_ylabel(r"$Wind\:speed\:[m/s]$",fontsize=26)
ax.set_xlabel(r'$West-East\:[m]$', size=26, weight='bold')
ax.set_ylabel(r'$North-South\:[m]$', fontsize=26, weight='bold')
ax.tick_params(labelsize=16)

Q = ax.quiver(-1200.00, 2700.00, V_mean, -U_mean,pivot='middle', scale=110, units='width', color='k',zorder=2)
circle = plt.Circle((-1200.00, 2700.00), 450, edgecolor='k', facecolor= 'white', fill=True, zorder=1)
ax.add_artist(circle)
fig.tight_layout()    
      
##########################    

    contours = ax.contour(grdyv, grdxv, np.fliplr(vort),20, colors='black')
    im = ax.contourf(grdy, grdx, np.fliplr(np.sqrt((u)**2+(v)**2)), 20, cmap='bwr') 
#    plt.colorbar(im)
    
#    im = ax.contourf(grd[0], grd[1], v_ave, 20,cmap='jet') 
    
    plt.pause(.5)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
contours = ax.contour(grd[0], grd[1], vort_m[i],20, colors='black')
##################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.01)
fig.set_size_inches(8,8)
r_i = range(800,950)
i = 900
ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
#for i in r_i: #enumerate([filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=15) for i in r_i]):   
#    r_i = range(i,i+100)
grdx, grdy, U = shrink(grd, U_out[i])
_, _, V = shrink(grd, V_out[i])
grdxv, grdyv, v = shrink(grd, vort_m[i])
U_mean,V_mean = np.nanmean(U), np.nanmean(V)

#    x, y = grdx.flatten(), grdy.flatten()
#    done = griddata(np.c_[x,y], done.flatten(), (grdx.flatten(), grdy.flatten()), method='linear')
#    done = np.reshape(done,grdx.shape)
s = (np.min(dfL_phase2.loc[(dfL_phase2.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)*1.1/ds)+1
v = filterfft(v,np.isnan(v),sigma = s)
ax.cla()  
levels = np.linspace(np.nanmin(v), np.nanmax(v), 10)
levels = levels[np.abs(levels) > np.nanmax(v)*.01]        
contours = ax.contour(grdyv, grdxv, np.fliplr(v), levels, colors='black',alpha=.8,linewidths = 1) 
ax.clabel(contours, inline=True, fontsize=12) 
im = ax.contourf(grdy, grdx, np.fliplr(np.sqrt(U**2+V**2)),20,cmap='jet') 
contours = ax.contour(grdyv, grdxv, np.fliplr(v), levels = [.00], colors='black',linewidths = 3) 
#ax.clabel(contours, inline=True, inline_spacing  = 0, fontsize=20,fmt = '%1.0f') 
divider = make_axes_locatable(ax)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax)
cb.ax.set_ylabel(r"$Wind speed\:[m/s]$",fontsize=26)
ax.set_xlabel(r'$West-East\:[m]$', size=26, weight='bold')
ax.set_ylabel(r'$North-South\:[m]$', fontsize=26, weight='bold')
ax.tick_params(labelsize=16)

Q = ax.quiver(-1200.00, 2700.00, V_mean, -U_mean,pivot='middle', scale=110, units='width', color='k',zorder=2)
circle = plt.Circle((-1200.00, 2700.00), 450, edgecolor='k', facecolor= 'white', fill=True, zorder=1)
ax.add_artist(circle)
fig.tight_layout()
    
#    im = ax.contourf(grdx, grdy, v,20,cmap='RdGy') 
    
    plt.pause(.5)


contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();


#vort_m, cont_m = [], []
#for i in range(len(U_out)):
#    print(i)
#    vort, cont =  vort_cont(U_out[i], V_out[i], grd)
#    vort_m.append(vort)
#    cont_m.append(cont)
#
#vort_m = [item for sublist in U_out for item in sublist]
#cont_m = [item for sublist in V_out for item in sublist]

#with open('vort_rec_08_08_2016.pkl', 'wb') as field:
#     pickle.dump((vort_m, cont_m), field)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grdx, grdy, vort, np.linspace(-.05,.05,20),cmap='jet') 
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grdx[n-1:,m-1:], grdy[n-1:,m-1:], done, cmap='jet') 
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grdx, grdy, np.sqrt(U**2+V**2),np.linspace(10,20,20),cmap='jet') 
fig.colorbar(im)
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
for i in range(180,220):    
    ax.cla()          
    ax.contourf(grd[0], grd[1], np.sqrt(U_out[i]**2+V_out[i]**2), 20,cmap='RdGy')  
    plt.pause(.4)

s = 47462
plt.figure()
plt.contourf(r_0*np.cos(phi_0), r_0*np.sin(phi_0),df0.ws.loc[df0.scan==s],cmap='jet')

s = 46811
plt.figure()
plt.contourf(r_1*np.cos(phi_1), r_1*np.sin(phi_1),df1.ws.loc[df1.scan==s],cmap='jet')

              