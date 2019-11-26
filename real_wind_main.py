# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:20:43 2019

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
from scipy.signal import find_peaks
import pickle
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
import spectralfitting.spectralfitting as sf


import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

import emcee
import corner
from scipy.stats import chi

# In[Data loading]
root = tkint.Tk()
file_lidar_path_0 = tkint.filedialog.askopenfilenames(parent=root,title='Choose first lidar file')
root.destroy()

root = tkint.Tk()
file_lidar_path_1 = tkint.filedialog.askopenfilenames(parent=root,title='Choose second lidar file')
root.destroy()

root = tkint.Tk()
file_mask_path_0 = tkint.filedialog.askopenfilenames(parent=root,title='Choose first mask file')
root.destroy()

root = tkint.Tk()
file_mask_path_1 = tkint.filedialog.askopenfilenames(parent=root,title='Choose second mask file')
root.destroy()
# In[Data to dataframe structure]
# Columns names

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(198):

    labels = np.concatenate((labels,vel_lab))
    
# Loading data and creating dataframes

df_0 = pd.read_csv(file_lidar_path_0[0],sep=";", header=None) # Sirocco
df_1 = pd.read_csv(file_lidar_path_1[0],sep=";", header=None) # Vara

df_0.columns = labels
df_1.columns = labels

df_0['scan'] = df_0.groupby('azim').cumcount()
df_1['scan'] = df_1.groupby('azim').cumcount()

######

loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1

phi0 = df_0.azim.unique()
phi1 = df_1.azim.unique()

r0 = np.array(df_0.iloc[(df_0.azim==min(phi0)).nonzero()[0][0]].range_gate)
r1 = np.array(df_1.iloc[(df_1.azim==min(phi0)).nonzero()[0][0]].range_gate)

r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
r_1, phi_1 = np.meshgrid(r1, np.pi-np.radians(phi1)) # meshgrid

tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1),-d)
    
# In[Loading masks and filtering]    
with open(file_mask_path_0[0], 'rb') as m:
     mask_df_0 = pickle.load(m)

with open(file_mask_path_1[0], 'rb') as m:
     mask_df_1 = pickle.load(m)
     
mask_CNR_0 = (df_0.CNR>-24) & (df_0.CNR<-8)
mask_CNR_1 = (df_1.CNR>-24) & (df_1.CNR<-8)

mask_CNR_0.columns =  mask_df_0.columns
mask_CNR_1.columns =  mask_df_1.columns

mask_df_0.mask(mask_CNR_0,other=False,inplace=True)
mask_df_1.mask(mask_CNR_1,other=False,inplace=True)
             
df_0.ws = df_0.ws.mask(mask_df_0)
df_1.ws = df_1.ws.mask(mask_df_1)

# In[Reconstruction and mean wind estimation]
#
s=9000
start_scan = np.max([9000,s])
stop_scan = 11000
dscan = 10

start = np.arange(start_scan, stop_scan, dscan)
end = np.arange(start_scan+dscan, stop_scan+dscan, dscan) 

ind_0 = (df_0.scan>=start_scan) & (df_0.scan<=stop_scan)
ind_1 = (df_1.scan>=start_scan) & (df_1.scan<=stop_scan)

df_0 = df_0.loc[ind_0]
df_1 = df_1.loc[ind_1]

for s,e in zip(start, end):
    ind0 = (df_0.scan>=s) & (df_0.scan<e)
    ind1 = (df_1.scan>=s) & (df_1.scan<e)
    U_out, V_out, grd, _ = wr.direct_wf_rec(df_0.loc[ind0], df_1.loc[ind1],
                                            tri, d,N_grid = 512)   
    U_mean = np.nanmean(np.array(U_out))
    V_mean = np.nanmean(np.array(V_out))
    gamma = np.arctan2(V_mean,U_mean)        
    print(gamma*180/np.pi)
    current_scan = s
    for U,V in zip(U_out, V_out):
        print(current_scan)
        tau,eta,r_u,r_v,r_uv,_,_,_,_ = sc.spatial_autocorr_sq(grd,U,V,
                                       gamma=gamma, transform = False,
                                       transform_r = True,e_lim=.1,refine=32)
        valid_name = 'valid_scan_' + str(int(current_scan))
        tau_name = 'tau_scan_' + str(int(current_scan))
        eta_name = 'eta_scan_' + str(int(current_scan))
        r_u_name = 'r_u_scan_' + str(int(current_scan))
        r_v_name = 'r_v_scan_' + str(int(current_scan))
        r_uv_name = 'r_uv_scan_' + str(int(current_scan))
        current_scan+=1
        (tau.flatten()).astype(np.float32).tofile(tau_name)
        (eta.flatten()).astype(np.float32).tofile(eta_name)
        (r_u.flatten()).astype(np.float32).tofile(r_u_name)
        (r_v.flatten()).astype(np.float32).tofile(r_v_name)
        (r_uv.flatten()).astype(np.float32).tofile(r_uv_name)    

# In[]
current_scan = 9000
tau_name = 'tau_scan_' + str(int(current_scan))
eta_name = 'eta_scan_' + str(int(current_scan))
r_u_name = 'r_u_scan_' + str(int(current_scan))
r_v_name = 'r_v_scan_' + str(int(current_scan))
r_uv_name = 'r_uv_scan_' + str(int(current_scan))
tau = np.fromfile(tau_name, dtype=np.float32)
eta = np.fromfile(eta_name, dtype=np.float32)
r_u = np.fromfile(r_u_name, dtype=np.float32)
r_v = np.fromfile(r_v_name, dtype=np.float32)
r_uv = np.fromfile(r_uv_name, dtype=np.float32) 
tau_int = np.linspace(np.min(tau[tau>0]),np.max(tau[tau>0]),256)
tau_int = np.r_[-np.flip(tau_int),0,tau_int]
eta_int = np.linspace(np.min(eta[eta>0]),np.max(eta[eta>0]),256)
eta_int = np.r_[-np.flip(eta_int),0,eta_int]
tau_int, eta_int = np.meshgrid(tau_int,eta_int)          
_,_,ru_i = sc.autocorr_interp_sq(r_u, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,rv_i = sc.autocorr_interp_sq(r_v, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,ruv_i = sc.autocorr_interp_sq(r_uv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)          
ru_i[np.isnan(ru_i)]=0
rv_i[np.isnan(rv_i)]=0
ruv_i[np.isnan(ruv_i)]=0   
ru_i[tau_int<0]=np.flip(ru_i[tau_int>0])
rv_i[tau_int<0]=np.flip(rv_i[tau_int>0])           
ru_i[eta_int<0]=np.flip(ru_i[eta_int>0])
rv_i[eta_int<0]=np.flip(rv_i[eta_int>0])



        
# In[]    
s=9000
start_scan = np.max([9000,s])
stop_scan = 11000
dscan = 10

start = np.arange(start_scan, stop_scan, dscan)
end = np.arange(start_scan+dscan, stop_scan+dscan, dscan) 

root = tkint.Tk()
file_in_path_r = tkint.filedialog.askdirectory(parent=root,title='Choose a autocorr. Input dir')
root.destroy()

onlyfiles_r = [f for f in listdir(file_in_path_r) if isfile(join(file_in_path_r, f))]


for current_scan in range(start_scan,stop_scan):
                          
    valid_name = 'valid_scan_' + str(int(current_scan))
    tau_name = 'tau_scan_' + str(int(current_scan))
    eta_name = 'eta_scan_' + str(int(current_scan))
    r_u_name = 'r_u_scan_' + str(int(current_scan))
    r_v_name = 'r_v_scan_' + str(int(current_scan))
    r_uv_name = 'r_uv_scan_' + str(int(current_scan))
    if tau_name in onlyfiles_r:
        print(current_scan)
        tau = np.fromfile(tau_name, dtype=np.float32)
        eta = np.fromfile(eta_name, dtype=np.float32)
        r_u = np.fromfile(r_u_name, dtype=np.float32)
        r_v = np.fromfile(r_v_name, dtype=np.float32)
        r_uv = np.fromfile(r_uv_name, dtype=np.float32)       
        r_u[tau<0] = np.flip(r_u[tau>0])
        r_v[tau<0] = np.flip(r_v[tau>0])           
        tau_int = np.linspace(np.min(tau[tau>0]),np.max(tau[tau>0]),256)
        tau_int = np.r_[-np.flip(tau_int),0,tau_int]
        eta_int = np.linspace(np.min(eta[eta>0]),np.max(eta[eta>0]),256)
        eta_int = np.r_[-np.flip(eta_int),0,eta_int]
        tau_int, eta_int = np.meshgrid(tau_int,eta_int)          
        _,_,ru_i = sc.autocorr_interp_sq(r_u, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
        _,_,rv_i = sc.autocorr_interp_sq(r_v, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
        _,_,ruv_i = sc.autocorr_interp_sq(r_uv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)          
        ru_i[np.isnan(ru_i)]=0
        rv_i[np.isnan(rv_i)]=0
        ruv_i[np.isnan(ruv_i)]=0   
        ru_i[tau_int<0]=np.flip(ru_i[tau_int>0])
        rv_i[tau_int<0]=np.flip(rv_i[tau_int>0])           
        ru_i[eta_int<0]=np.flip(ru_i[eta_int>0])
        rv_i[eta_int<0]=np.flip(rv_i[eta_int>0])
        ku_r,kv_r,Su_r,Sv_r,Suv_r = sc.spectra_fft((tau_int,eta_int),ru_i,rv_i,ruv_i,K=0)            
        ku_r_name = 'ku_r_name'+ str(int(current_scan))
        kv_r_name = 'kv_r_name'+ str(int(current_scan))
        Su_r_name = 'Su_r_name'+ str(int(current_scan))
        Sv_r_name = 'Sv_r_name'+ str(int(current_scan))
        Suv_r_name = 'Suv_r_name'+ str(int(current_scan))
        (ku_r.flatten()).astype(np.float32).tofile(ku_r_name)
        (kv_r.flatten()).astype(np.float32).tofile(kv_r_name)
        (np.real(Su_r).flatten()).astype(np.float32).tofile(Su_r_name)
        (np.real(Sv_r).flatten()).astype(np.float32).tofile(Sv_r_name)
        (np.real(Suv_r).flatten()).astype(np.float32).tofile(Suv_r_name)
        (np.imag(Suv_r).flatten()).astype(np.float32).tofile(Suv_r_name+'imag') 


# In[]
onlyfiles_r = [f for f in listdir(file_in_path_r) if isfile(join(file_in_path_r, f))]
Shr1D_ave = []
k1r1D_ave = []
Su_ave = []
Sv_ave = []
for current_scan in range(start_scan,stop_scan):
    print(current_scan)
    ku_r_name = 'ku_r_name'+ str(int(current_scan))
    kv_r_name = 'kv_r_name'+ str(int(current_scan))
    Su_r_name = 'Su_r_name'+ str(int(current_scan))
    Sv_r_name = 'Sv_r_name'+ str(int(current_scan))
    Suv_r_name = 'Suv_r_name'+ str(int(current_scan))
    if ku_r_name in onlyfiles_r:         
        
        k_u_r = np.fromfile(join(file_in_path_r,ku_r_name), dtype=np.float32)
        k_v_r = np.fromfile(join(file_in_path_r,kv_r_name), dtype=np.float32)
        S_u_r = np.fromfile(join(file_in_path_r,Su_r_name), dtype=np.float32)
        S_v_r = np.fromfile(join(file_in_path_r,Sv_r_name), dtype=np.float32)
        S_uv_r= np.fromfile(join(file_in_path_r,Suv_r_name), dtype=np.float32)
        
        kur,kvr = np.meshgrid(k_u_r,k_v_r)
        S_u_r = np.reshape(S_u_r,kur.shape)
        S_v_r = np.reshape(S_v_r,kur.shape)
        S_uv_r = np.reshape(S_uv_r,kur.shape)
        
        Sur_ave=sc.spectra_average(S_u_r,(k_u_r, k_v_r),bins=30).S
        Svr_ave=sc.spectra_average(S_v_r,(k_u_r, k_v_r),bins=30).S
        
        Su_ave.append(Sur_ave)
        Sv_ave.append(Svr_ave)        
        Shr1D_ave.append(sp.integrate.simps(.5*(Sur_ave+Svr_ave),k_v_r,axis=0))
        k1r1D_ave.append(k_u_r)


# In[]
s=9000
start_scan = np.max([9000,s])
stop_scan = 11000
dscan = 10
ind_0 = (df_0.scan>=start_scan) & (df_0.scan<=stop_scan)
ind_1 = (df_1.scan>=start_scan) & (df_1.scan<=stop_scan)
df_0 = df_0.loc[ind_0]
df_1 = df_1.loc[ind_1]
U_out, V_out, grd, _ = wr.direct_wf_rec(df_0, df_1,
                                            tri, d,N_grid = 512)


U_out = []
V_out = []

with open('U_rec_300.pkl', 'rb') as V_t:
     grd,U = pickle.load(V_t)
U_out.append(U)

with open('U_rec_600.pkl', 'rb') as V_t:
     grd,U = pickle.load(V_t)
U_out.append(U)

with open('U_rec_900.pkl', 'rb') as V_t:
     grd,U = pickle.load(V_t)
U_out.append(U)

with open('V_rec_300.pkl', 'rb') as V_t:
     grd,V = pickle.load(V_t)
V_out.append(V)

with open('V_rec_600.pkl', 'rb') as V_t:
     grd,V = pickle.load(V_t)
V_out.append(V)

with open('V_rec_900.pkl', 'rb') as V_t:
     grd,V = pickle.load(V_t)
V_out.append(V)

U_out = [u for i in U_out for u in i]
V_out = [u for i in V_out for u in i]

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
levels=np.linspace(10,21,20)
_,_,U = sc.shrink(grd,U_out[0])
grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[0])
grd_shr = (grd_shr_x,grd_shr_y)

#im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=levels,cmap='rainbow')
im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
divider = make_axes_locatable(ax)
cb = fig.colorbar(im)
cb.ax.set_ylabel("Wind speed [m/s]")
#Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
ax.set_xlabel(r'West-East [m]', fontsize=12, weight='bold')
ax.set_ylabel(r'North-South [m]', fontsize=12, weight='bold') 


def update(scan_n):   
    ax.cla()   
    
    from datetime import datetime, timedelta
    
    date = datetime(1904,1,1) # January 1st, 1904 at midnight   
    delta = timedelta(seconds = df_0.loc[df_0.scan==scan_n+9000].stop_time.max())   
    newdate = date + delta
    
    ax.set_title(str(newdate))  
    
    _,_,U = sc.shrink(grd,U_out[scan_n])
    grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[scan_n])
    grd_shr = (grd_shr_x,grd_shr_y)
    im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
    
    #check if there is more than one axes
    if len(fig.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts = fig.axes[-1].get_position().get_points()
        # and its label
        label = fig.axes[-1].get_ylabel()
        # and then remove the axes
        fig.axes[-1].remove()
        # then we draw a new axes a the extents of the old one
        divider = make_axes_locatable(ax)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.set_ylabel("Wind speed [m/s]")
        ax.set_xlabel(r'West-East [m]', fontsize=12, weight='bold')
        ax.set_ylabel(r'North-South [m]', fontsize=12, weight='bold') 
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        fig.colorbar(im)
    
    return ax

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(650, 850), interval=500)
    #if len(sys.argv) > 1 and sys.argv[1] == 'save':
    anim.save('U_night.gif', dpi=80, writer='imagemagick')


