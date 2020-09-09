# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:32:52 2020

@author: lalc
"""
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import scipy as sp
from scipy import ndimage
import pickle
import tkinter as tkint
import tkinter.filedialog
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import datetime
from datetime import timedelta
import h5py


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
import spectralfitting.spectralfitting as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from scipy.spatial import Delaunay

# In[Functions]
def fm2(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def synch_df(df0,df1,dtscan=45):
    date = datetime.datetime(1904, 1, 1, 0, 0) 
    s0 = df0.scan.unique()
    s1 = df1.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    dt = [(t0[i]-t1[j]).total_seconds() for i in range(len(t0)) for j in range(len(t1))]
    s = np.array([[s_0,s_1] for s_0 in s0 for s_1 in s1])
    t = np.array([[t_0,t_1] for t_0 in t0 for t_1 in t1])
    ind_synch = np.abs(dt)<dtscan
    if np.sum(ind_synch)==0:
       sync = []
       time = []
       off0 = s0[-1]
       off1 = s1[-1]
    else:
       sync = s[ind_synch,:]
       time = t[ind_synch,:]
       # Complete scans   
       n_scan_0 = np.array([(df0.scan==s).sum() for s in sync[:,0]]) 
       n_scan_1 = np.array([(df1.scan==s).sum() for s in sync[:,1]])
       sync = sync[(n_scan_0==45) & (n_scan_1==45),:]
       time = time[(n_scan_0==45) & (n_scan_1==45),:]
       off0 = sync[-1,0]+1
       off1 = sync[-1,1]+1 
    return (sync,time, off0*45,off1*45)

import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)

def mov_con(x,N):
    x = np.r_[np.ones(N)*x[0],x,np.ones(N)*x[-1]]
    return np.convolve(x, np.ones((N,))/N,mode='same')[N:-N]

def datetimeDF(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

def field_rot(x, y, U, V, gamma = None, grid = [], tri_calc = False, tri_del = []):    
    if gamma is None:
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
    U = vel[0,:]
    V = vel[1,:]
    mask = ~np.isnan(U)
    mask_int = []   
    if not grid:
            grid = np.meshgrid(x,y)       
    xtrans = (x[0]+x[-1])/2
    ytrans = (y[0]+y[-1])/2
    T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
    T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
    T = np.dot(np.dot(T1,R),T2)
    Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
    Xx = np.dot(T,Xx)   
    if tri_calc:        
        tri_del = Delaunay(np.c_[Xx[0,:][mask],Xx[1,:][mask]])
        mask_int = ~(tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()]) == -1)       
    return (U, V, mask, mask_int, tri_del, gamma, Xx)

def U_rot(grid, U, V, gamma = None, tri_calc = True, tri_del = [], mask_int = [], mask = []):
    x = grid[0][0,:]
    y = grid[1][:,0]
    if tri_calc:
        U, V, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, gamma = gamma, grid = grid,
                                                         tri_calc = tri_calc, tri_del = tri_del)
    else:
        U, V, mask_c, _, tri_del, gamma, Xx = field_rot(x, y, U, V, gamma = gamma, grid = grid,
                                                         tri_calc = tri_calc, tri_del = tri_del) 
    if tri_del.points.shape[0] == np.sum(mask_c):   
        if len(U)>0:
            U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
            U[~mask_int] = np.nan
            U = np.reshape(U, grid[0].shape)
        if len(V)>0:
            V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
            V[~mask_int] = np.nan
            V = np.reshape(V, grid[0].shape)
    else:
        if len(U)>0:
            ind_nan = np.isnan(U[mask])
            vertex_nan = np.array(range(0,len(U[mask])))[ind_nan]
            simplices_n = np.array(range(0,len(tri_del.simplices)))
            simplices_n = np.c_[simplices_n, simplices_n, simplices_n]
            ind_simp = np.isin(tri_del.simplices.flatten(), vertex_nan)
            ind_simp = np.unique(simplices_n.flatten()[ind_simp])
            ind_simp = ~np.isin(np.array(range(0,len(tri_del.simplices))), ind_simp)
            ind_simp = ind_simp & ~(circleratios(tri_del)<.05)
            simp_grid = -np.ones(grid[1].flatten().shape)
            simp_grid[mask_int] = tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
            mask_simp = np.isin(simp_grid,np.array(range(0,len(tri_del.simplices)))[ind_simp])     
            U_tesa = np.zeros(U.shape)*np.nan
            U_tesa[mask] = sp.interpolate.NearestNDInterpolator(tri_del.points[~ind_nan,:],U[mask][~ind_nan])(Xx.T[mask,0],Xx.T[mask,1])
            U[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del,U_tesa[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_simp])
            U[~mask_simp] = np.nan
            U = np.reshape(U, grid[0].shape)  
        if len(V)>0:   
            ind_nan = np.isnan(V[mask])
            vertex_nan = np.array(range(0,len(V[mask])))[ind_nan]
            simplices_n = np.array(range(0,len(tri_del.simplices)))
            simplices_n = np.c_[simplices_n, simplices_n, simplices_n]
            ind_simp = np.isin(tri_del.simplices.flatten(), vertex_nan)
            ind_simp = np.unique(simplices_n.flatten()[ind_simp])
            ind_simp = ~np.isin(np.array(range(0,len(tri_del.simplices))), ind_simp)
            ind_simp = ind_simp & ~(circleratios(tri_del)<.05)
            simp_grid = -np.ones(grid[1].flatten().shape)
            simp_grid[mask_int] = tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
            mask_simp = np.isin(simp_grid,np.array(range(0,len(tri_del.simplices)))[ind_simp])     
            V_tesa = np.zeros(V.shape)*np.nan
            V_tesa[mask] = sp.interpolate.NearestNDInterpolator(tri_del.points[~ind_nan,:],V[mask][~ind_nan])(Xx.T[mask,0],Xx.T[mask,1])
            V[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del,V_tesa[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_simp])
            V[~mask_simp] = np.nan
            V = np.reshape(V, grid[0].shape)
    return (U, V)

def circleratios(tri):       
    pts = tri.points
    tri_pts = pts[tri.simplices.copy()]
    # Computes the 3 side lengths
    a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
    c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
    a = np.sqrt(a[:, 0]**2 + a[:, 1]**2)
    b = np.sqrt(b[:, 0]**2 + b[:, 1]**2)
    c = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
    # circumcircle and incircle radii
    s = (a+b+c)*0.5
    prod = s*(a+b-s)*(a+c-s)*(b+c-s)
    # We have to deal with flat triangles with infinite circum_radius
    bool_flat = (prod == 0.)
    if np.any(bool_flat):
        # Pathologic flow
        ntri = tri_pts.shape[0]
        circum_radius = np.empty(ntri, dtype=np.float64)
        circum_radius[bool_flat] = np.inf
        abc = a*b*c
        circum_radius[~bool_flat] = abc[~bool_flat] / (
            4.0*np.sqrt(prod[~bool_flat]))
    else:
        # Normal optimized flow
        circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
    in_radius = (a*b*c) / (4.0*circum_radius*s)
    circle_ratio = in_radius/circum_radius
    #mask = self._triangulation.mask
    #if mask is None:
    return circle_ratio
    
def interp(grdu,ulist,vlist,N=100):
    chunk = ulist[0].shape[1]
    Ulist = [ulist[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    Vlist = [vlist[i] for i in range(len(vlist)) if np.sum(~np.isnan(ulist[i])) > N]
    U_arr_pod = np.vstack([Ulist[i] for i in range(len(Ulist))])
    V_arr_pod = np.vstack([Vlist[i] for i in range(len(Vlist))])      
    umean = np.nanmean(U_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0)
    maskpod = ~np.isnan(np.nanmean(U_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0))    
    ut = np.vstack([Ulist[i][maskpod] for i in range(len(Ulist))])
    vt = np.vstack([Vlist[i][maskpod] for i in range(len(Ulist))])
    x = np.vstack([grdu[0][maskpod] for i in range(len(Ulist))])
    y = np.vstack([grdu[1][maskpod] for i in range(len(Ulist))])
    t = np.vstack([np.ones(np.sum(maskpod))*i*45 for i in range(len(Ulist))])   
    ind = ~np.isnan(ut)    
    from sklearn.neighbors import KDTree
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors =  26,weights='distance',algorithm='auto', leaf_size=30,n_jobs=1)
    neigh.fit(np.c_[x[ind], y[ind], t[ind]], ut[ind])
    ut[~ind] = neigh.predict(np.c_[x[~ind], y[~ind], t[~ind]])
    neigh = KNeighborsRegressor(n_neighbors =  26,weights='distance',algorithm='auto', leaf_size=30,n_jobs=1)
    neigh.fit(np.c_[x[ind], y[ind], t[ind]], vt[ind])
    vt[~ind] = neigh.predict(np.c_[x[~ind], y[~ind], t[~ind]])    
    return (ut,vt,maskpod)

def filterfft(vorti, mask, sigma=20):
    vort = vorti.copy()
    vort[np.isnan(vort)] = np.nanmean(vort)   
    input_ = np.fft.fft2(vort)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.real(np.fft.ifft2(result))
    result[mask] = np.nan
    return result


# In[Wind field reconstruction]
def wfrec(db0,db1,dy,hms,loc = [], merge_db=False, filesave = False, filedir = [],N=128):
    # Location
    if len(loc)==0:
        loc0 = np.array([0,6322832.3])#-d
        loc1 = np.array([0,6327082.4])# d
    else:
        loc0 = loc[0]
        loc1 = loc[1]
    d = loc1-loc0
    # Time
    date = datetime.datetime(1904, 1, 1, 0, 0) 
    tpick = dy[0]+hms[0]#''.join(pick_rnd1.index.values[k])
    t = pd.to_datetime(tpick)
    t_init = pd.to_datetime(dy[0]+hms[0])#t - timedelta(minutes=minutes)
    t_end = pd.to_datetime(dy[1]+hms[1])#t + timedelta(minutes=minutes)
    hms_i = hms[0]#t_init.strftime("%H%M%S")
    hms_e = hms[1]#t_end.strftime("%H%M%S")
    dy0 = dy[0]#t_init.strftime("%Y%m%d")
    dy1 = dy[1]#t_end.strftime("%Y%m%d")
    
    ######################################
    t_0 = datetime.datetime(int(dy0[:4]), int(dy0[4:6]), int(dy0[6:]))
    t_0 = t_0+timedelta(seconds = int(hms_i[4:]))
    t_0 = t_0+timedelta(minutes = int(hms_i[2:4]))
    t_0 = t_0+timedelta(hours = int(hms_i[:2]))
    t_0 = str((t_0-date).total_seconds())
    #
    t_1 = datetime.datetime(int(dy1[:4]), int(dy1[4:6]), int(dy1[6:]))
    t_1 = t_1+timedelta(seconds = int(hms_e[4:]))
    t_1 = t_1+timedelta(minutes = int(hms_e[2:4]))
    t_1 = t_1+timedelta(hours = int(hms_e[:2]))
    t_1= str((t_1-date).total_seconds())
    
    ######################################
    ### labels
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
    
    # labels for query
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
    # Reconstruction of chuncks of 1 hour scans?
    U_out_c, V_out_c, su_c = [], [], [] 
    query_fil_0 = selec_fil+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
    query_raw_0 = selec_raw+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
    query_fil_1 = selec_fil+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
    query_raw_1 = selec_raw+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1   
    # First database loading
    df_0 = pd.read_sql_query(query_fil_0, db0)
    if merge_db:
        df = pd.read_sql_query(query_raw_0, db0)
        # Retrieving good CNR values from un-filtered scans
        for i in range(198):
            ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
            df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
        df = None
        df_0.drop(columns = labels_CNR,inplace=True)
    
    df_0.columns = labels_short
    
    df_1 = pd.read_sql_query(query_fil_1, db1)
    if merge_db:
        df = pd.read_sql_query(query_raw_1, db1)
        for i in range(198):
            ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
            df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
        df = None   
    df_1.columns = labels_short
    #################Reconstruction#####################
    switch = 0
    chunk = N
    U_out, V_out, su = [], [], [] 
    U_list = []
    V_list = [] 
    s_syn,t_scan,_,_ = synch_df(df_0,df_1,dtscan=45/2)
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
        t_1h = pd.date_range(tmin, tmax,freq='10T')#.strftime('%Y%m%d%H%M)
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
                u, v, grdu, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk)  
                su.append(s)
                U_out.append(u)
                V_out.append(v)
    #############################################################################################            
                tri_calc = True
                x = grdu[0][0,:]
                y = grdu[1][:,0]
                U_arr = np.vstack([u[i] for i in range(len(u))])
                V_arr = np.vstack([v[i] for i in range(len(v))])
                scan = s_syn[:,0]
                U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
                V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
                ur, vr, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 
                for j in range(len(u)):
                    print(j)
                    U, V = U_rot(grdu, u[j], v[j], gamma = gamma, tri_calc = False, 
                                              tri_del = tri_del, mask_int = mask_int, mask = mask) 
                    U_list.append(U)
                    V_list.append(V)
    #############################################################################################  
    U_out = [item for sublist in U_out for item in sublist]
    V_out = [item for sublist in V_out for item in sublist]
    su    = [item for sublist in su for item in sublist]
    inds = np.isin(s_syn[:,0],su)
    print(inds.shape,np.sum(inds),len(su))
    t_scan = t_scan[inds,:]
    ut,vt,maskpod = interp(grdu,U_list,V_list,N=100)
    if filesave:
        filename = filedir+tpick+'.pkl'
        joblib.dump((ut,vt,maskpod,U_list,V_list,grdu), filename)  
    return (ut, vt, maskpod, U_list, V_list, U_out, V_out, grdu, t_scan, tri)

# In[Wind field advection]
def expand_field_point_time(grd, U_list, V_list, point, n = 12, r = 1000,
                            tree=True, treetype = 'Kn', grid_i = [], nx = 100, alpha = [0, 2*np.pi],
                            beta = 1, al = 10, t_scale = 1, probe='circle', Luy = [], time_scan = [],
                            t_int = [], interp_t = False, tri_dom = [], part = 10):
    #Output arrays
    xfin, yfin, tfin, time, times, Ufin, Vfin = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    # Input time and coordinates
    date = datetime.datetime(1904, 1, 1) 
    dx = grd[0][0,1]-grd[0][0,0]
    dy = grd[1][1,0]-grd[1][0,0]
    dr = np.sqrt(dx**2+dy**2)
    print('means')
    # Local mean wind speed  parameters
    ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2) 
    Umeans = [filterfft(u,np.isnan(u),sigma=Luy/ds) for u in U_list]
    
    print(len(Umeans))
    
    Vmeans = [filterfft(u,np.isnan(u),sigma=Luy/ds) for u in V_list]   
    
    if interp_t:
        print('interpolation in time')
        xp = grid_i[0].flatten()
        yp = grid_i[1].flatten()
        points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]        
        # time steps from t_int, first time bins
        t = np.max(time_scan, axis=1)-date
        #t = np.array([(ti).seconds+ (ti).microseconds/1000000 for ti in t])
        t = np.array([(ti).total_seconds() for ti in t])
        t_int_s = t_int-date
        t_int_s = np.array([(ti).total_seconds() for ti in t_int_s]) 
        t_init = t[0]
        t_int_s = t_int_s-t_init
        t = t-t_init
        t_int_s = t_int_s[t_int_s<=t[-1]] 
        print(t_init)
        dt = np.diff(t)
        xg = np.dot(np.array(xp)[:,None],np.ones(t_int_s.shape)[None,:])
        yg = np.dot(np.array(yp)[:,None],np.ones(t_int_s.shape)[None,:])
        tg = np.dot(np.ones(np.array(xp).shape)[:,None], t_int_s[None,:])        
        for i in range(len(U_list)-1):            
            #Advecting the current wind field  
            mask = np.isnan(U_list[i])
            x0 = grd[0][~mask] + Umeans[i][~mask]*dt[i]
            y0 = grd[1][~mask] + Vmeans[i][~mask]*dt[i]         
            ind_tri = tri_dom.find_simplex(np.c_[x0,y0])==-1          
            for j,p in enumerate(points):
                if j == 0:           
                    indy = (y0<p[1]+al*dr) & (y0>p[1]-al*dr)
                else:
                    indy = indy | (y0<p[1]+al*dr) & (y0>p[1]-al*dr)
            ind_tri = ind_tri & indy
            x0 = grd[0][~mask][ind_tri]
            y0 = grd[1][~mask][ind_tri]   
            x0_min = np.min(x0)
            U_min = Umeans[i][~mask][ind_tri][np.argmin(x0)]
            dtmax = np.array([])
            for p in points:
                dtmax = np.r_[dtmax, (p[0]-x0_min)/U_min]
            tmax = np.max(dtmax)+t[i]  
            if np.sum(t>=tmax)>0:
                tmax = t[t>=tmax][0]
            ind_t = (t_int_s>=t[i]) & (t_int_s<=tmax)
            print(np.sum(ind_t), tmax, t[i])
            #Check the end of the current scan within the domain           
            x0 = np.dot(grd[0][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])
            y0 = np.dot(grd[1][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:]) 
            xnew = np.dot(Umeans[i][~mask][ind_tri][:,None], (t_int_s[ind_t]-t[i])[None,:]) + x0
            ynew = np.dot(Vmeans[i][~mask][ind_tri][:,None], (t_int_s[ind_t]-t[i])[None,:])+ y0                              
            for j,p in enumerate(points):
                T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
                xy = np.array(np.c_[xnew.flatten(),ynew.flatten(),np.ones(len(xnew.flatten()))]).T
                xy = np.dot(T1,xy).T
                if probe == 'square':
                    if j == 0:
                        ind = (xy[:,0] <= beta*dr) & (xy[:,1] <= beta*dr)
                    else:
                        ind = ind | (xy[:,0] <= beta*dr) & (xy[:,1] <= beta*dr)
                if probe == 'circle':
                    if j == 0:
                        ind = (xy[:,0]**2 + xy[:,1]**2) <= (beta*dr)**2
                    else:
                        ind = ind | ((xy[:,0]**2 + xy[:,1]**2) <= (beta*dr)**2)              
            ind = np.reshape(ind, xnew.shape)
            xnew = xnew[ind]
            ynew = ynew[ind]                           
            tnew = np.dot(np.ones(Umeans[i][~mask][ind_tri].shape)[:,None], (t_int_s[ind_t])[None,:])[ind]
            Unew = np.dot(U_list[i][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])[ind]
            Vnew = np.dot(V_list[i][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])[ind]
            print(ind.shape,tnew.shape, Unew.shape, Vnew.shape, xnew.shape, ynew.shape)                
            xfin = np.r_[xfin, xnew]
            yfin = np.r_[yfin, ynew]
            tfin = np.r_[tfin, tnew]
            Ufin = np.r_[Ufin, Unew]
            Vfin = np.r_[Vfin, Vnew]
        print('tree')
        if tree:  
            # xg, yg, tg = np.meshgrid(xp,yp,t_int_s)
            Uex = np.zeros(xg.shape)*np.nan
            Vex = np.zeros(xg.shape)*np.nan
            neigh = KNeighborsRegressor(n_neighbors =  26, weights='distance',algorithm='auto', leaf_size=30,n_jobs=1)
            X = np.c_[xfin, yfin, tfin*t_scale]
            Xg = np.c_[xg.flatten(), yg.flatten(), tg.flatten()*t_scale]
            print('fit U')
            neigh.fit(X, Ufin)                
            print('Predict U')
            Uex = neigh.predict(Xg) 
            print('fit V')
            neigh.fit(X, Vfin)
            print('Predict V')
            Vex = neigh.predict(Xg)           
            tout = tg.flatten()
            tout = tout + t_init
            tout = np.array([timedelta(seconds=ti) for ti in tout])
            tout = date+tout
            tout = np.reshape(tout,tg.shape)   
        else:
            from scipy.interpolate import Rbf
            t_part = np.linspace(np.min(tfin), np.max(tfin)+1, part+1)
            # xg, yg, tg = np.meshgrid(xp,yp,t_int_s)
            Uex = np.ones(xg.shape)*np.nan
            Vex = np.ones(xg.shape)*np.nan
            for i in range(len(t_part)-1):
                print(i)
                indt_s = (tfin>=t_part[i]) & (tfin<=t_part[i+1])
                indt_g = (tg>=t_part[i]) & (tg<=t_part[i+1])
                rbfiu = Rbf(xfin[indt_s], yfin[indt_s], tfin[indt_s]*t_scale, Ufin[indt_s], function = 'cubic')
                rbfiv = Rbf(xfin[indt_s], yfin[indt_s], tfin[indt_s]*t_scale, Vfin[indt_s], function = 'cubic')
                print('Interpolation')
                Uex[indt_g] = rbfiu(xg[indt_g] , yg[indt_g] , tg[indt_g]*t_scale)
                Vex[indt_g] = rbfiv(xg[indt_g] , yg[indt_g] , tg[indt_g]*t_scale)   
            tout = tg.flatten()
            tout = tout + t_init
            tout = np.array([timedelta(seconds=ti) for ti in tout])
            tout = date+tout
            tout = np.reshape(tout,tg.shape) 
        return (xg, yg, tg, tout, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape), np.c_[xp,yp])  
    else:
        print('interpolation in space')
        # Time steps for spatial interpolation (one time, many spatial points)
        t = np.max(time_scan, axis=1)-date
        t = np.array([(ti).total_seconds() for ti in t])
        t_int = t_int-date
        t_int = t_int.total_seconds()
        t_init = t[0]
        t_int = t_int-t_init
        t = t-t_init

        # bin for interpolation
        ti = np.min(t[t>=t_int])
        te = np.max(t[t<=t_int])
        tbin = [ti, te]
        dt = np.diff(t)   
        print('probes') 
        chunk = 128
        Uarr = np.vstack([Umeans[i] for i in range(len(Umeans))])
        Umask = ~np.isnan(np.nanmean(Uarr.reshape((int(len(Uarr)/chunk),chunk,chunk)),axis=0))
        xmask = grd[0][Umask]
        ymask = grd[1][Umask]
        center = .5*np.r_[grd[0][0,0]+grd[0][0,-1], grd[1][0,0]+grd[1][-1,0]]  
        R = np.array([])
        for p in point:
            R = np.r_[R, np.sqrt(np.sum((p[:2]-center)**2))]
        p = point[np.argmax(R)]
        # alpha0 = np.arctan2((point[0][:2]-center)[1], (point[0][:2]-center)[0])
        # alpha1 = np.arctan2((point[1][:2]-center)[1], (point[1][:2]-center)[0])
        alpha = np.linspace(alpha[0],alpha[1], n+1)   
        xp = center[0]+np.max(R)*np.cos(alpha)
        yp = center[1]+np.max(R)*np.sin(alpha)
        xp = xp[(yp>np.nanmin(grd[1][Umask]))&(yp<np.nanmax(grd[1][Umask]))]
        yp = yp[(yp>np.nanmin(grd[1][Umask]))&(yp<np.nanmax(grd[1][Umask]))]
        tree_p = KDTree(np.c_[xmask, ymask])
        ri, _ = tree_p.query(np.c_[xp,yp], k = 1, return_distance=True)
        points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]
        print('advection to time t_int')    
        Ui = U_list[np.nonzero(t==tbin[0])[0][0]]
        Ue = U_list[np.nonzero(t==tbin[1])[0][0]]
        mask = np.isnan(Ui)
        Ui = Ui[~mask]
        xi = grd[0][~mask]
        yi = grd[1][~mask]
        mask = np.isnan(Ue)
        Ue = Ue[~mask]
        xe = grd[0][~mask]
        ye = grd[1][~mask]
        #
        Vi = V_list[np.nonzero(t==tbin[0])[0][0]]
        Ve = V_list[np.nonzero(t==tbin[1])[0][0]]
        mask = np.isnan(Vi)
        Vi = Vi[~mask]
        mask = np.isnan(Ve)
        Ve = Ve[~mask]
        #
        ti = np.ones(Ui.shape)*tbin[0]
        te = np.ones(Ue.shape)*tbin[1]
        #
        xfin = np.r_[xfin, xi, xe]
        yfin = np.r_[yfin, yi, ye]
        time = np.r_[time, ti, te]
        times = np.r_[times, ti, te]+t_init
        
        Ufin = np.r_[Ufin, Ui, Ue]
        Vfin = np.r_[Vfin, Vi, Ve]                 
        for i in range(len(t[t<tbin[1]])):

            print(i,dt[i])

            mask = np.isnan(U_list[i])
            xnew = grd[0][~mask]+Umeans[i][~mask]*dt[i]
            ynew = grd[1][~mask]+Vmeans[i][~mask]*dt[i]
            #sampling
            for j,p in enumerate(points):
                T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
                xy = np.array(np.c_[xnew,ynew,np.ones(len(xnew))]).T
                xy = np.dot(T1,xy).T
                if probe == 'square':
                    if j == 0:
                        ind = (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
                    else:
                        ind = ind | (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
                if probe == 'circle':
                    if j == 0:
                          ind = (xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2
                    else:
                          ind = ind | ((xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2) 
            
            xi = xnew[ind]+Umeans[i][~mask][ind]*(t_int-t[i+1])#(tbin[0]-t[i+1])
            yi = ynew[ind]+Vmeans[i][~mask][ind]*(t_int-t[i+1])#(tbin[0]-t[i+1])
            Ui = U_list[i][~mask][ind]
            Vi = V_list[i][~mask][ind]
            ti = np.ones(xi.shape)*tbin[0]
            tis = np.ones(Ui.shape)*(t[i]+t_init)
            
            xfin = np.r_[xfin, xi]#, xe]
            yfin = np.r_[yfin, yi]#, ye]
            time = np.r_[time, ti]#, te]
            times = np.r_[times, tis]#, te]
            Ufin = np.r_[Ufin, Ui]#, Ue]
            Vfin = np.r_[Vfin, Vi]#, Ve]        
        if len(grid_i)==0:
            xg = np.arange(np.nanmin(xfin), np.nanmax(xfin)+dx, dx)
            yg = np.arange(np.nanmin(yfin), np.nanmax(yfin)+dy, dy) 
            xg, yg = np.meshgrid(xg, yg)
        else:
            xg, yg = grid_i[0], grid_i[1]     
        tg = t_int*np.ones(xg.shape)   
        print('envelope')
        x_bin = np.linspace(np.nanmin(xfin), np.nanmax(xfin), nx)  
        ########################## 
        print(xg.shape,xfin.shape, yg.shape,yfin.shape)        
        ##########################
        indyg = np.zeros(xg.shape, dtype=bool)    
        for i in range(len(x_bin)-1):
            indx = (xfin >= x_bin[i]) & (xfin < x_bin[i+1])
            print(np.sum(indx))
            indxg = (xg >= x_bin[i]) & (xg < x_bin[i+1])
            indyg = (((yg > np.nanmin(yfin[indx])) & (yg < np.nanmax(yfin[indx]))) & indxg) | indyg       
        print('points')
        X = np.c_[xfin, yfin, time]
        Xg = np.c_[xg[indyg],yg[indyg],tg[indyg]]
        # plt.figure()
        # plt.scatter(xg[indyg],yg[indyg])
        
        print('tree')
        if tree:  
            Uex = np.zeros(xg.shape)*np.nan
            Vex = np.zeros(xg.shape)*np.nan
            tex = np.zeros(xg.shape)*np.nan
            h = 3*dr
            func = lambda x: (1/np.sqrt(2*np.pi)/h)*np.exp(-x**2/(2*h**2))
            neigh = KNeighborsRegressor(n_neighbors =  26, weights=func,algorithm='auto', leaf_size=30,n_jobs=1)
            print('fit U')
            neigh.fit(X, Ufin)                
            print('Predict U')
            Uex[indyg] = neigh.predict(Xg) 
            print('fit V')
            neigh.fit(X, Vfin)
            print('Predict V')
            Vex[indyg] = neigh.predict(Xg)
            print('fit t')
            neigh.fit(X, times)
            print('Predict t')
            tex[indyg] = neigh.predict(Xg)
        else:
            print('triangulation')
            tri_exp = Delaunay(X)
            print('Interpolation')
            Uex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Ufin)(Xg)
            Vex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Vfin)(Xg)               
        return (xg, yg, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape),np.reshape(tex,xg.shape), np.c_[xp,yp])
# In[spec and auto average]
def spec_auto_ave(spec,auto,dy,hms,N=256):    
    k1grid = spec.loc[dy,:].k1.values
    k2grid = spec.loc[dy,:].k1.values
    k1grid = np.linspace(np.min(k1grid),np.max(k1grid),N)
    k2grid = np.linspace(np.min(k2grid),np.max(k2grid),N)
    k1grid, k2grid = np.meshgrid(k1grid, k2grid)    
    # lags
    taugrid = auto.loc[dy,:].tau.values
    etagrid = auto.loc[dy,:].eta.values
    taugrid = np.linspace(np.min(taugrid),np.max(taugrid),N)
    etagrid = np.linspace(np.min(etagrid),np.max(etagrid),N)
    taugrid, etagrid = np.meshgrid(taugrid, etagrid)    
    Suul = []
    Svvl = []
    Shl  = []    
    ruul = []
    rvvl = []
    ruvl = []    
    for h in hms:
        print(h)
        k1 = (spec.loc[dy,h].k1.values).reshape((N,N))
        k2 = (spec.loc[dy,h].k2.values).reshape((N,N))
        Suu = (spec.loc[dy,h].Suu.values).reshape((N,N))
        Svv = (spec.loc[dy,h].Svv.values).reshape((N,N))
        Sh = (spec.loc[dy,h].Sh.values).reshape((N,N))
        tau = auto.loc[dy,h].tau.values
        eta = auto.loc[dy,h].eta.values
        ruu = auto.loc[dy,h].ru.values
        rvv = auto.loc[dy,h].rv.values
        ruv = auto.loc[dy,h].ruv.values
        _,_,Suui = sc.autocorr_interp_sq(Suu, k1, k2, N = N, tau_lin = k1grid, eta_lin = k2grid)
        _,_,Svvi = sc.autocorr_interp_sq(Svv, k1, k2, N = N, tau_lin = k1grid, eta_lin = k2grid)
        _,_,Shi = sc.autocorr_interp_sq(Sh, k1grid, k2grid, N = N, tau_lin = k1grid, eta_lin = k2grid)   
        _,_,ruui = sc.autocorr_interp_sq(ruu, tau, eta, N = N, tau_lin = taugrid, eta_lin = etagrid)
        _,_,rvvi = sc.autocorr_interp_sq(rvv, tau, eta, N = N, tau_lin = taugrid, eta_lin = etagrid)
        _,_,ruvi = sc.autocorr_interp_sq(ruv, tau, eta, N = N, tau_lin = taugrid, eta_lin = etagrid)  
        Suul.append(Suui.T)
        Svvl.append(Svvi.T)
        Shl.append(Shi.T)        
        ruul.append(ruui.T)
        rvvl.append(rvvi.T)
        ruvl.append(ruvi.T)          
    chunk = N
    Suu_arr = np.vstack([Suul[i] for i in range(len(Suul))])
    Svv_arr = np.vstack([Svvl[i] for i in range(len(Svvl))])
    Sh_arr = np.vstack([Shl[i] for i in range(len(Shl))])
    Suum =  np.nanmean(Suu_arr.reshape((int(len(Suu_arr)/chunk),chunk,chunk)),axis=0)
    Svvm =  np.nanmean(Svv_arr.reshape((int(len(Svv_arr)/chunk),chunk,chunk)),axis=0)
    Shm =  np.nanmean(Sh_arr.reshape((int(len(Sh_arr)/chunk),chunk,chunk)),axis=0)    
    
    ruu_arr = np.vstack([ruul[i] for i in range(len(ruul))])
    rvv_arr = np.vstack([rvvl[i] for i in range(len(rvvl))])
    ruv_arr = np.vstack([ruvl[i] for i in range(len(ruvl))])
    ruum =  np.nanmean(ruu_arr.reshape((int(len(ruu_arr)/chunk),chunk,chunk)),axis=0)
    rvvm =  np.nanmean(rvv_arr.reshape((int(len(rvv_arr)/chunk),chunk,chunk)),axis=0)
    ruvm=  np.nanmean(ruv_arr.reshape((int(len(ruv_arr)/chunk),chunk,chunk)),axis=0)     
    return (k1grid, k2grid,Suum,Svvm,Shm,taugrid, etagrid,ruum,rvvm,ruvm)
# In[1D Spectra for time series]
def spectra1D_t_series(U,dt):
    m, n = U.shape
    N = 2**np.ceil(np.log2(m)).astype('int')
    k = 2*np.pi*np.fft.fftshift((np.fft.fftfreq(N, d=dt)))/np.nanmean(U,axis=0)[0]
    U = U-np.nanmean(U,axis=0)
    fs = 1/dt
    from itertools import combinations
    fft = np.zeros((N,n))+1j*np.zeros((N,n))
    index = [[a,a] for a in list(np.arange(n))]+[list(p) for p in combinations(list(np.arange(n)),2)]
    Phi = np.zeros((len(index),N))
    for i in range(n):
        fft[:,i] = np.fft.fftshift(np.fft.fft(U[:,i],N))
    for i, ind in enumerate(index):
        print(U.shape,fft.shape,n,m,N,len(index))
        print(fft.shape,Phi.shape)
        Phi[i,:] = np.real((1/(2*np.pi*N*fs))*fft[:,ind[0]]*np.conj(fft[:,ind[1]]))
    return (k, Phi)

def smoothing(Su,k=[],bin_dec =[], islist = False):
    if islist:
        k = np.hstack([tup[0] for tup in Su])
        Su = np.hstack([tup[1] for tup in Su])
        Su = Su[:, np.argsort(k)]
        k = k[np.argsort(k)]
    Su = Su[:,k>0]
    k = k[k>0]    
    Su = np.real(Su)
    n, m = Su.shape
    decades = np.arange(np.floor(np.log10(k[0])),np.ceil(np.log10(k[-1]))+1)    
    bins = 10**np.unique(np.array([np.linspace(decades[i],
               decades[i+1],bin_dec+1) for i in range(len(decades)-1)]).flatten())
    print(decades,np.log10(bins))
    k_s = np.sqrt(bins[:-1]*bins[1:])
    S_s = np.zeros((n,len(k_s)))
    for j in range(n):
        for i in range(len(k_s)):       
            S_s[j,:] = [np.nanmean(Su[j,(k>bins[i]) & (k<bins[i+1])])  for i in range(len(bins)-1)]
    
    print(S_s.shape,~np.isnan(S_s).any(axis=0))
    k_s = k_s[~np.isnan(S_s).any(axis=0)]
    S_s = S_s[:,~np.isnan(S_s).any(axis=0)]
    return (k_s, S_s)

def off_w(u,v,w):  
    #Data filtering anc correcting for 37m
    from sklearn import mixture
    from scipy.interpolate import UnivariateSpline
    # There are three posibilities:
        # 1 No offset
        # 2 Offset in the whole series
        # 3 Offset on part of the data    

    components = np.array([1,2])
    gic = []
    ind0=[]
    comp=2
    for i in components:
        gmm = mixture.GaussianMixture(n_components=i, covariance_type='full')
        gmm.fit(w.reshape(-1, 1))
        gic.append(gmm.bic(w.reshape(-1, 1)))
    gic = np.array(gic)
    labels=gmm.predict(w.reshape(-1, 1))
    if np.diff(gic)/np.abs(gic[0])>-.15:
        wmean = np.nanmean(w)
        if np.abs(wmean)>1:
            stat = 2
            ind0 = np.ones(w.shape).astype(bool)
        else:
            stat = 1
            ind0 = np.ones(w.shape).astype(bool)    
    else:       
        stat = 3
        for l in np.unique(labels):
            ind = labels==l            
            if np.abs(np.nanmean(w[ind]))>1:
                ind0 = ind
        if len(ind0)==0:
            stat=2
            ind0 = np.ones(w.shape).astype(bool)
    if stat == 3:
        meanu = np.nanmean(u[~ind0])
        meanv = np.nanmean(v[~ind0])
        w[ind0] = w[ind0]-np.nanmean(w[ind0])
        u[ind0] = u[ind0]-np.nanmean(u[ind0])
        u[~ind0] = u[~ind0]-np.nanmean(u[~ind0])
        v[ind0] = v[ind0]-np.nanmean(v[ind0])
        v[~ind0] = v[~ind0]-np.nanmean(v[~ind0])
        u = u+meanu
        v = v+meanv        
    print(stat)
    return (u,v,w,ind0)

# In[Plots]
def wf_anim(ut, maskpod, grdu, t_scan,p = 1):  
    fig0, ax0 = plt.subplots(figsize=(8, 8))
    ax0.set_aspect('equal')
    ax0.use_sticky_edges = False
    ax0.margins(0.07)
    newdate = []    
    uscan = grdu[0]*np.nan
    umin,umax = np.quantile(ut.flatten(),q = [1-.997,.997])
    for j in range(ut.shape[0]): 
        print(j)
        new_date = t_scan[j,0].strftime("%Y-%m-%d %H:%M:%S")
        newdate.append(new_date[:11]+ '\:'+new_date[11:])        
    # meanU = []
    # meanV = []
    for j in range(ut.shape[0]):
        ax0.cla()    
        # meanU.append(np.nanmean(ut1[j,:]))
        # meanV.append(np.nanmean(vt1[j,:]))
        uscan[maskpod] = ut[j,:]
        ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20)   
        im0 = ax0.contourf(grdu[0], grdu[1], uscan, np.linspace(umin,umax,10), cmap='jet')
        ax0.tick_params(axis='both', which='major', labelsize=24)
        ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
        ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
        # ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
        
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
    
        # fig0.tight_layout()
        # fig0.savefig(file_out_path_u_field+file_pre+'/'+str(j)+'.png')
        plt.pause(p)
    return
def adv_plot(xg,yg,U,V,Luy,dx,lx=1,Lx=10,pl = 1):
    dudy, dudx = np.gradient(filterfft(U,np.isnan(U), [lx*Luy[1]/dx,lx*Luy[0]/dx]), yg[:,0], xg[0,:]) 
    dvdy, dvdx = np.gradient(filterfft(V,np.isnan(V), [lx*Luy[1]/dx,lx*Luy[0]/dx]), yg[:,0], xg[0,:])  
    contsh = dudx + dvdy
    vort = dvdx-dudy
    ums = filterfft(U,np.isnan(U),Lx*Luy/dx)
    vms = filterfft(V,np.isnan(V),Lx*Luy/dx)
    a = .12  
    b = 1       
    cmin,cmax = np.quantile(contsh.flatten()[~np.isnan(contsh.flatten())],q = [1-.997,.997]) 
    levelsc = np.linspace(cmin,cmax,20)   
    umin,umax = np.quantile((U-ums).flatten()[[~np.isnan((U-ums).flatten())]],q = [1-.997,.997]) 
    levelsu = np.linspace(umin,umax,20)                                 
    fig, ax = plt.subplots(2,1,figsize=(10, 6))
    ax[0].set_aspect('equal')
    ax[0].use_sticky_edges = False
    ax[0].margins(0.07)
    ax[1].set_aspect('equal')
    ax[1].use_sticky_edges = False
    ax[1].margins(0.07)
    im0 = ax[0].contourf(xg, yg, contsh, levelsc,cmap='bwr',norm=MidpointNormalize(midpoint=0,
                                                                          vmin = np.nanmin(contsh)*b,
                                                                          vmax=np.nanmax(contsh)*b))   
    ax[0].contour(xg, yg, U-ums, levels = [np.nanmin(U-ums)*a], colors = 'k',
                  linewidths = 2, linestyles = 'solid') 
    im1 = ax[1].contourf(xg,yg,U-ums,levelsu,cmap='bwr', norm=MidpointNormalize(midpoint=0,
                                                                          vmin = np.nanmin(U-ums),
                                                                          vmax=np.nanmax(U-ums)))
    ax[0].set_xlabel(r'$x_1\:[m]$',fontsize=20)
    ax[0].set_ylabel(r'$x_2\:[m]$',fontsize=20)
    ax[1].set_xlabel(r'$x_1\:[m]$',fontsize=20)
    ax[1].set_ylabel(r'$x_2\:[m]$',fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax= divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fmt))
    cb.ax.set_ylabel(r"$div_h\:[1/s]$",fontsize=24)
    divider = make_axes_locatable(ax[1])
    cax= divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm2))
    cb.ax.set_ylabel(r"$u_1\:[m/s]$",fontsize=24)
    
    if pl ==2:
    
        fig, ax = plt.subplots(2,1,figsize=(10, 6))
        ax[0].set_aspect('equal')
        ax[0].use_sticky_edges = False
        ax[0].margins(0.07)
        ax[1].set_aspect('equal')
        ax[1].use_sticky_edges = False
        ax[1].margins(0.07)
        im0 = ax[0].contourf(xg, yg, vort, 50,cmap='bwr',norm=MidpointNormalize(midpoint=0,
                                                                              vmin = np.nanmin(vort)*b,
                                                                              vmax=np.nanmax(vort)*b))   
        ax[0].contour(xg, yg, U-ums, levels = [np.nanmin(U-ums)*a], colors = 'k',
                      linewidths = 2, linestyles = 'solid') 
        im1 = ax[1].contourf(xg,yg,V-vms,50,cmap='bwr', norm=MidpointNormalize(midpoint=0,
                                                                              vmin = np.nanmin(V-vms),
                                                                              vmax=np.nanmax(V-vms)))
        ax[0].set_xlabel(r'$x_1\:[m]$',fontsize=20)
        ax[0].set_ylabel(r'$x_2\:[m]$',fontsize=20)
        ax[1].set_xlabel(r'$x_1\:[m]$',fontsize=20)
        ax[1].set_ylabel(r'$x_2\:[m]$',fontsize=20)
        divider = make_axes_locatable(ax[0])
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm2))
        cb.ax.set_ylabel(r"$div_h\:[1/s]$",fontsize=24)
        divider = make_axes_locatable(ax[1])
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
        cb.ax.set_ylabel(r"$u_1\:[m/s]$",fontsize=24)
    return

def auto_plot(tau, eta, ru, rv, ruv, comp, text = [], lim = [], z_i=1000, dim = 1):  
    suu = np.max(ru)
    svv = np.max(rv)
    suv = (suu*svv)**(1/2)    
    lab = [r'$\rho_{uu}', r'$\rho_{vv}', r'$\rho_{uv}', r'$\rho_{u^2+v^2}']
    rl = [ru/suu, rv/svv, ruv/suv, .5*(ru+rv)/np.max(.5*(ru+rv))]
    N = ru.shape[0]
    if dim ==2:
        for l,r in zip(lab,rl):
        
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_aspect('equal')
            ax1.use_sticky_edges = False
            ax1.margins(0.07)
            im1 = ax1.contour(tau/zi,eta/zi, r,np.linspace(np.nanmin(r),np.nanmax(r),50),colors = 'k')# cmap='jet')
            ax1.tick_params(axis='both', which='major', labelsize=24)
            ax1.set_xlabel('$\\tau/z_i$', fontsize=24)
            ax1.set_ylabel('$\eta/z_i$', fontsize=24)
            ax1.text((lim[0]+300)/zi, 3200,'(a)',fontsize=30,color='w')
            ax1.set_xlim(lim[0]/z_i,lim[1]/z_i)
            ax1.set_ylim(lim[0]/z_i,lim[1]/z_i)
            divider = make_axes_locatable(ax1)
            cax= divider.append_axes("right", size="5%", pad=0.05)
            cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
            cb.ax.set_ylabel(l+'$',fontsize=24)
            cb.ax.tick_params(labelsize=24)
            fig1.tight_layout()
    if dim ==1:    
        fig, ax = plt.subplots(figsize=(8, 8))
        #fig.set_size_inches(8,8)
        ax.use_sticky_edges = False
        ax.margins(0.07)
        ax.plot(eta[:,0]/z_i, rl[comp].T[:,int(N/2)],'k',label = lab[comp]+'(\\tau,0)$')
        ax.plot(tau[0,:]/z_i, rl[comp].T[int(N/2),:],'r',label = lab[comp]+'(0,\\eta)$')
        ax.set_xlabel('$\\tau/z_i,\:\\eta/z_i$',fontsize=24)
        ax.set_ylabel(lab[comp]+'$',fontsize=24)
        ax.legend(fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_xlim(lim[0]/z_i,lim[1]/z_i)
        ax.text((lim[0]+300)/z_i, .95,text,fontsize=30,color='k')
        # ax.set_xlim(-4000,4000)
        fig.tight_layout()    
    return

def spec_1D_plot(k1_int,k2_int,Su,Sv,Sh, axis = 0, premul=True,scale = 0,
                 kcut = 10**-2,zi=1000,z=200,norm=True):                  
        Suu_int = sp.integrate.simps(Su,k2_int[:,0],axis=axis)
        su = .5*sp.integrate.simps(Suu_int,k1_int[0,:])
        Svv_int = sp.integrate.simps(Sv,k2_int[:,0],axis=axis)
        sv = .5*sp.integrate.simps(Svv_int,k1_int[0,:])
        Sh_int = sp.integrate.simps(Sh,k2_int[:,0],axis=axis)
        tabs='S_{ii}[rad\:m/s^2]$'
        tab1 = ['$k_1\:[rad/m]$','$k_2\:[rad/m]$']
        tab2 = ['$k_1','$k_2']
        if norm:
            Suu_int = Suu_int/su
            Svv_int = Svv_int/sv
            tabs = 'S_{ii}/\sigma_{i}$'
            k1_int = k1_int*zi/(2*np.pi)
            tab1 = ['$k_1z_i$','$k_2z_i$']
            tab2 = ['$k_1z_i','$k_2z_i']
            z_i = 1
        else:
            z_i = 2*np.pi/zi
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.use_sticky_edges = False
        ax.margins(0.07)
        
        if premul:
            Suu_int = np.abs(k1_int[0,:])*Suu_int
            Svv_int = np.abs(k1_int[0,:])*Svv_int
            limy = np.r_[np.min(np.r_[Suu_int,Svv_int]),np.max(np.r_[Suu_int,Svv_int])]*2
            limx = np.r_[np.min(k1_int),np.max(k1_int)]*.97 
            min_k = np.argmin(np.abs(k1_int[0,:]-kcut))
            c = (Suu_int)[min_k]/k1_int[0,min_k]**(-2/3)
            ax.plot(k1_int[0,:],Suu_int,'b',label = '$S_{uu}$')
            ax.plot(k1_int[0,:],Svv_int,'r',label = '$S_{vv}$')
            ax.plot([z_i,z_i],limy ,'--k',label = '$z_i$')
            ax.plot(k1_int[0,:],c*k1_int[0,:]**(-2/3) ,'-k')
            ax.set_xlabel(tab1[axis],fontsize=24)
            ax.set_ylabel(tab2[axis]+tabs,fontsize=24)
            
        else: 
            limy = np.r_[np.min(np.r_[Suu_int,Svv_int]),np.max(np.r_[Suu_int,Svv_int])]*2
            limx = np.r_[np.min(k1_int),np.max(k1_int)]*.97 
            min_k = np.argmin(np.abs(k1_int[0,:]-kcut))
            c = (Suu_int)[min_k]/k1_int[0,min_k]**(-5/3)   
            ax.plot(k1_int[0,:],Suu_int ,'b',label = '$S_{uu}$')
            ax.plot(k1_int[0,:],Svv_int ,'r',label = '$S_{vv}$')
            ax.plot([z_i,z_i],limy ,'--k',label = '$z_i$')
            ax.plot(k1_int[0,:],c*k1_int[0,:]**(-5/3) ,'-k')            
            ax.set_xlabel(tab1[axis],fontsize=24)
            ax.set_ylabel(tab2[axis]+'S_{ii}[rad\:m/s^2]$',fontsize=24)            
        ax.legend(fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        if scale==0:
            ax.set_xscale('log')
            ax.set_yscale('log')
        elif scale == 1:
            ax.set_xscale('log')
        print(limx,limy)
        ax.set_xlim(limx)
        ax.set_ylim(limy)
        fig.tight_layout()
        return

# In[Data file paths and data loading]
# Length scales, stability conditions and scans
file1 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 1/West/L30min1.pkl'
file2 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 2/West/L30min2.pkl'
L30min1 = joblib.load(file1)
L30min2 = joblib.load(file2)

# Databases for wind field reconstruction
file1_0 = 'D:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 1/west/raw_filt_0_phase1.db'
file1_1 = 'D:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 1/west/raw_filt_1_phase1.db'
csv_database_0_ph1 = create_engine('sqlite:///'+file1_0)
csv_database_1_ph1 = create_engine('sqlite:///'+file1_1) 

file2_0 = 'D:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 2/west/raw_filt_0_phase2.db'
file2_1 = 'D:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 2/west/raw_filt_1_phase2.db'
csv_database_0_ph2 = create_engine('sqlite:///'+file2_0)
csv_database_1_ph2 = create_engine('sqlite:///'+file2_1)   


file_PBL = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/OsterPBLH_RMOL_WS_WD_T.dat'

# In[PBL form reanalysis]
h_pbl = [50., 75., 100., 150., 200., 250., 500.]
col_pbl = []
for lab in ['$U_pbl_','$D_pbl_','$T_pbl_']:
    for h in h_pbl:
        col_pbl.append(lab+str(int(h))+'$')

df_PBL = pd.read_table(file_PBL, sep='\t')
pbl_col = ['$time$','$z_i$', '$1/L$']+col_pbl
df_PBL.columns = pbl_col
df_PBL['$time$'] = pd.to_datetime(df_PBL['$time$'],format='%Y-%m-%d_%H:%M:%S')
df_PBL.set_index('$time$',inplace=True)

# phase 1

t1 = df_PBL.index.values
tL = pd.to_datetime([datetime.datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L30min1.index.values]])
L30min1[pbl_col[1:]] = pd.DataFrame(columns=pbl_col[1:], data = np.nan*np.zeros((len(tL),len(pbl_col[1:]))))
Lph1 = pd.DataFrame(columns=pbl_col[1:], data = np.nan*np.zeros((len(tL),len(pbl_col[1:]))))
Lph1.index = L30min1.index
for i in range(len(t1)-1):
    ind = (L30min1.time>=t1[i]) & (L30min1.time<=t1[i+1])
    if ind.sum()>0:        
        aux = df_PBL[pbl_col[1:]].loc[df_PBL.index ==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
        print(t1[i],t1[i+1])
        print(ind.sum())
L30min1[pbl_col[1:]] = Lph1[pbl_col[1:]]


t1 = df_PBL.index.values
tL = pd.to_datetime([datetime.datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L30min2.index.values]])
L30min2[pbl_col[1:]] = pd.DataFrame(columns=pbl_col[1:], data = np.nan*np.zeros((len(tL),len(pbl_col[1:]))))
Lph1 = pd.DataFrame(columns=pbl_col[1:], data = np.nan*np.zeros((len(tL),len(pbl_col[1:]))))
Lph1.index = L30min2.index
for i in range(len(t1)-1):
    ind = (L30min2.time>=t1[i]) & (L30min2.time<=t1[i+1])
    if ind.sum()>0:        
        aux = df_PBL[pbl_col[1:]].loc[df_PBL.index ==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
        print(t1[i],t1[i+1])
        print(ind.sum())
L30min2[pbl_col[1:]] = Lph1[pbl_col[1:]]

# In[Richarson number limits and cases selection]
# Columns and labels
Ric = np.array(['$Ri_{241}$', '$Ri_{175}$', '$Ri_{103}$','$Ri_{37}$', '$Ri_{7}$'])
U_l = np.array(['$U_{241}$', '$U_{175}$', '$U_{103}$','$U_{37}$', '$U_{7}$'])
uu_l = np.array(['$uu_{241}$', '$uu_{175}$', '$uu_{103}$','$uu_{37}$', '$uu_{7}$'])
vv_l = np.array(['$vv_{241}$', '$vv_{175}$', '$vv_{103}$','$vv_{37}$', '$vv_{7}$'])
ww_l = np.array(['$ww_{241}$', '$ww_{175}$', '$ww_{103}$','$ww_{37}$', '$ww_{7}$'])
uw_l = np.array(['$uw_{241}$', '$uw_{175}$', '$uw_{103}$','$uw_{37}$', '$uw_{7}$'])
vw_l = np.array(['$vw_{241}$', '$vw_{175}$', '$vw_{103}$','$vw_{37}$', '$vw_{7}$'])
uv_l = np.array(['$uv_{241}$', '$uv_{175}$', '$uv_{103}$','$uv_{37}$', '$uv_{7}$'])
wth_l = np.array(['$wth_{241}$', '$wth_{175}$', '$wth_{103}$','$wth_{37}$', '$wth_{7}$'])
us_l = np.array(['$u_{star,241}$', '$u_{star,175}$', '$u_{star,103}$','$u_{star,37}$', '$u_{star,7}$'])
TH_l = np.array(['$TH_{241}$', '$TH_{175}$', '$TH_{103}$','$TH_{37}$', '$TH_{7}$'])
D_l = np.array(['$D_{241}$', '$D_{175}$', '$D_{103}$','$D_{37}$', '$D_{7}$'])
V_l = np.array(['$V_{241}$', '$V_{175}$', '$V_{103}$','$V_{37}$', '$V_{7}$'])
angle_l = np.array(['$angle_{241}$', '$angle_{175}$', '$angle_{103}$','$angle_{37}$', '$angle_{7}$'])
zL = np.array(['$zL_{241}$', '$zL_{175}$', '$zL_{103}$','$zL_{37}$', '$zL_{7}$'])
Sspeed_0 = ['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS']
Sdir_0 = ['Wdir_244m_LMS', 'Wdir_40m_LMS']
Ric = np.array(['$Ri_{241}$', '$Ri_{175}$', '$Ri_{103}$','$Ri_{37}$', '$Ri_{7}$'])
TH = np.array(['$TH_{241}$', '$TH_{175}$', '$TH_{103}$','$TH_{37}$', '$TH_{7}$'])
zL = np.array(['$zL_{241}$', '$zL_{175}$', '$zL_{103}$','$zL_{37}$', '$zL_{7}$'])
L = np.array(['$L_{241}$', '$L_{175}$', '$L_{103}$','$L_{37}$', '$L_{7}$'])

# In[Cases selection by wind direction, reliability and stability]
i = 6
di = 1
direction1 = L30min1.loc[~L30min1[Sdir_0[di]].isna()][Sdir_0[di]]
direction2 = L30min2.loc[~L30min2[Sdir_0[di]].isna()][Sdir_0[di]]
speed1 = L30min1.loc[~L30min1[Sspeed_0[i]].isna()][Sspeed_0[i]]
speed2 = L30min2.loc[~L30min2[Sspeed_0[i]].isna()][Sspeed_0[i]]
speed1 = L30min1[U_l[-2]]
speed2 = L30min2[U_l[-2]]
j=-2
alpha_cen = 270
dalpha = 30
s1 = [5,300]
s2 = [5,300]
limL = 100000
indd1 = ((direction1>=alpha_cen-dalpha) & (direction1<=alpha_cen+dalpha)) 
inds1 = (speed1>s1[0]) & (speed1<s1[1]) 
indz1 = L30min1[zL[j]].abs()<1
indL1 = L30min1[L[j]].abs()<limL
indr1 = L30min1.relscan>.25
ind1 = indd1 & inds1 & indz1 & indL1 & indr1
indd2 = ((direction2>=alpha_cen-dalpha) & (direction2<=alpha_cen+dalpha)) 
inds2 = (speed2>s2[0]) & (speed2<s2[1]) 
indz2 = L30min2[zL[j]].abs()<1
indL2 = L30min2[L[j]].abs()<limL
indr2 = L30min2.relscan >.25
ind2 = indd2 & inds2 & indz2 & indL2 & indr2
# Richarson number limits
Rilim = [[-np.inf,-.25], [-.25,.0], [.0,.21], [.21,np.inf]] 
r_l = 1
alfa = .8
betax = .1
betay = .1
j=-2
indri1 = (L30min1[Ric[j]]>=Rilim[r_l][0]) & (L30min1[Ric[j]]<=Rilim[r_l][1])
indf1 = ind1 & indri1
lux_mean1 = L30min1['$L_{h,x_1}$'].loc[indf1].mean()
luy_mean1 = L30min1['$L_{h,x_2}$'].loc[indf1].mean()
offx1 = betax*L30min1['$L_{h,x_1}$'].loc[indf1].std()
offy1 = betay*L30min1['$L_{h,x_2}$'].loc[indf1].std()
indlx1 = (L30min1['$L_{h,x_1}$']>alfa*(lux_mean1+offx1)) & (L30min1['$L_{h,x_1}$']<(2-alfa)*(lux_mean1+offx1))
indly1 = (L30min1['$L_{h,x_2}$']>alfa*(luy_mean1+offy1)) & (L30min1['$L_{h,x_2}$']<(2-alfa)*(luy_mean1+offy1))
ind_1 = indlx1 & indly1 & indf1 

indri2 = (L30min2[Ric[j]]>=Rilim[r_l][0]) & (L30min2[Ric[j]]<=Rilim[r_l][1])
indf2 = ind2 & indri2
lux_mean2 = L30min2['$L_{h,x_1}$'].loc[indf2].mean()
luy_mean2 = L30min2['$L_{h,x_2}$'].loc[indf2].mean()
offx2 = betax*L30min2['$L_{h,x_1}$'].loc[indf2].std()
offy2 = betay*L30min2['$L_{h,x_2}$'].loc[indf2].std()
indlx2 = (L30min2['$L_{h,x_1}$']>alfa*(lux_mean2+offx2)) & (L30min2['$L_{h,x_1}$']<(2-alfa)*(lux_mean2+offx2))
indly2 = (L30min2['$L_{h,x_2}$']>alfa*(luy_mean2+offy2)) & (L30min2['$L_{h,x_2}$']<(2-alfa)*(luy_mean2+offy2))
ind_2 = indlx2 & indly2 & indf2 

# In[Phase 1 cases]
#####################################################################################################
#####################################################################################################
#####################################################################################################
pick_rnd1 = L30min1.loc[indf1]
days0 = np.unique(np.array([i[0] for i in pick_rnd1.index]))
for dy in days0:
    dy = '20160504'
    win=5
    umean = (L30min1.loc[dy,:][U_l[-2]].rolling(window=win,center=True).median()).mean()
    ax = L30min1.loc[dy,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().plot(ylim=(0,900))  
    L30min1.loc[dy,:][us_l[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
    L30min1.loc[dy,:][wth_l[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
    L30min1.loc[dy,:][Ric[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
    (L30min1.loc[dy,:][U_l[-2]]/umean).rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
    ax.right_ax.set_ylim(-3,2)
    ax.set_title(dy+', '+format(umean, '.2f'))
    plt.legend()

########################################################################################
# In[Unstable and high wind cases]
days_U_0 = days0
days_U_ph1_jump = days_U_0[[0,2,3,6]]
days_U_ph1_high = days_U_0[[1,4,5,8]]
# Neutral-Unstable
days_NU_0 = days0[~np.isin(days0,days_U_0)]
file_days_ph1 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 1/West/U_NU_days.pkl'
# joblib.dump((days_U_ph1_jump,days_U_ph1_high,days_U_0,days_NU_0),file_days_ph1)
days_U_ph1_jump,days_U_ph1_high,days_U_0,days_NU_0 = joblib.load(file_days_ph1)

# In[Time series cases phase 1]
import matplotlib.dates as mdates
# dy0 = '20160418'
# dy1 = '20160419'
# dy2 = '20160420'
dy0 = '20160421'

# dy0 = '20160501'
# dy1 = '20160502'
dy1 = '20160503'
# dy3 = '20160504'

h1 = '170000'
h2 = '190000'
date1 = [datetime.datetime.strptime(dy0+h1, '%Y%m%d%H%M%S'),
         datetime.datetime.strptime(dy0+h2, '%Y%m%d%H%M%S')]
win=5
hour0 = L30min1.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
hour0 = np.array([datetime.datetime.strptime(dy0+h, '%Y%m%d%H%M%S') for h in hour0])
Lh0 = L30min1.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
z_i0 = L30min1.loc[dy0,:]['$z_i$'].rolling(window=win,center=True).median().values
Linv0 = L30min1.loc[dy0,:]['$1/L$'].rolling(window=win,center=True).median().values
Ri0 = L30min1.loc[dy0,:][Ric[-2]].rolling(window=win,center=True).median()
U0 = L30min1.loc[dy0,:][U_l[-2]].rolling(window=win,center=True).median()
Upbl0 = L30min1.loc[dy0,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
Us0 = L30min1.loc[dy0,:][us_l[-2]].rolling(window=win,center=True).median()
uu0 = L30min1.loc[dy0,:][uu_l[-2]].rolling(window=win,center=True).median()
vv0 = L30min1.loc[dy0,:][vv_l[-2]].rolling(window=win,center=True).median()
ww0 = L30min1.loc[dy0,:][ww_l[-2]].rolling(window=win,center=True).median()

ind0 = ~np.isnan(Ri0)

h1 = '100000'
h2 = '140000'
date2 = [datetime.datetime.strptime(dy1+h1, '%Y%m%d%H%M%S'),
         datetime.datetime.strptime(dy1+h2, '%Y%m%d%H%M%S')]

hour1 = L30min1.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
hour1 = np.array([datetime.datetime.strptime(dy1+h, '%Y%m%d%H%M%S') for h in hour1])
Lh1 = L30min1.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
z_i1 = L30min1.loc[dy1,:]['$z_i$'].rolling(window=win,center=True).median().values
Linv1 = L30min1.loc[dy1,:]['$1/L$'].rolling(window=win,center=True).median().values
Ri1 = L30min1.loc[dy1,:][Ric[-2]].rolling(window=win,center=True).median()
U1 = L30min1.loc[dy1,:][U_l[-2]].rolling(window=win,center=True).median()
Upbl1 = L30min1.loc[dy1,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
Us1 = L30min1.loc[dy1,:][us_l[-2]].rolling(window=win,center=True).median()
uu1 = L30min1.loc[dy1,:][uu_l[-2]].rolling(window=win,center=True).median()
vv1 = L30min1.loc[dy1,:][vv_l[-2]].rolling(window=win,center=True).median()
ww1 = L30min1.loc[dy1,:][ww_l[-2]].rolling(window=win,center=True).median()

# hour2 = L30min1.loc[dy2,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
# hour2 = np.array([datetime.datetime.strptime(dy2+h, '%Y%m%d%H%M%S') for h in hour2])
# Lh2 = L30min1.loc[dy2,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
# z_i2 = L30min1.loc[dy2,:]['$z_i$'].rolling(window=win,center=True).median().values
# Ri2 = L30min1.loc[dy2,:][Ric[-2]].rolling(window=win,center=True).median()
# U2 = L30min1.loc[dy2,:][U_l[-2]].rolling(window=win,center=True).median()
# Upbl2 = L30min1.loc[dy2,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
# Us2 = L30min1.loc[dy2,:][us_l[-2]].rolling(window=win,center=True).median()
# uu2 = L30min1.loc[dy2,:][uu_l[-2]].rolling(window=win,center=True).median()
# vv2 = L30min1.loc[dy2,:][vv_l[-2]].rolling(window=win,center=True).median()
# ww2 = L30min1.loc[dy2,:][ww_l[-2]].rolling(window=win,center=True).median()

# hour3 = L30min1.loc[dy3,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
# hour3 = np.array([datetime.datetime.strptime(dy3+h, '%Y%m%d%H%M%S') for h in hour3])
# Lh3 = L30min1.loc[dy3,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
# z_i3 = L30min1.loc[dy3,:]['$z_i$'].rolling(window=win,center=True).median().values
# Ri3 = L30min1.loc[dy3,:][Ric[-2]].rolling(window=win,center=True).median()
# U3 = L30min1.loc[dy3,:][U_l[-2]].rolling(window=win,center=True).median()
# Upbl3 = L30min1.loc[dy3,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
# Us3 = L30min1.loc[dy3,:][us_l[-2]].rolling(window=win,center=True).median()
# uu3 = L30min1.loc[dy3,:][uu_l[-2]].rolling(window=win,center=True).median()
# vv3 = L30min1.loc[dy3,:][vv_l[-2]].rolling(window=win,center=True).median()
# ww3 = L30min1.loc[dy3,:][ww_l[-2]].rolling(window=win,center=True).median()

# hour = np.r_[hour0,hour1,hour2,hour3]
# Lh = np.vstack([Lh0,Lh1,Lh2,Lh3])
# z_i = np.r_[z_i0,z_i1,z_i2,z_i3]
# Ri = np.r_[Ri0,Ri1,Ri2,Ri3]
# U = np.r_[U0,U1,U2,U3]
# Upbl = np.r_[Upbl0,Upbl1,Upbl2,Upbl3]#np.r_[U0,U1,U2,U3]
# Us = np.r_[Us0,Us1,Us2,Us3]
# Us = np.r_[Us0,Us1,Us2,Us3]
# uu = np.r_[uu0,uu1,uu2,uu3]
# vv = np.r_[vv0,vv1,vv2,vv3]
# ww = np.r_[ww0,ww1,ww2,ww3]

ind1 = ~np.isnan(Ri1)

fig0, ax0 = plt.subplots(3,1,figsize=(8, 8))
ax0[0].use_sticky_edges = False
ax0[0].margins(0.07)
ax0[0].plot(hour0[ind0],Lh0[ind0,0],label='$L_{h,x_1}$')
ax0[0].plot(hour0[ind0],Lh0[ind0,1],label = '$L_{h,x_2}$')
ax0[0].plot(hour0[ind0],z_i0[ind0],label = '$z_i$')
ax0[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax0[0].fill_between(date1, [0,0], [1200,1200], facecolor='red', alpha=.2)
# ax0[0].fill_between(date2, [0,0], [1200,1200], facecolor='grey', alpha=.2)
ax0[0].set_ylim(0,1200)
ax0[0].set_ylabel('$L_{h,x_i}$',fontsize=20)
ax0[0].legend(fontsize=12)

labels = [item.get_text() for item in ax0[0].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax0[0].set_xticklabels(empty_string_labels)

ax0[1].use_sticky_edges = False
ax0[1].margins(0.07)
ax0[1].plot(hour0[ind0],Linv0[ind0],color='r',label = Ric[-2])
# ax0[1].fill_between(date1, [-.6,-.6], [.25,.25], facecolor='red', alpha=.2)
# ax0[1].fill_between(date2, [-2.5,-2.5], [.25,.25], facecolor='grey', alpha=.2)
# ax0[1].set_ylim(-.6,.25)
ax0[1].set_ylabel(Ric[-2],fontsize=20)

labels = [item.get_text() for item in ax0[1].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax0[1].set_xticklabels(empty_string_labels)

ax0[2].use_sticky_edges = False
ax0[2].margins(0.07)
ax0[2].plot(hour0[ind0],U0[ind0],color='g',label = U_l[-2])
ax0[2].plot(hour0[ind0],Upbl0[ind0],color='r',label = '$U_pbl_50$')
ax0[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax0[2].fill_between(date1, [2.5,2.5], [15,15], facecolor='red', alpha=.2)
# ax[2].fill_between(date2, [2.5,2.5], [15,15], facecolor='grey', alpha=.2)
ax0[2].set_ylim(2.5,20)
ax0[2].set_ylabel(U_l[-2],fontsize=20)
ax0[2].tick_params(axis='both', which='major', labelsize=12)

##############################################################################

fig1, ax1 = plt.subplots(3,1,figsize=(8, 8))
ax1[0].use_sticky_edges = False
ax1[0].margins(0.07)
ax1[0].plot(hour1[ind1],Lh1[ind1,0],label='$L_{h,x_1}$')
ax1[0].plot(hour1[ind1],Lh1[ind1,1],label = '$L_{h,x_2}$')
ax1[0].plot(hour1[ind1],z_i1[ind1],label = '$z_i$')
ax1[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1[0].fill_between(date1, [0,0], [1200,1200], facecolor='red', alpha=.2)
ax1[0].fill_between(date2, [0,0], [1200,1200], facecolor='grey', alpha=.2)
ax1[0].set_ylim(0,1200)
ax1[0].set_ylabel('$L_{h,x_i}$',fontsize=20)
ax1[0].legend(fontsize=12)

labels = [item.get_text() for item in ax1[0].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax1[0].set_xticklabels(empty_string_labels)

ax1[1].use_sticky_edges = False
ax1[1].margins(0.07)
ax1[1].plot(hour1[ind1],Linv1[ind1],color='r',label = Ric[-2])
# ax1[1].fill_between(date1, [-2.5,-2.5], [.25,.25], facecolor='red', alpha=.2)
# ax1[1].fill_between(date2, [-.6,-.5], [.25,.25], facecolor='grey', alpha=.2)
# ax1[1].set_ylim(-.6,.25)
ax1[1].set_ylabel(Ric[-2],fontsize=20)

labels = [item.get_text() for item in ax1[1].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax1[1].set_xticklabels(empty_string_labels)

ax1[2].use_sticky_edges = False
ax1[2].margins(0.07)
ax1[2].plot(hour1[ind1],U1[ind1],color='g',label = U_l[-2])
ax1[2].plot(hour1[ind1],Upbl1[ind1],color='r',label = '$U_pbl_50$')
ax1[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1[2].fill_between(date1, [2.5,2.5], [15,15], facecolor='red', alpha=.2)
ax1[2].fill_between(date2, [2.5,2.5], [15,15], facecolor='grey', alpha=.2)
ax1[2].set_ylim(2.5,20)
ax1[2].set_ylabel(U_l[-2],fontsize=20)
ax1[2].tick_params(axis='both', which='major', labelsize=12)

#####################################################################
###############################
# In[Vertical profiles phase 1]
heights = [241,175,103,37,7]

dy0 = '20160421'
dy1 = '20160503'

h1 = '100000'
h2 = '140000'
ind0 = (L30min1.loc[dy0,:].index>h1) & (L30min1.loc[dy0,:].index<h2)
ind1 = (L30min1.loc[dy1,:].index>h1) & (L30min1.loc[dy1,:].index<h2)

Ri0_p = np.nanmean(L30min1.loc[dy0,:][Ric].loc[ind0],axis=0)
Us0_p = np.nanmean(L30min1.loc[dy0,:][us_l].loc[ind0],axis=0)
uu0_p = np.nanmean(L30min1.loc[dy0,:][uu_l].loc[ind0],axis=0)
vv0_p = np.nanmean(L30min1.loc[dy0,:][vv_l].loc[ind0],axis=0)
ww0_p = np.nanmean(L30min1.loc[dy0,:][ww_l].loc[ind0],axis=0)
uw0_p = np.nanmean(L30min1.loc[dy0,:][uw_l].loc[ind0],axis=0)
uv0_p = np.nanmean(L30min1.loc[dy0,:][uv_l].loc[ind0],axis=0)
vw0_p = np.nanmean(L30min1.loc[dy0,:][vw_l].loc[ind0],axis=0)
wth0_p = np.nanmean(L30min1.loc[dy0,:][wth_l].loc[ind0],axis=0)
U0_p = np.nanmean(L30min1.loc[dy0,:][U_l].loc[ind0],axis=0)
Us0_p = np.nanmean(L30min1.loc[dy0,:][us_l].loc[ind0],axis=0)
TH0_p = np.nanmean(L30min1.loc[dy0,:][TH_l].loc[ind0],axis=0)

Ri1_p = np.nanmean(L30min1.loc[dy1,:][Ric].loc[ind1],axis=0)
Us1_p = np.nanmean(L30min1.loc[dy1,:][us_l].loc[ind1],axis=0)
uu1_p = np.nanmean(L30min1.loc[dy1,:][uu_l].loc[ind1],axis=0)
vv1_p = np.nanmean(L30min1.loc[dy1,:][vv_l].loc[ind1],axis=0)
ww1_p = np.nanmean(L30min1.loc[dy1,:][ww_l].loc[ind1],axis=0)
uw1_p = np.nanmean(L30min1.loc[dy1,:][uw_l].loc[ind1],axis=0)
uv1_p = np.nanmean(L30min1.loc[dy1,:][uv_l].loc[ind1],axis=0)
vw1_p = np.nanmean(L30min1.loc[dy1,:][vw_l].loc[ind1],axis=0)
wth1_p = np.nanmean(L30min1.loc[dy1,:][wth_l].loc[ind1],axis=0)
U1_p = np.nanmean(L30min1.loc[dy1,:][U_l].loc[ind1],axis=0)
Us1_p = np.nanmean(L30min1.loc[dy1,:][us_l].loc[ind1],axis=0)
TH1_p = np.nanmean(L30min1.loc[dy1,:][TH_l].loc[ind1],axis=0)

k0 = (uu0_p+vv0_p+ww0_p)
k1 = (uu1_p+vv1_p+ww1_p)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax1 = ax0.twiny()
ax1.plot(TH0_p-TH0_p[-1],heights,'-r',lw=2,label = r'$\Theta^{\ast}$')
ax0.plot(U0_p/Us0_p[-2],heights,'-k',lw=2,label=r'$U^{\ast}$')
fig0.legend(fontsize = 20,loc=[.2,.5])
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_xlabel(r'$U^{\ast}\:=\:\frac{U_1}{u_{\ast}}\:[-]$',fontsize=24)
ax0.set_ylabel(r'$z\:[m]$',fontsize=24)
ax0.set_ylim(0,280)
ax0.text(8, 260,'(a)',fontsize=30,color='k')
ax0.plot(U1_p/Us1_p[-2],heights,'--k',lw=2)
ax1.plot(TH1_p-TH1_p[-1],heights,'--r',lw=2)
ax1.set_xlabel(r'$\Theta^{\ast}\:=\:\Theta(z)-\Theta(0)\:[K]$',fontsize=24)
ax1.tick_params(axis='both', which='major', labelsize=20)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.plot(uu0_p/k0,heights,'-k',lw=2,label = r'$\frac{<u_1^2>}{k}$')
ax0.plot(vv0_p/k0,heights,'-r',lw=2,label = r'$\frac{<u_2^2>}{k}$')
ax0.plot(ww0_p/k0,heights,'-b',lw=2,label = r'$\frac{<u_3^2>}{k}$')
ax0.legend(fontsize = 20,ncol=4,loc=[0.15,.88])
ax0.set_xlabel(r'$\frac{<u_iu_j>}{k}\:[-]$',fontsize=24)
ax0.plot(uu1_p/k1,heights,'--k',lw=2,label = r'$\frac{<u_1^2>_u}{k}$')
ax0.plot(vv1_p/k1,heights,'--r',lw=2,label = r'$\frac{<u_2^2>_u}{k}$')
ax0.plot(ww1_p/k1,heights,'--b',lw=2,label = r'$\frac{<u_3^2>_u}{k}$')
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_ylim(0,280)
ax0.set_xlim(0,.6)
ax0.text(.025, 260,'(b)',fontsize=30,color='k')

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.plot(-uw0_p/Us0_p[-2]**2,heights,'-k',lw=2,label = r'$\frac{-<u_1u_3>}{u_{\ast}^2}$')
ax1 = ax0.twiny()
ax1.plot(wth0_p/wth0_p[-1],heights,'-r',lw=2,label = r'$\frac{-<u_2u_3>}{u_{\ast}^2}$')
# ax0.legend(fontsize = 16,loc=[0.035,.9])
ax0.set_xlabel(r'$\frac{-<u_1u_3>}{u_{\ast}^2}\:[-]$',fontsize=24)
ax0.plot(-uw1_p/Us1_p[-2]**2,heights,'--k',lw=2,label = r'$\frac{<u_1u_3>_u}{k}$')
ax0.tick_params(axis='both', which='major', labelsize=20)
ax1.plot(wth1_p/wth1_p[-1],heights,'--r',lw=2,label = r'$\frac{-<u_2u_3>}{u_{\ast}^2}$')
ax1.set_xlabel(r'$\frac{<u_3\theta>(z)}{<u_3\theta>(0)}\:[-]$',fontsize=24)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax0.set_ylim(0,280)
ax0.set_xlim(0,1.1)
ax0.text(.05, 260,'(c)',fontsize=30,color='k')

#####################################################################################################
# In[Wind Filed Reconstruction, phase 1]
dy0 = '20160421'
dy1 = '20160503'
db0 = csv_database_0_ph1
db1 = csv_database_1_ph1
h1 = '170000'
h2 = '190000'
ut1p1,vt1p1,maskpod1p1,U_list1p1,V_list1p1,U_out1p1,V_out1p1,grdu,tscan1p1,tri1p1 = wfrec(db0,db1,[dy0, dy0],
                                                                   [h1, h2],loc = [], merge_db=False)
h1 = '120000'
h2 = '140000'
ut2p1,vt2p1,maskpod2p1,U_list2p1,V_list2p1,U_out2p1,V_out2p1,grdu,tscan2p1,tri2p1 = wfrec(db0,db1,[dy1, dy1],
                                                                   [h1, h2],loc = [], merge_db=False)

wf_anim(ut1p1, maskpod1p1, grdu, tscan1p1,p=.1)
wf_anim(ut2p1, maskpod2p1, grdu, tscan2p1,p=.1)

wf_anim(vt1p1, maskpod1p1, grdu, tscan1p1,p=.1)
wf_anim(vt2p1, maskpod2p1, grdu, tscan2p1,p=.1)

# In[Advection phase 1]
from itertools import compress
dy0 = '20160421'
dy1 = '20160503'
loc0 = np.array([0,6322832.3])#-d
loc1 = np.array([0,6327082.4])# d
d = loc1-loc0

h1 = '170000'
h2 = '173000'
tin1 = datetime.datetime.strptime(dy0+h1, '%Y%m%d%H%M%S')
ten1 = datetime.datetime.strptime(dy0+h2, '%Y%m%d%H%M%S')
ind = (tscan1p1[:len(U_list1p1)][:,0]>=tin1) & (tscan1p1[:len(U_list1p1)][:,0]<=ten1)

tri_calc = True
chunk = 128
x = grdu[0][0,:]
y = grdu[1][:,0]
U_arr = np.vstack(list(compress(U_out1p1, ind)))
V_arr = np.vstack(list(compress(V_out1p1, ind)))

U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
dr0 = np.dot(T,dr0)  
dr1 = np.array(np.r_[d/2,np.ones(1)]).T
dr1 = np.dot(T,dr1) 

Xx = np.array(np.c_[tri1p1.x, tri1p1.y, np.ones(tri1p1.x.shape)]).T
Xx = np.dot(T,Xx)  
tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

Luy1p1 = L30min1.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
Luy1p1 = Luy1p1.loc[(Luy1p1.index>h1)&(Luy1p1.index<h2)].max().values   

beta=1.2   

xg1p1, yg1p1, Uex_t1p1, Vex_t1p1,t_ext1p1, points1p1 = expand_field_point_time(grdu, list(compress(U_list1p1, ind)), list(compress(V_list1p1, ind)),
                                                   [dr0, dr1], Luy = 3*Luy1p1, nx=100,
                                                   n = 36, alpha = [0, 2*np.pi],
                                                   time_scan = tscan1p1[:len(U_list1p1)][ind,:], t_int = tscan1p1[:len(U_list1p1)][ind,:][-1,0],
                                                   beta = beta, tri_dom = tri_dom)
h1 = '120000'
h2 = '130000'
tin2 = datetime.datetime.strptime(dy1+h1, '%Y%m%d%H%M%S')
ten2 = datetime.datetime.strptime(dy1+h2, '%Y%m%d%H%M%S')
ind = (tscan2p1[:len(U_list2p1)][:,0]>=tin2) & (tscan2p1[:len(U_list2p1)][:,0]<=ten2)

U_arr = np.vstack(list(compress(U_out2p1, ind)))
V_arr = np.vstack(list(compress(V_out2p1, ind)))
U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
dr0 = np.dot(T,dr0)  
dr1 = np.array(np.r_[d/2,np.ones(1)]).T
dr1 = np.dot(T,dr1) 

Xx = np.array(np.c_[tri1p1.x, tri1p1.y, np.ones(tri1p1.x.shape)]).T
Xx = np.dot(T,Xx)  
tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

Luy2p1 = L30min1.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
Luy2p1 = Luy2p1.loc[(Luy2p1.index>h1)&(Luy2p1.index<h2)].max().values    

beta=1.2

xg2p1, yg2p1, Uex_t2p1, Vex_t2p1, t_ext2p1, points2p1 = expand_field_point_time(grdu, list(compress(U_list2p1, ind)), list(compress(V_list2p1, ind)),
                                                   [dr0, dr1], Luy = 3*Luy2p1, nx=512,
                                                   n = 240, alpha = [0, 2*np.pi],
                                                   time_scan = tscan2p1[:len(U_list2p1)][ind,:],
                                                   t_int = tscan2p1[:len(U_list2p1)][ind,:][-1,0],
                                                   beta = beta, tri_dom = tri_dom)
# In[Advection plot]

dx = grdu[0][0,1]-grdu[0][0,0]
adv_plot(xg1p1,yg1p1,Uex_t1p1,Vex_t1p1,Luy1p1,dx,lx=1,Lx=3)

dx = grdu[0][0,1]-grdu[0][0,0]
adv_plot(xg2p1,yg2p1,Uex_t2p1,Vex_t2p1,Luy2p1,dx,lx=1,Lx=3)

#####################################################################################################
# In[Autocorrelation and spectra phase1]
# days
# days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ph1).values
# days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ph1).values
# days0 = np.squeeze(days0)
# days1 = np.squeeze(days1)
# days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])
# daysph1 = days

# In[Spectra phase 1]

daysph1 = np.array(['20160415', '20160418', '20160419', '20160420', '20160421',
       '20160422', '20160423', '20160424', '20160425', '20160426',
       '20160501', '20160502', '20160503', '20160504', '20160505',
       '20160506', '20160507', '20160508', '20160509', '20160513',
       '20160514', '20160515', '20160516', '20160517', '20160518',
       '20160520', '20160523', '20160524', '20160606', '20160608',
       '20160609', '20160610', '20160611', '20160614', '20160615',
       '20160616', '20160617'])
dy0 = '20160421'
dy1 = '20160503'

dy = dy0
h1 = '170000'
h2 = '180000'

file_path = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/west/phase_1/'
filename = file_path+'corr_spec_phase1_d'+ str(np.nonzero(daysph1==dy)[0][0]+1).zfill(2)+ '.h5'
ind =(L30min1.loc[dy,:].index>h1) & (L30min1.loc[dy,:].index<h2)
ziph1_1 = L30min1.loc[dy,:].loc[ind]['$z_i$'].mean()
dirph1_1 = L30min1.loc[dy,:].loc[ind]['$z_i$'].mean()
specph1_1 = pd.read_hdf(filename, key='Spec', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
autoph1_1 = pd.read_hdf(filename, key='Corr_10min', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
hms = np.unique(specph1_1.index.get_level_values(1))
k1gridph1_1,k2gridph1_1,Suum1ph1_1,Svvm1ph1_1,Shm1ph1_1,taugridph1_1,etagridph1_1,ruum1ph1_1,rvvm1ph1_1,ruvm1ph1_1 = spec_auto_ave(specph1_1,autoph1_1,dy,hms,N=256)

# second case
dy = dy1
h1 = '110000'
h2 = '120000'
file_path = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/west/phase_1/'
filename = file_path+'corr_spec_phase1_d'+ str(np.nonzero(daysph1==dy)[0][0]+1).zfill(2)+ '.h5'
ind =(L30min1.loc[dy,:].index>h1) & (L30min1.loc[dy,:].index<h2)
ziph1_2 = L30min1.loc[dy,:].loc[ind]['$z_i$'].mean()
specph1_2 = pd.read_hdf(filename,key='Spec', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
autoph1_2 = pd.read_hdf(filename, key='Corr_10min', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
hms = np.unique(specph1_2.index.get_level_values(1))
k1gridph1_2,k2gridph1_2,Suum1ph1_2,Svvm1ph1_2,Shm1ph1_2,taugridph1_2,etagridph1_2,ruum1ph1_2,rvvm1ph1_2,ruvm1ph1_2 = spec_auto_ave(specph1_2,autoph1_2,dy,hms,N=256)

# In[Spectra and autocorrelation plots,phase 1]
# Case 1
n1 = ziph1_1/2/np.pi
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Suum1ph1_1),
              label_S = "$\log_{10}{S_{uu}}$", C =n1*10**-3,fig_num='a',nl=10, minS = 0, S_lim = 5.1)
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Svvm1ph1_1),
              label_S = "$\log_{10}{S_{vv}}$", C =n1*10**-3,fig_num='b',nl=10, minS = 0, S_lim = 5.1)
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Shm1ph1_1),
              label_S = "$\log_{10}{S_{h}}$", C =n1*10**-3,fig_num='a',nl=10, minS = -1.5, S_lim = 5.1)
spec_1D_plot(k1gridph1_1,k2gridph1_1, np.abs(Suum1ph1_1), np.abs(Svvm1ph1_1), np.abs(Shm1ph1_1),
             axis=0, kcut=2,zi=ziph1_1,z=50,scale=0)
spec_1D_plot(k1gridph1_1,k2gridph1_1, np.abs(Suum1ph1_1), np.abs(Svvm1ph1_1), np.abs(Shm1ph1_1),
             axis=1, kcut=2,zi=ziph1_1,z=50,scale=0)
auto_plot(taugridph1_1,etagridph1_1, ruum1ph1_1, rvvm1ph1_1, ruvm1ph1_1,
          0,text = '(a)',lim = [-6000,6000],z_i = ziph1_1,dim=2)
# Case 2
n1 = ziph1_2/2/np.pi
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Suum1ph1_2),
              label_S = "$\log_{10}{S_{uu}}$", C =n1*10**-3,fig_num='a',nl=10, minS = 0, S_lim = 5.1)
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Svvm1ph1_2),
              label_S = "$\log_{10}{S_{vv}}$", C =n1*10**-3,fig_num='a',nl=10, minS = 0, S_lim = 5.1)
sc.plot_log2D((k1gridph1_1*n1,k2gridph1_1*n1), np.abs(Shm1ph1_2),
              label_S = "$\log_{10}{S_{h}}$", C =n1*10**-3,fig_num='a',nl=10, minS = -1.5, S_lim = 5.1)
spec_1D_plot(k1gridph1_2,k2gridph1_2, np.abs(Suum1ph1_2), np.abs(Svvm1ph1_2), np.abs(Shm1ph1_2),
             axis=0,kcut=2,zi=ziph1_2,z=50)
spec_1D_plot(k1gridph1_2,k2gridph1_2, np.abs(Suum1ph1_2), np.abs(Svvm1ph1_2), np.abs(Shm1ph1_2),
             axis=1,kcut=2,zi=ziph1_2,z=50)
auto_plot(taugridph1_2,etagridph1_2, ruum1ph1_2, rvvm1ph1_2, ruvm1ph1_2,
          0,text = '(b)',lim = [-7000,7000],z_i = ziph1_2,dim=2)

# In[Phase 2 cases]
pick_rnd2 = L30min2.loc[indf2]
days1 = np.unique(np.array([i[0] for i in pick_rnd2.index]))
for dy in days1:
    win=5
    umean = (L30min2.loc[dy,:][U_l[-2]].rolling(window=win).median()).mean()  
    ax = L30min2.loc[dy,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win).median().plot()
    # L30min2.loc[dy,:][us_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    L30min2.loc[dy,:][wth_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    L30min2.loc[dy,:][Ric[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    # (L30min2.loc[dy,:][U_l[-2]]/umean).rolling(window=win).median().plot(ax = ax, secondary_y=True)
    ax.set_title(dy+', $U_{mean}$ = '+format(umean, '.2f'))
    plt.legend()

#####################################################################################################
# In[days]
days_U_1 = days1
days_U_ph2_steady = days1[[0,2,4,5,6,7]]
# Neutral-Unstable
days_NU_1 = days1[~np.isin(days1,days_U_1)]
days_NU_ph2_steady = days_NU_1[[3,4,5]]
# In[]
file_days_ph2 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 2/West/U_NU_days.pkl'
#joblib.dump((days_U_ph2_steady,days_NU_1,days_NU_ph2_steady),file_days_ph2)
days_U_ph2_steady,days_NU_1,days_NU_ph2_steady = joblib.load(file_days_ph2)

for dy in days_NU_ph2_steady:
    win=5
    umean = (L30min2.loc[dy,:][U_l[-2]].rolling(window=win).median()).mean()    
    ax = L30min2.loc[dy,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win).median().plot()
    L30min2.loc[dy,:][us_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    L30min2.loc[dy,:][wth_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    L30min2.loc[dy,:][Ric[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
    # (L30min2.loc[dy,:][U_l[-2]].div(L30min2.loc[dy,:][us_l[-2]])).plot(ax = ax, secondary_y=True)
    (L30min2.loc[dy,:][U_l[-2]]/umean).rolling(window=win).median().plot(ax = ax, secondary_y=True)
    ax.set_title(dy+', $U_{mean}$ = '+format(umean, '.2f'))
    ax.set_ylim(0,900)
    plt.legend()

# In[Time series cases phase 2]
dy0 = '20160805'

dy1 = '20160806'
h1 = '100000'
h2 = '140000'

date1 = [datetime.datetime.strptime(dy0+h1, '%Y%m%d%H%M%S'),
         datetime.datetime.strptime(dy0+h2, '%Y%m%d%H%M%S')]

h1 = '180000'
h2 = '200000'

date2 = [datetime.datetime.strptime(dy1+h1, '%Y%m%d%H%M%S'),
         datetime.datetime.strptime(dy1+h2, '%Y%m%d%H%M%S')]

win=10

hour0 = L30min2.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
hour0 = np.array([datetime.datetime.strptime(dy0+h, '%Y%m%d%H%M%S') for h in hour0])
Lh0 = L30min2.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
z_i0 = L30min2.loc[dy0,:]['$z_i$'].rolling(window=win,center=True).median().values
Ri0 = L30min2.loc[dy0,:][Ric[-2]].rolling(window=win,center=True).median()
U0 = L30min2.loc[dy0,:][U_l[-2]].rolling(window=win,center=True).median()
Upbl0 = L30min2.loc[dy0,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
Us0 = L30min2.loc[dy0,:][us_l[-2]].rolling(window=win,center=True).median()

h1 = '180000'
h2 = '200000'

hour1 = L30min2.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().index
hour1 = np.array([datetime.datetime.strptime(dy1+h, '%Y%m%d%H%M%S') for h in hour1])
Lh1 = L30min2.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().values
z_i1 = L30min2.loc[dy1,:]['$z_i$'].rolling(window=win,center=True).median().values
Ri1 = L30min2.loc[dy1,:][Ric[-2]].rolling(window=win,center=True).median()
U1 = L30min2.loc[dy1,:][U_l[-2]].rolling(window=win,center=True).median()
Upbl1 = L30min2.loc[dy1,:]['$U_pbl_50$'].rolling(window=win,center=True).median()
Us1 = L30min2.loc[dy1,:][us_l[-2]].rolling(window=win,center=True).median()

hour = np.r_[hour0,hour1]#,hour2]#,hour3]
Lh = np.vstack([Lh0,Lh1])#,Lh2])#,Lh3])
Ri = np.r_[Ri0,Ri1]#,Ri2]#,Ri3]
U = np.r_[U0,U1]#,U2]#,U3]
Upbl = np.r_[Upbl0,Upbl1]#,U2]#,U3]
Us = np.r_[Us0,Us1]#,Us2]#,Us3]
z_i = np.r_[z_i0,z_i1]

ind = ~np.isnan(Ri)

fig0, ax = plt.subplots(3,1,figsize=(8, 8))
ax[0].use_sticky_edges = False
ax[0].margins(0.07)
ax[0].plot(hour[ind],Lh[ind,0],label='$L_{h,x_1}$')
ax[0].plot(hour[ind],Lh[ind,1],label = '$L_{h,x_2}$')
ax[0].plot(hour[ind],z_i[ind],label = '$z_i$')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[0].fill_between(date1, [0,0], [1500,1500], facecolor='red', alpha=.2)
ax[0].fill_between(date2, [0,0], [1500,1500], facecolor='grey', alpha=.2)
ax[0].set_ylim(0,1500)
ax[0].set_ylabel('$L_{h,x_i}$',fontsize=20)
ax[0].legend(fontsize=12)

labels = [item.get_text() for item in ax[0].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax[0].set_xticklabels(empty_string_labels)

ax[1].use_sticky_edges = False
ax[1].margins(0.07)
ax[1].plot(hour[ind],Ri[ind],color='r',label = Ric[-2])
ax[1].fill_between(date1, [-.3,-.3], [.1,.1], facecolor='red', alpha=.2)
ax[1].fill_between(date2, [-.3,-.3], [.1,.1], facecolor='grey', alpha=.2)
ax[1].set_ylim(-.3,.1)
ax[1].set_ylabel(Ric[-2],fontsize=20)

labels = [item.get_text() for item in ax[1].get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax[1].set_xticklabels(empty_string_labels)

ax[2].use_sticky_edges = False
ax[2].margins(0.07)
ax[2].plot(hour[ind],U[ind],color='k',label = U_l[-2])
ax[2].plot(hour[ind],Upbl[ind],color='r',label = U_l[-2])
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[2].fill_between(date1, [2.5,2.5], [15,15], facecolor='red', alpha=.2)
ax[2].fill_between(date2, [2.5,2.5], [15,15], facecolor='grey', alpha=.2)
ax[2].set_ylim(2.5,15)
ax[2].set_ylabel(U_l[-2],fontsize=20)
ax[2].tick_params(axis='both', which='major', labelsize=12)


# In[Vertical profiles phase 2]
heights = [241,175,103,37,7]

h1 = '100000'
h2 = '140000'
ind0 = (L30min2.loc[dy1,:].index>h1) & (L30min2.loc[dy1,:].index<h2)
ind1 = (L30min2.loc[dy0,:].index>h1) & (L30min2.loc[dy0,:].index<h2)

Ri0_p = np.nanmean(L30min2.loc[dy1,:][Ric].loc[ind0],axis=0)
Us0_p = np.nanmean(L30min2.loc[dy1,:][us_l].loc[ind0],axis=0)
uu0_p = np.nanmean(L30min2.loc[dy1,:][uu_l].loc[ind0],axis=0)
vv0_p = np.nanmean(L30min2.loc[dy1,:][vv_l].loc[ind0],axis=0)
ww0_p = np.nanmean(L30min2.loc[dy1,:][ww_l].loc[ind0],axis=0)
uw0_p = np.nanmean(L30min2.loc[dy1,:][uw_l].loc[ind0],axis=0)
uv0_p = np.nanmean(L30min2.loc[dy1,:][uv_l].loc[ind0],axis=0)
vw0_p = np.nanmean(L30min2.loc[dy1,:][vw_l].loc[ind0],axis=0)
wth0_p = np.nanmean(L30min2.loc[dy1,:][wth_l].loc[ind0],axis=0)
U0_p = np.nanmean(L30min2.loc[dy1,:][U_l].loc[ind0],axis=0)
Us0_p = np.nanmean(L30min2.loc[dy1,:][us_l].loc[ind0],axis=0)
TH0_p = np.nanmean(L30min2.loc[dy1,:][TH_l].loc[ind0],axis=0)

Ri1_p = np.nanmean(L30min2.loc[dy0,:][Ric].loc[ind1],axis=0)
Us1_p = np.nanmean(L30min2.loc[dy0,:][us_l].loc[ind1],axis=0)
uu1_p = np.nanmean(L30min2.loc[dy0,:][uu_l].loc[ind1],axis=0)
vv1_p = np.nanmean(L30min2.loc[dy0,:][vv_l].loc[ind1],axis=0)
ww1_p = np.nanmean(L30min2.loc[dy0,:][ww_l].loc[ind1],axis=0)
uw1_p = np.nanmean(L30min2.loc[dy0,:][uw_l].loc[ind1],axis=0)
uv1_p = np.nanmean(L30min2.loc[dy0,:][uv_l].loc[ind1],axis=0)
vw1_p = np.nanmean(L30min2.loc[dy0,:][vw_l].loc[ind1],axis=0)
wth1_p = np.nanmean(L30min2.loc[dy0,:][wth_l].loc[ind1],axis=0)
U1_p = np.nanmean(L30min2.loc[dy0,:][U_l].loc[ind1],axis=0)
Us1_p = np.nanmean(L30min2.loc[dy0,:][us_l].loc[ind1],axis=0)
TH1_p = np.nanmean(L30min2.loc[dy0,:][TH_l].loc[ind1],axis=0)

k0 = (uu0_p+vv0_p+ww0_p)
k1 = (uu1_p+vv1_p+ww1_p)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax1 = ax0.twiny()
ax1.plot(TH0_p-TH0_p[-1],heights,'-r',lw=2,label = r'$\Theta^{\ast}$')
ax0.plot(U0_p/Us0_p[-2],heights,'-k',lw=2,label=r'$U^{\ast}$')
fig0.legend(fontsize = 20,loc=[.2,.5])
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_xlabel(r'$U^{\ast}\:=\:\frac{U_1}{u_{\ast}}\:[-]$',fontsize=24)
ax0.set_ylabel(r'$z\:[m]$',fontsize=24)
ax0.set_ylim(0,280)
ax0.text(8, 260,'(a)',fontsize=30,color='k')
ax0.plot(U1_p/Us1_p[-2],heights,'--k',lw=2)
ax1.plot(TH1_p-TH1_p[-1],heights,'--r',lw=2)
ax1.set_xlabel(r'$\Theta^{\ast}\:=\:\Theta(z)-\Theta(0)\:[K]$',fontsize=24)
ax1.tick_params(axis='both', which='major', labelsize=20)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.plot(uu0_p/k0,heights,'-k',lw=2,label = r'$\frac{<u_1^2>}{k}$')
ax0.plot(vv0_p/k0,heights,'-r',lw=2,label = r'$\frac{<u_2^2>}{k}$')
ax0.plot(ww0_p/k0,heights,'-b',lw=2,label = r'$\frac{<u_3^2>}{k}$')
ax0.legend(fontsize = 20,ncol=4,loc=[0.15,.88])
ax0.set_xlabel(r'$\frac{<u_iu_j>}{k}\:[-]$',fontsize=24)
ax0.plot(uu1_p/k1,heights,'--k',lw=2,label = r'$\frac{<u_1^2>_u}{k}$')
ax0.plot(vv1_p/k1,heights,'--r',lw=2,label = r'$\frac{<u_2^2>_u}{k}$')
ax0.plot(ww1_p/k1,heights,'--b',lw=2,label = r'$\frac{<u_3^2>_u}{k}$')
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_ylim(0,280)
ax0.set_xlim(0,.6)
ax0.text(.025, 260,'(b)',fontsize=30,color='k')

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.plot(-uw0_p/Us0_p[-2]**2,heights,'-k',lw=2,label = r'$\frac{-<u_1u_3>}{u_{\ast}^2}$')
ax1 = ax0.twiny()
ax1.plot(wth0_p/wth0_p[-1],heights,'-r',lw=2,label = r'$\frac{-<u_2u_3>}{u_{\ast}^2}$')
# ax0.legend(fontsize = 16,loc=[0.035,.9])
ax0.set_xlabel(r'$\frac{-<u_1u_3>}{u_{\ast}^2}\:[-]$',fontsize=24)
ax0.plot(-uw1_p/Us1_p[-2]**2,heights,'--k',lw=2,label = r'$\frac{<u_1u_3>_u}{k}$')
ax0.tick_params(axis='both', which='major', labelsize=20)
ax1.plot(wth1_p/wth1_p[-1],heights,'--r',lw=2,label = r'$\frac{-<u_2u_3>}{u_{\ast}^2}$')
ax1.set_xlabel(r'$\frac{<u_3\theta>(z)}{<u_3\theta>(0)}\:[-]$',fontsize=24)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax0.set_ylim(0,280)
ax0.set_xlim(0,1.1)
ax0.text(.05, 260,'(c)',fontsize=30,color='k')

#####################################################################################################
#####################################################################################################
# In[Wind Filed Reconstruction, phase 2]
db0 = csv_database_0_ph2
db1 = csv_database_1_ph2
dy0 = '20160805'
dy1 = '20160806'
h1 = '100000'
h2 = '120000'
ut1p2,vt1p2,maskpod1p2,U_list1p2,V_list1p2,U_out1p2,V_out1p2,grdu,tscan1p2,tri1p2 = wfrec(db0,db1,[dy0, dy0],
                                                                   [h1, h2],loc = [], merge_db=False)
h1 = '180000'
h2 = '200000'
ut2p2,vt2p2,maskpod2p2,U_list2p2,V_list2p2,U_out2p2,V_out2p2,grdu,tscan2p2,tri2p2 = wfrec(db0,db1,[dy1, dy1],
                                                                   [h1, h2],loc = [], merge_db=False)

wf_anim(ut1p2-np.nanmean(ut1p2), maskpod1p2, grdu, tscan1p2,p=.1)
wf_anim(ut2p2-np.nanmean(ut2p2), maskpod2p2, grdu, tscan2p2,p=.1)
wf_anim(vt1p2, maskpod1p2, grdu, tscan1p2,p=.1)
wf_anim(vt2p2, maskpod2p2, grdu, tscan2p2,p=.1)

# In[Advection phase 2]
from itertools import compress

loc0 = np.array([0,6322832.3])#-d
loc1 = np.array([0,6327082.4])# d
d = loc1-loc0

dy0 = '20160805'
h1 = '103000'
h2 = '113000'
tin1 = datetime.datetime.strptime(dy0+h1, '%Y%m%d%H%M%S')
ten1 = datetime.datetime.strptime(dy0+h2, '%Y%m%d%H%M%S')
ind = (tscan1p2[:len(U_list1p2)][:,0]>=tin1) & (tscan1p2[:len(U_list1p2)][:,0]<=ten1)

tri_calc = True
chunk = 128
x = grdu[0][0,:]
y = grdu[1][:,0]
U_arr = np.vstack(list(compress(U_out1p2, ind)))
V_arr = np.vstack(list(compress(V_out1p2, ind)))

U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
dr0 = np.dot(T,dr0)  
dr1 = np.array(np.r_[d/2,np.ones(1)]).T
dr1 = np.dot(T,dr1) 

Xx = np.array(np.c_[tri1p2.x, tri1p2.y, np.ones(tri1p2.x.shape)]).T
Xx = np.dot(T,Xx)  
tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

Luy1p2 = L30min2.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
Luy1p2 = Luy1p2.loc[(Luy1p2.index>h1)&(Luy1p2.index<h2)].max().values   

beta=1   

xg1p2, yg1p2, Uex_t1p2, Vex_t1p2,t_ext1p2, points1p2 = expand_field_point_time(grdu, list(compress(U_list1p2, ind)), list(compress(V_list1p2, ind)),
                                                   [dr0, dr1], Luy = 3*Luy1p2, nx=100,
                                                   n = 36, alpha = [0, 2*np.pi],
                                                   time_scan = tscan1p2[:len(U_list1p2)][ind,:], t_int = tscan1p2[:len(U_list1p2)][ind,:][-1,0],
                                                   beta = beta, tri_dom = tri_dom)
dy1 = '20160806'
h1 = '183000'
h2 = '190000'
tin2 = datetime.datetime.strptime(dy1+h1, '%Y%m%d%H%M%S')
ten2 = datetime.datetime.strptime(dy1+h2, '%Y%m%d%H%M%S')
ind = (tscan2p2[:len(U_list2p2)][:,0]>=tin2) & (tscan2p2[:len(U_list2p2)][:,0]<=ten2)

U_arr = np.vstack(list(compress(U_out2p2, ind)))
V_arr = np.vstack(list(compress(V_out2p2, ind)))
U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
dr0 = np.dot(T,dr0)  
dr1 = np.array(np.r_[d/2,np.ones(1)]).T
dr1 = np.dot(T,dr1) 

Xx = np.array(np.c_[tri1p2.x, tri1p2.y, np.ones(tri1p2.x.shape)]).T
Xx = np.dot(T,Xx)  
tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

Luy2p2 = L30min2.loc[dy1,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
Luy2p2 = Luy2p2.loc[(Luy2p2.index>h1)&(Luy2p2.index<h2)].max().values    

beta=1.2

xg2p2, yg2p2, Uex_t2p2, Vex_t2p2,t_ext2p2, points2p2 = expand_field_point_time(grdu, list(compress(U_list2p2, ind)), list(compress(V_list2p2, ind)),
                                                   [dr0, dr1], Luy = 3*Luy2p2, nx=512,
                                                   n = 240, alpha = [0, 2*np.pi],
                                                   time_scan = tscan2p2[:len(U_list2p2)][ind,:], t_int = tscan2p2[:len(U_list2p2)][ind,:][-1,0],
                                                   beta = beta, tri_dom = tri_dom)
# In[plots advection]
dx = grdu[0][0,1]-grdu[0][0,0]
adv_plot(xg1p2,yg1p2,Uex_t1p2,Vex_t1p2,Luy1p2,dx,lx=.5,Lx=3)
dx = grdu[0][0,1]-grdu[0][0,0]
adv_plot(xg2p2,yg2p2,Uex_t2p2,Vex_t2p2,Luy2p2,dx,lx=.5,Lx=3)

# In[Autocorrelation and spectra phase2]
# days
days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ph2).values
days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ph2).values
days0 = np.squeeze(days0)
days1 = np.squeeze(days1)
days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])
daysph2 = days

# In[Spectra and autocorr phase 2]
daysph2 = np.array(['20160629', '20160630', '20160701', '20160702', '20160703',
       '20160704', '20160705', '20160706', '20160708', '20160709',
       '20160710', '20160711', '20160712', '20160714', '20160715',
       '20160717', '20160718', '20160720', '20160721', '20160722',
       '20160723', '20160724', '20160725', '20160726', '20160727',
       '20160729', '20160730', '20160731', '20160801', '20160802',
       '20160803', '20160804', '20160805', '20160806', '20160807',
       '20160808', '20160809', '20160810', '20160811', '20160812'])

dy0 = '20160805'
dy1 = '20160806'
h1 = '120000'
h2 = '140000'

# Combine .h5 files
dy = dy0

file_path = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/west/phase_2/'
filename = file_path+'corr_spec_phase2_d'+ str(np.nonzero(daysph2==dy)[0][0]+1).zfill(2)+ '.h5'
indph2_1 =(L30min2.loc[dy,:].index>h1) & (L30min2.loc[dy,:].index<h2)
ziph2_1 = L30min2.loc[dy,:].loc[indph2_1]['$z_i$'].mean()
specph2_1 = pd.read_hdf(filename, key='Spec', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
autoph2_1 = pd.read_hdf(filename, key='Corr_10min', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
hms = np.unique(specph2_1.index.get_level_values(1))
k1gridph2_1,k2gridph2_1,Suum1ph2_1,Svvm1ph2_1,Shm1ph2_1,taugridph2_1,etagridph2_1,ruum1ph2_1,rvvm1ph2_1,ruvm1ph2_1 = spec_auto_ave(specph2_1,autoph2_1,dy,hms,N=256)

dy = dy1
h1 = '180000'
h2 = '200000'

file_path = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/west/phase_2/'
filename = file_path+'corr_spec_phase2_d'+ str(np.nonzero(daysph2==dy)[0][0]+1).zfill(2)+ '.h5'
indph2_2 =(L30min2.loc[dy,:].index>h1) & (L30min2.loc[dy,:].index<h2)
ziph2_2 = L30min2.loc[dy,:].loc[indph2_2]['$z_i$'].mean()
specph2_2 = pd.read_hdf(filename, key='Spec', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
autoph2_2 = pd.read_hdf(filename, key='Corr_10min', where = "name = '"+ dy + "' and hms <= '"+ h2 + "' and hms >= '" + h1 +"'")
hms = np.unique(specph2_2.index.get_level_values(1))
k1gridph2_2,k2gridph2_2,Suum1ph2_2,Svvm1ph2_2,Shm1ph2_2,taugridph2_2,etagridph2_2,ruum1ph2_2,rvvm1ph2_2,ruvm1ph2_2 = spec_auto_ave(specph2_2,autoph2_2,dy,hms,N=256)

# In[plots phase 2]
# Case 1
sc.plot_log2D((k1gridph2_1,k2gridph2_1), np.abs(Suum1ph2_1),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='a',nl=30, minS = -1.5)
sc.plot_log2D((k1gridph2_1,k2gridph2_1), np.abs(Svvm1ph2_1),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='b',nl=30, minS = -1.5)
sc.plot_log2D((k1gridph2_1,k2gridph2_1), np.abs(Shm1ph2_1),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='c',nl=30, minS = -1.5)
spec_1D_plot(k1gridph1_2,k2gridph2_1, np.abs(Suum1ph2_1), np.abs(Svvm1ph2_1), np.abs(Shm1ph2_1),
             axis=0,kcut=2*10**-2,zi=ziph2_1,z=200,scale=1)
spec_1D_plot(k1gridph1_2,k2gridph2_1, np.abs(Suum1ph2_1), np.abs(Svvm1ph2_1), np.abs(Shm1ph2_1),
             axis=1,kcut=2*10**-2,zi=ziph2_1,z=200,scale=1)
auto_plot(taugridph2_2,etagridph2_1, ruum1ph2_1, rvvm1ph2_1, ruvm1ph2_1,
          0,text = '(a)',lim = [-6000,6000],z_i = ziph2_1)

# Case 2
sc.plot_log2D((k1gridph2_2,k2gridph2_2), np.abs(Suum1ph2_2),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='a',nl=30, minS = -1.5)
sc.plot_log2D((k1gridph2_2,k2gridph2_2), np.abs(Svvm1ph2_2),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='a',nl=30, minS = -1.5)
sc.plot_log2D((k1gridph2_2,k2gridph2_2), np.abs(Shm1ph2_2),
              label_S = "$\log_{10}{S}$", C =10**-3,fig_num='a',nl=30, minS = -1.5)
spec_1D_plot(k1gridph1_2,k2gridph2_2, np.abs(Suum1ph2_2), np.abs(Svvm1ph2_2), np.abs(Shm1ph2_2),
             axis=0,lim=[[-6*10**-2,6*10**-2],[10**-2,10**0]],kcut=10**-2,zi=ziph2_2,z=200)
spec_1D_plot(k1gridph1_2,k2gridph2_2, np.abs(Suum1ph2_2), np.abs(Svvm1ph2_2), np.abs(Shm1ph2_2),
             axis=1,lim=[[-6*10**-2,6*10**-2],[10**-2,10**0]],kcut=10**-2,zi=ziph2_2,z=200)
auto_plot(taugridph2_2,etagridph2_2, ruum1ph2_2, rvvm1ph2_2, ruvm1ph2_2,
          0,text = '(b)',lim = [-6000,6000],z_i = ziph2_2)
# 

# In[MySQL Østerild sonic and lidar data phase 1 and phase 2]
osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')
Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
Sdir_1 = ['Sdir_241m_LMN','Sdir_175m_LMN','Sdir_103m_LMN','Sdir_37m_LMN','Sdir_7m_LMN']
Name = ['Name']
T_0 = ['T_241m_LSM','T_175m_LSM','T_103m_LMS','T_37m_LMS','T_7m_LMS']
X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
P_0 = ['Press_241m_LMS','Press_7m_LMS']
## In[Spectra 1D from sonics]
heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
b = 15
## In[Phase 1]

dy0 = '20160421'
dy1 = '20160503'
#case1
dy = dy0
h1 = '170000'
h2 = '190000'

stampi = dy+h1
stampe = dy+h2

#for j,stamp in enumerate(t_ph1):
#    print(stamp,j)    
table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
query_name_i = 'select name' + table_20Hz + where + ' limit 1'
where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
query_name_e = 'select name' + table_20Hz + where + ' limit 1'
name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
where = 'where name > ' + name_i + ' and name < ' + name_e
sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_0.fillna(value=pd.np.nan, inplace=True)

# correcting one signal

x = -stab_20hz_0[X_0].values[:,-2].copy()
y = stab_20hz_0[Y_0].values[:,-2].copy()
z = stab_20hz_0[Z_0].values[:,-2].copy()

x,y,z,_ = off_w(x,y,z)

gammax_0_ph1_0 = mov_con(np.arctan2(-stab_20hz_0[X_0].values[:,0],stab_20hz_0[Y_0].values[:,0]),N=30*60*20)

S11 = np.cos(gammax_0_ph1_0)
S12 = np.sin(gammax_0_ph1_0)





u_s_0_ph1_0 = np.zeros((len(gammax_0_ph1_0),len(heights)))
v_s_0_ph1_0 = np.zeros((len(gammax_0_ph1_0),len(heights)))
w_s_0_ph1_0 = stab_20hz_0[Z_0].values

for i in range(len(heights)):  
    if i ==3:
        vel0 = np.c_[y,-x]
        w_s_0_ph1_0[:,i] = z 
    else:
        vel0 = np.c_[stab_20hz_0[Y_0].values[:,i], -stab_20hz_0[X_0].values[:,i]]
    u_s_0_ph1_0[:,i] = S11*vel0[:,0]+S12*vel0[:,1]
    v_s_0_ph1_0[:,i] = -S12*vel0[:,0]+S11*vel0[:,1]

for i in range(len(heights)):  
    
    u = np.c_[u_s_0_ph1_0[:,i],v_s_0_ph1_0[:,i],w_s_0_ph1_0[:,i]]
    k0_0, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivv0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiww0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivw0_0 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu0_0[i,:] = Phi[0,:]
    Phivv0_0[i,:] = Phi[1,:]
    Phiww0_0[i,:] = Phi[2,:]
    Phiuv0_0[i,:] = Phi[3,:]
    Phiuw0_0[i,:] = Phi[4,:]
    Phivw0_0[i,:] = Phi[5,:]
    
k_s0_ph1_0, S_s_uu_0_ph1_0 = smoothing(Phiuu0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph1_0, S_s_vv_0_ph1_0 = smoothing(Phivv0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph1_0, S_s_ww_0_ph1_0 = smoothing(Phiww0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph1_0, S_s_uv_0_ph1_0 = smoothing(Phiuv0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph1_0, S_s_uw_0_ph1_0 = smoothing(Phiuw0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph1_0, S_s_vw_0_ph1_0 = smoothing(Phivw0_0,k=k0_0,bin_dec = b, islist = False)

# case 2 Neutral

dy = dy1
h1 = '120000'
h2 = '140000'
stampi = dy+h1
stampe = dy+h2


#for j,stamp in enumerate(t_ph1):
#    print(stamp,j)    
table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
query_name_i = 'select name' + table_20Hz + where + ' limit 1'
where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
query_name_e = 'select name' + table_20Hz + where + ' limit 1'
name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
where = 'where name > ' + name_i + ' and name < ' + name_e
sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_0.fillna(value=pd.np.nan, inplace=True)
# sql_query = 'select ' + ", ".join(T_1+X_1+Y_1+Z_1+Sspeed_1+Sdir_1+Name) + table_20Hz +  where
# stab_20hz_1 = pd.read_sql_query(sql_query[0],osterild_database)  
# stab_20hz_1.fillna(value=pd.np.nan, inplace=True)

gammax_1_ph1_0 = mov_con(np.arctan2(-stab_20hz_0[X_0].values[:,0],stab_20hz_0[Y_0].values[:,0]),N=30*60*20)

S11 = np.cos(gammax_1_ph1_0)
S12 = np.sin(gammax_1_ph1_0)

u_s_0_ph1_1 = np.zeros((len(gammax_1_ph1_0),len(heights)))
v_s_0_ph1_1 = np.zeros((len(gammax_1_ph1_0),len(heights)))
w_s_0_ph1_1 = stab_20hz_0[Z_0].values


for i in range(len(heights)):  

    vel0 = np.c_[stab_20hz_0[Y_0].values[:,i], -stab_20hz_0[X_0].values[:,i]]
    u_s_0_ph1_1[:,i] = S11*vel0[:,0]+S12*vel0[:,1]
    v_s_0_ph1_1[:,i] = -S12*vel0[:,0]+S11*vel0[:,1]

for i in range(len(heights)):  
    
    u = np.c_[u_s_0_ph1_1[:,i],v_s_0_ph1_1[:,i],w_s_0_ph1_1[:,i]]
    k0_1, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivv0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiww0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivw0_1 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu0_1[i,:] = Phi[0,:]
    Phivv0_1[i,:] = Phi[1,:]
    Phiww0_1[i,:] = Phi[2,:]
    Phiuv0_1[i,:] = Phi[3,:]
    Phiuw0_1[i,:] = Phi[4,:]
    Phivw0_1[i,:] = Phi[5,:]
    

k_s0_ph1_1, S_s_uu_0_ph1_1 = smoothing(Phiuu0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph1_1, S_s_vv_0_ph1_1 = smoothing(Phivv0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph1_1, S_s_ww_0_ph1_1 = smoothing(Phiww0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph1_1, S_s_uv_0_ph1_1 = smoothing(Phiuv0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph1_1, S_s_uw_0_ph1_1 = smoothing(Phiuw0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph1_1, S_s_vw_0_ph1_1 = smoothing(Phivw0_1,k=k0_1,bin_dec = b, islist = False)

## In[phase 2]

Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
Sdir_1 = ['Sdir_241m_LMN','Sdir_175m_LMN','Sdir_103m_LMN','Sdir_37m_LMN','Sdir_7m_LMN']
Name = ['Name']
T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
X_1 = ['X_241m_LMN','X_175m_LMN','X_103m_LMN','X_37m_LMN','X_7m_LMN']
Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
Y_1 = ['Y_241m_LMN','Y_175m_LMN','Y_103m_LMN','Y_37m_LMN','Y_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']
P_0 = ['Press_241m_LMS','Press_7m_LMS']
P_1 = ['Press_241m_LMN','Press_7m_LMN']

dy0 = '20160805'
dy1 = '20160806'
#case1

dy = dy0
h1 = '100000'
h2 = '140000'

stampi = dy+h1
stampe = dy+h2

table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
query_name_i = 'select name' + table_20Hz + where + ' limit 1'
where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
query_name_e = 'select name' + table_20Hz + where + ' limit 1'
name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
where = 'where name > ' + name_i + ' and name < ' + name_e
sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_0.fillna(value=pd.np.nan, inplace=True)
sql_query = 'select ' + ", ".join(T_1+X_1+Y_1+Z_1+Sspeed_1+Sdir_1+Name) + table_20Hz +  where
stab_20hz_1 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_1.fillna(value=pd.np.nan, inplace=True)

gammax_0_ph2_0 = mov_con(np.arctan2(-stab_20hz_0[X_0].values[:,0],stab_20hz_0[Y_0].values[:,0]),N=30*60*20)
gammax_1_ph2_0 = mov_con(np.arctan2(-stab_20hz_1[X_1].values[:,0],stab_20hz_1[Y_1].values[:,0]),N=30*60*20)
gammax_ph2_0 = .5*(gammax_0_ph2_0+gammax_1_ph2_0)

S11 = np.cos(gammax_ph2_0)
S12 = np.sin(gammax_ph2_0)

u_s_0_ph2_0 = np.zeros((len(gammax_ph2_0),len(heights)))
u_s_1_ph2_0 = np.zeros((len(gammax_ph2_0),len(heights)))
v_s_0_ph2_0 = np.zeros((len(gammax_ph2_0),len(heights)))
v_s_1_ph2_0 = np.zeros((len(gammax_ph2_0),len(heights)))
w_s_0_ph2_0 = stab_20hz_0[Z_0].values
w_s_1_ph2_0 = stab_20hz_1[Z_1].values

for i in range(len(heights)):  

    vel0 = np.c_[stab_20hz_0[Y_0].values[:,i], -stab_20hz_0[X_0].values[:,i]]
    vel1 = np.c_[stab_20hz_1[Y_1].values[:,i], -stab_20hz_1[X_1].values[:,i]]
    u_s_0_ph2_0[:,i] = S11*vel0[:,0]+S12*vel0[:,1]
    v_s_0_ph2_0[:,i] = -S12*vel0[:,0]+S11*vel0[:,1]
    u_s_1_ph2_0[:,i] = S11*vel1[:,0]+S12*vel1[:,1]
    v_s_1_ph2_0[:,i] = -S12*vel1[:,0]+S11*vel1[:,1]

for i in range(len(heights)):  
    
    u = np.c_[u_s_0_ph2_0[:,i],v_s_0_ph2_0[:,i],w_s_0_ph2_0[:,i]]
    k0_0, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivv0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiww0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw0_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivw0_0 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu0_0[i,:] = Phi[0,:]
    Phivv0_0[i,:] = Phi[1,:]
    Phiww0_0[i,:] = Phi[2,:]
    Phiuv0_0[i,:] = Phi[3,:]
    Phiuw0_0[i,:] = Phi[4,:]
    Phivw0_0[i,:] = Phi[5,:]
    
    u = np.c_[u_s_1_ph2_0[:,i],v_s_1_ph2_0[:,i],w_s_1_ph2_0[:,i]]
    k1_0, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu1_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivv1_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiww1_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv1_0 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw1_0 = np.zeros((len(heights), Phi.shape[1]))
        Phivw1_0 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu1_0[i,:] = Phi[0,:]
    Phivv1_0[i,:] = Phi[1,:]
    Phiww1_0[i,:] = Phi[2,:]
    Phiuv1_0[i,:] = Phi[3,:]
    Phiuw1_0[i,:] = Phi[4,:]
    Phivw1_0[i,:] = Phi[5,:]       


k_s0_ph2_0, S_s_uu_0_ph2_0 = smoothing(Phiuu0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph2_0, S_s_vv_0_ph2_0 = smoothing(Phivv0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph2_0, S_s_ww_0_ph2_0 = smoothing(Phiww0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph2_0, S_s_uv_0_ph2_0 = smoothing(Phiuv0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph2_0, S_s_uw_0_ph2_0 = smoothing(Phiuw0_0,k=k0_0,bin_dec = b, islist = False)
k_s0_ph2_0, S_s_vw_0_ph2_0 = smoothing(Phivw0_0,k=k0_0,bin_dec = b, islist = False)

k_s1_ph2_0, S_s_uu_1_ph2_0 = smoothing(Phiuu1_0,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_0, S_s_vv_1_ph2_0 = smoothing(Phivv1_0,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_0, S_s_ww_1_ph2_0 = smoothing(Phiww1_0,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_0, S_s_uv_1_ph2_0 = smoothing(Phiuv1_0,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_0, S_s_uw_1_ph2_0 = smoothing(Phiuw1_0,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_0, S_s_vw_1_ph2_0 = smoothing(Phivw1_0,k=k1_0,bin_dec = b, islist = False)

# case 2 Neutral

dy = dy1
h1 = '180000'
h2 = '200000'
stampi = dy+h1
stampe = dy+h2


#for j,stamp in enumerate(t_ph1):
#    print(stamp,j)    
table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
query_name_i = 'select name' + table_20Hz + where + ' limit 1'
where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
query_name_e = 'select name' + table_20Hz + where + ' limit 1'
name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
where = 'where name > ' + name_i + ' and name < ' + name_e
sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_0.fillna(value=pd.np.nan, inplace=True)
sql_query = 'select ' + ", ".join(T_1+X_1+Y_1+Z_1+Sspeed_1+Sdir_1+Name) + table_20Hz +  where
stab_20hz_1 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_1.fillna(value=pd.np.nan, inplace=True)

heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806

gammax_0_ph2_1 = mov_con(np.arctan2(-stab_20hz_0[X_0].values[:,0],stab_20hz_0[Y_0].values[:,0]),N=30*60*20)
gammax_1_ph2_1 = mov_con(np.arctan2(-stab_20hz_1[X_1].values[:,0],stab_20hz_1[Y_1].values[:,0]),N=30*60*20)
gammax_ph2_1 = .5*(gammax_0_ph2_1+gammax_1_ph2_1)

S11 = np.cos(gammax_ph2_1)
S12 = np.sin(gammax_ph2_1)

u_s_0_ph2_1 = np.zeros((len(gammax_ph2_1),len(heights)))
u_s_1_ph2_1 = np.zeros((len(gammax_ph2_1),len(heights)))
v_s_0_ph2_1 = np.zeros((len(gammax_ph2_1),len(heights)))
v_s_1_ph2_1 = np.zeros((len(gammax_ph2_1),len(heights)))
w_s_0_ph2_1 = stab_20hz_0[Z_0].values
w_s_1_ph2_1 = stab_20hz_1[Z_1].values

for i in range(len(heights)):  

    vel0 = np.c_[stab_20hz_0[Y_0].values[:,i], -stab_20hz_0[X_0].values[:,i]]
    vel1 = np.c_[stab_20hz_1[Y_1].values[:,i], -stab_20hz_1[X_1].values[:,i]]
    u_s_0_ph2_1[:,i] = S11*vel0[:,0]+S12*vel0[:,1]
    v_s_0_ph2_1[:,i] = -S12*vel0[:,0]+S11*vel0[:,1]
    u_s_1_ph2_1[:,i] = S11*vel1[:,0]+S12*vel1[:,1]
    v_s_1_ph2_1[:,i] = -S12*vel1[:,0]+S11*vel1[:,1]

for i in range(len(heights)):  
    
    u = np.c_[u_s_0_ph2_1[:,i],v_s_0_ph2_1[:,i],w_s_0_ph2_1[:,i]]
    k0_1, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivv0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiww0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw0_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivw0_1 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu0_1[i,:] = Phi[0,:]
    Phivv0_1[i,:] = Phi[1,:]
    Phiww0_1[i,:] = Phi[2,:]
    Phiuv0_1[i,:] = Phi[3,:]
    Phiuw0_1[i,:] = Phi[4,:]
    Phivw0_1[i,:] = Phi[5,:]
    
    u = np.c_[u_s_1_ph2_1[:,i],v_s_1_ph2_1[:,i],w_s_1_ph2_1[:,i]]
    k1_0, Phi = spectra1D_t_series(u, 1/20)
    
    if i == 0:
        Phiuu1_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivv1_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiww1_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuv1_1 = np.zeros((len(heights), Phi.shape[1]))
        Phiuw1_1 = np.zeros((len(heights), Phi.shape[1]))
        Phivw1_1 = np.zeros((len(heights), Phi.shape[1]))
    
    Phiuu1_1[i,:] = Phi[0,:]
    Phivv1_1[i,:] = Phi[1,:]
    Phiww1_1[i,:] = Phi[2,:]
    Phiuv1_1[i,:] = Phi[3,:]
    Phiuw1_1[i,:] = Phi[4,:]
    Phivw1_1[i,:] = Phi[5,:]       

k_s0_ph2_1, S_s_uu_0_ph2_1 = smoothing(Phiuu0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph2_1, S_s_vv_0_ph2_1 = smoothing(Phivv0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph2_1, S_s_ww_0_ph2_1 = smoothing(Phiww0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph2_1, S_s_uv_0_ph2_1 = smoothing(Phiuv0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph2_1, S_s_uw_0_ph2_1 = smoothing(Phiuw0_1,k=k0_1,bin_dec = b, islist = False)
k_s0_ph2_1, S_s_vw_0_ph2_1 = smoothing(Phivw0_1,k=k0_1,bin_dec = b, islist = False)

k_s1_ph2_1, S_s_uu_1_ph2_1 = smoothing(Phiuu1_1,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_1, S_s_vv_1_ph2_1 = smoothing(Phivv1_1,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_1, S_s_ww_1_ph2_1 = smoothing(Phiww1_1,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_1, S_s_uv_1_ph2_1 = smoothing(Phiuv1_1,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_1, S_s_uw_1_ph2_1 = smoothing(Phiuw1_1,k=k1_0,bin_dec = b, islist = False)
k_s1_ph2_1, S_s_vw_1_ph2_1 = smoothing(Phivw1_1,k=k1_0,bin_dec = b, islist = False)

## In[Saving the 1D spectra]
file_path_sp_ph1 = 'D:/PhD/Python Code/Balcony/data_process/results/spectra/one_dimensional/phase1/west/'
file_path_sp_ph2 = 'D:/PhD/Python Code/Balcony/data_process/results/spectra/one_dimensional/phase2/west/'
# phase 1
# case 1

file_case = 'Spec_case1_unstable'
to_save = (u_s_0_ph1_0, v_s_0_ph1_0, w_s_0_ph1_0,
           k_s0_ph1_0, S_s_uu_0_ph1_0,
           k_s0_ph1_0, S_s_vv_0_ph1_0,
           k_s0_ph1_0, S_s_ww_0_ph1_0,
           k_s0_ph1_0, S_s_uv_0_ph1_0,
           k_s0_ph1_0, S_s_uw_0_ph1_0,
           k_s0_ph1_0, S_s_vw_0_ph1_0)
joblib.dump(to_save,file_path_sp_ph1+file_case)
file_case = 'Spec_case2_neutral'
to_save = (u_s_0_ph1_1, v_s_0_ph1_1, w_s_0_ph1_1,
           k_s0_ph1_1, S_s_uu_0_ph1_1,
           k_s0_ph1_1, S_s_vv_0_ph1_1,
           k_s0_ph1_1, S_s_ww_0_ph1_1,
           k_s0_ph1_1, S_s_uv_0_ph1_1,
           k_s0_ph1_1, S_s_uw_0_ph1_1,
           k_s0_ph1_1, S_s_vw_0_ph1_1)
joblib.dump(to_save,file_path_sp_ph1+file_case)

# phase 2

file_case = 'Spec_case1_unstable'
to_save = (u_s_0_ph2_0, v_s_0_ph2_0, w_s_0_ph2_0,
           k_s0_ph2_0, S_s_uu_0_ph2_0,
           k_s0_ph2_0, S_s_vv_0_ph2_0,
           k_s0_ph2_0, S_s_ww_0_ph2_0,
           k_s0_ph2_0, S_s_uv_0_ph2_0,
           k_s0_ph2_0, S_s_uw_0_ph2_0,
           k_s0_ph2_0, S_s_vw_0_ph2_0, 
           u_s_1_ph2_0, v_s_1_ph2_0, w_s_1_ph2_0,
           k_s1_ph2_0, S_s_uu_1_ph2_0,
           k_s1_ph2_0, S_s_vv_1_ph2_0,
           k_s1_ph2_0, S_s_ww_1_ph2_0,
           k_s1_ph2_0, S_s_uv_1_ph2_0,
           k_s1_ph2_0, S_s_uw_1_ph2_0,
           k_s1_ph2_0, S_s_vw_1_ph2_0)

joblib.dump(to_save,file_path_sp_ph2+file_case)
file_case = 'Spec_case2_neutral'
to_save = (u_s_0_ph2_1, v_s_0_ph2_1, w_s_0_ph2_1,
           k_s0_ph2_1, S_s_uu_0_ph2_1,
           k_s0_ph2_1, S_s_vv_0_ph2_1,
           k_s0_ph2_1, S_s_ww_0_ph2_1,
           k_s0_ph2_1, S_s_uv_0_ph2_1,
           k_s0_ph2_1, S_s_uw_0_ph2_1,
           k_s0_ph2_1, S_s_vw_0_ph2_1, 
           u_s_1_ph2_1, v_s_1_ph2_1, w_s_1_ph2_1,
           k_s1_ph2_1, S_s_uu_1_ph2_1,
           k_s1_ph2_1, S_s_vv_1_ph2_1,
           k_s1_ph2_1, S_s_ww_1_ph2_1,
           k_s1_ph2_1, S_s_uv_1_ph2_1,
           k_s1_ph2_1, S_s_uw_1_ph2_1,
           k_s1_ph2_1, S_s_vw_1_ph2_1)

joblib.dump(to_save,file_path_sp_ph2+file_case)

# In[plots]
# Phase 1

zi_0, k = np.meshgrid(np.array(heights)/ziph1_1,1/(k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,2]))/ziph1_1)



plt.figure()
plt.contourf(k_s0_ph1_0,np.array(heights)/ziph1_1,S_s_uu_0_ph1_0,cmap='jet')
plt.xscale('log')
plt.yscale('log')



pi = np.pi
i = 1
plt.figure()
plt.plot(k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i]), k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i])*S_s_uu_0_ph1_0[i,:],color='b',label='$Neutral$')
plt.plot(k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i]), k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i])*S_s_uu_0_ph1_1[i,:],color='r',label='$Unstable$')
plt.plot([1/ziph1_1,1/ziph1_1],[10**-3,10**-1],'--b')
plt.plot([1/ziph1_2,1/ziph1_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i]), 2*np.pi*k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i])*S_s_vv_0_ph1_0[i,:],color='b',label='$Neutral$')
plt.plot(2*np.pi*k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i]), 2*np.pi*k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i])*S_s_vv_0_ph1_1[i,:],color='r',label='$Unstable$')
plt.plot([2*pi/ziph1_1,2*pi/ziph1_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph1_2,2*pi/ziph1_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i]), 2*np.pi*k_s0_ph1_0/np.nanmean(u_s_0_ph1_0[:,i])*S_s_ww_0_ph1_0[i,:],color='b',label='$Neutral$')
plt.plot(2*np.pi*k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i]), 2*np.pi*k_s0_ph1_1/np.nanmean(u_s_0_ph1_1[:,i])*S_s_ww_0_ph1_1[i,:],color='r',label='$Unstable$')
plt.plot([2*pi/ziph1_1,2*pi/ziph1_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph1_2,2*pi/ziph1_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')
#####################################
# Phase 2
# Southern
plt.figure()
plt.plot(2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i]), 2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i])*S_s_uu_0_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i]), 2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i])*S_s_uu_0_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i]), 2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i])*S_s_vv_0_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i]), 2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i])*S_s_vv_0_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i]), 2*np.pi*k_s0_ph2_0/np.nanmean(u_s_0_ph2_0[:,i])*S_s_ww_0_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i]), 2*np.pi*k_s0_ph2_1/np.nanmean(u_s_0_ph2_1[:,i])*S_s_ww_0_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

# Northern

plt.figure()
plt.plot(2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i]), 2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i])*S_s_uu_1_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i]), 2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i])*S_s_uu_1_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i]), 2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i])*S_s_vv_1_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i]), 2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i])*S_s_vv_1_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i]), 2*np.pi*k_s1_ph2_0/np.nanmean(u_s_1_ph2_0[:,i])*S_s_ww_1_ph2_0[i,:],color='r',label='$Unstable$')
plt.plot(2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i]), 2*np.pi*k_s1_ph2_1/np.nanmean(u_s_1_ph2_1[:,i])*S_s_ww_1_ph2_1[i,:],color='b',label='$Neutral$')
plt.plot([2*pi/ziph2_1,2*pi/ziph2_1],[10**-3,10**-1],'--b')
plt.plot([2*pi/ziph2_2,2*pi/ziph2_2],[10**-3,10**-1],'--r')
plt.xscale('log')
plt.yscale('log')

# In[wind fields phase 1]
# # file_pre = '/case_'+pick_rnd.name.values[0]+pick_rnd.hms.values[0]
# # os.mkdir(file_out_path_u_field+file_pre)
# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# newdate = []

# uscan = grdu[0]*np.nan
# vscan = grdu[0]*np.nan

# for j in range(len(U_list2)): 
#     print(j)
#     new_date = t_scan2[j,0].strftime("%Y-%m-%d %H:%M:%S")
#     newdate.append(new_date[:11]+ '\:'+new_date[11:])
    
# meanU = []
# meanV = []
# for j in range(len(U_list2)):
#     ax0.cla()    
#     # meanU.append(np.nanmean(ut1[j,:]))
#     # meanV.append(np.nanmean(vt1[j,:]))
#     uscan[maskpod2] = ut2[j,:]
#     vscan[maskpod2] = vt2[j,:]
#     ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20) 
#     # ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
#     # ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
#     im0 = ax0.contourf(grdu[0], grdu[1], uscan, np.linspace(5,8,10), cmap='jet')
#     # ax0.streamplot(grdu[0], grdu[1], uscan-np.nanmean(ut1[j,:]), vscan, density=[4, 4])
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
#     ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
#     ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    
#     if len(fig0.axes) > 1: 
#         # if so, then the last axes must be the colorbar.
#         # we get its extent
#         pts = fig0.axes[-1].get_position().get_points()
#         # and its label
#         label = fig0.axes[-1].get_ylabel()
#         # and then remove the axes
#         fig0.axes[-1].remove()
#         # then we draw a new axes a the extents of the old one
#         divider = make_axes_locatable(ax0)
#         cax= divider.append_axes("right", size="5%", pad=0.05)
#         cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
#         cb.ax.tick_params(labelsize=24)
#         cb.ax.set_ylabel(r'$U\:[m/s]$', fontsize = 24)
#         # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=24, )
#         # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=24, weight='bold') 
#         # unfortunately the aspect is different between the initial call to colorbar 
#         #   without cax argument. Try to reset it (but still it's somehow different)
#         #cbar.ax.set_aspect(20)
#     else:
#         divider = make_axes_locatable(ax0)
#         cax= divider.append_axes("right", size="5%", pad=0.05)
#         cb = fig0.colorbar(im0, cax=cax, format=ticker.FuncFormatter(fm))
#         cb.ax.set_ylabel('$U\:[m/s]$', fontsize = 24)
#         cb.ax.tick_params(labelsize=24)
#         # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=12, weight='bold')
#         # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=12, weight='bold') 

#     # fig0.tight_layout()
#     # fig0.savefig(file_out_path_u_field+file_pre+'/'+str(j)+'.png')
#     plt.pause(.005)

 # In[wind fields phase 1]
# # file_pre = '/case_'+pick_rnd.name.values[0]+pick_rnd.hms.values[0]
# # os.mkdir(file_out_path_u_field+file_pre)

# pick_rndf1 = L30min1.loc[indf1]
# pick_rndf2 = L30min2.loc[indf2]

# tpick1 = ''.join(pick_rnd1.index.values[7])
# tpick2 = ''.join(pick_rnd2.index.values[2])
# file1 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 1/west/case_ph1_'+tpick1+'.pkl'
# file2 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 2/west/case_ph2_'+tpick2+'.pkl'
# pick1,ut1,vt1,maskpod1,U_list1,V_list1,grdu = joblib.load(file1)
# pick2,ut2,vt2,maskpod2,U_list2,V_list2,grdu = joblib.load(file2)

# phi1, A1, lam1 = pod(ut1,vt1)
# phi2, A2, lam2 = pod(ut2,vt2)

# N_nodes = 1000
# N_init = 1
# plt.figure(figsize=(8,8))
# plt.plot(lam1[N_init:N_nodes]/np.sum(np.abs(lam1[N_init:N_nodes])), '-o', label = r'$Ph\:1\:Unstable,\:Ri\:=\:'+'%.2f' % pick1['$Ri_{37}$']+'$')
# plt.plot(lam2[N_init:N_nodes]/np.sum(np.abs(lam2[N_init:N_nodes])), '-o', label = r'$Ph\:2\:Unstable,\:Ri\:=\:'+'%.2f' % pick2['$Ri_{37}$']+'$')
# plt.legend(fontsize = 20)
# plt.xlabel(r'$Mode\:number\:i$',fontsize = 24)
# plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.tight_layout()


# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# newdate = []

# for j in range(len(U_list2)): 
#     new_date = t_scan2[j,0].strftime("%Y-%m-%d %H:%M:%S")
#     newdate.append(new_date[:11]+ '\:'+new_date[11:])
    
# meanU = []
# meanV = []
# for j in range(len(U_list2)):
#     ax0.cla()    
#     meanU.append(np.nanmean(U_list2[j]))
#     meanV.append(np.nanmean(V_list2[j]))
#     ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20) 
#     # ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
#     # ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
#     im0 = ax0.contourf(grdu[0], grdu[1], U_list2[j], np.linspace(5,12,10), cmap='jet')
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
#     ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
#     ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    
#     if len(fig0.axes) > 1: 
#         # if so, then the last axes must be the colorbar.
#         # we get its extent
#         pts = fig0.axes[-1].get_position().get_points()
#         # and its label
#         label = fig0.axes[-1].get_ylabel()
#         # and then remove the axes
#         fig0.axes[-1].remove()
#         # then we draw a new axes a the extents of the old one
#         divider = make_axes_locatable(ax0)
#         cax= divider.append_axes("right", size="5%", pad=0.05)
#         cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
#         cb.ax.tick_params(labelsize=24)
#         cb.ax.set_ylabel(r'$U\:[m/s]$', fontsize = 24)
#         # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=24, )
#         # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=24, weight='bold') 
#         # unfortunately the aspect is different between the initial call to colorbar 
#         #   without cax argument. Try to reset it (but still it's somehow different)
#         #cbar.ax.set_aspect(20)
#     else:
#         divider = make_axes_locatable(ax0)
#         cax= divider.append_axes("right", size="5%", pad=0.05)
#         cb = fig0.colorbar(im0, cax=cax, format=ticker.FuncFormatter(fm))
#         cb.ax.set_ylabel('$U\:[m/s]$', fontsize = 24)
#         cb.ax.tick_params(labelsize=24)
#         # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=12, weight='bold')
#         # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=12, weight='bold') 

#     # fig0.tight_layout()
#     # fig0.savefig(file_out_path_u_field+file_pre+'/'+str(j)+'.png')
#     plt.pause(.005)




# In[]
#Sun
# plt.figure()
# j=-5
# kk = .5*(pick_rnd1[uu_l[j]]**2+pick_rnd1[vv_l[j]]**2+pick_rnd1[ww_l[j]]**2)**.5
# plt.scatter(pick_rnd1[U_l[j]],kk)

# plt.figure()
# j=-5
# kk = .5*(pick_rnd2[uu_l[j]]**2+pick_rnd2[vv_l[j]]**2+pick_rnd2[ww_l[j]]**2)**.5
# plt.scatter(pick_rnd2[U_l[j]],kk)
# ##########

# #c1 = [2,5,11,-1]
# c1 = [3,7]
# c2 = [2,7]

###################################
############# Cases ###############
###################################

#### Neutrally-unstable#####
###Phase1
# dy = '20160418'
# dy = '20160419'
# dy = '20160516'
# dy = '20160517'
###Phase2
#dy = '20160801'
#dy = '20160806'

#### Unstable#####
#Phase 1
# dy = '20160420' Changes in Lhx/Lhy duew to changes in u_star and Ri
# dy = '20160423' (this one is interesting neg. corr between Ri and Lhx)
# dy = '20160503' Similar to NU same negative correlations
# dy = '20160513' Very low speed, no changes in ustar, Lhx and Lhy more or less constant ******
# dy = '20160516' Not very uncorrelated, take a look increases with unstability. This case is very unstable. but ustar and Lhx still negatively correlated
# dy = '20160520' very very low wind speed (free convection?) see reconstruction!
# dy = '20160609' very stable length scales, and u_star

#Phase 2
# dy = '20160630'
 

#phase 1
# 0                   20160731 150436
# 1 20160421 180009   20160421 173009
# 2 20160421 184009
# 3 20160419 225036
# 4 20160504 190041

# phase 2
# 0 20160704 113020   20160731 170041
# 1 20160805 073626   20160726 150950
# 2 20160801 190022


# In[]
# ax = (L30min1.loc[dy,:]['$L_{h,x_1}$']/L30min1.loc[dy,:]['$L_{h,x_2}$']).rolling(window=win).median().plot()

# ax = L30min1.loc[dy,:][['$L_{h,x_1}$','$L_{h,x_2}$']].rolling(window=win,center=True).median().plot(ylim=(0,900))  
# L30min1.loc[dy,:][us_l[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
# L30min1.loc[dy,:][wth_l[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
# L30min1.loc[dy,:][Ric[-2]].rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
# (L30min1.loc[dy,:][U_l[-2]]/umean).rolling(window=win,center=True).median().plot(ax = ax, secondary_y=True)
# ax.right_ax.set_ylim(-4,2)


# # (L30min2.loc[dy,:][U_l[-2]].div(L30min2.loc[dy,:][us_l[-2]])).plot(ax = ax, secondary_y=True)
# # L30min2.loc[dy,:][U_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True)
# ax.set_title(dy+', '+format(umean, '.2f'))
# plt.legend()


########################################################################################


# fig, ax = plt.subplots(figsize=(8, 8))
# ax.use_sticky_edges = False
# ax.margins(0.07)
# L30min1.loc[dy,:][wth_l[-2]].rolling(window=win).median().plot(ax = ax, legend = True)
# (L30min1.loc[dy,:][us_l[-2]]**2).rolling(window=win).median().plot(ax = ax, secondary_y=True, legend = True)
# L30min1.loc[dy,:][U_l[-2]].rolling(window=win).median().plot(ax = ax, secondary_y=True, legend = True)



# dy = '20160630'
# indh = (L30min2.loc[dy,:].index>'140000') & (L30min2.loc[dy,:].index<'180000')
# ust = L30min2.loc[dy,:].loc[indh][us_l].values[:,-2]**2

# plt.plot((L30min2.loc[dy,:].loc[indh][uu_l].values/ust[:,None]).mean(axis=0),heights,'-.')
# plt.plot((L30min2.loc[dy,:].loc[indh][vv_l].values/ust[:,None]).mean(axis=0),heights,'-.')
# plt.plot((L30min2.loc[dy,:].loc[indh][ww_l].values/ust[:,None]).mean(axis=0),heights,'-.')



# In[Wind field reconstruction]
# # In[Database loading Phase1]
# # time definition

# day_jump = days_U_ph1_high[0]
# hour_jump = '100000'

# for k in c1:
#     date = datetime(1904, 1, 1, 0, 0)
#     tpick = day_jump+hour_jump#''.join(pick_rnd1.index.values[k])
#     t = pd.to_datetime(tpick)
#     minutes = 90
#     t_init = t - timedelta(minutes=minutes)
#     t_end = t + timedelta(minutes=minutes)
#     hms_i = t_init.strftime("%H%M%S")
#     hms_e = t_end.strftime("%H%M%S")
#     dy0 = t_init.strftime("%Y%m%d")
#     dy1 = t_end.strftime("%Y%m%d")
    
#     ######################################
#     t_0 = datetime(int(dy0[:4]), int(dy0[4:6]), int(dy0[6:]))
#     t_0 = t_0+timedelta(seconds = int(hms_i[4:]))
#     t_0 = t_0+timedelta(minutes = int(hms_i[2:4]))
#     t_0 = t_0+timedelta(hours = int(hms_i[:2]))
#     t_0 = str((t_0-date).total_seconds())
#     #
#     t_1 = datetime(int(dy1[:4]), int(dy1[4:6]), int(dy1[6:]))
#     t_1 = t_1+timedelta(seconds = int(hms_e[4:]))
#     t_1 = t_1+timedelta(minutes = int(hms_e[2:4]))
#     t_1 = t_1+timedelta(hours = int(hms_e[:2]))
#     t_1= str((t_1-date).total_seconds())
    
#     ######################################
#     ### labels
#     iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
#     labels = iden_lab
#     labels_new = iden_lab
#     #Labels for range gates and speed
#     labels_mask = []
#     labels_ws = []
#     labels_rg = []
#     labels_CNR = []
#     labels_Sb = []
#     for i in np.arange(198):
#         vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
#         mask_lab = np.array(['ws_mask'+str(i)])
#         labels_new = np.concatenate((labels_new,vel_lab))
#         labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
#         labels_mask = np.concatenate((labels_mask,mask_lab))
#         labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
#         labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
#         labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
#         labels_Sb = np.concatenate((labels_Sb,np.array(['Sb_'+str(i)])))
#     labels_new = np.concatenate((labels_new,np.array(['scan'])))    
    
#     # labels for query
#     labels_short = np.array([ 'stop_time', 'azim'])
#     for w,r in zip(labels_ws,labels_rg):
#         labels_short = np.concatenate((labels_short,np.array(['ws','range_gate', 'CNR', 'Sb'])))
#     labels_short = np.concatenate((labels_short,np.array(['scan'])))   
#     lim = [-8,-24]
#     i=0
#     col = 'SELECT '
#     col_raw = 'SELECT '
#     for w,r,c, s in zip(labels_ws,labels_rg,labels_CNR, labels_Sb):
#         if i == 0:
#             col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', ' + s + ', '
#             col_raw = col_raw  +  w  +  ', '
#         elif (i == len(labels_ws)-1):
#             col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', scan'
#             col_raw = col_raw + ' ' + w
#         else:
#             col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', ' 
#             col_raw = col_raw + ' ' + w + ', '
#         i+=1
    
#     selec_fil = col + ' FROM "table_fil"'
#     selec_raw = col_raw + ' FROM "table_raw"'
#     # Reconstruction of chuncks of 1 hour scans?
#     switch = 0
#     U_out_c, V_out_c, su_c = [], [], [] 
#     query_fil_0 = selec_fil+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
#     query_raw_0 = selec_raw+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
#     query_fil_1 = selec_fil+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
#     query_raw_1 = selec_raw+ ' where name >= ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
    
#     # First database loading
#     df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ph1)
#     # df = pd.read_sql_query(query_raw_0, csv_database_0_ph1)
#     # # Retrieving good CNR values from un-filtered scans
#     # for i in range(198):
#     #     ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
#     #     df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#     # df = None
#     # df_0.drop(columns = labels_CNR,inplace=True)
#     df_0.columns = labels_short
    
#     df_1 = pd.read_sql_query(query_fil_1, csv_database_1_ph1)
#     # df = pd.read_sql_query(query_raw_1, csv_database_1_ph1)
#     # for i in range(198):
#     #     ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
#     #     df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#     # df = None   
#     # #df_1.drop(columns = labels_CNR,inplace=True) 
#     df_1.columns = labels_short
    
#     loc0 = np.array([0,6322832.3])#-d
#     loc1 = np.array([0,6327082.4])# d
#     d = loc1-loc0  
#     switch = 0
    
#     ######################################
#     #In[Reconstruction]
#     chunk = 128
#     U_out_u, V_out_u, su_u = [], [], [] 
#     U_list1 = []
#     V_list1 = [] 
#     s_syn,t_scan1,_,_ = synch_df(df_0,df_1,dtscan=45/2)
#     print(s_syn)
#     if len(s_syn)>0:      
#         ind_df0 = df_0.scan.isin(s_syn[:,0])
#         df_0 = df_0.loc[ind_df0]
#         ind_df1 = df_1.scan.isin(s_syn[:,1])
#         df_1 = df_1.loc[ind_df1] 
#         # 1 hour itervals
#         s0 = s_syn[:,0]
#         s1 = s_syn[:,1]
#         t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])
#         t1 = np.array([date+timedelta(seconds = df_1.loc[df_1.scan==s].stop_time.min()) for s in s1])                                                  
#         tmin, tmax = np.min(np.r_[t0,t1]), np.max(np.r_[t0,t1])+timedelta(hours = 1)
#         t_1h = pd.date_range(tmin, tmax,freq='10T')#.strftime('%Y%m%d%H%M)
#         indt0 = [df_0.scan.isin(s0[(t0>=t_1h[i]) & (t0<t_1h[i+1])]) for i in range(len(t_1h)-1)]  
#         indt1 = [df_1.scan.isin(s1[(t1>=t_1h[i]) & (t1<t_1h[i+1])]) for i in range(len(t_1h)-1)]       
#         indt0 = [x for x in indt0 if np.sum(x) > 0]
#         indt1 = [x for x in indt1 if np.sum(x) > 0] 
#         if (len(indt0)>0)&(len(indt1)>0):
#             for i0,i1 in zip(indt0,indt1):
#                 df0 = df_0.loc[i0]
#                 df1 = df_1.loc[i1]
#                 if switch == 0:
#                     phi0 = df0.azim.unique()
#                     phi1 = df1.azim.unique()               
#                     r0 = df0.range_gate.iloc[0].values
#                     r1 = df1.range_gate.iloc[0].values                
#                     r_0, phi_0 = np.meshgrid(r0, np.pi/2-np.radians(phi0)) # meshgrid
#                     r_1, phi_1 = np.meshgrid(r0, np.pi/2-np.radians(phi1)) # meshgrid                
#                     tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
#                     switch = 1 
#                 u, v, grdu, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk)  
#                 su_u.append(s)
#     #############################################################################################            
#                 tri_calc = True
#                 x = grdu[0][0,:]
#                 y = grdu[1][:,0]
#                 U_arr = np.vstack([u[i] for i in range(len(u))])
#                 V_arr = np.vstack([v[i] for i in range(len(v))])
#                 scan = s_syn[:,0]
#                 U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
#                 V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
#                 ur, vr, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 
#                 for j in range(len(u)):
#                     print(j)
#                     U, V = U_rot(grdu, u[j], v[j], gamma = gamma, tri_calc = False, 
#                                               tri_del = tri_del, mask_int = mask_int, mask = mask) 
#                     U_list1.append(U)
#                     V_list1.append(V)
#     #############################################################################################  
#     # U_out_u = [item for sublist in U_out_u for item in sublist]
#     # V_out_u = [item for sublist in V_out_u for item in sublist]
#     # su_u    = [item for sublist in su_u for item in sublist]
#     ut1,vt1,maskpod1 = interp(grdu,U_list1,V_list1,N=100)
#     file1 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 1/west/case_ph1_U_'+tpick+'.pkl'
#     joblib.dump((ut1,vt1,maskpod1,U_list1,V_list1,grdu), file1)

# # In[Database loading Phase2]
# # time definition

# day_jump = days_U_ph2_steady[1]
# hour_jump = '200000'

# # for k in c2:
# date = datetime(1904, 1, 1, 0, 0)
# tpick = day_jump+hour_jump#''.join(pick_rnd2.index.values[k])
# t = pd.to_datetime(tpick)
# minutes = 90
# t_init = t - timedelta(minutes=minutes)
# t_end = t + timedelta(minutes=minutes)
# hms_i = t_init.strftime("%H%M%S")
# hms_e = t_end.strftime("%H%M%S")
# dy0 = t_init.strftime("%Y%m%d")
# dy1 = t_end.strftime("%Y%m%d")

# ######################################
# t_0 = datetime(int(dy0[:4]), int(dy0[4:6]), int(dy0[6:]))
# t_0 = t_0+timedelta(seconds = int(hms_i[4:]))
# t_0 = t_0+timedelta(minutes = int(hms_i[2:4]))
# t_0 = t_0+timedelta(hours = int(hms_i[:2]))
# t_0 = str((t_0-date).total_seconds())
# #
# t_1 = datetime(int(dy1[:4]), int(dy1[4:6]), int(dy1[6:]))
# t_1 = t_1+timedelta(seconds = int(hms_e[4:]))
# t_1 = t_1+timedelta(minutes = int(hms_e[2:4]))
# t_1 = t_1+timedelta(hours = int(hms_e[:2]))
# t_1= str((t_1-date).total_seconds())

# ######################################
# ### labels
# iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
# labels = iden_lab
# labels_new = iden_lab
# #Labels for range gates and speed
# labels_mask = []
# labels_ws = []
# labels_rg = []
# labels_CNR = []
# labels_Sb = []
# for i in np.arange(198):
#     vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
#     mask_lab = np.array(['ws_mask'+str(i)])
#     labels_new = np.concatenate((labels_new,vel_lab))
#     labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
#     labels_mask = np.concatenate((labels_mask,mask_lab))
#     labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
#     labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
#     labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
#     labels_Sb = np.concatenate((labels_Sb,np.array(['Sb_'+str(i)])))
# labels_new = np.concatenate((labels_new,np.array(['scan'])))    

# # labels for query
# labels_short = np.array([ 'stop_time', 'azim'])
# for w,r in zip(labels_ws,labels_rg):
#     labels_short = np.concatenate((labels_short,np.array(['ws','range_gate', 'CNR', 'Sb'])))
# labels_short = np.concatenate((labels_short,np.array(['scan'])))   
# lim = [-8,-24]
# i=0
# col = 'SELECT '
# col_raw = 'SELECT '
# for w,r,c, s in zip(labels_ws,labels_rg,labels_CNR, labels_Sb):
#     if i == 0:
#         col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', ' + s + ', '
#         col_raw = col_raw  +  w  +  ', '
#     elif (i == len(labels_ws)-1):
#         col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', scan'
#         col_raw = col_raw + ' ' + w
#     else:
#         col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', ' 
#         col_raw = col_raw + ' ' + w + ', '
#     i+=1

# selec_fil = col + ' FROM "table_fil"'
# selec_raw = col_raw + ' FROM "table_raw"'
# # Reconstruction of chuncks of 1 hour scans?
# switch = 0
# U_out_c, V_out_c, su_c = [], [], [] 
# query_fil_0 = selec_fil+ ' where name = ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
# query_raw_0 = selec_raw+ ' where name = ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
# query_fil_1 = selec_fil+ ' where name = ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1
# query_raw_1 = selec_raw+ ' where name = ' + dy0 + ' and name <= ' + dy1 + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1

# # First database loading
# df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ph2)
# df = pd.read_sql_query(query_raw_0, csv_database_0_ph2)
# # Retrieving good CNR values from un-filtered scans
# for i in range(198):
#     ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
#     df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
# df = None
# # df_0.drop(columns = labels_CNR,inplace=True)
# df_0.columns = labels_short

# df_1 = pd.read_sql_query(query_fil_1, csv_database_1_ph2)
# df = pd.read_sql_query(query_raw_1, csv_database_1_ph2)
# for i in range(198):
#     ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
#     df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
# df = None   
# #df_1.drop(columns = labels_CNR,inplace=True) 
# df_1.columns = labels_short

# loc0 = np.array([0,6322832.3])#-d
# loc1 = np.array([0,6327082.4])# d
# d = loc1-loc0  
# switch = 0

# ######################################
# #In[Reconstruction]
# chunk = 128
# U_out_u, V_out_u, su_u = [], [], [] 
# U_list2 = []
# V_list2 = [] 
# s_syn,t_scan2,_,_ = synch_df(df_0,df_1,dtscan=45/2)
# print(s_syn)
# if len(s_syn)>0:      
#     ind_df0 = df_0.scan.isin(s_syn[:,0])
#     df_0 = df_0.loc[ind_df0]
#     ind_df1 = df_1.scan.isin(s_syn[:,1])
#     df_1 = df_1.loc[ind_df1] 
#     # 1 hour itervals
#     s0 = s_syn[:,0]
#     s1 = s_syn[:,1]
#     t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])
#     t1 = np.array([date+timedelta(seconds = df_1.loc[df_1.scan==s].stop_time.min()) for s in s1])                                                  
#     tmin, tmax = np.min(np.r_[t0,t1]), np.max(np.r_[t0,t1])+timedelta(hours = 1)
#     t_1h = pd.date_range(tmin, tmax,freq='10T')#.strftime('%Y%m%d%H%M)
#     indt0 = [df_0.scan.isin(s0[(t0>=t_1h[i]) & (t0<t_1h[i+1])]) for i in range(len(t_1h)-1)]  
#     indt1 = [df_1.scan.isin(s1[(t1>=t_1h[i]) & (t1<t_1h[i+1])]) for i in range(len(t_1h)-1)]       
#     indt0 = [x for x in indt0 if np.sum(x) > 0]
#     indt1 = [x for x in indt1 if np.sum(x) > 0] 
#     if (len(indt0)>0)&(len(indt1)>0):
#         for i0,i1 in zip(indt0,indt1):
#             df0 = df_0.loc[i0]
#             df1 = df_1.loc[i1]
#             if switch == 0:
#                 phi0 = df0.azim.unique()
#                 phi1 = df1.azim.unique()               
#                 r0 = df0.range_gate.iloc[0].values
#                 r1 = df1.range_gate.iloc[0].values                
#                 r_0, phi_0 = np.meshgrid(r0, np.pi/2-np.radians(phi0)) # meshgrid
#                 r_1, phi_1 = np.meshgrid(r0, np.pi/2-np.radians(phi1)) # meshgrid                
#                 tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
#                 switch = 1 
#             u, v, grdu, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk)  
#             # U_out_u.append(u), V_out_u.append(v), su_u.append(s)
# #############################################################################################            
#             tri_calc = True
#             x = grdu[0][0,:]
#             y = grdu[1][:,0]
#             U_arr = np.vstack([u[i] for i in range(len(u))])
#             V_arr = np.vstack([v[i] for i in range(len(v))])
#             scan = s_syn[:,0]
#             U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
#             V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
#             ur, vr, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 
#             for j in range(len(u)):
#                 print(j)
#                 U, V = U_rot(grdu, u[j], v[j], gamma = gamma, tri_calc = False, 
#                                           tri_del = tri_del, mask_int = mask_int, mask = mask) 
#                 U_list2.append(U)
#                 V_list2.append(V)
# #############################################################################################  
# # U_out_u = [item for sublist in U_out_u for item in sublist]
# # V_out_u = [item for sublist in V_out_u for item in sublist]
# # su_u    = [item for sublist in su_u for item in sublist]   
# ut2,vt2,maskpod2 = interp(grdu,U_list2,V_list2,N=100)
# file2 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 2/west/case_ph2_U_'+tpick+'.pkl'
# joblib.dump((pick_rnd2.iloc[k],ut2,vt2,maskpod2,U_list2,V_list2,grdu), file2)


# In[Advection phase 2]
# loc0 = np.array([0,6322832.3])#-d
# loc1 = np.array([0,6327082.4])# d
# d = loc1-loc0

# dy0 = '20160805'
# dy1 = '20160806'
# h1 = '100000'
# h2 = '120000'

# tri_calc = True
# chunk = 128
# x = grdu[0][0,:]
# y = grdu[1][:,0]
# U_arr = np.vstack([U_out1p2[i] for i in range(len(U_out1p2))])
# V_arr = np.vstack([V_out1p2[i] for i in range(len(V_out1p2))])
# U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
# V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
# u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

# S11 = np.cos(gamma)
# S12 = np.sin(gamma)
# R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
# xtrans = (x[0]+x[-1])/2
# ytrans = (y[0]+y[-1])/2
# T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
# T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
# T = np.dot(np.dot(T1,R),T2)
# dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
# dr0 = np.dot(T,dr0)  
# dr1 = np.array(np.r_[d/2,np.ones(1)]).T
# dr1 = np.dot(T,dr1) 

# Xx = np.array(np.c_[tri1p2.x, tri1p2.y, np.ones(tri1p2.x.shape)]).T
# Xx = np.dot(T,Xx)  
# tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

# Luy1p2 = L30min2.loc[dy0,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
# Luy1p2 = Luy1p2.loc[(Luy1p2.index>h1)&(Luy1p2.index<h2)].max().values   

# beta=1.05   

# xg1p2, yg1p2, Uex_t1p2, Vex_t1p2,t_ext1p2, points1p2 = expand_field_point_time(grdu, U_list1p2, V_list1p2,
#                                                    [dr0, dr1], Luy = 10*Luy1p2, nx=100,
#                                                    n = 36, alpha = [0, 2*np.pi],
#                                                    time_scan = tscan1p2, t_int = tscan1p2[:len(U_list1p2)][-1,0],
#                                                    beta = beta, tri_dom = tri_dom)
# dx = grdu[0][0,1]-grdu[0][0,0]
# adv_plot(xg1p2,yg1p2,Uex_t1p2,Vex_t1p2,Luy1p2,dx,lx=.1,Lx=10)

# U_arr = np.vstack([U_out2p2[i] for i in range(len(U_out2p2))])
# V_arr = np.vstack([V_out2p2[i] for i in range(len(V_out2p2))])
# U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
# V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
# u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

# S11 = np.cos(gamma)
# S12 = np.sin(gamma)
# R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
# xtrans = (x[0]+x[-1])/2
# ytrans = (y[0]+y[-1])/2
# T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
# T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
# T = np.dot(np.dot(T1,R),T2)
# dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
# dr0 = np.dot(T,dr0)  
# dr1 = np.array(np.r_[d/2,np.ones(1)]).T
# dr1 = np.dot(T,dr1) 

# Xx = np.array(np.c_[tri1p2.x, tri1p2.y, np.ones(tri1p2.x.shape)]).T
# Xx = np.dot(T,Xx)  
# tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  

# Luy2p2 = L30min1.loc[dy2,:][['$L_{h,x_1}$','$L_{h,x_2}$']]
# Luy2p2 = Luy2p2.loc[(Luy2p2.index>h1)&(Luy2p2.index<h2)].max().values    

# xg2p2, yg2p2, Uex_t2p2, Vex_t2p2,t_ext2p2, points2p2 = expand_field_point_time(grdu, U_list2p2, V_list2p2,
#                                                    [dr0, dr1], Luy = 3*Luy2p2, nx=100,
#                                                    n = 36, alpha = [0, 2*np.pi],
#                                                    time_scan = tscan2p2, t_int = tscan2p2[:len(U_list2p2)][-1,0],
#                                                    beta = beta, tri_dom = tri_dom)
# dx = grdu[0][0,1]-grdu[0][0,0]
# adv_plot(xg2p2,yg2p2,Uex_t2p2,Vex_t2p2,Luy2p2,dx,lx=.1,Lx=10)



# In[]








