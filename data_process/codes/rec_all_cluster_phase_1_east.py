#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:22:58 2019

@author: leonardo alcayaga
"""
import numpy as np
import scipy as sp
import pandas as pd
import os
import sys
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
date = datetime(1904, 1, 1) 

# In[Functions]
def L_smooth(flux, N=6):    
    U = mov_con(flux[:,1], N)
    T = mov_con(flux[:,4], N)
    T = T+273.15
    uw = mov_con(flux[:,7], N)
    vw = mov_con(flux[:,9], N)
    wth = mov_con(flux[:,11], N)
    u_fric = (uw**2+vw**2)**.25
    k = .4
    g = 9.806
    L = -u_fric**3*T/(k*g*wth)
    #L[np.abs(L)>=500] = 500*np.sign(L[np.abs(L)>=500])
    stab = np.ones(L.shape)    
    indL = (L>0) & (L<100)
    stab[indL] = 0
    indL = (L>100) & (L<500)
    stab[indL] = 1
    indL = np.abs(L)>=500
    stab[indL] = 2
    indL = (L>-500) & (L<-100)
    stab[indL] = 3
    indL = (L>-100) & (L<0)
    stab[indL] = 4   
    return np.c_[u_fric, L, stab, U]

def mov_con(x,N):
    return np.convolve(x, np.ones((N,))/N,mode='same')
   
def datetimeDF(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

def synch_df(df0,df1,dtscan=45):
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

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])
def fm2(x, pos=None):
    return r'${}$'.format('{:.3f}'.format(x).split('f')[0])

def filterfft(_i, mask, sigma=20):
    vort = vorti.copy()
    vort[np.isnan(vort)] = np.nanmean(vort)   
    input_ = np.fft.fft2(vort)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.real(np.fft.ifft2(result))
    result[mask] = np.nan
    return result

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

# In[more functions]
def tot_L(r_u,r_v,r_h,tau,eta):
    tau = tau[0,:]
    eta = eta[:,0]
    Lu = np.array(sc.integral_lenght_scale(r_u,tau,eta))
    Lv = np.array(sc.integral_lenght_scale(r_v,tau,eta)) 
    Lh = np.array(sc.integral_lenght_scale(r_h,tau,eta)) 
    return np.r_[Lu,Lv,Lh]

# In[Files paths]    
file_in_path_0_db = '/mnt/mimer/lalc/db/scans/phase_1/east/raw_filt_0_east_phase1.db'
file_in_path_1_db = '/mnt/mimer/lalc/db/scans/phase_1/east/raw_filt_1_east_phase1.db'

# In[Database]
csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0_db)
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1_db)  

# In[day index]
index = int(sys.argv[1])
suf = str(index+1).zfill(2) 
file_name = '/mnt/mimer/lalc/results/correlations/east/phase_1/corr_spec_phase1_east_d'+suf+'.h5'

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

dy = days[index]

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
######################################
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
            uo, vo, grdu, so = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk) 
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
                # Correlations
                areafrac = [np.sum(~np.isnan(U))/len(U.flatten()) for U in ur]
                # gamma = gamma_10m(df0,df1,U_out,V_out,su)#np.arctan2(V_mean,U_mean)   
                tau_out = []
                eta_out = []    
                r_u_out = []
                r_v_out = []
                r_uv_out = []
                cont=0
                print(len(ur))
                for j in range(len(ur)):                 
                    tau,eta,r_u,r_v,r_uv,_,_,_,_ = sc.spatial_autocorr_sq(grdu,ur[j],vr[j],
                                                                          transform = False,
                                                                          transform_r = False,
                                                                          e_lim=.1,refine=32)
                    tau_out.append(tau.astype(np.float32))
                    eta_out.append(eta.astype(np.float32)) 
                    r_u_out.append(r_u.astype(np.float32))
                    r_v_out.append(r_v.astype(np.float32))
                    r_uv_out.append(r_uv.astype(np.float32))                    
                scan = np.array(so)[:,0]
                time = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in scan])    
                df0 = None
                df1 = None  
                r = np.vstack([np.c_[tau.flatten(),eta.flatten(),
                                     r_u.flatten(),r_v.flatten(),r_uv.flatten(),
                                     np.repeat(s,len(tau.flatten()))] for tau,eta,r_u,r_v,r_uv,s in zip(tau_out,eta_out,r_u_out,r_v_out,r_uv_out,scan)])
                t_stamp = np.hstack([np.repeat(t,len(tau.flatten())) for tau,t in zip(tau_out,time)])   
                
                l = np.vstack([np.r_[tot_L(r_u,r_v,.5*(r_u+r_v),tau,eta),np.nanmean(u), np.nanmean(v),af,s] for tau,eta,r_u,r_v,u,v,af,s in zip(tau_out,eta_out,r_u_out,r_v_out,ur,vr,areafrac,scan)])  
                floats = pd.DataFrame(r[:,:5].astype(np.float32))
                floats.columns = ['tau','eta','r_u','r_v','r_uv']
                ints =  pd.DataFrame(r[:,5].astype(np.int32))
                ints.columns = ['scan']
                strings = pd.DataFrame(t_stamp)
                strings.columns = ['time']
                L = pd.DataFrame(l)
                L.columns = ['L_ux', 'L_uy', 'L_vx', 'L_vy', 'L_hx', 'L_hy','Umean', 'Vmean', 'area_frac', 'scan']
                t =  pd.DataFrame(time)
                t.columns = ['time']
                aux0 = pd.concat([floats,ints,strings],axis=1)
                aux1 = pd.concat([L,t],axis=1)                
                timestp = pd.to_datetime(aux0.time.values).strftime("%Y%m%d")
                timestphms = pd.to_datetime(aux0.time.values).strftime("%H%M%S")
                aux0['name'] = timestp
                aux0['hms'] = timestphms
                aux0.set_index(['name','hms'], inplace=True)                
                timestp = pd.to_datetime(aux1.time.values).strftime("%Y%m%d")
                timestphms = pd.to_datetime(aux1.time.values).strftime("%H%M%S")
                aux1['name'] = timestp
                aux1['hms'] = timestphms
                aux1.set_index(['name','hms'], inplace=True)  
                columnshdf5 = aux0.iloc[[0]].reset_index().columns.tolist() 
                aux0.to_hdf(file_name,'corr',mode='a', data_columns = columnshdf5,format='table', append = True)
                columnshdf5 = aux1.iloc[[0]].reset_index().columns.tolist()
                aux1.to_hdf(file_name,'L',mode='a', data_columns = columnshdf5,format='table', append = True)
                
# In[]                            
                            
L = pd.read_hdf(file_name, key='L')

name_min, hms_min = L.index.min()
name_max, hms_max = L.index.max()
t_init = datetime.strptime(name_min+hms_min, '%Y%m%d%H%M%S')
t_end = datetime.strptime(name_max+hms_max, '%Y%m%d%H%M%S')+timedelta(hours = 1)
name_id= np.unique(pd.date_range(t_init, t_end,freq='10T').strftime('%Y%m%d'))[0]
t_arrayhms = pd.date_range(t_init, t_end,freq='10T').strftime('%H%M%S')
N = 256

for i in range(len(t_arrayhms)-1):
    hms1 = t_arrayhms[i]
    hms2 = t_arrayhms[i+1]
    corr = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='corr')
    L = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='L')
    
    if corr.empty:
        print('empty df')
    else:

        reslist = [corr[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[corr.scan == s].values for s in L.scan.values]
    
        tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist]
        eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist]
        ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist]
        rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist]
        ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist]
        resarray = np.vstack([r for r in reslist])
        tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)
    
        taumax = np.nanmax(np.abs(tau_arr))
        taui = np.linspace(0,taumax,int(N/2)+1)
        taui = np.r_[-np.flip(taui[1:]),taui]
        etamax = np.nanmax(np.abs(eta_arr))
        etai = np.linspace(0,etamax,int(N/2)+1)
        etai = np.r_[-np.flip(etai[1:]),etai]
        taui, etai = np.meshgrid(taui,etai)
    
        rui = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ru_list)])
        rvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, rv_list)])
        ruvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ruv_list)])
    
        ru_mean = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
        rv_mean = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
        ruv_mean = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)
    
        k1, k2, Suu, Svv, Suv = sc.spectra_fft((taui,etai),ru_mean,rv_mean,ruv_mean)
        Sh = .5*(Suu+Svv)
        k1, k2 = np.meshgrid(k1, k2)
        
        data_real = np.real(np.c_[k1.flatten(), k2.flatten(), Suu.flatten(), Svv.flatten(), Suv.flatten(), Sh.flatten()])
        data_imag = np.imag(np.c_[k1.flatten(), k2.flatten(), Suu.flatten(), Svv.flatten(), Suv.flatten(), Sh.flatten()])
        
        times = np.repeat(t_arrayhms[i],len(k1.flatten()))
        names_ids = np.repeat(name_id,len(k1.flatten()))
    
        columns = ['k1','k2','Suu','Svv','Suv', 'Sh']

        aux0 = pd.DataFrame(data_real, columns = columns)
        aux0['name'] = names_ids
        aux0['hms'] = times
        aux0.set_index(['name','hms'], inplace=True)
        
        aux1 = pd.DataFrame(data_imag, columns = columns)
        aux1['name'] = names_ids
        aux1['hms'] = times
        aux1.set_index(['name','hms'], inplace=True)
        
        cols_hdf5 = aux0.iloc[[0]].reset_index().columns.tolist()
     
        aux0.to_hdf(file_name,'Spec',mode='a', data_columns = cols_hdf5,
                    format='table', append = True)
        
        cols_hdf5 = aux1.iloc[[0]].reset_index().columns.tolist()
        
        aux1.to_hdf(file_name,'Spec_imag',mode='a', data_columns = cols_hdf5,
                    format='table', append = True)
# In[]

L = pd.read_hdf(file_name, key='L')

name_min, hms_min = L.index.min()
name_max, hms_max = L.index.max()
t_init = datetime.strptime(name_min+hms_min, '%Y%m%d%H%M%S')
t_end = datetime.strptime(name_max+hms_max, '%Y%m%d%H%M%S')+timedelta(hours = 1)
name_id= np.unique(pd.date_range(t_init, t_end,freq='10T').strftime('%Y%m%d'))[0]
t_arrayhms = pd.date_range(t_init, t_end,freq='10T').strftime('%H%M%S')
N = 256
dataL = []
timeL = []
namesL = []
for i in range(len(t_arrayhms)-1):
    hms1 = t_arrayhms[i]
    hms2 = t_arrayhms[i+1]
    corr = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='corr')
    L = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='L')
    
    if corr.empty:
        print('empty df')
    else:

        reslist = [corr[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[corr.scan == s].values for s in L.scan.values]
    
        tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist]
        eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist]
        ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist]
        rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist]
        ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist]
        resarray = np.vstack([r for r in reslist])
        tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)
    
        taumax = np.nanmax(np.abs(tau_arr))
        taui = np.linspace(0,taumax,int(N/2)+1)
        taui = np.r_[-np.flip(taui[1:]),taui]
        etamax = np.nanmax(np.abs(eta_arr))
        etai = np.linspace(0,etamax,int(N/2)+1)
        etai = np.r_[-np.flip(etai[1:]),etai]
        taui, etai = np.meshgrid(taui,etai)
    
        rui = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ru_list)])
        rvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, rv_list)])
        ruvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ruv_list)])
    
        ru_mean = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
        rv_mean = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
        ruv_mean = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)
        
        Luxm, Luym = sc.integral_lenght_scale(ru_mean,taui[0,:], etai[:,0])
        Lvxm, Lvym = sc.integral_lenght_scale(rv_mean,taui[0,:], etai[:,0])
        Lhxm, Lhym = sc.integral_lenght_scale(.5*(ru_mean+rv_mean), taui[0,:], etai[:,0])
            
        data = np.c_[taui.flatten(), etai.flatten(), ru_mean.flatten(), rv_mean.flatten(), ruv_mean.flatten()]

        times = np.repeat(t_arrayhms[i],len(taui.flatten()))
        names_ids = np.repeat(name_id,len(taui.flatten()))
    
        columns   = ['tau','eta','ru','rv','ruv'] 
        
        dataL.append(np.r_[Luxm, Luym, Lvxm, Lvym, Lhxm, Lhym, L.Umean.mean(), L.Vmean.mean()])
        timeL.append(t_arrayhms[i])
        namesL.append(name_id)

        aux0 = pd.DataFrame(data, columns = columns)
        aux0['name'] = names_ids
        aux0['hms'] = times
        aux0.set_index(['name','hms'], inplace=True)
        
        cols_hdf5 = aux0.iloc[[0]].reset_index().columns.tolist()
     
        aux0.to_hdf(file_name,'Corr_10min',mode='a', data_columns = cols_hdf5,
                    format='table', append = True)
        

columnsL = ['L_ux', 'L_uy', 'L_vx', 'L_vy', 'L_hx', 'L_hy','Umean', 'Vmean']
aux1 = pd.DataFrame(np.array(dataL), columns = columnsL)
aux1['name'] = np.array(namesL)
aux1['hms'] = np.array(timeL)
aux1.set_index(['name','hms'], inplace=True)

cols_hdf5 = aux1.iloc[[0]].reset_index().columns.tolist()

aux1.to_hdf(file_name,'L_10min',mode='a', data_columns = cols_hdf5,
            format='table', append = True)


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            