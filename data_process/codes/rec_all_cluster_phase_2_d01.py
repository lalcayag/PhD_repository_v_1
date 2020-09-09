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
file_in_path_0_db = '/mnt/mimer/lalc/db/scans/phase_2/raw_filt_0_phase2.db'
file_in_path_1_db = '/mnt/mimer/lalc/db/scans/phase_2/raw_filt_1_phase2.db'

# In[Database]
csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0_db)
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1_db)  

# In[Reconstruction and correlations]

index = int(sys.argv[1])

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
days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])


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
                S11 = np.cos(gamma)
                S12 = np.sin(gamma)
                R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
                xtrans = (x[0]+x[-1])/2
                ytrans = (y[0]+y[-1])/2
                T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
                T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
                T = np.dot(np.dot(T1,R),T2)
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
                
                if str(index)
                
                sufix = 
                 
                aux0.to_hdf('/mnt/mimer/lalc/results/correlations/west/phase_2/corr_phase2_d01.h5','corr',mode='a',
                            data_columns = aux0.columns,format='table', append = True)
                aux1.to_hdf('/mnt/mimer/lalc/results/correlations/west/phase_2/corr_phase2_d01.h5','L',mode='a',
                            data_columns = aux1.columns,format='table', append = True)
                            
                            
                # aux0.to_sql('corr', csv_database_r, if_exists='append',index_label = ['name','hms'])
                # aux1.to_sql('L', csv_database_r, if_exists='append',index_label = ['name','hms'])
 
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# corr = pd.read_hdf('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/corr_phase2.h5', key= 'corr')

# L = pd.read_hdf('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/corr_phase2.h5', key= 'L')
    
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################




# # In[Reindexing database]

# csv_database_r2 = create_engine('sqlite:///'+file_out_path+'/corr_uv_west_phase2_ind.db')
# chunk=1000000
# off = 0
# for i in range(300): 
#     dfcorr = pd.read_sql_query('select * from "corr" ' +' LIMIT ' + str(int(chunk)) +
#                                ' OFFSET ' + str(int(off)), csv_database_r)     
#     scans = dfcorr.scan.unique()
#     print(scans[-1])
#     if len(scans)==0:
#         break
#     for s in scans:
#         dfcorr.loc[dfcorr.scan==s,'time'] = np.repeat(np.unique(dfL_phase1.loc[dfL_phase1.scan == s].time.values),len(dfcorr.loc[dfcorr.scan==s].time))
#     timestp = pd.to_datetime(dfcorr.time.values).strftime("%Y%m%d").values
#     timestphms = pd.to_datetime(dfcorr.time.values).strftime("%H%M%S").values
#     dfcorr['name'] = timestp
#     dfcorr['hms'] = timestphms
    
#     dfcorr.set_index(['name','hms'], inplace=True)
#     dfcorr.to_sql('corr', csv_database_r2, if_exists='append',index_label = ['name','hms'])
#     off = off+chunk+1
    
# timestp = pd.to_datetime(dfL_phase1.time.values).strftime("%Y%m%d").values
# timestphms = pd.to_datetime(dfL_phase1.time.values).strftime("%H%M%S").values  
# dfL_phase1['name'] = timestp
# dfL_phase1['hms'] = timestphms  
# dfL_phase1.set_index(['name','hms'], inplace=True)      
# dfL_phase1.to_sql('L', csv_database_r2, if_exists='append',index_label = ['name','hms'])

# # In[Reindexing database filtered]
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
        
# # In[Correlations phase2 indexed]
# loc0 = np.array([6322832.3,0])
# loc1 = np.array([6327082.4,0])
# d = loc0-loc1  
  
# csv_database_r2 = create_engine('sqlite:///'+file_out_path+'/corr_uv_west_phase2_ind.db')
# csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0_phase2.db')
# csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1_phase2.db')     
# labels_short = np.array([ 'stop_time', 'azim'])
# for w,r in zip(labels_ws,labels_rg):
#     labels_short = np.concatenate((labels_short,np.array(['ws','range_gate'])))
# labels_short = np.concatenate((labels_short,np.array(['scan'])))   
# lim = [-8,-24]
# i=0
# col = 'SELECT '
# col_raw = 'SELECT '
# for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
#     if i == 0:
#         col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
#         col_raw = col_raw + w + ', '
#     elif (i == len(labels_ws)-1):
#         col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
#         col_raw = col_raw + ' ' + w
#     else:
#         col = col + ' ' + w + ', ' + r + ', ' + c + ', ' 
#         col_raw = col_raw + ' ' + w + ', '
#     i+=1

# selec_fil = col + ' FROM "table_fil"'
# selec_raw = col_raw + ' FROM "table_raw"'

# init = pd.read_sql_query('select name, hms, scan from "L" where scan = (select max(scan) from "L")', csv_database_r2)
# n_i, h_i, scan_i = init.name.values[0], init.hms.values[0], init.scan.values[0]
# chunk_scan = int(13*6)

# days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ind).values
# days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ind).values
# days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r2).values
# days_old = np.squeeze(days_old)
# days0 = np.squeeze(days0)
# days1 = np.squeeze(days1)
# days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])
# days = days[~np.isin(days,days_old)]
# switch = 0
# for dy in days:
#     print(dy)
#     query_fil = selec_fil+ ' where name = '+dy
#     query_raw = selec_raw+ ' where name = '+dy

#     df_0 = pd.read_sql_query(query_fil, csv_database_0_ind)
#     df = pd.read_sql_query(query_raw, csv_database_0_ind)
#     for i in range(198):
#         ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
#         df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#     df = None
#     df_0.drop(columns = labels_CNR,inplace=True)
#     df_0.columns = labels_short
    
#     day_r = pd.to_datetime(pd.read_sql_query('select time from L where name = '+
#                                               dy,csv_database_r2).time.values).strftime("%H%M%S").values   
#     if len(day_r)>0: 
#         s0 = df_0.scan.unique()
#         t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])                                         
#         ind = ~np.isin(pd.to_datetime(t0).strftime("%H%M%S").values,day_r)
#         s0 = s0[ind]
#         ind_df0 = df_0.scan.isin(s0)
#         df_0 = df_0.loc[ind_df0]
#         if np.sum(ind)==0:
#             print(dy,len(ind))
#             continue
        
#     df_1 = pd.read_sql_query(query_fil, csv_database_1_ind)
#     df = pd.read_sql_query(query_raw, csv_database_1_ind)
#     for i in range(198):
#         ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
#         df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#     df = None   
#     df_1.drop(columns = labels_CNR,inplace=True) 
#     df_1.columns = labels_short
    
#     #Synchronous?    
#     s_syn,_,_ = synch_df(df_0,df_1,dtscan=45/2)
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
#         t_1h = pd.date_range(tmin, tmax,freq='1H')#.strftime('%Y%m%d%H%M)
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
#                     r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
#                     r_1, phi_1 = np.meshgrid(r0, np.pi-np.radians(phi1)) # meshgrid                
#                     tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
#                     switch = 1 
#                 U_out, V_out, grd, su = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = 512)  
#                 if len(U_out)>0:
#                     areafrac = [np.sum(~np.isnan(U))/len(U.flatten()) for U in U_out]
#                     gamma = gamma_10m(df0,df1,U_out,V_out,su)#np.arctan2(V_mean,U_mean)   
#                     tau_out = []
#                     eta_out = []    
#                     r_u_out = []
#                     r_v_out = []
#                     r_uv_out = []
#                     cont=0
#                     for U,V in zip(U_out, V_out):
#                         tau,eta,r_u,r_v,r_uv,_,_,_,_ = sc.spatial_autocorr_sq(grd,U,V,
#                                                        gamma=gamma[cont], transform = False,
#                                                        transform_r = True,e_lim=.1,refine=32)
#                         tau_out.append(tau.astype(np.float32))
#                         eta_out.append(eta.astype(np.float32)) 
#                         r_u_out.append(r_u.astype(np.float32))
#                         r_v_out.append(r_v.astype(np.float32))
#                         r_uv_out.append(r_uv.astype(np.float32))
#                         cont=cont+1
#                     scan = np.array(su)[:,0]
#                     time = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in scan])    
#                     df0 = None
#                     df1 = None  
#                     r = np.vstack([np.c_[tau.flatten(),eta.flatten(),
#                                          r_u.flatten(),r_v.flatten(),r_uv.flatten(),
#                                          np.repeat(s,len(tau.flatten()))] for tau,eta,r_u,r_v,r_uv,s in zip(tau_out,eta_out,r_u_out,r_v_out,r_uv_out,scan)])
#                     t_stamp = np.hstack([np.repeat(t,len(tau.flatten())) for tau,t in zip(tau_out,time)])       
#                     l = np.vstack([np.r_[tot_L(r_u,r_v,tau,eta),af,s] for tau,eta,r_u,r_v,af,s in zip(tau_out,eta_out,r_u_out,r_v_out,areafrac,scan)])  
#                     U_out = None
#                     V_out = None  
#                     floats = pd.DataFrame(r[:,:5].astype(np.float32))
#                     floats.columns = ['tau','eta','r_u','r_v','r_uv']
#                     ints =  pd.DataFrame(r[:,5].astype(np.int32))
#                     ints.columns = ['scan']
#                     strings = pd.DataFrame(t_stamp)
#                     strings.columns = ['time']
#                     L = pd.DataFrame(l)
#                     L.columns = ['$L_{u,x}$', '$L_{u,y}$', '$L_{v,x}$', '$L_{v,y}$', 'area_frac', 'scan']
#                     t =  pd.DataFrame(time)
#                     t.columns = ['time']
#                     aux0 = pd.concat([floats,ints,strings],axis=1)
#                     aux1 = pd.concat([L,t],axis=1)
                    
#                     timestp = pd.to_datetime(aux0.time.values).strftime("%Y%m%d").values
#                     timestphms = pd.to_datetime(aux0.time.values).strftime("%H%M%S").values
#                     aux0['name'] = timestp
#                     aux0['hms'] = timestphms
#                     aux0.set_index(['name','hms'], inplace=True)
                    
#                     timestp = pd.to_datetime(aux1.time.values).strftime("%Y%m%d").values
#                     timestphms = pd.to_datetime(aux1.time.values).strftime("%H%M%S").values
#                     aux1['name'] = timestp
#                     aux1['hms'] = timestphms
#                     aux1.set_index(['name','hms'], inplace=True)
                    
#                     aux0.to_sql('corr', csv_database_r2, if_exists='append',index_label = ['name','hms'])
#                     aux1.to_sql('L', csv_database_r2, if_exists='append',index_label = ['name','hms'])
                    
# # In[Reliable scan phase 2]
# root = tkint.Tk()
# file_in_path_0_ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_in_path_1_ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_out_path_ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
# root.destroy()

# root = tkint.Tk()
# file_in_path_0_ph2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_in_path_1_ph2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_out_path_ph2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
# root.destroy()

# csv_database_r2_ph1 = create_engine('sqlite:///'+file_out_path_ph1+'/corr_uv_west_phase1_ind.db')
# csv_database_0_ind_ph1 = create_engine('sqlite:///'+file_in_path_0_ph1+'/raw_filt_0_phase1.db')
# csv_database_1_ind_ph1 = create_engine('sqlite:///'+file_in_path_1_ph1+'/raw_filt_1_phase1.db') 
# days_old_ph1 = pd.read_sql_query('select distinct name from "L"', csv_database_r2_ph1).values
# days_old_ph1 = np.squeeze(days_old_ph1)

# csv_database_r2_ph2 = create_engine('sqlite:///'+file_out_path_ph2+'/corr_uv_west_phase2_ind.db')
# csv_database_0_ind_ph2 = create_engine('sqlite:///'+file_in_path_0_ph2+'/raw_filt_0_phase2.db')
# csv_database_1_ind_ph2 = create_engine('sqlite:///'+file_in_path_1_ph2+'/raw_filt_1_phase2.db') 
# days_old_ph2 = pd.read_sql_query('select distinct name from "Lcorrected"', csv_database_r2_ph2).values
# days_old_ph2 = np.squeeze(days_old_ph2)

# col = 'SELECT '
# for i,c in enumerate(labels_CNR):
#     if i == 0:
#         col = col + 'stop_time,' + ' azim, ' + c + ', '
#     elif (i == len(labels_CNR)-1):
#         col = col + ' ' + c + ', scan'
#     else:
#         col = col + ' ' + c + ', ' 
#     i+=1

# selec_fil_rel = col + ' FROM "table_fil"'

# lab = ['stop_time', 'azim']+['CNR']*len(labels_CNR)+['scan']

# rel_scan_ph1 = []
                    
# for dy in days_old_ph1:
#     print(dy)
#     query_fil = selec_fil_rel+ ' where name = '+dy
#     df0 = pd.read_sql_query(query_fil, csv_database_0_ind_ph1)
#     df0.columns = lab    
#     scans0 = df0.scan.unique() 
#     df1 = pd.read_sql_query(query_fil, csv_database_1_ind_ph1)
#     df1.columns = lab    
#     scans1 = df1.scan.unique() 
#     s_syn,_,_ = synch_df(df0,df1,dtscan=45/2)
    
#     ind_df0 = df0.scan.isin(s_syn[:,0])
#     df0 = df0.loc[ind_df0]
#     ind_df1 = df1.scan.isin(s_syn[:,1])
#     df1 = df1.loc[ind_df1] 
    
#     chunk = len(df0.azim.unique())    
#     rt = df0.loc[df0.scan==s_syn[0,0]].CNR.values.flatten().shape[0]    
#     ind_cnr0 = np.sum(((df0.CNR >-24) & (df0.CNR < -8)).values, axis = 1)
#     ind_cnr1 = np.sum(((df1.CNR >-24) & (df1.CNR < -8)).values, axis = 1)
#     rel0 = ind_cnr0.reshape((int(len(ind_cnr0)/chunk),chunk)).sum(axis=1)/rt
#     rel1 = ind_cnr1.reshape((int(len(ind_cnr1)/chunk),chunk)).sum(axis=1)/rt
#     t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s_syn[:,0]])
#     t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s_syn[:,1]])
#     rel = np.array([np.min([r0,r1]) for r0,r1 in zip(rel0,rel1)])
#     rel_scan_ph1.append(np.c_[s_syn[:,0],t0,rel0,s_syn[:,1],t1,rel1,rel])

# relarray = np.vstack(rel_scan_ph1)
# timestp = pd.to_datetime(relarray[:,1]).strftime("%Y%m%d").values
# timestphms = pd.to_datetime(relarray[:,1]).strftime("%H%M%S").values   
# rel_df = pd.DataFrame(relarray)
# rel_df.columns = ['scan0','time0','rel0','scan1','time1','rel1','relscan']
# rel_df['name'] = timestp
# rel_df['hms'] = timestphms
# rel_df.set_index(['name','hms'], inplace=True) 
# rel_df.to_sql('reliable_scan', csv_database_r2_ph1, if_exists='append',index_label = ['name','hms'])  


# rel_scan_ph2 = []
                    
# for dy in days_old_ph2:
#     print(dy)
#     query_fil = selec_fil_rel+ ' where name = '+dy
#     df0 = pd.read_sql_query(query_fil, csv_database_0_ind_ph2)
#     df0.columns = lab    
#     scans0 = df0.scan.unique() 
#     df1 = pd.read_sql_query(query_fil, csv_database_1_ind_ph2)
#     df1.columns = lab    
#     scans1 = df1.scan.unique() 
#     s_syn,_,_ = synch_df(df0,df1,dtscan=45/2)
    
#     ind_df0 = df0.scan.isin(s_syn[:,0])
#     df0 = df0.loc[ind_df0]
#     ind_df1 = df1.scan.isin(s_syn[:,1])
#     df1 = df1.loc[ind_df1] 
    
#     chunk = len(df0.azim.unique())    
#     rt = df0.loc[df0.scan==s_syn[0,0]].CNR.values.flatten().shape[0]    
#     ind_cnr0 = np.sum(((df0.CNR >-24) & (df0.CNR < -8)).values, axis = 1)
#     ind_cnr1 = np.sum(((df1.CNR >-24) & (df1.CNR < -8)).values, axis = 1)
#     rel0 = ind_cnr0.reshape((int(len(ind_cnr0)/chunk),chunk)).sum(axis=1)/rt
#     rel1 = ind_cnr1.reshape((int(len(ind_cnr1)/chunk),chunk)).sum(axis=1)/rt
#     t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s_syn[:,0]])
#     t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s_syn[:,1]])
#     rel = np.array([np.min([r0,r1]) for r0,r1 in zip(rel0,rel1)])
#     rel_scan_ph2.append(np.c_[s_syn[:,0],t0,rel0,s_syn[:,1],t1,rel1,rel])

# relarray = np.vstack(rel_scan_ph2)
# timestp = pd.to_datetime(relarray[:,1]).strftime("%Y%m%d").values
# timestphms = pd.to_datetime(relarray[:,1]).strftime("%H%M%S").values   
# rel_df = pd.DataFrame(relarray)
# rel_df.columns = ['scan0','time0','rel0','scan1','time1','rel1','relscan']
# rel_df['name'] = timestp
# rel_df['hms'] = timestphms
# rel_df.set_index(['name','hms'], inplace=True) 
# rel_df.to_sql('reliable_scan', csv_database_r2_ph2, if_exists='append',index_label = ['name','hms'])  

# # In[Recover of L that are nan] 
# dfL_phase2 = pd.read_sql_query("""select * from "L" """,csv_database_r2)
# from scipy.spatial import Delaunay
# cond = dfL_phase2[['name','hms','time']].loc[dfL_phase2['$L_{u,x}$'].isna()]
# L = []
# col = pd.read_sql_query("""select * from "corr" limit 10""",csv_database_r2).columns
# colL = pd.read_sql_query("""select * from "L" limit 10""",csv_database_r2).columns

# selec = 'select ' +", ".join( col[2:8]) + ' from "corr" '
# cols = ['$L_{u,x}$' , '$L_{u,y}$' , '$L_{v,x}$' , '$L_{v,y}$']
# for i, c in enumerate(cond.values):
#     print(i)
#     print(c)
#     query = selec+ ' where (name, hms, time) = ' + '('+c[0]+', ' + c[1]+ ', "' + c[2] + '")'
#     dcorr = pd.read_sql_query(query,csv_database_r2)
#     if len(dcorr)>0:
#         dcorr = dcorr.loc[~dcorr.scan.isna()]
    
#         nt = len(dcorr.tau.unique())
#         ne = len(dcorr.eta.unique())
        
#         if len(dcorr) == nt*ne: 
#             tau = np.reshape(dcorr.tau.values,(ne,nt))
#             eta = np.reshape(dcorr.eta.values,(ne,nt))
#             ru = np.reshape(dcorr.r_u.values,(ne,nt))
#             rv = np.reshape(dcorr.r_v.values,(ne,nt))
#             indt = (dfL_phase2.name==c[0])&(dfL_phase2.hms==c[1])&(dfL_phase2.time==c[2])
#             dfL_phase2.loc[indt,cols] = tot_L(ru,rv,tau,eta)
#         else:
#             tau = dcorr.tau.values
#             eta = dcorr.eta.values
#             ru = dcorr.r_u.values
#             rv = dcorr.r_v.values
#             taug = dcorr.tau.unique()
#             etag = dcorr.eta.unique()
#             taug,etag = np.meshgrid(taug, etag)
            
#             ind = ~np.isnan(ru.flatten())
            
#             tri_tau = Delaunay(np.c_[tau.flatten()[ind],eta.flatten()[ind]])   
#             ru = sp.interpolate.CloughTocher2DInterpolator(tri_tau, ru.flatten()[ind])(np.c_[taug.flatten(),etag.flatten()])
#             rv = sp.interpolate.CloughTocher2DInterpolator(tri_tau, rv.flatten()[ind])(np.c_[taug.flatten(),etag.flatten()])
#             tau = taug
#             eta = etag
#             ru = np.reshape(ru,tau.shape)
#             rv = np.reshape(rv,tau.shape)
            
#             indt = (dfL_phase2.name==c[0])&(dfL_phase2.hms==c[1])&(dfL_phase2.time==c[2])
#             dfL_phase2.loc[indt,cols] = tot_L(ru,rv,tau,eta)
        
#         print(tot_L(ru,rv,tau,eta))
# dfL_phase2.set_index(['name','hms'], inplace=True)        
# dfL_phase2.to_sql('Lcorrected', csv_database_r2, if_exists='append',index_label = ['name','hms'])       

# # In[]

# root = tkint.Tk()
# file_out_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
# root.destroy()

# csv_database_r1 = create_engine('sqlite:///'+file_out_path_1+'/corr_uv_west_phase1_ind.db')

# # In[Reliable scan]
# root = tkint.Tk()
# file_in_path_0ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()
# root = tkint.Tk()
# file_in_path_1ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
# root.destroy()

# csv_database_0_indph1 = create_engine('sqlite:///'+file_in_path_0ph1+'/raw_filt_0_phase1.db')
# csv_database_1_indph1 = create_engine('sqlite:///'+file_in_path_1ph1+'/raw_filt_1_phase1.db')   

# days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_indph1).values
# days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_indph1).values
# days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r1).values
# days_old = np.squeeze(days_old)

# col = 'SELECT '
# for i,c in enumerate(labels_CNR):
#     if i == 0:
#         col = col + 'stop_time,' + ' azim, ' + c + ', '
#     elif (i == len(labels_CNR)-1):
#         col = col + ' ' + c + ', scan'
#     else:
#         col = col + ' ' + c + ', ' 
#     i+=1

# selec_fil_rel = col + ' FROM "table_fil"'

# lab = ['stop_time', 'azim']+['CNR']*len(labels_CNR)+['scan']

# rel_scan = []
                    
# for dy in days_old:
#     print(dy)
#     query_fil = selec_fil_rel+ ' where name = '+dy
#     df0 = pd.read_sql_query(query_fil, csv_database_0_indph1)
#     df0.columns = lab    
#     scans = df0.scan.unique()   
#     chunk = len(df0.azim.unique())    
#     rt = df0.loc[df0.scan==scans[0]].CNR.values.flatten().shape[0]    
#     ind_cnr = np.sum(((df0.CNR >-24) & (df0.CNR < -8)).values, axis = 1)
#     rel = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
#     t = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in scans])
#     rel_scan.append(np.c_[scans,t,rel])

# relarray = np.vstack(rel_scan)
# timestp = pd.to_datetime(relarray[:,1]).strftime("%Y%m%d").values
# timestphms = pd.to_datetime(relarray[:,1]).strftime("%H%M%S").values   
# rel_df = pd.DataFrame(relarray)
# rel_df.columns = ['scan','time','relscan']
# rel_df['name'] = timestp
# rel_df['hms'] = timestphms
# rel_df.set_index(['name','hms'], inplace=True) 
# rel_df.to_sql('relscan', csv_database_r1, if_exists='append',index_label = ['name','hms'])  

# # In[Plots, phase1]
# # In[]
# csv_database_r1 = create_engine('sqlite:///'+file_out_path_1+'/corr_uv_west_phase1_ind.db')
# csv_database_r2 = create_engine('sqlite:///'+file_out_path+'/corr_uv_west_phase2_ind.db')

# drel_phase2 = pd.read_sql_query("""select * from "relscan" """,csv_database_r2)
# dfL_phase2 = pd.read_sql_query("""select * from "Lcorr" """,csv_database_r2)
# rel2 = (drel_phase2.time.isin(dfL_phase2.time) & (drel_phase2.relscan>0.2)) & (dfL_phase2.area_frac>.8*dfL_phase2.area_frac.max())
# sns.jointplot(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase1.loc[rel2], kind="kde")
# sns.jointplot(x='$L_{v,x}$', y='$L_{v,y}$', data=dfL_phase1.loc[rel2], kind="kde")  

# drel_phase1 = pd.read_sql_query("""select * from "relscan" """,csv_database_r1)
# dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
# rel1 = (drel_phase1.time.isin(dfL_phase1.time) & (drel_phase1.relscan>0.2)) & (dfL_phase1.area_frac>.8*dfL_phase1.area_frac.max())
# sns.jointplot(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase1.loc[rel1], kind="kde")
# sns.jointplot(x='$L_{v,x}$', y='$L_{v,y}$', data=dfL_phase1.loc[rel1], kind="kde")
###########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
# In[]
######

#loc0 = np.array([6322832.3,0])
#loc1 = np.array([6327082.4,0])
#d = loc0-loc1
#date = datetime(1904, 1, 1)
#
#switch = 1
#
#csv_database_0 = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0.db')
#csv_database_1 = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1.db')  
##csv_databaseuv = create_engine('sqlite:///'+file_out_path+'/csv_database_UV.db')
#
#csv_database_r = create_engine('sqlite:///'+file_out_path+'/corr_uv_west_phase1.db')
#
#labels_short = np.array([ 'stop_time', 'azim'])
#for w,r in zip(labels_ws,labels_rg):
#    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate'])))
#labels_short = np.concatenate((labels_short,np.array(['scan'])))     
#
#col = 'SELECT '
#i = 0
#for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
#    if i == 0:
#        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
#    elif (i == len(labels_ws)-1):
#        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
#    else:
#        col = col + ' ' + w + ', ' + r + ',' + c + ', '        
#    i+=1
#   
#scann = int(13*6)#10*60/45)
#chunksize = 45*scann
#tot_scans = 55000
#tot_iter = int(tot_scans*45/chunksize)
#off0 = 812*45
#off1 = 812*45
#switch0 = 1
#switch1 = 1
#lim = [-24,-8]
#
#selec_fil = col + ' FROM "table_fil"'
#selec_raw = col + ' FROM "table_raw"'
#switch = 0
#for i in range(tot_iter):
#    print(off0/45,off1/45)
#    df_0 = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off0)), csv_database_0)
#    df = pd.read_sql_query(selec_raw+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off0)), csv_database_0)
#    for i in range(198):
#        ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
#        df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#    df = None
#    df_0.drop(columns = labels_CNR,inplace=True)
#    df_0.columns = labels_short
#    df_1 = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off1)), csv_database_1)
#    df = pd.read_sql_query(selec_raw+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off1)), csv_database_1)
#    for i in range(198):
#        ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
#        df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
#    df = None   
#    df_1.drop(columns = labels_CNR,inplace=True) 
#    df_1.columns = labels_short
#    #Synchronous?    
#    s_syn,off0,off1 = synch_df(df_0,df_1,dtscan=45)
#    print(s_syn)
#    if len(s_syn)>0:      
#        ind_df0 = df_0.scan.isin(s_syn[:,0])
#        df_0 = df_0.loc[ind_df0]
#        ind_df1 = df_1.scan.isin(s_syn[:,1])
#        df_1 = df_1.loc[ind_df1]       
#        if switch == 0:
#            phi0 = df_0.azim.unique()
#            phi1 = df_1.azim.unique()               
#            r0 = np.array(df_0.iloc[(df_0.azim==min(phi0)).nonzero()[0][0]].range_gate)
#            r1 = np.array(df_1.iloc[(df_1.azim==min(phi1)).nonzero()[0][0]].range_gate)                
#            r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
#            r_1, phi_1 = np.meshgrid(r0, np.pi-np.radians(phi1)) # meshgrid                
#            tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1),-d)
#            switch = 1 
#        U_out, V_out, grd, _ = wr.direct_wf_rec(df_0.astype(np.float32), df_1.astype(np.float32), tri, d,N_grid = 512) 
#        areafrac = [np.sum(~np.isnan(U))/len(U.flatten()) for U in U_out]
#        U_mean = np.nanmean(np.array(U_out))
#        V_mean = np.nanmean(np.array(V_out))
#        gamma = np.arctan2(V_mean,U_mean)
#        tau_out = []
#        eta_out = []    
#        r_u_out = []
#        r_v_out = []
#        r_uv_out = []  
#        if len(U_out)>0:
#            for U,V in zip(U_out, V_out):
#                tau,eta,r_u,r_v,r_uv,_,_,_ = sc.spatial_autocorr_sq(grd,U,V,
#                                               gamma=gamma, transform = False,
#                                               transform_r = True,e_lim=.1,refine=32)
#                tau_out.append(tau.astype(np.float32))
#                eta_out.append(eta.astype(np.float32)) 
#                r_u_out.append(r_u.astype(np.float32))
#                r_v_out.append(r_v.astype(np.float32))
#                r_uv_out.append(r_uv.astype(np.float32))
#            scan = df_0.scan.unique()
#            time = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in df_0.scan.unique()])    
#            df_0 = None
#            df_1 = None  
#            r = np.vstack([np.c_[tau.flatten(),eta.flatten(),
#                                 r_u.flatten(),r_v.flatten(),r_uv.flatten(),
#                                 np.repeat(s,len(tau.flatten()))] for tau,eta,r_u,r_v,r_uv,s in zip(tau_out,eta_out,r_u_out,r_v_out,r_uv_out,scan)])
#            t_stamp = np.hstack([np.repeat(t,len(tau.flatten())) for u,t in zip(tau_out,time)])       
#            l = np.vstack([np.r_[tot_L(r_u,r_v,tau,eta),af,s] for tau,eta,r_u,r_v,af,s in zip(tau_out,eta_out,r_u_out,r_v_out,areafrac,scan)])  
#            U_out = None
#            V_out = None  
#            floats = pd.DataFrame(r[:,:5].astype(np.float32))
#            floats.columns = ['tau','eta','r_u','r_v','r_uv']
#            ints =  pd.DataFrame(r[:,5].astype(np.int32))
#            ints.columns = ['scan']
#            strings = pd.DataFrame(t_stamp)
#            strings.columns = ['time']
#            L = pd.DataFrame(l)
#            L.columns = ['Lu_x','Lu_y','Lv_x','Lv_y','area_frac','scan']
#            t =  pd.DataFrame(time)
#            t.columns = ['time']
#            pd.concat([floats,ints,strings],axis=1).to_sql('corr', csv_database_r, if_exists='append',index = False)
#            pd.concat([L,t],axis=1).to_sql('L', csv_database_r, if_exists='append',index = False)
#            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            