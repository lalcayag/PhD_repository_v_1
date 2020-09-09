# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:39:47 2020

@author: lalc
"""
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import tkinter as tkint
import tkinter.filedialog
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import datetime
from datetime import datetime, timedelta
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
import spectralfitting.spectralfitting as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[Functions for stability]
# Richarson number
# Bulk Richardson number
# Ri(zm) = ((g/theta)*(Delta z_m)/(Delta U)**2)ln(z_2/z_1)
def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])
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
    
def stability_grad(t0, t1, x0, x1, y0, y1, z0, z1, p0, p1):
    u0 = np.sqrt(x0**2+y0**2)
    u1 = np.sqrt(x1**2+y1**2)
    theta0 = t0*(p0/1000)**.286
    theta1 = t1*(p1/1000)**.286
    dtheta = theta1 - theta0
    du = u1 - u0
    Ri = np.zeros(t0.shape)*np.nan
    L = np.zeros(t0.shape)*np.nan
    stab = np.zeros(t0.shape)*np.nan
    Ri = 9.806*dtheta*.5*(z1+z0)*np.log(z1/z0)/du**2/theta0
    Ri[Ri>0.2] = 0.2
    indRi = Ri <= 0
    L[indRi] = .5*(z1+z0)/Ri[indRi]
    L[L<-500] = -500
    indRi = (Ri > 0) & (Ri < .2)
    L[indRi] = .5*(z1+z0)*(1-5*Ri[indRi])/Ri[indRi]
    L[L>500] = 500
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
    return np.c_[Ri , L, stab]
 
def stability_bulk(t0, t1, x1, y1, z0, z1, p0, p1, zf = .1):    
    u1 = np.sqrt(x1**2+y1**2)
    theta0 = t0*(p0/1000)**.286
    theta1 = t1*(p1/1000)**.286
    dtheta = theta1 - theta0
    Ri = np.zeros(t0.shape)*np.nan
    # Initial guess
    L = np.zeros(t0.shape)*np.nan
    stab = np.zeros(t0.shape)*np.nan
    Ri = (z1*9.806*dtheta/theta1)/u1**2
    L = np.ones(Ri.shape)*z1
    from scipy.optimize import fsolve 
    for i in range(len(L)):
        func = lambda l: np.abs(l-z1/(Ri[i]*F_G(l,z1,z0,zf=.1)))
        L[i] =  fsolve(func, L[i])
    L[np.abs(L)>=500] = 500    
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
    return np.c_[Ri, L, stab] 

def F_G(L,z1,z0,zf):
    ef = (1-16*zf/L)**.25
    e1 = (1-16*z1/L)**.25
    l0 = (1-16*z0/L)**.5
    l1 = (1-16*z1/L)**.5
    indL = L<0
    F = np.zeros(L.shape)*np.nan
    F[indL] = (np.log((z1/zf)*((ef[indL]**2+1)/(e1[indL]**2+1))*((ef[indL]+1)/(e1[indL]+1))**2)+
                     2*np.arctan((ef[indL]-e1[indL])/(1+ef[indL]*e1[indL])))
    F[~indL] = (np.log((z1/zf))+5*z1/L[~indL])
    indL = L<=0
    G = np.zeros(L.shape)*np.nan
    G[indL] = (np.log((z1/z0)*((l0[indL]+1)/(l1[indL]+1))**2)) 
    G[~indL] = (np.log(z1/z0) + 5*(z1-z0)/L[~indL])
    return F**2/G        

def stability_flux(t0,t1, x0, y0, x1, y1, w1, z0, z1, p0, p1):
    #this will 10 Hz data
    theta_mean = .5*(np.nanmean(t0*(p0/1000)**.286)+np.nanmean(t1*(p1/1000)**.286))
    theta1 = t1*(p1/1000)**.286
    u0 = np.sqrt(x0**2+y0**2)
    U0 = np.nanmean(u0)
    w1 = w1-np.nanmean(w1)
    u1 = np.sqrt(x1**2+y1**2)
    U1 = np.nanmean(u1)
    zm = .5*(z1+z0)
    dudz = (U1-U0)/zm/np.log(z1/z0)
    wth = np.nanmean(w1*(theta1-np.nanmean(theta1)))
    uw = np.nanmean((u1-U1)*w1)
    Ri = 9.806*wth/(theta_mean*uw*dudz)
    L = zm/Ri
    stab = np.nan
    if (L>0) & (L<=100):
        stab = 0
    if (L>100) & (L<500):
        stab = 1
    if np.abs(L)>500:
        stab = 2
    if (L>-500) & (L<-100):
        stab = 3
    if (L>-100) & (L<0):
        stab = 4  
    return np.c_[Ri, L, stab,theta_mean,dudz,uw,wth] 

def valid_col(df_stab,cols):
    indextot=[]
    for j, c in enumerate(cols):
        index = []
        for i,cc in enumerate(c):
            value = df_stab[cc].values
            if np.sum(value == None)==0:
                index.append(i)
        indextot.append(index)    
    return indextot

def obukhov_L(t, x, y, w, z, p, name):
    theta = t*(p/1000)**.286
    X = np.nanmean(x)
    Y = np.nanmean(y)
    angle = np.arctan2(Y,X)
    # Components in matrix of coefficients
    S11 = np.cos(angle)
    S12 = np.sin(angle)
    R = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[x.flatten(),y.flatten()]).T
    vel = np.dot(R,vel)
    u = vel[0,:]
    v = vel[1,:]
    w = w-np.nanmean(w)
    u_fric = (np.nanmean(u*w)**2+np.nanmean(v*w)**2)**.25
    wth = np.nanmean(w*(theta-np.nanmean(theta)))
    k = .4
    g = 9.806
    L = -u_fric**3*np.nanmean(theta)/(k*g*wth)
    return [int(name),np.nanmean(u),u_fric,wth,L,np.nanmean(theta),z]

def fluxes(t, x, y, w, z, p, name):
    th = (t+273.15)*(p/1000)**.286
    X = np.nanmean(x)
    Y = np.nanmean(y)
    angle = np.arctan2(Y,X)
    # Components in matrix of coefficients
    S11 = np.cos(angle)
    S12 = np.sin(angle)
    R = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[x.flatten(),y.flatten()]).T
    vel = np.dot(R,vel)
    u = vel[0,:]
    v = vel[1,:]
    U = np.nanmean(u)
    V = np.nanmean(v)
    W = np.nanmean(w)
    T = np.nanmean(th)   
    u  = u-U
    v  = v-V
    w  = w-W
    th = th-T    
    uu  = np.nanmean(u*u)
    uv  = np.nanmean(u*v)
    uw  = np.nanmean(u*w)
    vv  = np.nanmean(v*v)
    vw  = np.nanmean(v*w)
    ww  = np.nanmean(w*w)
    wth = np.nanmean(w*th)   
    return [int(name), U, V, W, T, uu, uv, uw, vv, vw, ww, wth, z]

def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)

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

# In[ ]
file_in_path_corr1 = 'E:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 1/West'
file_in_path_corr2 = 'E:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 2/West'

file_in_path_0 = 'E:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 1/west/raw_filt_0_phase1.db'
file_in_path_1 = 'E:/PhD/Python Code/Balcony/data_process/results/filtering/masks/Phase 1/west/raw_filt_1_phase1.db'


# In[]

def autocorr_interp_sq(r, eta, tau, N = [], tau_lin = [], eta_lin = []):
    if (len(eta_lin) == 0) | (len(eta_lin) == 0):
        if len(N) == 0:
            N = 2**(int(np.ceil(np.log(np.max([tau.shape[1],eta.shape[0]]))/np.log(2)))+3)
        tau_lin = np.linspace(np.min(tau.flatten()),np.max(tau.flatten()),N)
        eta_lin = np.linspace(np.min(eta.flatten()),np.max(eta.flatten()),N)
        tau_lin, eta_lin = np.meshgrid(tau_lin,eta_lin)
    ind = ~np.isnan(r.flatten())
    tri_tau = Delaunay(np.c_[tau.flatten()[ind],eta.flatten()[ind]])   
    r_int = sp.interpolate.CloughTocher2DInterpolator(tri_tau, r.flatten()[ind])(np.c_[tau_lin.flatten(),eta_lin.flatten()])
    r_int[np.isnan(r_int)] = 0.0
    return (tau_lin,eta_lin,np.reshape(r_int,tau_lin.shape))

csv_database_r1 = create_engine('sqlite:///'+file_in_path_corr1+'/corr_uv_west_phase1_ind.db')
csv_database_r2 = create_engine('sqlite:///'+file_in_path_corr2+'/corr_uv_west_phase2_ind.db')

drel_phase1 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r1)
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r2)
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)


# L1 = dfL_phase1[['$L_{u,x}$', '$L_{u,y}$']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))].mean()
# L2 = dfL_phase2[['$L_{u,x}$', '$L_{u,y}$']].loc[dfL_phase2['name']=='20160806'].loc[dfL_phase2['scan'].isin(np.squeeze(scan2))]


# L_1 = dfL_phase1[['$L_{flux,103}$']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))]
# L_2 = dfL_phase2[['$L_{flux,103}$']].loc[dfL_phase2['name']=='20160806'].loc[dfL_phase2['scan'].isin(np.squeeze(scan2))]

scl = dfL_phase2[['scan']].loc[(dfL_phase2['$L_{u,x}$']>800) &(dfL_phase2['$L_{u,x}$']<900)].loc[dfL_phase2['name']=='20160806']


hms = drel_phase1[['scan0','hms']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))]
df_corr_phase1 = pd.read_sql_query("select * from 'corr' where name = '20160421' and hms >= '" +                                   
                                   hms.hms.min() + "' and hms <= '" +  hms.hms.max()+"'",csv_database_r1)


hms = drel_phase2[['scan0','hms']].loc[drel_phase2['name']=='20160806'].loc[drel_phase2['scan0'].isin(np.squeeze(scan2))]
df_corr_phase2 = pd.read_sql_query("select * from 'corr' where name = '20160806' and hms >= '" +                                   
                                   hms.hms.min() + "' and hms <= '" +  hms.hms.max()+"'",csv_database_r2)

# In[]

with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph1.pkl', 'rb') as reader:
    res_flux_1 = pickle.load(reader)   
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)      

drel_phase1 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r1)
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
dfL_phase1.dropna(inplace=True)
dfL_phase1.reset_index(inplace=True)
dfL_phase1.drop(['index'],axis=1,inplace=True)
dfL_phase1 = dfL_phase1.drop_duplicates(subset='time', keep='first')


drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r2)
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)
dfL_phase2.dropna(inplace=True)
dfL_phase2.reset_index(inplace=True)
dfL_phase2.drop(['index'],axis=1,inplace=True)
dfL_phase2 = dfL_phase2.drop_duplicates(subset='time', keep='first')

heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
    
L_list_1 = np.hstack([L_smooth(np.array(res_flux_1)[:,i,:]) for i in range(len(heights))])
L_list_2 = np.hstack([L_smooth(np.array(res_flux_2)[:,i,:]) for i in range(len(heights))])
t1 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_1)[:,2,0]])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_2)[:,2,0]])
    
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]

cols = [item for sublist in cols for item in sublist]

stab_phase1_df = pd.DataFrame(columns = cols, data = L_list_1)
stab_phase1_df['time'] = t1

stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2

dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
dfL_phase1[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))

Lph1 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph1.index = dfL_phase1.index
Lph2.index = dfL_phase2.index

aux1 = dfL_phase1.time
aux2 = dfL_phase2.time

dfL_phase1.time = pd.to_datetime(dfL_phase1.time)
dfL_phase2.time = pd.to_datetime(dfL_phase2.time)

for i in range(len(t1)-1):
    print(i,t1[i])
    ind = (dfL_phase1.time>=t1[i]) & (dfL_phase1.time<=t1[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase1_df[cols].loc[stab_phase1_df.time==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
    
for i in range(len(t2)-1):
    print(i,t2[i])
    ind = (dfL_phase2.time>=t2[i]) & (dfL_phase2.time<=t2[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values

dfL_phase1.time = aux1 
dfL_phase2.time = aux2

# In[Histograms of L and U]

dfL_phase1[cols] = Lph1
dfL_phase2[cols] = Lph2
dfL_phase1.sort_values(by=['time'], inplace = True)
dfL_phase2.sort_values(by=['time'], inplace = True)

ind1 = drel_phase1.time0.isin(dfL_phase1.time)
timein = pd.concat([drel_phase1.time0.loc[ind1],drel_phase1.relscan.loc[ind1]],axis = 1)
timein.index = dfL_phase1.index
dfL_phase1[['time0','rel']] = timein

ind1 = drel_phase2.time0.isin(dfL_phase2.time)
timein = pd.concat([drel_phase2.time0.loc[ind1],drel_phase2.relscan.loc[ind1]],axis = 1)
timein.index = dfL_phase2.index
dfL_phase2[['time0','rel']] = timein

colL = '$L_{flux,103}$'
fig,ax = plt.subplots(figsize=(8,8))
ax.hist(1/(dfL_phase1[colL].loc[dfL_phase1[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'k', label = '$Phase\:1$')
ax.hist(1/(dfL_phase2[colL].loc[dfL_phase2[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'r', label = '$Phase\:1$')
ax.set_xlabel('$1/L\:[1/m]$', fontsize=24)
ax.set_ylabel('$Frequency$', fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(labelsize=24)
fig.tight_layout()

colL = '$L_{flux,103}$'
fig,ax = plt.subplots(figsize=(8,8))
ax.hist(1/(dfL_phase1[colL].loc[dfL_phase1[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'k', label = '$Phase\:1$')
ax.hist(1/(dfL_phase2[colL].loc[dfL_phase2[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'r', label = '$Phase\:2$')
ax.set_xlabel('$1/L\:[1/m]$', fontsize=24)
ax.set_ylabel('$Frequency$', fontsize=24)
ax.text(-.095, 7000,'(a)',fontsize=30,color='k')
#ax.legend(fontsize=24)
ax.tick_params(labelsize=24)
fig.tight_layout()

colL = '$L_{flux,103}$'
fig,ax = plt.subplots(figsize=(8,8))
colU = '$U_{103}$'
ax.hist((dfL_phase1[colU].loc[dfL_phase1[colL].abs()>500].values),bins=50, histtype = 'step',lw = 2, color = 'k', label = '$Phase\:1\:103\:[m]$')
colU = '$U_{241}$'
ax.hist((dfL_phase2[colU].loc[dfL_phase2[colL].abs()>500].values),bins=50, histtype = 'step',lw = 2, color = 'r', label = '$Phase\:2,\:241\:[m]$')
ax.set_xlabel('$U\:[m/s]$', fontsize=24)
ax.set_ylabel('$Frequency$', fontsize=24)
ax.text(1, 450,'(b)',fontsize=30,color='k')
#ax.legend(fontsize=24)
ax.tick_params(labelsize=24)
fig.tight_layout()

############FIgures in the paper

colL = '$L_{flux,103}$'
fig,ax = plt.subplots(figsize=(8,8))
ax.hist(1/(stab_phase1_df[colL].loc[stab_phase1_df[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'k', label = '$Phase\:1$')
ax.hist(1/(stab_phase2_df[colL].loc[stab_phase2_df[colL].abs()>10].values),bins=50, histtype = 'step',lw = 2, color = 'r', label = '$Phase\:2$')
ax.set_xlabel('$1/L\:[1/m]$', fontsize=24)
ax.set_ylabel('$Counts$', fontsize=24)
ax.text(-.095, 1625,'(a)',fontsize=30,color='k')
ax.legend(fontsize=24)
ax.tick_params(labelsize=24)
fig.tight_layout()

colL = '$L_{flux,103}$'
fig,ax = plt.subplots(figsize=(8,8))
colU = '$U_{103}$'
ax.hist((stab_phase1_df[colU].loc[stab_phase1_df[colL].abs()>500].values),bins=50, histtype = 'step',lw = 2, color = 'k', label = '$103\:[m]$')
colU = '$U_{241}$'
ax.hist((stab_phase2_df[colU].loc[stab_phase2_df[colL].abs()>500].values),bins=50, histtype = 'step',lw = 2, color = 'r', label = '$241\:[m]$')
ax.set_xlabel('$U\:[m/s]$', fontsize=24)
ax.set_ylabel('$Counts$', fontsize=24)
ax.text(1.5, 87,'(b)',fontsize=30,color='k')
ax.legend(fontsize=24)
ax.tick_params(labelsize=24)
fig.tight_layout()


# In[]

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

dy = '20160421'
######################################
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

query_fil_0 = selec_fil+ ' where name = ' + dy

csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0)
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1)  

df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ind)
df_0.columns = labels_short
df_1 = pd.read_sql_query(query_fil_0, csv_database_1_ind)
df_1.columns = labels_short

date = datetime(1904, 1, 1) 

loc0 = np.array([0,6322832.3])
loc1 = np.array([0,6327082.4])
d = loc1-loc0  
switch = 0
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
            print(t_scan[0])  
            uo, vo, grdu, so, r_i_0, phi_i_0, r_i_1, phi_i_1, mask = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = chunk)

r_i = np.sqrt(grdu[0]**2 + grdu[1]**2)
phi_i = np.arctan2(grdu[1],grdu[0])


ind = tri.get_trifinder()(grdu[0].flatten(), grdu[1].flatten())!=-1


r_i_0, phi_i_0 = wr.translationpolargrid((r_i, phi_i),-d/2)
phi_i_0 = np.pi*0.5-phi_i_0
phi_i_0 = np.where(phi_i_0<0 , 2*np.pi+phi_i_0, phi_i_0)
r_i_1, phi_i_1 = wr.translationpolargrid((r_i, phi_i), d/2)
phi_i_1 = np.pi*0.5-phi_i_1
phi_i_1 = np.where(phi_i_1<0 , 2*np.pi+phi_i_1, phi_i_1)
plt.figure()
plt.contourf(grdu[0], grdu[1], phi_i_0*180/np.pi , 10, cmap='jet')
plt.colorbar()
plt.figure()
plt.contourf(grdu[0], grdu[1], phi_i_1*180/np.pi , 10, cmap='jet')
plt.colorbar()


# Angle of lidars

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], np.sin(phi_i_1-phi_i_0), 20, cmap='jet')
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax0)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\sin{(\theta_1-\theta_2)}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.tight_layout()
fig0.tight_layout()
ax0.set_xlim(-7000,0)
ax0.set_ylim(-3500,3500)
ax0.triplot(tri, lw=1, color='k')
fig0.tight_layout()

