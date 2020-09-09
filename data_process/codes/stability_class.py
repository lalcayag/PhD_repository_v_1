# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:35:45 2020

@author: lalc
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
from os.path import isfile, join, getsize
from os import listdir
import ppiscanprocess.windfieldrec as wr
import joblib
date = datetime(1904, 1, 1) 
import plotly.express as px
import seaborn as sns
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
# In[Functions]

def L_sm(L,ti,Llim=[],N=[]):
    L_s = np.ones(L.shape)
    date = datetime(1904, 1, 1, 0, 0)
    ind = (np.abs(L) <= Llim[1]) & (np.abs(L) >= Llim[0])
    time = np.array([str(int(n)) for n in ti])
    time = np.array([(datetime.strptime(t,"%Y%m%d%H%M")-date).total_seconds() for t in time])
    time = (time-time[0])/3600
    L_s[ind] = mov_con(np.r_[np.ones(N)*L[ind][0],L[ind], np.ones(N)*L[ind][-1]],N)[N:-N]
    L_s[~ind] = UnivariateSpline(time[ind],L_s[ind],k=3)(time[~ind])
    return (L_s,time)

def smoothing(res,hj,lim,N):
    resout = np.ones(res.shape)
    t,h,co = res.shape
    for i in range(h):
        for j in range(1,co-2): 
            print(j)
            resout[:,hj[i],j],time = L_sm(res[:,hj[i],j],res[:,hj[i],0],Llim=lim[j],N=N)
    resout[:,:,0] = res[:,:,0] 
    return (resout,time)

def phim(zL):
    phim = np.zeros(zL.shape)
    phim[zL>=0] = 1+4.8*zL[zL>=0]
    phim[zL<0] = (1-19.3*zL[zL<0])**(-.25)
    return phim

def mov_con(x,N):
    return np.convolve(x, np.ones((N,))/N,mode='same')

def L_smooth(flux, N=3):    
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

def kaiser_filter(x, f_s, fa, fb, width = .1, A_p = .01, A_s = 80):
    xm = np.mean(x)
    x = x-xm
    f_n = f_s/2
    pi=np.pi    
    Df = width
    w_a = 2*pi*fa/f_s
    w_b = 2*pi*fb/f_s
    delta_pass = (10**(A_p/20)-1)/(10**(A_p/20)+1)
    delta_stop = 10**(-A_s/20)
    delta = np.min([delta_pass,delta_stop])
    A = -20*np.log10(delta) 
    if A >= 50:
        alpha = 0.1102*(A-8.7)
    elif (A<50) & (A>21):
        alpha = 0.5842*(A-21)**0.4+0.07886*(A-21)
    else:
        alpha = 0    
    if A>21:
        D =  (A-7.95)/14.36
    else:
        D = 0.922
    N = 1+ 2*D/Df
    if N % 2 <1:
        N=np.ceil(N)
    else:
        N=np.ceil(N)+1
    print(delta_pass, delta_stop, A, D, N, fa, fb) 
    M = (N-1)/2
    n = np.arange(0,N)
    # Kaiser window
    w_n = np.i0(alpha*np.sqrt(n*(2*M-n))/M)/np.i0(alpha)  
    # Windowed impulse response    
    d_n_M = (np.sin(w_b*(n-M))-np.sin(w_a*(n-M)))/(pi*(n-M))    
    h_n = w_n*d_n_M
    h_n[np.isnan(h_n)]= (w_b/pi-w_a/pi)  
    ff = np.fft.fftshift(np.fft.fftfreq(len(h_n), d=1/f_s))
    H_w = np.fft.fftshift(np.abs(np.fft.fft(h_n)))[ff>0]
    ff = ff[ff>0]
    filt = np.convolve(x, h_n, mode = 'same')+xm
    return (filt,H_w,h_n,ff)


# In[Loading DataFrames, 10 min and scan]
Lcorr1,L10min1 = joblib.load('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/storeL.pkl')
Lcorr2,L10min2 = joblib.load('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/storeLph2.pkl')
Lcorr1.relscan = np.array(Lcorr1.relscan.values).astype(float)
Lcorr2.relscan = np.array(Lcorr2.relscan.values).astype(float)

# In[Obukov length corrected, 30min]
L_file1_30 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/L_ph1_30m.pkl'
L_file2_30 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/L_ph2_30m.pkl'
hj=[-1,-2,-3,-4, -5]
res_flux1_30 = joblib.load(L_file1_30)
res_flux2_30 = joblib.load(L_file2_30)
res_flux1_30 = np.array(res_flux1_30)
res_flux2_30= np.array(res_flux2_30)
heights = [241, 175, 103, 37, 7]
Llim = [(0,0),(0,40),(0,10),(0,10),(0,400),(0,10),(-10,10),(-10,10),(0,10),(-10,10),(0,10),(-10,10),
        (-100000,100000),(0,360),(0,360)]   

res_flux1_30_s, time1 = smoothing(res_flux1_30,hj,Llim,N=2) 
res_flux2_30_s, time2 = smoothing(res_flux2_30,hj,Llim,N=2) 

#np.c_[namei, U, V, W, TH, uu, uv, uw, vv, vw, ww, wth, L, D, angle, stat, flag]

L_list_1 = np.hstack([np.c_[(res_flux1_30_s[:,i,7]**2+res_flux1_30_s[:,i,9]**2)**.25,
                            res_flux1_30_s[:,i,1:]] for i in range(len(heights))])
L_list_2 = np.hstack([np.c_[(res_flux2_30_s[:,i,7]**2+res_flux2_30_s[:,i,9]**2)**.25,
                            res_flux2_30_s[:,i,1:]] for i in range(len(heights))])

t1 = pd.to_datetime([str(int(n)) for n in res_flux1_30[:,2,0]])
t2 = pd.to_datetime([str(int(n)) for n in res_flux2_30[:,2,0]])
    
cols = np.hstack([['$u_{star,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$', '$V_{'+str(int(h))+'}$',
                   '$W_{'+str(int(h))+'}$', '$TH_{'+str(int(h))+'}$',
         '$uu_{'+str(int(h))+'}$', '$uv_{'+str(int(h))+'}$',
         '$uw_{'+str(int(h))+'}$', '$vv_{'+str(int(h))+'}$', '$vw_{'+str(int(h))+'}$',
         '$ww_{'+str(int(h))+'}$', '$wth_{'+str(int(h))+'}$', '$L_{'+str(int(h))+'}$',
         '$D_{'+str(int(h))+'}$', '$angle_{'+str(int(h))+'}$', '$stat_{'+str(int(h))+'}$',
         '$flag_{'+str(int(h))+'}$'] for h in heights])

stab_phase1_30m = pd.DataFrame(columns = cols, data = L_list_1)
stab_phase1_30m['time'] = t1
stab_phase2_30m = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_30m['time'] = t2           

# In[]
col = ['L_ux', 'L_uy', 'L_vx', 'L_vy', 'L_hx', 'L_hy', 'Umean', 'Vmean','area_frac', 'relscan', 'time']
L30min1 = L10min1[col]
L30min1 = L30min1[~L30min1.index.duplicated(keep='first')]
L30min1.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L10min1.index.values]])
L30min1[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
Lph1 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
print(len(Lph1.index), len(L10min1.index), len(tL))
Lph1.index = L30min1.index
for i in range(len(t1)-1):
    print(t1[i],t1[i+1])
    ind = (L30min1.time>=t1[i]) & (L30min1.time<=t1[i+1])
    print(ind.sum())
    if ind.sum()>0:        
        aux = stab_phase1_30m[cols].loc[stab_phase1_30m.time==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
        print(ind.sum())

L30min1[cols] = Lph1[cols]
L30min1.sort_values(by=['time'], inplace = True)

col = ['L_ux', 'L_uy', 'L_vx', 'L_vy', 'L_hx', 'L_hy', 'Umean', 'Vmean','area_frac', 'relscan', 'time']
L30min2 = L10min2[col]
L30min2 = L30min2[~L30min2.index.duplicated(keep='first')]
L30min2.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L10min2.index.values]])
L30min2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
print(len(Lph2.index), len(L10min2.index), len(tL))
Lph2.index = L30min2.index
for i in range(len(t2)-1):
    print(t2[i],t2[i+1])
    ind = (L30min2.time>=t2[i]) & (L30min2.time<=t2[i+1])
    print(ind.sum())
    if ind.sum()>0:        
        aux = stab_phase2_30m[cols].loc[stab_phase2_30m.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values
        print(ind.sum())

L30min2[cols] = Lph2[cols]
L30min2.sort_values(by=['time'], inplace = True)

# In[]
################Direction in dataframes###################################### 
Sspeed_0 = ['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS']
Sdir_0 = ['Wdir_244m_LMS', 'Wdir_40m_LMS']
veldir_file = 'D:/PhD/Python Code/Balcony/data_process/results/vel_dir_ph1.pkl'
vel_direcph1 = joblib.load(veldir_file)
t1 = pd.to_datetime([str(int(n)) for n in vel_direcph1.index.values])
vel_direcph1['time'] = t1

L30min1 = L30min1[~L30min1.index.duplicated(keep='first')]
L30min1.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L30min1.index.values]])
L30min1[Sspeed_0+Sdir_0] = pd.DataFrame(columns=Sspeed_0+Sdir_0, data = np.nan*np.zeros((len(tL),len(Sspeed_0+Sdir_0))))
Lph1 = pd.DataFrame(columns=Sspeed_0+Sdir_0, data = np.nan*np.zeros((len(tL),len(Sspeed_0+Sdir_0))))
print(len(Lph1.index), len(L30min1.index), len(tL))
Lph1.index = L30min1.index
for i in range(len(t1)-1):
    print(t1[i],t1[i+1])
    ind = (L30min1.time>=t1[i]) & (L30min1.time<=t1[i+1])
    print(ind.sum())
    if ind.sum()>0:       
        aux = vel_direcph1[Sspeed_0+Sdir_0].loc[vel_direcph1.time==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
        print(ind.sum())
L30min1[Sspeed_0+Sdir_0] = Lph1[Sspeed_0+Sdir_0]
L30min1.sort_values(by=['time'], inplace = True)


veldir_file = 'D:/PhD/Python Code/Balcony/data_process/results/vel_dir_ph2.pkl'
vel_direcph2 = joblib.load(veldir_file)

t2 = pd.to_datetime([str(int(n)) for n in vel_direcph2.index.values])
vel_direcph2['time'] = t2
L30min2 = L30min2[~L30min2.index.duplicated(keep='first')]
L30min2.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L30min2.index.values]])
L30min2[Sspeed_0+Sdir_0] = pd.DataFrame(columns=Sspeed_0+Sdir_0, data = np.nan*np.zeros((len(tL),len(Sspeed_0+Sdir_0))))
Lph2 = pd.DataFrame(columns=Sspeed_0+Sdir_0, data = np.nan*np.zeros((len(tL),len(Sspeed_0+Sdir_0))))
print(len(Lph2.index), len(L30min2.index), len(tL))
Lph2.index = L30min2.index
for i in range(len(t2)-1):
    print(t2[i],t2[i+1])
    ind = (L30min2.time>=t2[i]) & (L30min2.time<=t2[i+1])
    print(ind.sum())
    if ind.sum()>0:       
        aux = vel_direcph2[Sspeed_0+Sdir_0].loc[vel_direcph2.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values
        print(ind.sum())

L30min2[Sspeed_0+Sdir_0] = Lph2[Sspeed_0+Sdir_0]
L30min2.sort_values(by=['time'], inplace = True)


# In[Loading DataFrames, 10 min and scan]
file1 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 1/West/L30min1.pkl'
file2 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 2/West/L30min2.pkl'
L30min1 = joblib.load(file1)
L30min2 = joblib.load(file2)

# In[Wind rose both phases]
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

veldir_file = 'D:/PhD/Python Code/Balcony/data_process/results/vel_dir_ph1.pkl'
veldir_file = 'D:/PhD/Python Code/Balcony/data_process/results/vel_dir_ph2.pkl'
vel_direcph1 = joblib.load(veldir_file)
vel_direcph2 = joblib.load(veldir_file)

i = 6
j = 1

wd = vel_direcph1.loc[~vel_direcph1[Sdir_0[j]].isna()][Sdir_0[j]].values
ws = vel_direcph1.loc[~vel_direcph1[Sspeed_0[i]].isna()][Sspeed_0[i]].values
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()

wd = vel_direcph2.loc[~vel_direcph2[Sdir_0[j]].isna()][Sdir_0[j]].values
ws = vel_direcph2.loc[~vel_direcph2[Sspeed_0[i]].isna()][Sspeed_0[i]].values
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()

wd = L30min1.loc[~L30min1[Sdir_0[j]].isna()][Sdir_0[j]].values
ws = L30min1.loc[~L30min1[Sspeed_0[i]].isna()][Sspeed_0[i]].values
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()

wd = L30min2.loc[~L30min2[Sdir_0[j]].isna()][Sdir_0[j]].values
ws = L30min2.loc[~L30min2[Sspeed_0[i]].isna()][Sspeed_0[i]].values
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()


# In[Richardson number and stability conditions phase 1]
s1=1
index = [0,1,2,3,4]
heights = np.array([241, 175, 103, 37, 7])
heights10 = np.array([244, 210, 178, 140, 106, 70, 40, 10])
Ls1 = np.array(['$L_{241}$', '$L_{175}$', '$L_{103}$','$L_{37}$', '$L_{7}$'])
us1 = np.array(['$u_{star,241}$', '$u_{star,175}$', '$u_{star,103}$','$u_{star,37}$', '$u_{star,7}$'])
Us1 = np.array(['$U_{241}$', '$U_{175}$', '$U_{103}$', '$U_{37}$','$U_{7}$'])
U10s_1 = np.array(['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS'])
heightsf1 = heights[index][::-1]
ks = 3
i = 6
j = 1
direction1 = L30min1.loc[~L30min1[Sdir_0[j]].isna()][Sdir_0[j]].values
speed1 = L30min1.loc[~L30min1[Sspeed_0[i]].isna()][Sspeed_0[i]].values
U1 = L30min1[Us1[index]].values
L1 = L30min1[Ls1[index]].values
ustar1 = L30min1[us1[index]].values
U10_1 = L30min1[U10s_1].values
U_prime1 = np.array([UnivariateSpline(heightsf1, U1[i,::-1], k=ks,s=s1)(heights) for i in range(U1.shape[0])])
U10_prime1 = np.array([UnivariateSpline(heights10[::-1], U10_1[i,::-1], k=4,s=100)(heights) for i in range(U1.shape[0])])
L_prime1 = np.array([UnivariateSpline(heightsf1, L1[i,::-1], k=ks)(heights) for i in range(U1.shape[0])])
u_star1 = np.array([UnivariateSpline(heightsf1, ustar1[i,::-1],k=ks,s=0)(heights) for i in range(U1.shape[0])])

dUdz1 = np.array([UnivariateSpline(heightsf1, U1[i,::-1],k=ks).derivative()(heights) for i in range(U1.shape[0])])
dU10dz1 = np.array([UnivariateSpline(heights10[::-1], U10_1[i,::-1],k=4).derivative()(heights) for i in range(U1.shape[0])])
k=.4
phi1 = k*heights[None,:]*dUdz1/u_star1
zL1 = heights[None,:]/L_prime1
zL1 = heights[None,:]/L1
Ri1 = zL1/phi1

###################################################################################################
# In[Richardson number and stability conditions phase 2]
s2 = 1
index = [0,1,2,3,4]
Ls2 = np.array(['$L_{241}$', '$L_{175}$', '$L_{103}$','$L_{37}$', '$L_{7}$'])
TH2 = np.array(['$TH_{241}$', '$TH_{175}$', '$TH_{103}$','$TH_{37}$', '$TH_{7}$'])
us2 = np.array(['$u_{star,241}$', '$u_{star,175}$', '$u_{star,103}$','$u_{star,37}$', '$u_{star,7}$'])
Us2 = np.array(['$U_{241}$', '$U_{175}$', '$U_{103}$', '$U_{37}$','$U_{7}$'])
U10s_2 = ['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS']
heightsf2 = heights[index][::-1]
ks = 3
i = 6
j = 1
direction2 = L30min2[Sdir_0[j]].values
speed2 = L30min2[Sspeed_0[i]].values
U2 = L30min2[Us2[index]].values
L2 = L30min2[Ls2[index]].values
ustar2 = L30min2[us2[index]].values
U10_2 = L30min2[U10s_2].values
U_prime2 = np.array([UnivariateSpline(heightsf2, U2[i,::-1], k=ks,s=s2)(heights) for i in range(U2.shape[0])])
U10_prime2 = np.array([UnivariateSpline(heights10[::-1], U10_2[i,::-1], k=4,s=s2)(heights) for i in range(U2.shape[0])])
L_prime2 = np.array([UnivariateSpline(heightsf2, L2[i,::-1], k=ks,s=s2)(heights) for i in range(U2.shape[0])])
u_star2 = np.array([UnivariateSpline(heightsf2, ustar2[i,::-1],k=ks,s=2)(heights) for i in range(U2.shape[0])])
dUdz2 = np.array([UnivariateSpline(heightsf2, U2[i,::-1],k=ks,s=s2).derivative()(heights) for i in range(U2.shape[0])])
dU10dz2 = np.array([UnivariateSpline(heights10[::-1], U10_2[i,::-1],k=4,s=s2).derivative()(heights) for i in range(U2.shape[0])])
k=.4
phi2 = k*heights[None,:]*dU10dz2/u_star2
# zL2 = heights[None,:]/L2
zL2 = heights[None,:]/L_prime2
Ri2= zL2/phi2

# In[]
j=-2
alpha_cen = 270
dalpha = 30
s1 = [4,50]
s2 = [4,50]
limL = 1000000
indd1 = ((direction1>=alpha_cen-dalpha) & (direction1<=alpha_cen+dalpha)) 
inds1 = (speed1>s1[0]) & (speed1<s1[1]) 
indp1 = np.abs(phi1[:,j])<4
indz1 = np.abs(zL1[:,j])<10
indL1 = np.abs(L_prime1[:,j])<limL
ind1 = indd1 & inds1 & indp1 & indz1 & indL1

indd2 = ((direction2>=alpha_cen-dalpha) & (direction2<=alpha_cen+dalpha)) 
inds2 = (speed2>s2[0]) & (speed2<s2[1]) 
indp2 = np.abs(phi2[:,j])<4
indz2 = np.abs(zL2[:,j])<10
indL2 = np.abs(L_prime2[:,j])<limL
ind2 = indd2 & inds2 & indp2 & indz2  & indL2

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.scatter(zL1[ind1,j],phi1[ind1,j],marker='o',facecolor='w',edgecolor='k',label='$Data$')
ax0.plot(np.sort(zL1[ind1,j]),phim(np.sort(zL1[ind1,j])),'-r', linewidth = 2,label='$Hogstrom$')
ax0.set_xlabel('$z/L$', fontsize=24)
ax0.set_ylabel('$\phi_m$', fontsize=24)
fig0.legend(fontsize=20,loc=[.6,.15])
ax0.set_xscale('log')
ax0.set_yscale('log')


ax0.set_xlim(-.4,.4)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.scatter(zL1[ind1,j],Ri1[ind1,j],marker='o',facecolor='w',edgecolor='k',label='$Data$')
ax1.plot(np.sort(zL1[ind1,j]),np.sort(zL1[ind1,j])/phim(np.sort(zL1[ind1,j])),'-r', linewidth = 2,label='$Hogstrom$')
ax1.set_xlabel('$z/L$', fontsize=24)
ax1.set_ylabel('$Ri_f$', fontsize=24)
#ax1.set_ylim(-2,.25)
fig1.legend(fontsize=20,loc=[.6,.15])



fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(zL2[ind2,j],phi2[ind2,j],marker='o',facecolor='w',edgecolor='k',label='$Data$')
ax2.plot(np.sort(zL2[ind2,j]),phim(np.sort(zL2[ind2,j])),'-r', linewidth = 2,label='$Hogstrom$')
ax2.set_xlabel('$z/L$', fontsize=24)
ax2.set_ylabel('$\phi_m$', fontsize=24)
fig2.legend(fontsize=20,loc=[.6,.15])
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.set_xlim(-.4,.4)

fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.scatter(zL2[ind2,j],Ri2[ind2,j],marker='o',facecolor='w',edgecolor='k',label='$Data$')
ax3.plot(np.sort(zL2[ind2,j]),np.sort(zL2[ind2,j])/phim(np.sort(zL2[ind2,j])),'-r', linewidth = 2,label='$Hogstrom$')
ax3.set_xlabel('$z/L$', fontsize=24)
ax3.set_ylabel('$Ri_f$', fontsize=24)
#ax1.set_ylim(-2,.25)
fig3.legend(fontsize=20,loc=[.6,.15])

# In[]

j=-2
alpha_cen = 270
dalpha = 30
s1 = [3,30]
s2 = [3,30]
limL = 5000
indd1 = ((direction1>=alpha_cen-dalpha) & (direction1<=alpha_cen+dalpha)) 
inds1 = (speed1>s1[0]) & (speed1<s1[1]) 
indp1 = np.abs(phi1[:,j])<10 
indz1 = np.abs(zL1[:,j])<1
indL1 = np.abs(L_prime1[:,j])<limL
ind1 = indd1 & inds1 & indp1 & indz1 & indL1

indd2 = ((direction2>=alpha_cen-dalpha) & (direction2<=alpha_cen+dalpha)) 
inds2 = (speed2>s2[0]) & (speed2<s2[1]) 
indp2 = np.abs(phi2[:,j])<10 
indz2 = np.abs(zL2[:,j])<1
indL2 = np.abs(L_prime2[:,j])<limL
ind2 = indd2 & inds2 & indp2 & indz2  & indL2

file_out_figures = 'C:/Users/lalc/Documents/Old Documents folder/PhD/Meetings/July 2020/'
file = ['U','UN','N','SN']  
limits = [[-np.inf,-.1], [-.1,-.01], [-.01,.01], [.01,.21], [.21,np.inf]]  
limits = [[-np.inf,-.1], [-.1,-.01], [-.01,.01], [.01,.21]]
     
relind = L30min1.relscan>.25
j = -2
for i,l in enumerate(limits):
    stabind = ((Ri1[:,j]>l[0]) & (Ri1[:,j]<l[1]))
    cols = np.r_[['$L_{u_1,x_1}$', '$L_{u_1,x_2}$','$L_{v_1,x_1}$', '$L_{v_1,x_2}$','$L_{h,x_1}$', '$L_{h,x_2}$'], L30min1.columns [6:]]
    L30min1.columns = cols
    xlim = 5*200
    ylim = 5*200
    g = sns.jointplot(x ='$L_{h,x_1}$', y = '$L_{h,x_2}$', data=L30min1.loc[relind & stabind & ind1], 
                            height = 8, kind="kde", cmap="jet", xlim = (0,xlim), ylim = (0,ylim),
                            color='k')#,cbar=True, cbar_kws={"format": formatter, "label": '$Density$'})
    g.set_axis_labels('$L_{h,x_1}$', '$L_{h,x_2}$', fontsize = 24)
    g.ax_joint.plot([0,xlim],[0,ylim],'--k', linewidth = 2)
    g.ax_joint.plot(L30min1.loc[relind & stabind & ind1]['$L_{h,x_1}$'].values,L30min1.loc[relind & stabind & ind1]['$L_{h,x_2}$'].values,'o', color = 'k', alpha=.2)
    g.ax_joint.text(100, 800,'$'+'%.2f' % l[0] +'<Ri_f<'+'%.2f' % l[1] +'$',fontsize=30,color='r')
    plt.tight_layout()
    plt.savefig(file_out_figures+file[i]+'_phase_1.png')


file = ['U','UN','N','SN','VS']       
relind = L30min2.relscan>.25
for i,l in enumerate(limits):
    stabind = ((Ri2[:,-2]>l[0]) & (Ri2[:,-2]<l[1]))
    cols = np.r_[['$L_{u_1,x_1}$', '$L_{u_1,x_2}$','$L_{v_1,x_1}$', '$L_{v_1,x_2}$','$L_{h,x_1}$', '$L_{h,x_2}$'], L30min2.columns [6:]]
    L30min2.columns = cols
    xlim = 5*200
    ylim = 5*200
    g = sns.jointplot(x ='$L_{h,x_1}$', y = '$L_{h,x_2}$', data=L30min2.loc[relind & stabind & ind2], 
                            height = 8, kind="kde", cmap="jet", xlim = (0,xlim), ylim = (0,ylim),
                            color='k')#,cbar=True, cbar_kws={"format": formatter, "label": '$Density$'})
    g.set_axis_labels('$L_{h,x_1}$', '$L_{h,x_2}$', fontsize = 24)
    g.ax_joint.plot([0,xlim],[0,ylim],'--k', linewidth = 2)
    g.ax_joint.plot(L30min2.loc[relind & stabind & ind2]['$L_{h,x_1}$'].values,L30min2.loc[relind & stabind & ind2]['$L_{h,x_2}$'].values,'o', color = 'k', alpha=.2)
    # g.ax_joint.plot(L10min1.loc[relind & stabind]['$L_{h,x_1}$'].values,L10min1.loc[relind & stabind]['$L_{h,x_2}$'].values,'+', color = 'r', alpha=.2)
    g.ax_joint.text(100, 800,'$'+'%.2f' % l[0] +'<Ri_f<'+'%.2f' % l[1] +'$',fontsize=30,color='r')
    plt.tight_layout()
    plt.savefig(file_out_figures+file[i]+'_phase_2.png')

# In[Richarson number in dataframes]

Ric = np.array(['$Ri_{241}$', '$Ri_{175}$', '$Ri_{103}$','$Ri_{37}$', '$Ri_{7}$'])
zL = np.array(['$zL_{241}$', '$zL_{175}$', '$zL_{103}$','$zL_{37}$', '$zL_{7}$'])

ri1 = pd.DataFrame(columns=Ric, data = Ri1)
ri1.index = L30min1.index
zL1 = pd.DataFrame(columns=zL, data = zL1)
zL1.index = L30min1.index
L30min1[Ric] = ri1
L30min1[zL] = zL1
# L30min1[L] = L_prime1

ri2 = pd.DataFrame(columns=Ric, data = Ri2)
ri2.index = L30min2.index
zL2 = pd.DataFrame(columns=zL, data = zL2)
zL2.index = L30min2.index
L30min2[Ric] = ri2
L30min2[zL] = zL2

# In[Saving DataFrames, 30 min and scan]
file1 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 1/West/L30min1.pkl'
file2 = 'D:/PhD/Python Code/Balcony/data_process/results/correlations/Phase 2/West/L30min2.pkl'
joblib.dump(L30min1, file1)
joblib.dump(L30min2, file2)

# In[Clustering of classes]
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KDTree
from scipy.signal import find_peaks
from sklearn import mixture
import matplotlib as mpl
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
nn=4
Ls1 = np.array(['$L_{241}$', '$L_{175}$', '$L_{103}$','$L_{37}$', '$L_{7}$'])
us1 = np.array(['$u_{star,241}$', '$u_{star,175}$', '$u_{star,103}$','$u_{star,37}$', '$u_{star,7}$'])
Us1 = np.array(['$U_{241}$', '$U_{175}$', '$U_{103}$', '$U_{37}$','$U_{7}$'])
U10s_1 = np.array(['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS'])
wths1 = np.array(['$wth_{241}$', '$wth_{175}$', '$wth_{103}$','$wth_{37}$', '$wth_{7}$'])
Ths1 = np.array(['$TH_{241}$', '$TH_{175}$', '$TH_{103}$','$TH_{37}$', '$TH_{7}$'])

X = 

X = L30min1[np.r_[Us1,us1,wths1]].values
X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
tree_X = KDTree(X)
d,i = tree_X.query(tree_X.data,k = nn) 
d=d[:,-1]
#non-zero values
d = d[d>0]
d = np.log(np.sort(d))
plt.plot(np.sort(d))
# x axis (point label)
l = np.arange(0,len(d)) 
rsmpl = 200
# Down sampling to speed up calculations
# same with point lables
l_resample = np.unique(np.r_[l[::int(len(l)/rsmpl)],l[-1]])
#    print(l_resample,l_resample.shape)
d_resample = d[l_resample]
# Cubic spline of resampled k-distances, lower memory usage and higher calculation speed.
#spl = UnivariateSpline(l_resample, d_resample,s=0.5)
std=.001*np.ones_like(d_resample)
# Changes in slope in the sorted, log transformed, k-distance graph
t = np.arange(l_resample.shape[0])    
fx = sp.interpolate.UnivariateSpline(t, l_resample/(-l_resample[0]+l_resample[-1]), k=4, w=1/std)
fy = sp.interpolate.UnivariateSpline(t, d_resample/(-d_resample[0]+d_resample[-1]), k=4, w=1/std)
x_1prime = fx.derivative(1)(t)
x_2prime = fx.derivative(2)(t)
y_1prime = fy.derivative(1)(t)
y_2prime = fy.derivative(2)(t) 
kappa = (x_1prime* y_2prime - y_1prime* x_2prime) / np.power(x_1prime**2 + y_1prime**2, 1.5)
ind_kappa, _ = find_peaks(kappa,prominence=1) 
# Just after half of the graph
ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
# The first knee...
l1 = l_resample[ind_kappa][0]
# the corresponding eps distance
eps0 = np.exp(d[l1])

outlier_detection = DBSCAN(min_samples = nn, eps = eps0)
clusters1 = outlier_detection.fit_predict(X)
plt.figure()
plt.hist(clusters1,bins=100)









