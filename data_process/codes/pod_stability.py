# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:59:11 2020

@author: lalc

POD

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

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, rand

# In[pod function]

def mov_con(x,N):
    return np.convolve(x, np.ones((N,))/N,mode='same')

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
    

def pod_tot(grdu, ulist,vlist,N = 100):
    print('interp')
    ut,vt,_ = interp(grdu,ulist,vlist,N=100)
    # utmean = np.mean(ut,axis=0)
    # ut= ut-utmean[None,:]
    # chunk = ulist[0].shape[1]   
    # Ulist = [ulist[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    # Vlist = [vlist[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    # x = grdu[0][0,:]
    # y = grdu[1][:,0]
    # U_arr_pod = np.vstack([Ulist[i] for i in range(len(Ulist))])
    # V_arr_pod = np.vstack([Vlist[i] for i in range(len(Vlist))])
    # maskpod = ~np.isnan(np.mean(U_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0))
    # ut = np.vstack([Ulist[i][maskpod] for i in range(len(Ulist))])
    # vt = np.vstack([Vlist[i][maskpod] for i in range(len(Ulist))])
    # print(np.sum(np.isnan(ut)))
    print('pod')
    X = np.c_[ut, vt]
    L, S, R = np.linalg.svd(X/np.sqrt(X.shape[0]-1))
    print(L.shape, S.shape, R.shape)
    return (R.T, X.dot(R.T), S**2,maskpod)
###################################################
def pod(ut,vt):
    print('pod')
    X = np.c_[ut, vt]
    L, S, R = np.linalg.svd(X/np.sqrt(X.shape[0]-1))
    print(L.shape, S.shape, R.shape)
    return (R.T, X.dot(R.T), S**2)
###################################################


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def spod_tot(ut,vt,N = 100):
    
    
    X = np.c_[ut2, vt2]       
    R = X.dot(X.T)/(2*np.sum(maskpod))
    S = R.copy()
    sigma = 10
    for o in range(-R.shape[0]+1,R.shape[0]):
        print(o)
        r = np.diagonal(R,offset=o) 
        w = sp.signal.gaussian(len(r),sigma)
        w = w/np.sum(w)
        rs = sp.ndimage.convolve1d(r,weights=w,mode='nearest')
        rows,columns = kth_diag_indices(R, o)
        S[rows,columns] = rs
 
    lam, A = np.linalg.eig(R)
    mui, bi = np.linalg.eig(S)
    
    plt.figure()
    plt.plot(np.sort(lam)[::-1],'-o')
    plt.plot(np.sort(mui)[::-1],'-o')
    
    plt.figure()  
    plt.plot(np.cumsum(np.sort(lamS)[::-1])/np.sum(lamS),'-o')
       
    phi_i = X.T.dot(bi[:,0])   
    
    U_k= U_list[0].copy()*np.nan
    V_k = V_list[0].copy()*np.nan
    
    U_k[maskpod] = phi_i[:np.sum(maskpod)]
    V_k[maskpod] = phi_i[np.sum(maskpod):]
    
    fig0, ax0 = plt.subplots(figsize=(8, 8))
    ax0.set_aspect('equal')
    ax0.use_sticky_edges = False
    ax0.margins(0.07)
    im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
    ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    fig0.colorbar(im0)
    fig0.tight_layout()
    
    Tx = bi[:-2,:].T
    Ty = bi[1:-1,:].T
    
    T = np.dot(Ty,np.linalg.pinv(Tx))
    
    vi,ci = np.linalg.eig(T)

    sig, ome = np.real(np.log(vi)/45), np.imag(np.log(vi)/45)
    
    Nc = 10
    bj = []
    j = 0 
    for i,t in enumerate(np.arange(0,bi.shape[0])*45):
        bj.append(np.sum(ci[j,:Nc]*np.exp((1j*ome[:Nc])*t)))
    plt.figure()
    plt.plot(bi[:,j])
    plt.plot(bj)
    
    C = np.zeros(T.shape)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            aux = ci[i,:Nc]*np.conj(ci[j,:Nc])*np.sign(np.imag(vi[:Nc]))
            aux = np.sum(aux)
            C[i,j] = np.imag(aux)
    
    index_list = np.transpose(np.unravel_index(np.argsort(C, axis=None), C.shape)).tolist()
    index_list = list(reversed(index_list))
    index_list = index_list[:int(len(index_list)/2)]
    index = np.array(index_list)
    
    plt.figure()
    plt.plot(bi[:,2])
    plt.plot(bi[:,24])
    
    c_hat = []

    for i in range(vi.shape[0]):
        ind = index[:,0]==i
        ii,jj = index[ind,:] [0,:]  
        c_hat.append(np.abs(ci[ii,i])**2 + np.abs(ci[jj,i])**2)
    
        
    plt.plot(np.real(vi))
    plt.plot(np.imag(vi[np.argsort(np.array(c_hat))]))
    
    print(np.sum(np.isnan(ut)))
    # vort = np.vstack([vorlist[i][maskpodv] for i in range(len(Ulist))])
    X = np.c_[ut, vt]
    L, S, R = np.linalg.svd(X/np.sqrt(X.shape[0]-1))
    print(L.shape, S.shape, R.shape)
    return (R.T, X.dot(R.T), S**2,maskpod)



def dmd_tot(grdu, ulist,vlist,r, N = 6500, dt=45):
    chunk = ulist[0].shape[1]   
    contsh = []
    vortsh = []
    for j in range(len(ulist)):
        dudy, dudx = np.gradient(ulist[j], grdu[1][:,0], grdu[0][0,:]) 
        dvdy, dvdx = np.gradient(vlist[j], grdu[1][:,0], grdu[0][0,:])    
        contsh.append(dudx + dvdy)
        vortsh.append(dvdx - dudy)
    Ulist = [ulist[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    Vlist = [vlist[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    clist = [contsh[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    vorlist = [vortsh[i] for i in range(len(ulist)) if np.sum(~np.isnan(ulist[i])) > N]
    x = grdu[0][0,:]
    y = grdu[1][:,0]
    U_arr_pod = np.vstack([Ulist[i] for i in range(len(Ulist))])
    V_arr_pod = np.vstack([Vlist[i] for i in range(len(Vlist))])
    c_arr_pod = np.vstack([clist[i] for i in range(len(Vlist))])
    v_arr_pod = np.vstack([vorlist[i] for i in range(len(Vlist))])
    maskpod = ~np.isnan(np.mean(U_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0))
    maskpodc = ~np.isnan(np.mean(c_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0))
    maskpodv = ~np.isnan(np.mean(v_arr_pod.reshape((int(len(U_arr_pod)/chunk),chunk,chunk)),axis=0))
    ut = np.vstack([Ulist[i][maskpod] for i in range(len(Ulist))])
    vt = np.vstack([Vlist[i][maskpod] for i in range(len(Ulist))])
    vort = np.vstack([vorlist[i][maskpodv] for i in range(len(Ulist))])
    X = np.c_[ut, vt][:-1].T
    X_prime = np.c_[ut, vt][1:].T    
    L, S, R = np.linalg.svd(X)
    L = L[:,:r]
    S = S[:r]
    R = R[:r,:]
    L = L[:,:len(S)]
    # continuous-time eigenvalues
    S = np.diag(S)
    Sinv = np.linalg.inv(S)   
    A_tilde = np.dot(X_prime,R.T)
    A_tilde = np.dot(A_tilde,Sinv)
    A_tilde = np.dot(np.conj(L.T),A_tilde)
    lam, W = np.linalg.eig(A_tilde)
    phi = np.dot(np.dot(np.dot(X_prime,R.T), Sinv),W) 
    omega = np.log(lam)/dt
    # amplitudes
    b = np.linalg.lstsq(np.real(phi),X[:,1])[0]
    # # DMD reconstruction
    m = X.shape[1]
    time_dyn = np.zeros((r,m))
    t = np.arange(m)*dt
    for i in range(m):
        time_dyn[:,i] = (b*np.exp(omega*t[i]))
    Xdmd = phi.dot(time_dyn)
    return (phi, lam, b, maskpod, Xdmd)
####################################################################################
##################################################################################
# In[]
tpick1 = ''.join(pick_rnd1.index.values[7])
tpick2 = ''.join(pick_rnd2.index.values[2])
file1 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 1/west/case_ph1_'+tpick1+'.pkl'
file2 = 'D:/PhD/Python Code/Balcony/data_process/results/wind_field/Phase 2/west/case_ph2_'+tpick2+'.pkl'
pick1,ut1,vt1,maskpod1,U_list1,V_list1,grdu = joblib.load(file1)
pick2,ut2,vt2,maskpod2,U_list2,V_list2,grdu = joblib.load(file2)

phi1, A1, lam1 = pod(ut1,vt1)
phi2, A2, lam2 = pod(ut2,vt2)

N_nodes = 1000
N_init = 1
plt.figure(figsize=(8,8))
plt.plot(lam1[N_init:N_nodes]/np.sum(np.abs(lam1[N_init:N_nodes])), '-o', label = r'$Ph\:1\:Unstable,\:Ri\:=\:'+'%.2f' % pick1['$Ri_{37}$']+'$')
plt.plot(lam2[N_init:N_nodes]/np.sum(np.abs(lam2[N_init:N_nodes])), '-o', label = r'$Ph\:2\:Unstable,\:Ri\:=\:'+'%.2f' % pick2['$Ri_{37}$']+'$')
plt.legend(fontsize = 20)
plt.xlabel(r'$Mode\:number\:i$',fontsize = 24)
plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

# In[]
##########################################################################    
k = 2
phi = phi1
A = A1
maskpod = maskpod1
ms = phi.shape[0]
U_mode = U_list1[0].copy()*np.nan
V_mode = V_list1[0].copy()*np.nan
U_mode[maskpod] =  np.real(phi[:int(ms/2),k])
V_mode[maskpod] =  np.real(phi[int(ms/2):,k])
vel_k = A[:,k][:,None].dot(phi[:,k][:,None].T)
U_k= U_list1[0].copy()*np.nan
V_k = V_list1[0].copy()*np.nan
U_k[maskpod] =  vel_k[np.argmax(A[:,k]),:int(ms/2)]
V_k[maskpod] =  vel_k[np.argmax(A[:,k]),int(ms/2):]

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], U_k, np.linspace(-2,2,10), cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[4, 4])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

phi = phi2
A = A2
maskpod = maskpod2
ms = phi.shape[0]
U_mode = U_list2[0].copy()*np.nan
V_mode = V_list2[0].copy()*np.nan
U_mode[maskpod] =  np.real(phi[:int(ms/2),k])
V_mode[maskpod] =  np.real(phi[int(ms/2):,k])
vel_k = A[:,k][:,None].dot(phi[:,k][:,None].T)
U_k= U_list2[0].copy()*np.nan
V_k = V_list2[0].copy()*np.nan
U_k[maskpod] =  vel_k[np.argmin(A[:,k]),:int(ms/2)]
V_k[maskpod] =  vel_k[np.argmin(A[:,k]),int(ms/2):]
U_k =U_mode
V_k =V_mode

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(b)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

####################################################################
    
# In[Load data]
##################################################################################

phi0, A0, lam0, maskpod0 = pod_tot(grdu, U_list0, V_list0, N = 6500)
phi1, A1, lam1, maskpod1 = pod_tot(grdu, U_list1, V_list1, N = 6800)
phi2, A2, lam2, maskpod2 = pod_tot(grdu, U_list2, V_list2, N = 6800)

N_nodes = 100
N_init = 1
plt.figure(figsize=(8,8))
plt.plot(lam0[N_init:N_nodes]/np.sum(np.abs(lam0[N_init:N_nodes])), '-o', label = r'$Unstable,\:L\:=\:'+'%.2f' % pick_rnd['$L_{175}$'].values[0]+'$')
plt.plot(lam1[N_init:N_nodes]/np.sum(np.abs(lam1[N_init:N_nodes])), '-o', label = r'$Stable,\:L\:=\:'+'%.2f' % pick_rnd1['$L_{flux,175}$'].values[0]+'$')
plt.plot(lam2[N_init:N_nodes]/np.sum(np.abs(lam2[N_init:N_nodes])), '-o', label = r'$Neutral,\:L\:=\:'+'%.2f' % pick_rnd2['$L_{flux,175}$'].values[0]+'$')
plt.legend(fontsize = 20)
plt.xlabel(r'$Mode\:number\:i$',fontsize = 24)
plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

plt.figure(figsize=(8,8))
plt.plot(np.cumsum(lam0[N_init:N_nodes])/np.sum(np.abs(lam0[N_init:N_nodes])), '-o', label = r'$Unstable,\:L\:=\:'+'%.2f' % pick_rnd0['$L_{flux,175}$'].values[0]+'$')
plt.plot(np.cumsum(lam1[N_init:N_nodes])/np.sum(np.abs(lam1[N_init:N_nodes])), '-o', label = r'$Stable,\:L\:=\:'+'%.2f' % pick_rnd1['$L_{flux,175}$'].values[0]+'$')
plt.plot(np.cumsum(lam2[N_init:N_nodes])/np.sum(np.abs(lam2[N_init:N_nodes])), '-o', label = r'$Neutral,\:L\:=\:'+'%.2f' % pick_rnd2['$L_{flux,175}$'].values[0]+'$')
plt.legend(fontsize = 20)
plt.xlabel(r'$Mode\:number\:i$',fontsize = 24)
plt.ylabel(r'$\frac{\sum_{i=0}^{j}\lambda_i}{\sum_{i=0}^{n} \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()


k = 2
phi = phi0
A = A0
maskpod = maskpod0
ms = phi.shape[0]
U_mode = U_list[0].copy()*np.nan
V_mode = V_list[0].copy()*np.nan
U_mode[maskpod] =  np.real(phi[:int(ms/2),k])
V_mode[maskpod] =  np.real(phi[int(ms/2):,k])
vel_k = A[:,k][:,None].dot(phi[:,k][:,None].T)
U_k= U_list[0].copy()*np.nan
V_k = V_list[0].copy()*np.nan
U_k[maskpod] =  vel_k[np.argmin(A[:,k]),:int(ms/2)]
V_k[maskpod] =  vel_k[np.argmin(A[:,k]),int(ms/2):]

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.plot(A[:,1],'-o')
ax0.plot(A[:,2],'-o')
ax0.plot(A[:,3],'-o')
ax0.plot(A[:,4],'-o')

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)   
im0 = ax0.contourf(grdu[0], grdu[1], V_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(b)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

plt.figure()
plt.plot(np.arange(0,A.shape[0])*45, np.real(A[:,1]), label = r'$\phi_1$')
# plt.plot(np.arange(0,A.shape[0])*45, np.real(A[:,2]), label = r'$\phi_2$')
# plt.plot(np.arange(0,A.shape[0])*45, np.real(A[:,3]), label = r'$\phi_3$')
plt.plot(np.arange(0,A.shape[0])*45, np.real(A[:,20]), label = r'$\phi_{20}$')
plt.xlabel('$t,\:[s]$', fontsize=20)
plt.ylabel('$A$', fontsize=20)
plt.legend(fontsize=20)



file_pre = '/case_pod_'+pick_rnd0.name.values[0]+pick_rnd0.hms.values[0]
os.mkdir(file_out_path_u_field+file_pre)

k = 1
k0 = 20
phi = phi0
A = A0
maskpod = maskpod0
ms = phi.shape[0]

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)

for j in range(40):
    ax0.cla()
    vel_k = A[:,k:k+k0].dot(phi[:,k:k+k0].T)
    U_k= U_list0[0].copy()*np.nan
    V_k = V_list0[0].copy()*np.nan
    U_k[maskpod] =  vel_k[j,:int(ms/2)]
    V_k[maskpod] =  vel_k[j,int(ms/2):]
    im0 = ax0.contourf(grdu[0], grdu[1], U_k, np.linspace(-2,2,10), cmap='jet')
    ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
 
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
    fig0.savefig(file_out_path_u_field+file_pre+'/'+str(j)+'.png')
    plt.pause(.01)
    
    
    
    
    
    
    
    
    
    
    
    # ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    # fig0.colorbar(im0)
    # fig0.tight_layout()
    plt.pause(.1)
    




phid0, lamd0, b0, maskpodd0, Xdmd0 = dmd_tot(grdu, U_list0, V_list0, 5, N = 6500)
phid1, lamd1, b1, maskpodd1, Xdmd1 = dmd_tot(grdu, U_list1, V_list1, 5, N = 6500)
phid2, lamd2, b2, maskpodd2, Xdmd2 = dmd_tot(grdu, U_list2, V_list2, 5, N = 6500)






k = 3
phi = phid2
maskpod = maskpodd2

ms = phi.shape[0]
U_mode = U_list0[0].copy()*np.nan
V_mode = V_list0[0].copy()*np.nan
U_mode[maskpod] =  np.real(phi[:int(ms/2),k])
V_mode[maskpod] =  np.real(phi[int(ms/2):,k])

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], U_mode, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_mode, V_mode, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)   
im0 = ax0.contourf(grdu[0], grdu[1], V_mode, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_mode, V_mode, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()








##################################################################################
# In[]
lam_s_vel0, lam_s_con0, lam_s_vor0, phi_s_vel0, phi_s_con0, phi_s_vor0, A_s_vel0, A_s_con0, A_s_vor0, maskpod0, maskpodc0, maskpodv0, ut0, vt0 =  pod_out(grdu, U_list0,V_list0, N = 6500)

lam_s_vel1, lam_s_con1, lam_s_vor1, phi_s_vel1, phi_s_con1, phi_s_vor1, A_s_vel1, A_s_con1, A_s_vor1, maskpod1, maskpodc1, maskpodv1, ut1, vt1 =  pod_out(grdu, U_list1,V_list1,N = 6500)

lam_s_vel2, lam_s_con2, lam_s_vor2, phi_s_vel2, phi_s_con2, phi_s_vor2, A_s_vel2, A_s_con2, A_s_vor2, maskpod2, maskpodc2, maskpodv2, ut2, vt2 =  pod_out(grdu, U_list2,V_list2,N = 6500)

# In[plots eigenvalues]
N_nodes = 100
N_init = 0
plt.figure(figsize=(8,8))
plt.plot(lam_s_vel0[N_init:N_nodes]/np.sum(np.abs(lam_s_vel0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(lam_s_vel1[N_init:N_nodes]/np.sum(np.abs(lam_s_vel1[N_init:N_nodes])), '-o', label = r'$Stable$')
plt.plot(lam_s_vel2[N_init:N_nodes]/np.sum(np.abs(lam_s_vel2[N_init:N_nodes])), '-o', label = r'$Neutral$')
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

plt.figure(figsize=(8,8))
plt.plot(np.cumsum(lam_s_vel0[N_init:N_nodes])/np.sum(np.abs(lam_s_vel0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(np.cumsum(lam_s_vel1[N_init:N_nodes])/np.sum(np.abs(lam_s_vel1[N_init:N_nodes])), '-o', label = r'$Stable$')
plt.plot(np.cumsum(lam_s_vel2[N_init:N_nodes])/np.sum(np.abs(lam_s_vel2[N_init:N_nodes])), '-o', label = r'$Neutral$')
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\sum_{i=0}^{j}\lambda_i}{\sum_{i=0}^{n} \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
 

plt.figure(figsize=(8,8))
plt.plot(lam_s_vor0[N_init:N_nodes]/np.sum(np.abs(lam_s_vor0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(lam_s_vor1[2:N_nodes]/np.sum(np.abs(lam_s_vor1[2:N_nodes])), '-o', label = r'$Stable$')
plt.plot(lam_s_vor2[:N_nodes]/np.sum(np.abs(lam_s_vor2[:N_nodes])), '-o', label = r'$Neutral$')
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

plt.figure(figsize=(8,8))
plt.plot(np.cumsum(lam_s_vor0[N_init:N_nodes])/np.sum(np.abs(lam_s_vor0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(np.cumsum(lam_s_vor1[N_init:N_nodes])/np.sum(np.abs(lam_s_vor1[2:N_nodes])), '-o', label = r'$Stable$')
plt.plot(np.cumsum(lam_s_vor2[N_init:N_nodes])/np.sum(np.abs(lam_s_vor2[:N_nodes])), '-o', label = r'$Neutral$')
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\sum_{i=0}^{j}\lambda_i}{\sum_{i=0}^{n} \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()


plt.figure(figsize=(8,8))
plt.plot(lam_s_con0[N_init:N_nodes]/np.sum(np.abs(lam_s_con0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(lam_s_con1[2:N_nodes]/np.sum(np.abs(lam_s_con1[:N_nodes])), '-o', label = r'$Stable$')
plt.plot(lam_s_con2[:N_nodes]/np.sum(np.abs(lam_s_con2[:N_nodes])), '-o', label = r'$Neutral$')
plt.legend()
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\lambda_i}{\sum_i \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

plt.figure(figsize=(8,8))
plt.plot(np.cumsum(lam_s_con0[N_init:N_nodes])/np.sum(np.abs(lam_s_con0[N_init:N_nodes])), '-o', label = r'$Unstable$')
plt.plot(np.cumsum(lam_s_con1[:N_nodes])/np.sum(np.abs(lam_s_con1[:N_nodes])), '-o', label = r'$Stable$')
plt.plot(np.cumsum(lam_s_con2[:N_nodes])/np.sum(np.abs(lam_s_con2[:N_nodes])), '-o', label = r'$Neutral$')
plt.legend(fontsize = 20)
plt.xlabel(r'$\lambda_i$',fontsize = 24)
plt.ylabel(r'$\frac{\sum_{i=0}^{j}\lambda_i}{\sum_{i=0}^{n} \lambda_i}$',fontsize = 24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

# In[] 

k = 3
vor_mode = U_list1[0].copy()*np.nan
vor_mode[maskpod] =  np.real(phi[:,k])
vor_k = A[:,k][:,None].dot(phi[:,k][:,None].T)
v_k= U_list1[0].copy()*np.nan
v_k[maskpod] =  vor_k[np.argmax(A[:,k]),:]


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], v_k, np.linspace(-.01,.01,10), cmap='bwr')
#ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

####################################################


k = 0
vor_mode = U_list2[0].copy()*np.nan
vor_mode[maskpodv2] =  np.real(phi_s_vor2[:,k])
vor_k = A_s_vor2[:,k][:,None].dot(phi_s_vor2[:,k][:,None].T)
v_k= U_list2[0].copy()*np.nan
v_k[maskpodv2] =  vor_k[np.argmax(A_s_vor2[:,k]),:]


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], v_k, np.linspace(-.015,.015,10), cmap='bwr')
#ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()


plt.figure()
plt.plot(np.real(A_s_vor2[:,0]))
plt.plot(np.real(A_s_vor2[:,1]))
plt.plot(np.real(A_s_vor2[:,2]))
plt.plot(np.real(A_s_vor2[:,3]))
plt.plot(np.real(A_s_vor2[:,4]))
plt.plot(np.real(A_s_vor2[:,5]))
plt.plot(np.real(A_s_vor2[:,6]))

####################################################


k = 0
vor_mode = U_list0[0].copy()*np.nan
vor_mode[maskpodv0] =  np.real(phi_s_vor0[:,k])
vor_k = A_s_vor0[:,k][:,None].dot(phi_s_vor0[:,k][:,None].T)
v_k= U_list0[0].copy()*np.nan
v_k[maskpodv0] =  vor_k[np.argmax(A_s_vor0[:,k]),:]


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], v_k, np.linspace(-.01,.01,10), cmap='bwr')
#ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

####################################

k = 0
ms = phi_s_vel0.shape[0]
U_mode = U_list0[0].copy()*np.nan
V_mode = V_list0[0].copy()*np.nan
U_mode[maskpod0] =  np.real(phi_s_vel0[:int(ms/2),k])
V_mode[maskpod0] =  np.real(phi_s_vel0[int(ms/2):,k])
vel_k = A_s_vel0[:,k][:,None].dot(phi_s_vel0[:,k][:,None].T)
U_k= U_list0[0].copy()*np.nan
V_k = V_list0[0].copy()*np.nan
U_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k]),:int(ms/2)]
V_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k]),int(ms/2):]

# plt.figure()
# plt.plot(np.real(A_s_vel0[:,0]))
# plt.plot(np.real(A_s_vel0[:,1]))
# plt.plot(np.real(A_s_vel0[:,2]))


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)   
im0 = ax0.contourf(grdu[0], grdu[1], V_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[3, 3])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

plt.figure()
# plt.plot(A_s_vel0[:,0])
# plt.plot(A_s_vel0[:,1])
# plt.plot(A_s_vel0[:,2])
plt.plot(A_s_vel0[:,3])
plt.plot(A_s_vel0[:,4])
####################################

k0 = 0
ms = phi_s_vel0.shape[0]
U_mode = U_list0[0].copy()*np.nan
V_mode = V_list0[0].copy()*np.nan
U_mode[maskpod0] =  np.real(phi_s_vel0[:int(ms/2),k0])
V_mode[maskpod0] =  np.real(phi_s_vel0[int(ms/2):,k0])

vel_k = A_s_vel0[:,k0][:,None].dot(phi_s_vel0[:,k0][:,None].T)
U_k= U_list0[0].copy()*np.nan
V_k = V_list0[0].copy()*np.nan
U_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k0]),:int(ms/2)]
V_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k0]),int(ms/2):]


ms = phi_s_vel0.shape[0]
U_mode = U_list0[0].copy()*np.nan
V_mode = V_list0[0].copy()*np.nan
U_mode[maskpod0] =  np.real(phi_s_vel0[:int(ms/2),k1])
V_mode[maskpod0] =  np.real(phi_s_vel0[int(ms/2):,k1])

vel_k = A_s_vel0[:,k0:k2].dot(phi_s_vel0[:,k0:k2].T)

U_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k0]),:int(ms/2)]
V_k[maskpod0] =  vel_k[np.argmax(A_s_vel0[:,k0]),int(ms/2):]

plt.figure()
plt.plot(A_s_vel0[:,k0])
plt.plot(A_s_vel0[:,k1])
plt.plot(A_s_vel0[:,k2])


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
for j in range(A_s_vel0.shape[0]):
    ax0.cla()      
    U_k[maskpod0] =  vel_k[j,:int(ms/2)]
    V_k[maskpod0] =  vel_k[j,int(ms/2):]
      
    im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
    ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[2, 2])
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    #fig0.colorbar(im0)
    #fig0.tight_layout()
    plt.pause(.5)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()       
im0 = ax0.contourf(grdu[0], grdu[1], V_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[2, 2])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()


#############################

k = 4
ms = phi_s_vel1.shape[0]
U_mode = U_list1[0].copy()*np.nan
V_mode = V_list1[0].copy()*np.nan
U_mode[maskpod1] =  np.real(phi_s_vel1[:int(ms/2),k])
V_mode[maskpod1] =  np.real(phi_s_vel1[int(ms/2):,k])
vel_k = A_s_vel1[:,k][:,None].dot(phi_s_vel1[:,k][:,None].T)
U_k= U_list1[0].copy()*np.nan
V_k = V_list1[0].copy()*np.nan
U_k[maskpod1] =  vel_k[np.argmax(A_s_vel1[:,k]),:int(ms/2)]
V_k[maskpod1] =  vel_k[np.argmax(A_s_vel1[:,k]),int(ms/2):]

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()      
  
im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[1, 1])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()      
ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
im0 = ax0.contourf(grdu[0], grdu[1], V_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[1, 1])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

#############################

k = 0
ms = phi_s_vel2.shape[0]
U_mode = U_list2[0].copy()*np.nan
V_mode = V_list2[0].copy()*np.nan
U_mode[maskpod2] =  np.real(phi_s_vel2[:int(ms/2),k])
V_mode[maskpod2] =  np.real(phi_s_vel2[int(ms/2):,k])

vel_k = A_s_vel2[:,k][:,None].dot(phi_s_vel2[:,k][:,None].T)
U_k = U_list2[0].copy()*np.nan
V_k = V_list2[0].copy()*np.nan
U_k[maskpod2] =  vel_k[np.argmax(A_s_vel2[:,k]),:int(ms/2)]
V_k[maskpod2] =  vel_k[np.argmax(A_s_vel2[:,k]),int(ms/2):]

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()      
   
im0 = ax0.contourf(grdu[0], grdu[1], U_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[1, 1])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()      
ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
im0 = ax0.contourf(grdu[0], grdu[1], V_k, 10, cmap='jet')
ax0.streamplot(grdu[0], grdu[1], U_k, V_k, density=[1, 1])
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.colorbar(im0)
fig0.tight_layout()


# In[]
###################################################

k = 0

velpod_s = A_s[:,k][:,None].dot(phi_s[:,k][:,None].T)

#############################################
k = 0
ms = vel_arr_pod.shape[1]
U_mode = U_pod.copy()*np.nan
U_mode[maskpodc] =  np.real(phi_s[:,k])

V_mode = V_pod.copy()*np.nan
U_mode[maskpodc] =  np.real(phi_s[:,k])

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
ax0.cla()      
ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
im0 = ax0.contourf(grdu[0], grdu[1], U_mode, 10, cmap='jet')
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.tight_layout()
fig0.colorbar(im0)


plt.figure()
plt.plot(np.abs(lam_s)/np.sum(np.abs(lam_s)), '-o')

plt.figure()
plt.plot(np.cumsum(lam_s/np.sum(lam_s)), '-o')

plt.figure()
plt.plot(A_s[:,0])
plt.plot(A_s[:,1])
plt.plot(A_s[:,2])
plt.plot(A_s[:,3])
plt.plot(A_s[:,4])
plt.plot(A_s[:,5])


########################################
k0 = 0
max_corr = []
U_mode0 = U_pod.copy()*np.nan
U_mode0[maskpod] =  np.real(phi_s[:,k0])
for k in range(1,50):
    print(k)
    U_mode1 = U_pod.copy()*np.nan
    U_mode1[maskpod] =  np.real(phi_s[:,k])
    
    tp,ep,r1p,r2p,r12p,_,_,_,_ = sc.spatial_autocorr_sq(grdu, U_mode0, U_mode1,
                                                              transform = False,
                                                              transform_r = False,
                                                              e_lim=.1,refine=32)
    max_corr.append(np.nanmax(np.abs(r12p[:,int(r12p.shape[1]/2)+1]/np.sqrt(np.nanmax(r1p)*np.nanmax(r2p)))))
    
    
plt.plot(max_corr)
    
plt.plot(2*np.pi*ep[:,0]/(2*np.max(ep[:,0])), r12p[:,int(r12p.shape[1]/2)+1]/np.sqrt(np.nanmax(r1p)*np.nanmax(r2p)))

###########################################
file_name = '/pod_case_'+pick_rnd.name.values[0]+pick_rnd.hms.values[0]+'.pkl'
joblib.dump((pick_rnd, phi_s, A_s, U_list, V_list, chunk), file_out_path_u_field+file_name)

#################################################





Upod_s = velpod_s[:,:int(ms/2)]
Vpod_s = velpod_s[:,int(ms/2):]

U1_s = U_pod.copy()*np.nan
V1_s = U_pod.copy()*np.nan


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)

umin = np.nanmin(Upod_s)
umax = np.nanmax(Upod_s)

for j in range(len(U_list)):
    ax0.cla()    


    
    U1_s[maskpod] = Upod_s[j,:]
    U1_s = U1_s+U_pod
    
    V1_s[maskpod] = Vpod_s[j,:]
    V1_s = V1_s+V_pod
    
    ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
    ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
    im0 = ax0.contourf(grdu[0], grdu[1], np.real(U1_s), np.linspace(umin,umax,10), cmap='jet')
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    
    
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
        cb = fig.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
        cb.ax.tick_params(labelsize=24)
        cb.ax.set_ylabel(r'$\Delta U\:[m/s]$', fontsize = 24)
        # ax0.set_xlabel(r'$x_1\:[m]$', fontsize=24, )
        # ax0.set_ylabel(r'$x_2\:[m]$', fontsize=24, weight='bold') 
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        divider = make_axes_locatable(ax0)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im0, cax=cax, format=ticker.FuncFormatter(fm))
        cb.ax.set_ylabel('$U\:[m/s]$', fontsize = 24)
        cb.ax.tick_params(labelsize=24)

    fig0.tight_layout()
    plt.pause(.5)






















from scipy.sparse import csr_matrix, rand
import tables as tb

a = csr_matrix(u_arr_pod.copy().T)
b = a.T
l, m, n = a.shape[0], a.shape[1], b.shape[1]
f = tb.open_file('dot.h5', 'w')
filters = tb.Filters(complevel=5, complib='blosc')
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(l, n), filters=filters)
bl = 100 #this is the number of rows we calculate each loop
#this may not the most efficient value
#look into buffersize usage in PyTables and adopt the buffersite of the
#carray accordingly to improve specifically fetching performance
b = b.tocsc() #we slice b on columns, csc improves performance
#this can also be changed to slice on rows instead of columns
for i in range(0, l, bl):
  print(i)
  out[:,i:min(i+bl, l)] = (a.dot(b[:,i:min(i+bl, l)])).toarray()
f.close()

h5 = tb.open_file('dot.h5', 'r')
a = h5.root.data

C = a[:,:]

h5.close()


# ##########################


# def matrix_append_row(fm,tp,mat):
#     #check if number of columns is the same
#     rows= fm.shape[0] + mat.shape[0]
#     new_fm = np.memmap(fm.filename, mode='r+', dtype= tp, shape=(rows,fm.shape[1]))
#     new_fm[fm.shape[0]:rows,:]= mat[:]
#     return new_fm

# def generate_and_store_data(cols,batch,iter,tp):
#     #create memmap file and append generated data in cycle

#     #can't create empty array - need to store first entry by copy not by append
#     fn= np.memmap('A.npy', dtype=tp, mode='w+', shape=(batch,cols))

#     for i in range(iter):
#         data= np.random.rand(batch,cols)*100  # [0-1]*100
#         # print i
#         # t0= time.time()
#         if i==0:
#             fn[:]= data[:]
#         else:   
#             fn= matrix_append_row(fn,tp,data)
#         # print (time.time()-t0)
#         # fm.flush()
#         # print fm.shape
#     return fn

# A= generate_and_store_data(l, 10, 10,'float16')
# #cov= A.T*A

# # A memmaped file
# print A.shape
# M= A.shape[0]
# N= A.shape[1]
# #create cov mat



# #A.T don't copy data just view?
# t0= time.time()
# np.dot(a,b,out= cov)
print (time.time()-t0)


# In[]


fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
newdate = []

for j in range(len(U_list2)): 
    new_date = t_scan2[j,0].strftime("%Y-%m-%d %H:%M:%S")
    newdate.append(new_date[:11]+ '\:'+new_date[11:])
    
meanU = []
meanV = []
for j in range(len(U_list2)):
    print(j)
    ax0.cla()    
    meanU.append(np.nanmean(ut2[j,:]))
    meanV.append(np.nanmean(vt2[j,:]))
    ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20) 
    # ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
    # ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
    u = np.ones(grdu[0].shape)*np.nan
    u[maskpod2] = ut2[j,:]
    im0 = ax0.contourf(grdu[0], grdu[1], u, np.linspace(4,15,10), cmap='jet')
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
    ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    
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
    plt.pause(.005)

