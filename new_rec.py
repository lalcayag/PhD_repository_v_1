# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:13:10 2018

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
import ppiscanprocess.windfieldrec as wf
import ppiscanprocess.spectraconstruction as sc
import spectralfitting.spectralfitting as sf

from importlib import reload  # Python 3.4+ only.


import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[Load files, after filtering, input files]

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

# In[Columns names]

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(198):

    labels = np.concatenate((labels,vel_lab))

# In[Load data and creating dataframes]


df_0 = pd.read_csv(file_lidar_path_0[0],sep=";", header=None) 
df_1 = pd.read_csv(file_lidar_path_1[0],sep=";", header=None) 

df_0.columns = labels
df_1.columns = labels

df_0['scan'] = df_0.groupby('azim').cumcount()
df_1['scan'] = df_1.groupby('azim').cumcount()


# In[]

loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1

phi0 = df_0.azim.unique()
phi1 = df_1.azim.unique()

r0 = np.array(df_0.iloc[(df_0.azim==min(phi0)).nonzero()[0][0]].range_gate)
r1 = np.array(df_1.iloc[(df_1.azim==min(phi0)).nonzero()[0][0]].range_gate)

r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
r_1, phi_1 = np.meshgrid(r1, np.pi-np.radians(phi1)) # meshgrid

tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wf.grid_over2((r_0, phi_0),(r_1,phi_1),-d)
    
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


# In[Interpolation]

df_0_int = data_interp_kdtree(df_0,45,col_r='range_gate',col_phi='azim')

# In[Reconstruction]

U, V = direct_wf_rec(df_0, df_1, tri, -d,N_grid = 256)

 # In[]
 


scan_n = 9000

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(900,1200):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_0.ws.loc[df_0.scan==scan_n].values,100,cmap='jet')
    plt.pause(.001)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(375):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_int.ws.loc[scan_n].values,100,cmap='jet')
    plt.pause(.001)

#   
#for scan_n in range(0,18):
#
#mask_CNR = (ws3_w_df.CNR>-24) & (ws3_w_df.CNR<-8)
#
#mask_CNR.columns =  mask.columns
#
#mask.mask(mask_CNR,other=False,inplace=True)
            
#
#im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df.ws.loc[df.scan==scan_n].values,100,cmap='jet')
#fig.colorbar(im)


p0 = df_0.azim.unique()
r0 = np.array(df_0.iloc[(df_0.azim==
                               min(p0)).nonzero()[0][0]].range_gate)

r_g0, p_g0 = np.meshgrid(r0, np.pi-np.radians(p0)) # meshgrid

r_g_t0, p_g_t0 = translationpolargrid((r_g0, p_g0), d/2)
p1 = df_1.azim.unique()
r1 = np.array(df_1.iloc[(df_1.azim==
                               min(p1)).nonzero()[0][0]].range_gate)

r_g1, p_g1 = np.meshgrid(r1, np.pi-np.radians(p1)) # meshgrid

r_g_t1, p_g_t1 = translationpolargrid((r_g1, p_g1), -d/2)


scan_n = 8625

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(tri,lw=2,color='grey',alpha=0.5)
im=ax.contourf(r_g_t0*np.cos(p_g_t0),r_g_t0*np.sin(p_g_t0),df_0.ws.loc[df_0.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri,lw=2,color='grey',alpha=0.5)
#im=ax.contourf(r_g_t0*np.cos(p_g_t0),r_g_t0*np.sin(p_g_t0),p_g0*180/np.pi,100,cmap='jet')
#fig.colorbar(im)



fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(tri,lw=2,color='grey', alpha = 0.5)
im=ax.contourf(r_g_t1*np.cos(p_g_t1),r_g_t1*np.sin(p_g_t1),df_1.ws.loc[df_1.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri,lw=2,color='grey', alpha = 0.5)
#im=ax.contourf(r_g_t1*np.cos(p_g_t1),r_g_t1*np.sin(p_g_t1),p_g1*189/np.pi,100,cmap='jet')
#fig.colorbar(im)

plt.figure()
plt.contourf(grd[0],grd[1],V[94],100,cmap='jet')
plt.colorbar()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(grd[0],grd[1],U[0],100,cmap='jet')

for scan_n in range(5):
    print(scan_n)
    ax.cla()
    ax.contourf(grd[0],grd[1],U[scan_n],100,cmap='jet')
    plt.pause(.001)

plt.figure()
plt.contourf(grd[0],grd[1],V[:,:,0],300,cmap='jet')
plt.colorbar()


phi_1 = np.pi-np.radians(df_1.loc[df_1['scan']==scan_n]['azim'].unique())
r_1 = np.unique(df_1.loc[df_1['scan']==scan_n]['range_gate'].values)
r_g_1, phi_g_1 = np.meshgrid(r_1,phi_1)
r_t_1, phi_t_1 = translationpolargrid((r_g_1, phi_g_1),-d/2)
x1 = r_t_1*np.cos(phi_t_1)
y1 = r_t_1*np.sin(phi_t_1)
v_los_1 = df_1['ws'].loc[df_1['scan']==scan_n].values
ind1 = ~np.isnan(v_los_1.flatten())
x_1 = x1.flatten()[ind1]
y_1 = y1.flatten()[ind1]

#################

trid = Delaunay(np.c_[x_1,y_1])
areas = areatriangles(trid, delaunay = True)

maskt = circleratios(trid)<.05
maska = areas> np.mean(areas) + 3*np.std(areas)
mask0 = maskt | maska

triangle_ind = np.arange(0,len(trid.simplices))

indtr = np.isin(trid.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()]),triangle_ind[~mask0])

grid = np.meshgrid(np.linspace(np.min(x1.flatten()),np.max(x1.flatten()),200),np.linspace(np.min(y1.flatten()),np.max(y1.flatten()),200))

v_sq_1 = np.zeros(grid[0].shape).flatten()*np.nan

v_sq_1[indtr] = sp.interpolate.griddata(np.c_[x_1,y_1],v_los_1.flatten()[ind1], (grid[0].flatten()[indtr],grid[1].flatten()[indtr]), method='cubic')

v_sq_1 = np.reshape(v_sq_1,grid[0].shape)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(tri,lw=2,color='grey', alpha = 0.5)
im=ax.contourf(grid[0],grid[1],v_sq_1,200,cmap='jet')
fig.colorbar(im)

plt.figure()

plt.triplot(x_1, y_1, trid.simplices.copy())

plt.triplot(x_1, y_1, trid.simplices.copy()[~mask0])

plt.scatter(grid[0].flatten()[indtr],grid[1].flatten()[indtr],color='r',s=1)

#################
tri_t1 = Triangulation(x_1,y_1)


maskt=TriAnalyzer(tri_t1).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05) 
maska = areatriangles(tri_t1)> np.mean(areatriangles(tri_t1)) + 2*np.std(areatriangles(tri_t1))
mask1 = maskt | maska
tri_t1.set_mask(mask1)

pts_in_1 = tri_t1.get_trifinder()(x_i,y_i) !=-1

tri_t1.set_mask(None)
##################
start = 8500
end = 8700
ind0 = (df_0.scan>=start) & (df_0.scan<=end)
ind1 = (df_1.scan>=start) & (df_1.scan<=end)

U_out, V_out, grd, _ = wf.direct_wf_rec(df_0.loc[ind0], df_1.loc[ind1], tri, d,N_grid = 512)

x = grd[0][0,:]
y = grd[1][:,0]

k_1 = []
k_2 = []
Su_u = []
Sv_v = []
Su_v = []


for scan_i in range(len(U_out)):
    print(scan_i)
    k1,k2,Su,Sv,Suv = sc.spatial_spec_sq(x,y,U_out[scan_i],V_out[scan_i],transform = True)
    k_1.append(k1)
    Su_u.append(Su)
    Sv_v.append(Sv)
    Su_v.append(Suv)
    k_2.append(k2)



k_1_r = []
k_2_r = []
Su_u_r = []
Sv_v_r = []
Su_v_r = []


for scan_i in range(len(U_out)):
    print(scan_i)
    ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y,U_out[scan_i],V_out[scan_i],transform = True, ring=True)
    k_1_r.append(ku)
    k_2_r.append(kv)
    Su_u_r.append(Su)
    Sv_v_r.append(Sv)
    Su_v_r.append(Suv)



with open('S_sq2.pkl', 'wb') as V_t:
     pickle.dump((Su_u,Sv_v,Su_v,k_1,k_2),V_t)
     
with open('S_ring.pkl', 'wb') as V_t:
     pickle.dump((Su_u_r,Sv_v_r,Su_v_r,k_1_r,k_2_r),V_t)
     
with open('UV_out2', 'wb') as V_t:
     pickle.dump((U_out,V_out),V_t)

scan_i = 30
ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y,U_out[scan_i],V_out[scan_i],transform = True, ring=True)

plt.figure()
plt.plot(k_1[scan_i],k_1[scan_i]*Su_u[scan_i])
plt.plot(ku,ku*Su)
plt.plot(ku,ku**(-2/3)/100,'-o')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(k_1[scan_i],k_1[scan_i]*Sv_v[scan_i])
plt.plot(ku,ku*Sv)
plt.plot(ku,ku**(-2/3)/100,'-o')
plt.xscale('log')
plt.yscale('log')


plt.figure()
plt.plot(k_1[scan_i],.5*k_1[scan_i]*(Su_u[scan_i]+Sv_v[scan_i]))
plt.plot(ku,.5*ku*(Sv+Su))
plt.plot(ku,ku**(-2/3)/100,'-o')
plt.xscale('log')
plt.yscale('log')
#U_tri = []
#V_tri = []
#
#for scan_n in range(max(df_0.loc[ind0].scan.min(),df_1.loc[ind1].scan.min()),min(df_0.loc[ind0].scan.max(),df_1.loc[ind1].scan.max())):
#    print(scan_n)
#    tot_s = (8910-mask_df_0[df_0.scan==scan_n].sum().sum())/8910
#    tot_v = (8910-mask_df_1[df_1.scan==scan_n].sum().sum())/8910
#    if (tot_s>.5) & (tot_v>.5):
#        print(scan_n)    
#        Lidar0 = (df_0.ws.loc[df_0.scan==scan_n],phi_0,w0,neigh0,index0) 
#        Lidar1 = (df_1.ws.loc[df_1.scan==scan_n],phi_1,w1,neigh1,index1)
#        auxU, auxV= wind_field_rec(Lidar0, Lidar1, tree, tri, -d)
#        U_tri.append(auxU) 
#        V_tri.append(auxV)
#    else:
#        U_tri.append([]) 
#        V_tri.append([])
#
#U_tri = [u for u in U_tri if len(u)>0]
#V_tri = [u for u in V_tri if len(u)>0]
#
#U_tri, V_tri = data_interp_triang(U_tri,V_tri,tri.x,tri.y,45)



#k1 = []
#k2 = []
#Su_u = []
#Su_u2 = []
#Sv_v = []
#Su_v = []
#Fu_v = []
#Sv_v2 = []
#Su_v2 = []
#
#for i in range(len(U_out)):
#    print(i)
#    k,k2,Su,Sv,Suv,Fuv,Su_u2, Sv_v2, Su_v2 = spatial_spec_sq(x,y,U_out[i],V_out[i],transform = True)
#    k1.append(k)
#    Su_u.append(Su)
#    Sv_v.append(Sv)
#    Su_v.append(Suv)
#    k2.append(k)
#    Su_u2.append(Su)
#    Sv_v2.append(Sv)
#    Su_v2.append(Suv)
#    Fu_v.append(Fuv)

#k1_tri = []
#Su_u_tri = []
#Sv_v_tri = []
#Su_v_tri = []
#Fu_v_tri = []
#
#for i in range(len(U_tri)):
#    print(i)
#    if len(U_tri[i])>0:
#        Su,Sv,Suv,k1,k2=spatial_autocorr_fft(tri,U_tri[i],V_tri[i],transform = True,N_grid=512,interp='cubic')
#        Su_u_tri.append(sp.integrate.simps(Su,k2,axis=1))
#        Sv_v_tri.append(sp.integrate.simps(Sv,k2,axis=1))
#        Su_v_tri.append(sp.integrate.simps(Suv,k2,axis=1))
#        k1_tri.append(k1)
#

#plt.figure()
#plt.plot(k_1[scan_i],k_1[scan_i]*Su_u[scan_i])
#plt.plot(k_1[scan_i],k_1[scan_i]*Sv_v[scan_i])
##plt.plot(k_1[scan_i][ind],F_obs[scan_i]k_1[scan_i][ind])
#plt.plot(k_1[scan_i],0.5*k_1[scan_i]*(Su_u[scan_i]+Sv_v[scan_i]))
##plt.plot(k_2[scan_i],k_2[scan_i]*Su_u2[scan_i])
##plt.plot(k1_tri[scan_i],k1_tri[scan_i]*Su_u_tri[scan_i])
#plt.xscale('log')
#plt.yscale('log')
     
###########################

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter("%.2f")    

from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

f = 24

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])
#############    
#MLC 

#k_star = ku[np.argmax(ku*F_obs)]
#
#param_ind = [1,3,5,7,9]
#param_fix = [param_fix_tot[i] for i in param_ind_not]
#
#z_i = np.sqrt(2)/k_star
       
#General

bound_tot = [(1000,4000),(0,2),(0,2), (0,2),(100,300), (0.2,1), (0,2), (0,.1),(10,1000),(0,10)]
param_fix_tot = [1000,0,0.85,2,200,0,1.6,.2*1.45**.5,300,2]
param_init_tot = param_fix_tot
lab_tot = ['$z_i$', '$w_*$', '$c_1^f$', '$c_2^f$', '$z$', '$u_*$', '$c_1^n$', '$c_2^n$', '$w$','$n$']
param_tot = np.zeros(10)
param_set = np.arange(10)
fig_letter = ['$(a)$','$(b)$','$(c)$','$(d)$','$(e)$','$(f)$']
fonts = matplotlib.font_manager.FontProperties(weight='black', size=24)
file_tot = ['z_i', 'w_star', 'c_1_f', 'c_2_f', 'z', 'u_star', 'c_1_n', 'c_2_n', 'Leq','N']



param_ind = [1,3,5,7,9]
param_ind_not = [e for e in param_set if e not in set(param_ind)]
param_init = [param_init_tot[i] for i in param_ind]
param_fix = [param_fix_tot[i] for i in param_ind_not]
bound = [bound_tot[i] for i in param_ind]
lab = [lab_tot[i] for i in param_ind]
ndim, nwalkers = len(param_init), 60
file = [file_tot[i] for i in param_ind]

# Model 0
param_ind0 = [5,7,9]
param_ind_not0 = [e for e in param_set if e not in set(param_ind0)]
param_init0 = [param_init_tot[i] for i in param_ind0]
param_fix0 = [param_fix_tot[i] for i in param_ind_not0]
bound0 = [bound_tot[i] for i in param_ind0]
lab0 = [lab_tot[i] for i in param_ind0]
ndim0 = len(param_init0)
file0 = [file_tot[i] for i in param_ind0]
# Model 1
param_ind1 = [1,3,5,7,9]
param_ind_not1 = [e for e in param_set if e not in set(param_ind1)]
param_init1 = [param_init_tot[i] for i in param_ind1]
param_fix1 = [param_fix_tot[i] for i in param_ind_not1]
bound1 = [bound_tot[i] for i in param_ind1]
lab1 = [lab_tot[i] for i in param_ind1]
ndim1= len(param_init1)
file1 = [file_tot[i] for i in param_ind1]

#l_f, s_f, c1_f, c2_f, l_n, s_n, c1_n, c2_n, w, n

n_run = 3000

sampler_mcmc0 = []
param_mcmc0 = []
param_mle0 = []
Cov_mle0 = []

for scan_i in range(len(Su_u_r)):
    print(scan_i)
    kb= 4*10**-3
    ind = ku<kb#k_1[scan_i]<kb
    F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
#    ################
#    #MCMC
#
    pos = [(np.array(bound0)[:,0] + np.diff(np.array(bound0),axis=1).T*np.random.rand(ndim0)).squeeze() for i in range(nwalkers)] 
    sampleri = emcee.EnsembleSampler(nwalkers, ndim0, LPST, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],F_obs[ind],bound0),))
    sampleri.run_mcmc(pos, n_run)
    
    pmcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(sampleri.chain[:, int(n_run*.15):, :].\
                                 reshape((-1, ndim0)), [16, 50, 84],axis=0)))
    pmcmc = np.array(list(pmcmc))
    param_mcmc0.append(pmcmc)
    
    res = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],param_init = pmcmc[:,0],bound= bound0)
    param_mle0.append(res.x)
    Cov_mle0.append(res.hess_inv(np.eye(ndim0)))

    sampler_mcmc0.append(sampleri.chain)

with open('Sampler_0.pkl', 'wb') as V_t:
     pickle.dump((sampler_mcmc0,param_mcmc0,param_mle0,Cov_mle0),V_t)

##########################
     
sampler_mcmc1 = []
param_mcmc1 = []
param_mle1 = []
Cov_mle1 = []
for scan_i in range(len(Su_u_r)):
    print(scan_i)
    kb= 4*10**-3
    ind = ku<kb#k_1[scan_i]<kb
    F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])  
#    ################
#    #MCMC
#
    pos = [(np.array(bound1)[:,0] + np.diff(np.array(bound1),axis=1).T*np.random.rand(ndim1)).squeeze() for i in range(nwalkers)] 
    sampleri = emcee.EnsembleSampler(nwalkers, ndim1, LPST, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],F_obs[ind],bound1),))
    sampleri.run_mcmc(pos, n_run)
    
    pmcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(sampleri.chain[:, int(n_run*.15):, :].\
                                 reshape((-1, ndim1)), [16, 50, 84],axis=0)))
    pmcmc = np.array(list(pmcmc))
    param_mcmc1.append(pmcmc)
    
    res = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],param_init = pmcmc[:,0],bound= bound)
    param_mle1.append(res.x)
    Cov_mle1.append(res.hess_inv(np.eye(ndim1)))
    
#    
    sampler_mcmc1.append(sampleri.chain)

    
with open('Sampler_1.pkl', 'wb') as V_t:
     pickle.dump((sampler_mcmc1,param_mcmc1,param_mle1,Cov_mle1),V_t)         
     


################################ 
#Loading of saved files
    
with open('Sampler_0.pkl', 'rb') as V_t:
     sampler_mcmc0,param_mcmc0,param_mle0,Cov_mle0 = pickle.load(V_t) 
with open('Sampler_1.pkl', 'rb') as V_t:
     sampler_mcmc1,param_mcmc1,param_mle1,Cov_mle1 = pickle.load(V_t) 
#################################
#To numpy arrays,     
param_mle_std0 = []
for scan_i in range(len(Su_u)):
    param_mle_std0.append(np.sqrt(np.diag(Cov_mle0[scan_i])))
parammcmc0=np.array(param_mcmc0)
parammle0=np.array(param_mle0)
parammlestd0=np.array(param_mle_std0)


param_mle_std1 = []
for scan_i in range(len(Su_u)):
    param_mle_std1.append(np.sqrt(np.diag(Cov_mle1[scan_i])))
parammcmc1=np.array(param_mcmc1)
parammle1=np.array(param_mle1)
parammlestd1=np.array(param_mle_std1)

################################
#Spectra plots
kb= 4*10**-3
ind = ku<kb
f, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
prod0 = np.array([-1,1,1])
prod1 = np.array([-1,1,-1,1,1])
for scan_i in range(len(Su_u)):
    ax1.cla()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
    ax1.plot(ku[ind],ku[ind]*F_obs[ind])
    ax1.plot(ku[ind],ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0],args=(param_fix0,param_ind0,ku[ind],)),'-o') 
    ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0]+
                    prod0*parammcmc0[scan_i,:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0]-
                    prod0*parammcmc0[scan_i,:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('$kF_{UU}(k)$')
    
    ax2.cla()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot(ku[ind],ku[ind]*F_obs[ind])
    ax2.plot(ku[ind],ku[ind]*spectra_peltier2(parammle[scan_i,:],args=(param_fix1,param_ind1,ku[ind],)),'-o') 
    ax2.fill_between(ku[ind],ku[ind]*spectra_peltier2(parammcmc1[scan_i,:,0]+
                    prod1*parammcmc1[scan_i,:,1],args=(param_fix1,param_ind1,ku[ind],)),ku[ind]*spectra_peltier2(parammcmc1[scan_i,:,0]-
                    prod1*parammcmc1[scan_i,:,2],args=(param_fix1,param_ind1,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
    ax2.set_xlabel('$k$')
    plt.pause(0.1)

kb= 4*10**-3
ind = ku<kb
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
for scan_i in range(len(Su_u)):
    ax.cla()
    ax.set_xscale('log')
    ax.set_yscale('log')
    F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
    ax.plot(ku[ind],ku[ind]*F_obs[ind])
    ax.plot(ku[ind],ku[ind]*spectra_peltier2(parammle[scan_i,:],args=(param_fix,param_ind,ku[ind],)),'-o') 
    ax.fill_between(ku[ind],ku[ind]*spectra_peltier2(parammcmc[scan_i,:,0]+
                    parammcmc[scan_i,:,1],args=(param_fix,param_ind,ku[ind],)),ku[ind]*spectra_peltier2(parammcmc[scan_i,:,0]-
                    parammcmc[scan_i,:,2],args=(param_fix,param_ind,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
    #ax.plot(ku[ind],ku[ind]*spectra_peltier2(parammcmc[scan_i,:,0],args=(np.r_[param_fix[:-1],1],param_ind,ku[ind],)),'--')
    plt.pause(0.1)

######################
#most likely
aux0 = np.zeros((len(Su_u),ndim0))
for scan_i in range(len(Su_u)):
    for dim in range(ndim0):
        h,b = np.histogram(sampler_mcmc0[scan_i][:,int(n_run*.15):,dim].flatten(),bins=50)
        b0 = .5*(b[1:]+b[:-1])[np.argmax(h)]
        aux0[scan_i,dim] = b0
#parammcmc0 = np.dstack((parammcmc0, aux))

#most likely
aux1 = np.zeros((len(Su_u),ndim1))
for scan_i in range(len(Su_u)):
    for dim in range(ndim1):
        h,b = np.histogram(sampler_mcmc1[scan_i][:,int(n_run*.15):,dim].flatten(),bins=50)
        b0 = .5*(b[1:]+b[:-1])[np.argmax(h)]
        aux1[scan_i,dim] = b0
#parammcmc1 = np.dstack((parammcmc1, aux)) 
    
######################
    
    
    
    
scan_i = 79
ft = 24
f = 24
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(x,y,U_out[scan_i],300,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)

#########################33     
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(x,y,V_out[scan_i],300,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_y$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200) 

fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
prod0 = np.array([-1,1,1])
prod1 = np.array([-1,1,-1,1,1])
ax1.cla()
ax1.set_xscale('log')
ax1.set_yscale('log')
F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
ax1.plot(ku[ind],ku[ind]*F_obs[ind],'o', markerfacecolor = 'w',  markeredgecolor = 'k', label = '$F_{UU, observed}$')
#ax1.plot(ku[ind],ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0],args=(param_fix0,param_ind0,ku[ind],)),'--',color = 'k', lw = 2, label = '$F_{UU,\:M_0,\:mean}$') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(parammle0[scan_i,:],args=(param_fix0,param_ind0,ku[ind],)),'-',color = 'k', label = '$F_{h,\:M_0}(k_1)$')
ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0]+
                prod0*parammcmc0[scan_i,:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(parammcmc0[scan_i,:,0]-
                prod0*parammcmc0[scan_i,:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k_1$', fontsize = ft)
ax1.set_ylabel('$kF_{h}(k_1)$', fontsize = ft)
ax1.legend(fontsize = ft-2,loc=4)
ax1.tick_params(labelsize=ft)
ax2.cla()
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.plot(ku[ind],ku[ind]*F_obs[ind],'o', markerfacecolor = 'w',  markeredgecolor = 'k', label = '$F_{UU, observed}$')
#ax2.plot(ku[ind],ku[ind]*spectra_peltier2(parammcmc1[scan_i,:,0],args=(param_fix1,param_ind1,ku[ind],)),'--',color = 'k', lw = 2, label = '$F_{UU, model\:M_1, mean}$') 
ax2.plot(ku[ind],ku[ind]*spectra_peltier2(parammle1[scan_i,:],args=(param_fix1,param_ind1,ku[ind],)),'-',color = 'k', label = '$F_{h,\:M_1}(k_1)$')
ax2.fill_between(ku[ind],ku[ind]*spectra_peltier2(parammcmc1[scan_i,:,0]+
                prod1*parammcmc1[scan_i,:,1],args=(param_fix1,param_ind1,ku[ind],)),ku[ind]*spectra_peltier2(parammcmc1[scan_i,:,0]-
                prod1*parammcmc1[scan_i,:,2],args=(param_fix1,param_ind1,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax2.set_xlabel('$k_1$', fontsize = ft)
ax2.legend(fontsize = ft-2,loc=4)
ax2.tick_params(labelsize=ft)

###########################################

scan_i = 79

fig = corner.corner(sampler_mcmc0[scan_i][:, int(n_run*.5):, :].reshape((-1, ndim0)),\
                        labels=lab0,\
                        quantiles=[0.16, 0.5, 0.84],truths=parammle0[scan_i,:],show_titles=True,
                        title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})

axes = np.array(fig.axes).reshape((ndim0, ndim0))

# Loop over the diagonal
for i in range(ndim0):
    ax = axes[i, 0]
    ax.set_ylabel(lab0[i],fontsize=18)
for i in range(ndim0):
    ax = axes[0, i]
    ax.set_ylabel(lab0[i],fontsize=18)

fig = corner.corner(sampler_mcmc1[scan_i][:, int(n_run*.5):, :].reshape((-1, ndim1)),\
                        labels=lab1,\
                        quantiles=[0.16, 0.5, 0.84],truths=parammle1[scan_i,:],show_titles=True)

for par in range(len(param_ind0)):
    plt.figure()  
    [plt.plot(sampler_mcmc0[scan_i][j,:,par],'-',color='grey') for j in range(60)]
    plt.plot(np.mean(sampler_mcmc0[scan_i][:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step')
    plt.ylabel(lab0[par])
    
    

for par in range(len(param_ind1)):
    plt.figure()  
    [plt.plot(sampler_mcmc1[scan_i][j,:,par],'-',color='grey') for j in range(60)]
    plt.plot(np.mean(sampler_mcmc1[scan_i][:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step')
    plt.ylabel(lab1[par])

#################################
#Parameters evolution
root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()


for paramind in range(ndim0):

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc0[:,paramind,0],lw=4,color='k',label='Expected value')
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc0[:,paramind,0]+parammcmc0[:,paramind,1],lw=2,color='grey')
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc0[:,paramind,0]-parammcmc0[:,paramind,2],lw=2,color='grey')
    ax.fill_between(np.arange(0,scan_i+1)+1,parammcmc0[:,paramind,0]+
                    parammcmc0[:,paramind,1],parammcmc0[:,paramind,0]-parammcmc0[:,paramind,2], color="grey",alpha=0.5, edgecolor="")
    ax.set_xlabel('Scan number')
    ax.set_ylabel(lab0[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')
    path = file_in_path+'/'+file0[paramind]+'_mcmc0'
    fig.savefig(path, bbox_inches='tight')
    
for paramind in range(ndim1):

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc1[:,paramind,0],lw=4,color='k',label='Expected value')
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc1[:,paramind,0]+parammcmc1[:,paramind,1],lw=2,color='grey')
    ax.plot(np.arange(0,scan_i+1)+1,parammcmc1[:,paramind,0]-parammcmc1[:,paramind,2],lw=2,color='grey')
    ax.fill_between(np.arange(0,scan_i+1)+1,parammcmc1[:,paramind,0]+
                    parammcmc1[:,paramind,1],parammcmc1[:,paramind,0]-parammcmc1[:,paramind,2], color="grey",alpha=0.5, edgecolor="")
    ax.set_xlabel('Scan number')
    ax.set_ylabel(lab1[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')
    path = file_in_path+'/'+file1[paramind]+'_mcmc1'
    fig.savefig(path, bbox_inches='tight')

scan_i = 70
for paramind in range(ndim0):

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,scan_i)+1,parammle0[:scan_i,paramind],lw=4,color='k',label='Expected value')
    ax.plot(np.arange(0,scan_i)+1,parammle0[:scan_i,paramind]+parammlestd0[:scan_i,paramind],lw=2,color='grey')
    ax.plot(np.arange(0,scan_i)+1,parammle0[:scan_i,paramind]-parammlestd0[:scan_i,paramind],lw=2,color='grey')
    ax.fill_between(np.arange(0,scan_i)+1,parammle0[:scan_i,paramind]+parammlestd0[:scan_i,paramind],parammle0[:scan_i,paramind]-parammlestd0[:scan_i,paramind], color="grey",alpha=0.5, edgecolor="")
    
    ax.fill_between(np.arange(0,scan_i)+1,parammcmc0[:scan_i,paramind,0]+
                    parammcmc0[:scan_i,paramind,1],parammcmc0[:scan_i,paramind,0]-parammcmc0[:scan_i,paramind,2], color="pink",alpha=0.5, edgecolor="")
    
    ax.set_xlabel('Scan number')
    ax.set_ylabel(lab0[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')
    path = file_in_path+'/'+file0[paramind]+'_mle0'
    ax.set_ylim(bound0[paramind])
    fig.savefig(path, bbox_inches='tight')
    

for paramind in range(ndim1):

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,scan_i)+1,parammle1[:scan_i,paramind],lw=4,color='k',label='Expected value')
    ax.plot(np.arange(0,scan_i)+1,parammle1[:scan_i,paramind]+parammlestd1[:scan_i,paramind],lw=2,color='grey')
    ax.plot(np.arange(0,scan_i)+1,parammle1[:scan_i,paramind]-parammlestd1[:scan_i,paramind],lw=2,color='grey')
    ax.fill_between(np.arange(0,scan_i)+1,parammle1[:scan_i,paramind]+parammlestd1[:scan_i,paramind],parammle1[:scan_i,paramind]-parammlestd1[:scan_i,paramind], color="grey",alpha=0.5, edgecolor="")
    
    ax.fill_between(np.arange(0,scan_i)+1,parammcmc1[:scan_i,paramind,0]+
                    parammcmc1[:scan_i,paramind,1],parammcmc1[:scan_i,paramind,0]-parammcmc1[:scan_i,paramind,2], color="pink",alpha=0.5, edgecolor="")
    
    ax.set_xlabel('Scan number')
    ax.set_ylabel(lab1[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')
    path = file_in_path+'/'+file1[paramind]+'_mle1'
    ax.set_ylim(bound1[paramind])
    fig.savefig(path, bbox_inches='tight')
    



###############
    
###############
    
###############
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
param_init0 = [.85,23,1000,1, 1.6,.009,200,.36,500,2]
param_init = param_init0
#param_init = [.5,10,.1,.001, .1,0,10,.1,10,2]
bound = [(.845,.855),(22.5,23.5),(0.001,10000),(10**-4,30),(1.55,1.65),(0.0085,.0095),(1,1000),(10**-4,1),(10,10000),(0,4)]     
param = np.zeros((1000,len(param_init)))
for scan_i in range(0,945):
    print(scan_i)
    kb= 10**-2
    ind = k_1[scan_i]<kb
    F_obs = .5*(Su_u[scan_i] + Sv_v[scan_i])

    res = spectra_fitting(F_obs[ind],spectra_peltier,spectra_noise,
                      k_1[scan_i][ind],param_init = param_init,bounds= bound)   
    param[scan_i,:] = res.x
    param_init = .5*(param_init0 + res.x)

fig, ax = plt.subplots()

for scan_i in range(0,945):
    ax.cla()
    kb= 10**-2
    ind = k_1[scan_i]<kb
    F_obs = .5*(Su_u[scan_i] + Sv_v[scan_i])
    ax.set_title('Scan  %i' %scan_i)
    ax.plot(k_1[scan_i][ind],k_1[scan_i][ind]*F_obs[ind])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(k_1[scan_i][ind],k_1[scan_i][ind]*spectra_peltier(param[scan_i,:10],args=(k_1[scan_i][ind],)),'-o')
    plt.pause(0.01)

plt.figure()
plt.plot(k1[scan_i],k1[scan_i]*Sv_v[scan_i])
plt.plot(k1_tri[scan_i],k1_tri[scan_i]*Sv_v_tri[scan_i])
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(k1[scan_i],k1[scan_i]*Su_v[scan_i])
plt.plot(k1_tri[scan_i],k1_tri[scan_i]*Su_v_tri[scan_i])
plt.xscale('log')
plt.yscale('log')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

U_mean = np.nanmean(U_out[scan_i].flatten())
V_mean = np.nanmean(V_out[scan_i].flatten())
# Wind direction
gamma = np.arctan2(V_mean,U_mean)
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U_out[scan_i].flatten(),V_out[scan_i].flatten()]).T
vel = np.dot(T,vel)
grid_s = np.meshgrid(x,y)
X = np.array(np.c_[grid_s[0].flatten(),grid_s[1].flatten()]).T
X = np.dot(T,X)   
U_t = np.reshape(vel[0,:],U_out[scan_i].shape)
V_t = np.reshape(vel[1,:],V_out[scan_i].shape)            
xp = X[0,:]
yp = X[1,:]

xp = np.reshape(xp,grid_s[0].shape)
yp = np.reshape(yp,grid_s[0].shape)

plt.figure()
plt.contourf(xp,yp,U_t,100,cmap='jet')
fig.colorbar(im)


U_mean = np.nanmean(U_tri[scan_i].flatten())
V_mean = np.nanmean(V_tri[scan_i].flatten())
# Wind direction
gamma = np.arctan2(V_mean,U_mean)
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U_tri[scan_i].flatten(),V_tri[scan_i].flatten()]).T
vel = np.dot(T,vel)
X = np.array(np.c_[tri.x,tri.y]).T
X = np.dot(T,X)   
U_t = vel[0,:]
V_t = vel[1,:]             
tri_t = Triangulation(X[0,:],X[1,:])

plt.figure()
plt.triplot(tri_t)
plt.tricontourf(tri_t,U_t,100,cmap='jet')
fig.colorbar(im)

plt.figure()
plt.tricontourf(tri,V_tri[scan_i],300,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(x,y,U_out[i],300,cmap='jet')
plt.colorbar()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
for scan_n in range(170):
    print(scan_n)
    ax.cla()
    ax.contourf(grd[0],grd[1],V_out[scan_n],np.linspace(7,18,300),cmap='jet')
    plt.pause(.001)











