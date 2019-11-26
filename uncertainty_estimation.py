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

# In[Plot text format]
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
end = 8750
ind0 = (df_0.scan>=start) & (df_0.scan<=end)
ind1 = (df_1.scan>=start) & (df_1.scan<=end)

U_out, V_out, grd, _ = wf.direct_wf_rec(df_0.loc[ind0], df_1.loc[ind1], tri, d,N_grid = 512)

x = grd[0][0,:]
y = grd[1][:,0]

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
     
with open('UV_out2', 'rb') as V_t:
     U_out, V_out = pickle.load(V_t)

###########################  
#MLC 


bound_tot = [(1000,4000),(0,2),(0,2), (0,2),(100,300), (0.2,3), (0,2), (0,.1),(10,1000),(0,10)]
#bound_tot = [(1000,4000),(0,10),(0,2), (0,2),(100,300), (1,3), (0,2), (0,.1),(10,1000),(0,10)]
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

##############################
def get_convergence_statistic(i, sampler_chain, nwalkers, convergence_length=10,
                              convergence_period=10):
    s = sampler_chain[:, i-convergence_length+1:i+1, :]
    within_std = np.mean(np.var(s, axis=1), axis=0)
    per_walker_mean = np.mean(s, axis=1)
    mean = np.mean(per_walker_mean, axis=0)
    between_std = np.sqrt(np.mean((per_walker_mean-mean)**2, axis=0))
    W = within_std
    B_over_n = between_std**2 / convergence_period
    Vhat = ((convergence_period-1.)/convergence_period * W
            + B_over_n + B_over_n / float(nwalkers))
    c = Vhat/W
    return i - convergence_period/2, c

###############################

scan_i = 50
kb= 4*10**-3
ind = ku<kb#k_1[scan_i]<kb
F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
n_run = 10000
nwalkers = 300 
param_ind1 = [1,3,5,7,9]
param_ind_not1 = [e for e in param_set if e not in set(param_ind1)]
param_init1 = [param_init_tot[i] for i in param_ind1]
param_fix1 = [param_fix_tot[i] for i in param_ind_not1]
bound1 = [bound_tot[i] for i in param_ind1]
lab1 = [lab_tot[i] for i in param_ind1]
ndim1= len(param_init1)
file1 = [file_tot[i] for i in param_ind1]

pos = [(np.array(bound1)[:,0] + np.diff(np.array(bound1),axis=1).T*np.random.rand(ndim1)).squeeze() for i in range(nwalkers)] 
sampleri1 = emcee.EnsembleSampler(nwalkers, ndim1, LPST, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],F_obs[ind],bound1),))
sampleri1.run_mcmc(pos, n_run)

##############
#Potential Scale reduction factor
c_length = 10
c_period = 10
PSRF1 = []
PSRFx1 = []
for i in range(c_length,n_run-c_length):
    idx, c = get_convergence_statistic(i, sampleri1.chain, nwalkers, 
                                       convergence_length=c_length, convergence_period=c_period)
    PSRF1.append(c)
    PSRFx1.append(idx)
PSRF1 = np.array(PSRF1)
PSRFx1 = np.array(PSRFx1)
f, axes = plt.subplots(ndim1,1,sharey=True)
for dim in range(ndim1):
    axes[dim].plot(PSRFx1,np.array(PSRF1)[:,dim],label=lab1[dim], color='k')

    axes[dim].set_ylabel('$R($'+lab1[dim]+'$)$',fontsize=18)

axes[dim].set_xlabel('$N$',fontsize=18)
#############

plt.figure()
plt.hist(sampleri1.acceptance_fraction,bins=20,edgecolor='k',facecolor='grey') 
plt.xlabel('$Acceptance\:rate$',fontsize=18) 
plt.ylabel('$Counts$',fontsize=18)  
ind_accep1 = (sampleri1.acceptance_fraction<.5)&(sampleri1.acceptance_fraction>.2)
plt.text(0.01, 0.98, '(b)', transform=ax.transAxes, fontsize=24,verticalalignment='top')
    
pmcmc1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampleri1.chain[ind_accep1 , int(n_run*.15):, :].\
                             reshape((-1, ndim1)), [16, 50, 84],axis=0)))

pmcmc1 = np.array(list(pmcmc1))

res1 = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],param_init = pmcmc1[:,0],bound= bound1)

aux1 = np.zeros(res1.x.shape)
for dim in range(ndim1):
    h,b = np.histogram(sampleri1.chain[ind_accep1, int(n_run*.15):, dim].flatten(),bins=100)
    b1 = .5*(b[1:]+b[:-1])[np.argmax(h)]
    aux1[dim] = b1

fig = corner.corner(sampleri1.chain[ind_accep1, int(n_run*.15):, :].reshape((-1, ndim1)),\
                        labels=lab1,\
                        quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        truths=res1.x,title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})

tau1 = sampleri1.get_autocorr_time(low=10, high=None, step=1, c=2, fast=True)


tau = np.zeros((ndim1,30))
for dim in range(ndim1):
    for i,f in enumerate(np.linspace(.5,1,30)):    
        tau[dim,i] = emcee.autocorr.integrated_time(np.mean(sampleri1.chain[:,:int(f*n_run),dim],axis=0),
           low=10, high=None, step=1, c=3, full_output=False, axis=0, fast=False)
    plt.plot((np.linspace(.3,1,30)*n_run).astype(int),tau[dim,:]/tau1[dim],label=lab1[dim])
    plt.xlabel('$N$')
    plt.ylabel('$t / t_0$')
    plt.legend()
# MCMC error
sigma1 = np.zeros((ndim1,nwalkers))
for dim in range(ndim1):
    plt.figure()
    sigma1[dim,:] = np.var(sampleri1.chain[:,:,dim],axis=1)/len(sampleri1.chain[:,:,dim])*tau1[dim]/np.mean(sampleri1.chain[:,:,dim],axis=1)
    plt.hist(sigma1[dim,:],bins=20,edgecolor='k',facecolor='grey') 
    plt.xlabel('$\sigma_{chain}\:/\:\mu,\:$' +lab1[dim],fontsize=18) 
    plt.ylabel('$Counts$',fontsize=18)  

    


for par in range(len(param_ind1)):
    plt.figure()  
    [plt.plot(sampleri1.chain[j,:,par],'-',color='grey') for j in np.arange(nwalkers)[ind_accep1]]
    plt.plot(np.mean(sampleri1.chain[:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step',fontsize=18)
    plt.ylabel(lab1[par],fontsize=18)
    
np.var(sampleri0.chain[:,:5000,0].flatten())/len(sampleri0.chain[:,:5000,0].flatten())*tau0[1]
    

prod1 = np.array([-1,1,-1,1,1])
kb= 4*10**-3
ind = ku<kb
f, ax1 = plt.subplots()
prod0 = np.array([-1,1,1])
ax1.set_xscale('log')
ax1.set_yscale('log')
F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res1.x,args=(param_fix1,param_ind1,ku[ind],)),'-o') 
#ax1.plot(ku[ind],ku[ind]*spectra_peltier2(aux1,args=(param_fix1,param_ind1,ku[ind],)),'-o') 
ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc1[:,0]+
                prod1*pmcmc1[:,1],args=(param_fix1,param_ind1,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc1[:,0]-
                prod1*pmcmc1[:,2],args=(param_fix1,param_ind1,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')


    

##############################
scan_i = 30
kb= 4*10**-3
ind = ku<kb#k_1[scan_i]<kb
F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
n_run = 10000
nwalkers = 300    
param_ind0 = [5,7,9]
param_ind_not0 = [e for e in param_set if e not in set(param_ind0)]
param_init0 = [param_init_tot[i] for i in param_ind0]
param_fix0 = [param_fix_tot[i] for i in param_ind_not0]
bound0 = [bound_tot[i] for i in param_ind0]
lab0 = [lab_tot[i] for i in param_ind0]
ndim0 = len(param_init0)
file0 = [file_tot[i] for i in param_ind0]

pos = [(np.array(bound0)[:,0] + np.diff(np.array(bound0),axis=1).T*np.random.rand(ndim0)).squeeze() for i in range(nwalkers)]
#pos = emcee.utils.sample_ball(pmcmc0[:,0] , pmcmc0[:,1], size=nwalkers) 
sampleri0 = emcee.EnsembleSampler(nwalkers, ndim0, LPST, a = 5, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],F_obs[ind],bound0),))
sampleri0.run_mcmc(pos, n_run)


##############
c_length = 30
c_period = 10
PSRF0 = []
PSRFx0 = []
for i in range(c_length,n_run-c_length):
    idx, c = get_convergence_statistic(i, sampleri0.chain, nwalkers, 
                                       convergence_length=c_length, convergence_period=c_length)
    PSRF0.append(c)
    PSRFx0.append(idx)
PSRF0 = np.array(PSRF0)
PSRFx0 = np.array(PSRFx0)
f, axes = plt.subplots(ndim0,1,sharey=True)
for dim in range(ndim0):
    axes[dim].plot(PSRFx0,np.array(PSRF0)[:,dim],label=lab0[dim],color='k')

    axes[dim].set_ylabel('$R($'+lab0[dim]+'$)$',fontsize=18)
axes[dim].set_xlabel('$N$',fontsize=18)
    #axes[dim].legend(fontsize=18,loc=1)
#############
    

    

plt.figure()
plt.hist(sampleri0.acceptance_fraction,bins=20,edgecolor='k',facecolor='grey') 
plt.xlabel('$Acceptance\:rate$',fontsize=18) 
plt.ylabel('$Counts$',fontsize=18) 
ind_accep0 = (sampleri0.acceptance_fraction<.5)&(sampleri0.acceptance_fraction>.2)
plt.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=24,verticalalignment='top')

pmcmc0 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampleri0.chain[ind_accep , int(n_run*.15):, :].\
                             reshape((-1, ndim0)), [16, 50, 84],axis=0)))

pmcmc0 = np.array(list(pmcmc0))

res0 = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],param_init = pmcmc0[:,0],bound= bound0)

aux0 = np.zeros(res.x.shape)
for dim in range(ndim0):
    h,b = np.histogram(sampleri0.chain[ind_accep, int(n_run*.15):, dim].flatten(),bins=100)
    b0 = .5*(b[1:]+b[:-1])[np.argmax(h)]
    aux0[dim] = b0

fig = corner.corner(sampleri0.chain[ind_accep0, int(n_run*.15):, :].reshape((-1, ndim0)),\
                        labels=lab0,\
                        quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        truths=aux0,title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})

tau0 = sampleri0.get_autocorr_time(low=10, high=None, step=1, c=5, fast=True)

tau = np.zeros((ndim0,30))
for dim in range(ndim0):
    for i,f in enumerate(np.linspace(.3,1,30)):    
        tau[dim,i] = emcee.autocorr.integrated_time(np.mean(sampleri0.chain[:,:int(f*n_run),dim],axis=0),
           low=10, high=None, step=1, c=5, full_output=False, axis=0, fast=False)
    plt.plot((np.linspace(.3,1,30)*n_run).astype(int),tau[dim,:]/tau0[dim],label=lab0[dim])
    plt.xlabel('$N$')
    plt.ylabel('$t / t_0$')
    plt.legend()

    
    
# MCMC error
sigma0 = np.zeros((ndim0,nwalkers))
for dim in range(ndim0):
    plt.figure()
    sigma0[dim,:] = np.var(sampleri0.chain[:,:,dim],axis=1)/len(sampleri0.chain[:,:,dim])*tau0[dim]/np.mean(sampleri0.chain[:,:,dim],axis=1)
    plt.hist(sigma0[dim,:],bins=20,edgecolor='k',facecolor='grey') 
    plt.xlabel('$\sigma_{chain}\:/\:\mu,\:$' +lab0[dim],fontsize=18) 
    plt.ylabel('$Counts$',fontsize=18) 

for par in range(len(param_ind0)):
    plt.figure()  
    [plt.plot(sampleri0.chain[j,:,par],'-',color='grey') for j in np.arange(nwalkers)[ind_accep0]]
    plt.plot(np.mean(sampleri0.chain[:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step',fontsize=18)
    plt.ylabel(lab0[par],fontsize=18)
    
    

prod0 = np.array([-1,1,1])
kb= 4*10**-3
ind = ku<kb
f, ax1 = plt.subplots()
prod0 = np.array([-1,1,1])
ax1.set_xscale('log')
ax1.set_yscale('log')
F_obs = .5*(Su_u_r[scan_i] + Sv_v_r[scan_i])
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res.x,args=(param_fix0,param_ind0,ku[ind],)),'-o') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(aux0,args=(param_fix0,param_ind0,ku[ind],)),'-o') 
ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc[:,0]+
                prod0*pmcmc[:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc[:,0]-
                prod0*pmcmc[:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')


#################################    
    
    
##################################
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
im = ax.contourf(y,x,np.flipud(V_out[scan_i].T),300,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_xlabel(r'West-East [m]', fontsize=f)#, weight='bold')
ax.set_ylabel(r'North-South [m]', fontsize=f)#, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_ylim(-3500,3500)
ax.set_xlim(-7000,-200) 

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
    ax = axes[ndim0-1, i]
    ax.set_xlabel(lab0[i],fontsize=18)

fig = corner.corner(sampler_mcmc1[scan_i][:, int(n_run*.5):, :].reshape((-1, ndim1)),\
                        labels=lab1,\
                        quantiles=[0.16, 0.5, 0.84],truths=parammle1[scan_i,:],show_titles=True,
                        title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})

axes = np.array(fig.axes).reshape((ndim1, ndim1))

# Loop over the diagonal
for i in range(ndim1):
    ax = axes[i, 0]
    ax.set_ylabel(lab1[i],fontsize=18)
for i in range(ndim1):
    ax = axes[ndim1-1, i]
    ax.set_xlabel(lab1[i],fontsize=18)
    
    

for par in range(len(param_ind0)):
    plt.figure()  
    [plt.plot(sampler_mcmc0[scan_i][j,:,par],'-',color='grey') for j in range(60)]
    plt.plot(np.mean(sampler_mcmc0[scan_i][:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step',fontsize=18)
    plt.ylabel(lab0[par],fontsize=18)
    
    

for par in range(len(param_ind1)):
    plt.figure()  
    [plt.plot(sampler_mcmc1[scan_i][j,:,par],'-',color='grey') for j in range(60)]
    plt.plot(np.mean(sampler_mcmc1[scan_i][:,:,par],axis = 0),lw=2,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)+np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    #plt.plot(np.mean(sampler_mcmc[scan_i][:,:,par],axis = 0)-np.std(sampler_mcmc[scan_i][:,:,par],axis = 0),'--',alpha=0.5,color='k')
    plt.xlabel('Step',fontsize=18)
    plt.ylabel(lab1[par],fontsize=18)

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
    










