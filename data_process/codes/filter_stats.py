# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize, abspath
from os import listdir
import sys
import tkinter as tkint

import tkinter.filedialog

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


import ppiscanprocess.filtering as filt
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectraconstruction as spec

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from os import listdir
from os.path import isfile, join

import matplotlib.ticker as ticker

from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import RobustScaler



# In[]

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

# In[Data loading]
# To do: it is necessary to link the mask file with the source file
root = tkint.Tk()
file_df = tkint.filedialog.askopenfilenames(parent=root,title='Choose a data file')
root.destroy()
root = tkint.Tk()
file_mask = tkint.filedialog.askopenfilenames(parent=root,title='Choose a mask file')
root.destroy()

# In[]

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
vel_lab = np.array(['range_gate','ws','CNR','Sb'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))
sirocco_loc = np.array([6322832.3,0])
vara_loc = np.array([6327082.4,0])
d = vara_loc-sirocco_loc

# In[]

DF = []

for f,m in zip(file_df,file_mask):
    with open(m, 'rb') as reader:
        mask = pickle.load(reader)        
    df = pd.read_csv(f,sep=";", header=None) 
    df.columns = labels      
    df['scan'] = df.groupby('azim').cumcount()
  
    mask_CNR = (df.CNR>-24) & (df.CNR<-8)
    mask_CNR.columns =  mask.columns
    mask.mask(mask_CNR,other=False,inplace=True)
    df.ws=df.ws.mask(mask)
    DF.append(df)
    
################ Figure for CNR comb #############
ind_can = (df.scan > 9000) & (df.scan < 9030)

ax = df.loc[ind_can].plot.scatter(x='ws', y='CNR', c='grey', s=20, edgecolor='k')  
ax.set_xlabel('$V_{LOS}$',fontsize=16) 
ax.set_ylabel('$CNR$',fontsize=16)  
ax.plot([-40,40],[-27,-27],c='r',lw=2)
ax.set_xlim([-35,35])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')

ind_can = (df.scan > 9000) & (df.scan < 9030)
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax.plot_surface(ws_plane,r_plane,CNR_plane, color = 'r', alpha = .5, zorder=1)
#face1 = mp3d.art3d.Poly3DCollection([top], alpha=0.5, linewidth=1)
#alpha = 0.5
#face1.set_facecolor((0, 0, 1, alpha))
#ax.add_collection3d(face1)
idxcnr = df.loc[ind_can].CNR.values.flatten() >-27
idxws = (df.loc[ind_can].ws.values.flatten()>-23) & (df.loc[ind_can].ws.values.flatten()<0)
idxr = (df.loc[ind_can].range_gate.values.flatten()>4000)
idxwsr = idxws & ~idxcnr & idxr
colors = np.array(['grey']*len(idxcnr))
colors[~idxcnr] = 'red'
colors[idxwsr] = 'blue'
#
#ax.scatter(df.loc[ind_can].ws.values.flatten()[~idxcnr], df.loc[ind_can].range_gate.values.flatten()[~idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[~idxcnr], c = 'red', alpha=0.2, s=20, edgecolor='k',zorder = 1)
#ax.scatter(df.loc[ind_can].ws.values.flatten()[idxcnr], df.loc[ind_can].range_gate.values.flatten()[idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[idxcnr], c = 'grey',s=20, edgecolor='k',zorder = 10)   

ax.scatter(df.loc[ind_can].ws.values.flatten(), df.loc[ind_can].range_gate.values.flatten(),
           df.loc[ind_can].CNR.values.flatten(), c = colors, s=20, edgecolor='k')
 
ax.set_xlabel('$V_{LOS}$',fontsize=16) 
ax.set_zlabel('$CNR$',fontsize=16)  
ax.set_ylabel('$Range\:gate$',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.grid(False)
##################################################
       
phi0w = DF[0].azim.unique()
phi1w = DF[0].azim.unique()
r0w = np.array(DF[0].iloc[(DF[0].azim==min(phi0w)).nonzero()[0][0]].range_gate)
r1w = np.array(DF[0].iloc[(DF[0].azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_vaw, phi_vaw = np.meshgrid(r0w, np.radians(phi0w)) # meshgrid
r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

treew,triw,wvaw,neighvaw,indexvaw,wsiw,neighsiw,indexsiw = wr.grid_over2((r_vaw,
                                                   phi_vaw),(r_siw, phi_siw),d)
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(10000,13000):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),DF[1].ws.loc[DF[1].scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(0,98):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df.ws.loc[df.scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(0,98):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),dfp.ws.loc[dfp.scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)
               
# In[Histograms]
    
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Alef']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

df_raw = pd.read_csv(file_df[0],sep=";", header=None)
df_raw.columns = labels      
df_raw['scan'] = df_raw.groupby('azim').cumcount()
df_clust2 = df_raw.copy()
df_clust2.ws=df_clust2.ws.mask(mask)  
df_median=filt.data_filt_median(df_raw,lim_m=6,lim_g=100) 

ws_raw = df_raw.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
ws_clust2 = df_clust2.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
ws_median = df_median.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
ws_raw_g = df_raw.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
ws_clust2_g = df_clust2.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
ws_median_g = df_median.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]

#ws_median = ws_median[~np.isnan(ws_median)]
#ws_clust2 = ws_clust2[~np.isnan(ws_clust2)]
#
#ws_median_g = ws_median_g[~np.isnan(ws_median_g)]
#ws_clust2_g = ws_clust2_g[~np.isnan(ws_clust2_g)]

r = df_raw.range_gate.values
a = (np.ones((r.shape[1],1))*df_raw.azim.values.flatten()).transpose()
r = r.flatten().astype(int)
a = a.flatten().astype(int)
n = np.max(df_raw.scan.values)
lim_m = [4,6,8]
ws_median = np.zeros((len(lim_m),len(ws_raw)))
ws_median_g = np.zeros((len(lim_m),len(ws_raw_g)))
recovery_median = np.zeros((len(lim_m),45*198))

for i,l in enumerate(lim_m[1:],1):
    df_median=filt.data_filt_median(df_raw,lim_m=l,lim_g=20)
    mask_median = np.isnan(df_median.ws).values
    mask_median = (~mask_median.flatten()).astype(int)
    ws_median[i,:] = df_median.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
    ws_median_g[i,:] = df_median.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
    df_median = None
    df_s_median = pd.DataFrame({'r': r, 'a': a, 'm': mask_median})
    recovery_median[i,:] = df_s_median.groupby(['a', 'r'])['m'].agg('sum').values/n
    df_s_median = None

den=False

i = 2

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=30,alpha=0.5,density=den)
h_med,_,_ = plt.hist(ws_median[i,:].flatten(),bins=bine,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust2.flatten(),bins=bine,alpha=0.5,density=den)

h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=30,alpha=0.5,density=den)
h_med_g,_,_ = plt.hist(ws_median_g[i,:].flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust2_g.flatten(),bins=bine_g,alpha=0.5,density=den)

bine=np.linspace(-5,5,50)
plt.figure()
plt.hist(((ws_median-np.mean(ws_median))/np.std(ws_median)).flatten(),bins=bine,alpha=0.5,density=True,histtype='step',color='blue',lw=2)
plt.hist(((ws_clust2-np.mean(ws_clust2))/np.std(ws_clust2)).flatten(),bins=bine,alpha=0.5,density=True,histtype='step',color='red',lw=2)
plt.plot(bine, np.exp(-bine**2/2)/np.sqrt(2*np.pi),'--k',lw=2)
plt.yscale('log')
plt.ylim(10**-3,1)
plt.xlabel(r'$V_{LOS}/\sigma$')
plt.ylabel('Prob. Density')

# In[]

phi1w = df_raw.azim.unique()
r1w = np.array(df_raw.iloc[(df_raw.azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

mask_median = np.isnan(df_median.ws).values

mask_median = (~mask_median.flatten()).astype(int)

df_s_median = pd.DataFrame({'r': r, 'a': a, 'm': mask_median})
mask_median = []

recovery_median = df_s_median.groupby(['a', 'r'])['m'].agg('sum').values/n
recovery_median = np.reshape(recovery_median, r_siw.shape) 

f, ax3 = plt.subplot(2, 2, 3)
im = ax3.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_median[2,:],phi_siw.shape),levels=np.linspace(.7,1,100),cmap='jet')
f.colorbar(im)
ax3.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax3.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 
    
r = df_raw.range_gate.values
a = (np.ones((r.shape[1],1))*df_raw.azim.values.flatten()).transpose()
r_clust = r[:mask.shape[0],:].astype(int)
a_clust = a[:mask.shape[0],:].astype(int)
mask = ((~mask.values)).astype(int)


df_s_clust = pd.DataFrame({'r': r_clust.flatten(), 'a': a_clust.flatten(), 'mask': mask.flatten()})
recovery_clust=np.reshape(df_s_clust.groupby(['a', 'r'])['mask'].agg('sum').values,r_siw.shape)
recovery_clust=recovery_clust/n
plt.figure()
plt.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),recovery_clust,levels=np.linspace(.6,1,100),cmap='jet')
plt.colorbar()

df_ws_clust = pd.DataFrame({'r': r_clust.flatten(), 'a': a_clust.flatten(), 'm': df_clust2.ws.values[:mask.shape[0],:].flatten()})
recovery_clust_std=np.reshape(df_ws_clust.groupby(['a', 'r'])['m'].agg('std').values,r_siw.shape)

df_ws_median = pd.DataFrame({'r': r, 'a': a, 'm': df_median.ws.values.flatten()})
recovery_median_std = df_ws_median.groupby(['a', 'r'])['m'].agg('std').values

# In[]
plt.figure()

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=30,alpha=0.5,density=den)
h_med,_,_ = plt.hist(ws_median.flatten(),bins=bine,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust2.flatten(),bins=bine,alpha=0.5,density=den)

h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=30,alpha=0.5,density=den)
h_med_g,_,_ = plt.hist(ws_median_g.flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust2_g.flatten(),bins=bine_g,alpha=0.5,density=den)


fmt = FormatScalarFormatter("%.2f")

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

f, axs = plt.subplots(2,2, sharey='row')
axs=axs.flatten()
ax1=axs[0];ax2=axs[1];ax3=axs[2];ax4=axs[3]

#plt.step(.5*(bine[1:]+bine[:-1]),h_raw,color='blue',lw=3,label=r'Raw data')
#ax1 = plt.subplot(2, 2, 1)
ax1.step(.5*(bine[1:]+bine[:-1]),h_med/h_raw,color='black',lw=3,label=r'Median')
ax1.step(.5*(bine[1:]+bine[:-1]),h_clust/h_raw,color='red',lw=3,label=r'Clustering')
#plt.yscale('log')
ax1.set_xlabel(r'$V_{LOS}$',fontsize=24)
ax1.set_ylabel(r'Data recovery fraction',fontsize=24)
ax1.legend(loc=(.65,.6),fontsize=24)
ax1.set_xlim(-30,30)
ax1.tick_params(labelsize=24)

#plt.step(.5*(bine[1:]+bine[:-1]),h_raw,color='blue',lw=3,label=r'Raw data')
ax2.step(.5*(bine[1:]+bine[:-1]),h_med_g/h_raw_g,color='black',lw=3,label=r'Median')
ax2.step(.5*(bine[1:]+bine[:-1]),h_clust_g/h_raw_g,color='red',lw=3,label=r'Clustering')
#plt.yscale('log')
ax2.set_xlabel(r'$V_{LOS}$',fontsize=24)
#ax2.set_ylabel(r'Data recovery fraction',fontsize=18)
ax2.legend(loc=(.65,.6),fontsize=24)
ax2.set_xlim(-35,30)
ax2.tick_params(labelsize=24)


im1 = ax3.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_median,phi_siw.shape),levels=np.linspace(.7,1,10),cmap='jet')

#divider1 = make_axes_locatable(ax3)
#cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#cbar1 = f.colorbar(im1, cax=cax1)
#cbar1.ax.set_ylabel("Data recovery fraction")
ax3.set_ylabel(r'West-East [m]', fontsize=24, weight='bold')
ax3.set_xlabel(r'North-South [m]', fontsize=24, weight='bold') 
ax3.tick_params(labelsize=24)
    
#f, ax4 = plt.subplot(2, 2, 4)
im2 = ax4.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_clust,phi_siw.shape),levels=np.linspace(.7,1,10),cmap='jet')
divider2 = make_axes_locatable(ax4)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = f.colorbar(im2, cax=cax2,format=ticker.FuncFormatter(fm))

cbar2.ax.set_ylabel("Data recovery fraction", fontsize=24)
cbar2.ax.tick_params(labelsize=24)
#ax4.set_ylabel(r'West-East [m]', fontsize=18, weight='bold')
ax4.set_xlabel(r'North-South [m]', fontsize=24, weight='bold') 
ax4.tick_params(labelsize=24)


ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=24,
        verticalalignment='top')
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=24,
       verticalalignment='top')
ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=24,
        verticalalignment='top')
ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontsize=24,
        verticalalignment='top') 


# In[Synthetic lidars filtering test]
############################## 
##############################    
# In[Synthetic data loading]
root = tkint.Tk()
file_vlos = tkint.filedialog.askopenfilenames(parent=root,title='Choose vlos scans files')
root.destroy()

root = tkint.Tk()
file_vlos_noise = tkint.filedialog.askopenfilenames(parent=root,title='Choose contaminated vlos scans files')
root.destroy()

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

###############
# Dataframe creation

r_0 = np.linspace(150,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(150,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)

ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
utot = np.linspace(15,25,5)
Dir = np.linspace(90,270,5)*np.pi/180

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

#####CNR#######
DT = int(600/45)
scanlist = np.arange(0,df.scan.max(),DT)
CNR = df.CNR.values
r = df.range_gate.values
scan = np.array(list(df.scan.values)*r.shape[1]).transpose()
azim = np.array(list(df.azim.values)*r.shape[1]).transpose()
r_unique = r[0,:]
azim_unique = df.azim.unique()
range_gate,azim = np.meshgrid(r_unique,azim_unique)
K_CNR = []
for r_i in r_unique:
    print(r_i)
    K_CNR_t = []
    for t in scanlist[:-1]:
        print(t)
        ind_r = r.flatten() == r_i
        ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
        ind_tot = (ind_r) & (ind_t)
        K_CNR_t.append(sp.stats.gaussian_kde(CNR.flatten()[ind_tot], bw_method='silverman'))
    K_CNR.append(K_CNR_t)
    
for r_i in r_unique:                                 # random scan 
    t = np.random.randint(scanlist[0],scanlist[-1])

    ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
    #ind_tot = (ind_r) & (ind_t)
    x = r.flatten()*np.cos(azim.flatten()*np.pi/180)
    y = r.flatten()*np.sin(azim.flatten()*np.pi/180)
    X = np.c_[CNR.flatten()[ind_t],x[ind_t],y[ind_t]]
    transformer = RobustScaler(quantile_range=(25, 75))
    transformer.fit(X)
    X = transformer.transform(X)  
    
    from sklearn.model_selection import GridSearchCV
    
    bandwidths = 10 ** np.linspace(-2, 0, 20)
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=2)
    grid.fit(X)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=.021)
    kde.fit(X)
    samples = kde.sample(n_samples=45*198)
    samples = transformer.inverse_transform(samples)
    x_int = r_0_g*np.cos(phi_0_g)
    y_int = r_0_g*np.sin(phi_0_g)
    X_int = sp.interpolate.griddata(samples[:,1:],samples[:,0], (x_int.flatten(), y_int.flatten()), method='linear')

im=plt.contourf(x_int, y_int, np.reshape(X_int,r_0_g.shape),cmap='jet')
plt.colorbar(im)

with open('K_CNR.pkl', 'wb') as writer:
    pickle.dump(K_CNR,writer)   
#####Dataframes

iden_lab = np.array(['azim'])
labels = iden_lab

vel_lab = np.array(['range_gate','ws','CNR'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))
df_vlos0 = pd.DataFrame(columns=labels)
df_vlos0_noise = pd.DataFrame(columns=labels)
df_vlos1 = pd.DataFrame(columns=labels)
df_vlos1_noise = pd.DataFrame(columns=labels)

r_unique0 = r_0_g[0,:]
azim_unique0 = phi_0_g[:,0]*180/np.pi
r_unique1 = r_1_g[0,:]
azim_unique1 = phi_1_g[:,0]*180/np.pi
aux_df = np.zeros((len(azim_unique),len(labels)))
aux_df_noise = np.zeros((len(azim_unique),len(labels)))

#############CNR
count=0
for dir_mean in Dir:
    for u_mean in utot:
        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            vlos0_file = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos1_file = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos0_noise_file = 'noise0_'+vlos0_file
            vlos1_noise_file = 'noise1_'+vlos1_file
            if vlos0_file in onlyfiles:                
                vlos0 = np.reshape(np.fromfile(vlos0_file, dtype=np.float32),r_0_g.shape)
                vlos0_noise = np.reshape(np.fromfile(vlos0_noise_file, dtype=np.float32),r_0_g.shape)
                n = 1
                aux_df[:,0] = azim_unique
                aux_df_noise[:,0] = azim_unique                   
                # random scan 
                t = np.random.randint(scanlist[0],scanlist[-1])
                ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
                x = r.flatten()*np.cos(azim.flatten()*np.pi/180)
                y = r.flatten()*np.sin(azim.flatten()*np.pi/180)
                X = np.c_[CNR.flatten()[ind_t],x[ind_t],y[ind_t]]
                transformer = RobustScaler(quantile_range=(25, 75))
                transformer.fit(X)
                X = transformer.transform(X)  
                kde = KernelDensity(kernel='gaussian', bandwidth=.021)
                kde.fit(X)
                samples = kde.sample(n_samples=45*198)
                samples = transformer.inverse_transform(samples)
                x_int = r_0_g*np.cos(phi_0_g)
                y_int = r_0_g*np.sin(phi_0_g)
                X_int = sp.interpolate.griddata(samples[:,1:],samples[:,0], (x_int.flatten(), y_int.flatten()), method='linear')
                samples_CNR = np.reshape(X_int,r_0_g.shape)                  
                for i in range(len(r_unique)):
                    #aux_df[:,n:n+3] = np.c_[r_0_g[:,i],vlos0[:,i],samples_CNR]
                    aux_df_noise[:,n:n+3] = np.c_[r_0_g[:,i],vlos0_noise[:,i],samples_CNR[:,i]]
                    n = n+3
                df_vlos0_noise = pd.concat([df_vlos0_noise, pd.DataFrame(data=aux_df_noise,columns=labels)])
                #df_vlos0 = pd.concat([df_vlos0, pd.DataFrame(data=aux_df,columns=labels)])
                count+=1
                print(count)
                
                
df_vlos0_noise['scan'] = df_vlos0_noise.groupby('azim').cumcount()


# Smoothing of CNR
df_vlos0_noise_med = df_vlos0_noise.copy()
#median filter in radial direction
df_vlos0_noise_med.CNR = df_vlos0_noise.CNR.rolling(15,axis=1, min_periods = 1).median()
#median filter in azimuth direction, not necessary
#df_vlos0_noise_med.CNR = df_vlos0_noise_med.CNR.rolling(3,axis=0, min_periods = 1).median()

# Contaminate CNR with noise-like values also!!

#with open('df_vlos_noise.pkl', 'wb') as writer:
#    pickle.dump(df_vlos0_noise,writer)

with open('df_vlos0_noise.pkl', 'rb') as reader:
    df_vlos0_noise = pickle.load(reader)

ind_scan = df_vlos0_noise.scan == 50

plt.figure()

plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise.ws.loc[ind_scan ].values, 30, cmap='jet')

plt.figure()

plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise.CNR.loc[ind_scan ].values, 30, cmap='jet')


dfcopy = df.copy()

dfcopy = fl.df_ws_grad(dfcopy,grad='dvdr')

mask_CNR = ((df.CNR>-24) & (df.CNR<-8))

mask_CNR.columns = mask.columns

mask_tot = mask.copy()

mask_tot.mask(mask_CNR,other=False,inplace=True)

dfcopy.ws=df.ws.mask(mask_tot)

ind_scan = (df.scan >= 10) & (df.scan <= 90)

plt.figure()
im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.ws.loc[ind_scan].values, 30, cmap='jet')
plt.colorbar(im)

#plt.figure()
#im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), dfcopy.ws.loc[ind_scan].values, 30, cmap='jet')
#plt.colorbar(im)

plt.figure()
im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.CNR.loc[ind_scan].values, 30, cmap='jet')
plt.colorbar(im)

########### Noise CNR
#########################

noisy_CNR = df.CNR.loc[ind_scan].values.flatten()

noisy_CNR = noisy_CNR[(noisy_CNR<-24)&(noisy_CNR>-40)]

transformer = RobustScaler(quantile_range=(25, 75))
transformer.fit(noisy_CNR.reshape(-1, 1))
X = transformer.transform(noisy_CNR.reshape(-1, 1))

from sklearn.model_selection import GridSearchCV
bandwidths = 10**np.linspace(-2, 0, 20)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=2)
grid.fit(X)

kde_CNR = sp.stats.gaussian_kde(noisy_CNR, bw_method='silverman')
CNR_noise = kde_CNR.resample(size=None)[source]

plt.figure()
plt.hist(noisy_CNR,bins=100)

###########################


ind_scan = (df.scan > 9000) & (df.scan < 12000)
ind_filtered = np.isnan(dfcopy.ws.loc[ind_scan].values)
ind_non_filtered = ~np.isnan(dfcopy.ws.loc[ind_scan].values)

filtered_CNR = df.CNR.loc[ind_scan].values[ind_filtered]
non_filtered_CNR = df.CNR.loc[ind_scan].values[ind_non_filtered]

ind_CNR = filtered_CNR<-24#~((filtered_CNR>-5) & (filtered_CNR<-30))

filtered_ws = df.ws.loc[ind_scan].values[ind_filtered]
non_filtered_ws = df.ws.loc[ind_scan].values[ind_non_filtered]

filtered_dvdr = dfcopy.loc[ind_scan].dvdr.values[ind_filtered]
non_filtered_dvdr = dfcopy.loc[ind_scan].dvdr.values[ind_non_filtered]

plt.figure()
plt.hist(filtered_ws[ind_CNR],bins=30)
plt.figure()
plt.hist(filtered_CNR[ind_CNR],bins=30)
plt.figure()
plt.hist(np.abs(filtered_dvdr[ind_CNR]),bins=30)

plt.figure()
plt.hist(non_filtered_ws[ind_CNR],bins=30)
plt.figure()
plt.hist(non_filtered_CNR[ind_CNR],bins=30)
plt.figure()
plt.hist(np.abs(non_filtered_dvdr),bins=30)

plt.figure()
plt.scatter(np.abs(filtered_dvdr[ind_CNR]),filtered_CNR[ind_CNR])
























