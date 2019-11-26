# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:13:10 2018

@author: lalc
"""

import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize
from os import listdir
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import ppiscanprocess.filtering as fl
import pickle

import numpy as np
import pandas as pd
import scipy as sp

import importlib

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

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

fmt = FormatScalarFormatter("%.2f")    

from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

f = 24

def fm(x, pos=None):
    return r'${}$'.format('{:.1f}'.format(x).split('f')[0])

# In[]

root = tkint.Tk()
file_in_path_wf = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

root = tkint.Tk()
file_out_path_df = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataFrame')
root.destroy()

root = tkint.Tk()
file_out_path_db = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataBase')
root.destroy()

root = tkint.Tk()
file_in_path_raw = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir Raw')
root.destroy()

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]
onlyfiles_raw = [f for f in listdir(file_in_path_raw) if isfile(join(file_in_path_raw, f))]
onlyfiles_wf = [f for f in listdir(file_in_path_wf) if isfile(join(file_in_path_wf, f))]

# In[column labels]

iden_lab = np.array(['num1','num2','start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(99):

    labels = np.concatenate((labels,vel_lab))
    
# In[]
    
filelist_s = [(filename,getsize(join(file_in_path,filename)))
             for filename in listdir(file_in_path) if getsize(join(file_in_path,filename))>1000]
size_s = list(list(zip(*filelist_s))[1])
filelist_s = list(list(zip(*filelist_s))[0])

# In[Different features of the DBSCAN filter]

feat0 = ['range_gate','ws','azim']
feat1 = ['range_gate','ws','dvdr']
feat2 = ['range_gate','azim','ws','dvdr']
    
# In[Geometry of scans]

r_0 = np.linspace(105,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(105,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)

# In[Routine for noisy dataframe and database creation]
############ Routine for noisy dataframe and database creation ################

iden_lab = np.array(['azim'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws']))) 
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan'])))            
   
# Mann-model parameters
Dir = np.linspace(90,270,7)*np.pi/180
u_mean = 15
ae = [0.025, 0.05, 0.075]
L = [62,62.5,125,250,500,750,1000]
G = [0,1,2,2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)        
scan = 0
m = np.arange(3,597,3)
azim_unique = phi_0_g[:,0]*180/np.pi
df_noise = pd.DataFrame(columns=labels)
df_raw = pd.DataFrame(columns=labels)
aux_df_noise = np.zeros((len(azim_unique),len(labels)))
aux_df_raw = np.zeros((len(azim_unique),len(labels)))
indx = 0
scan = 0 
################              
param = []
for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([int(dir_mean*180/np.pi),u_mean,ae_i,L_i,G_i,seed_i,scan])
            scan+=1
param = np.array(param)         
#######################            
for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([dir_mean,u_mean,ae_i,L_i,G_i,seed_i,scan])
            n = 1
            aux_df_noise[:,0] = azim_unique
            aux_df_raw[:,0] = azim_unique
            dataws = np.reshape(np.fromfile(file_in_path+'/noise0_'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
            dataws_raw = np.reshape(np.fromfile(file_in_path_raw+'/'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
            datar = r_0_g
            for i in range(datar.shape[1]):
                aux_df_noise[:,n:n+2] = np.c_[datar[:,i],dataws[:,i]]
                aux_df_raw[:,n:n+2] = np.c_[datar[:,i],dataws_raw[:,i]]                       
                n = n+2
            df_noise = pd.concat([df_noise, pd.DataFrame(data=aux_df_noise,
                       index = indx+np.arange(datar.shape[0]),columns = labels)])  
            df_raw   = pd.concat([df_raw, pd.DataFrame(data=aux_df_raw,
                       index = indx+np.arange(datar.shape[0]),columns = labels)])  
            scan+=1
            indx = indx + np.arange(datar.shape[0]) + 1
            print(scan)

for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([dir_mean,u_mean,ae_i,L_i,G_i,seed_i,scan])
            scan+=1
            
df_noise['scan'] = df_noise.groupby('azim').cumcount()
df_raw['scan'] = df_raw.groupby('azim').cumcount()

df_noise.reset_index(inplace = True)
df_raw.reset_index(inplace = True)

# In[Filtering of synthetic and storing in database]
##############################################################################
#with open(file_out_path_df+'/df0_noise.pkl', 'wb') as writer:
#    pickle.dump(df_noise,writer)
#with open(file_out_path_df+'/df0_raw.pkl', 'wb') as writer:
#    pickle.dump(df_raw,writer)    


with open(file_out_path_df+'/df0_noise.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)
with open(file_out_path_df+'/df0_raw.pkl', 'rb') as reader:
    df_raw = pickle.load(reader)

from sqlalchemy import create_engine

csv_database0 = create_engine('sqlite:///'+file_out_path+'/synthetic.db')

t_step = 3
mask0 = pd.DataFrame()
mask1 = pd.DataFrame()
mask2 = pd.DataFrame()
ind=np.unique(df_noise.scan.values)%t_step==0
times= np.unique(np.append(np.unique(df_noise.scan.values)[ind], df_noise.scan.values[-1]))  
     
for i in range(len(times)-1):
    print(times[i],times[i+1])
    if i == len(times)-2:
        loc = (df_noise.scan>=times[i]) & (df_noise.scan<=times[i+1])
    else:
        loc = (df_noise.scan>=times[i]) & (df_noise.scan<times[i+1])
    mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)])
    mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)])
    mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
mask0.columns = df_noise.ws.columns
mask1.columns = df_noise.ws.columns
mask2.columns = df_noise.ws.columns
df_noise.columns = np.concatenate((np.array(['index']),labels_new))
df_raw.columns = np.concatenate((np.array(['index']),labels_new))
df_raw.to_sql('raw', csv_database0, if_exists='append')
df_noise.to_sql('noise', csv_database0, if_exists='append')
df_noise.columns = np.concatenate((np.array(['index']),labels,np.array(['scan'])))

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask0.ws)
#with open(file_out_path_df+'/df0_fil0.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered0', csv_database0, if_exists='append')

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask1.ws)
#with open(file_out_path_df+'/df0_fil1.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered1', csv_database0, if_exists='append')

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask2.ws)
#with open(file_out_path_df+'/df0_fil2.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered2', csv_database0, if_exists='append')
  
df_median = fl.data_filt_median(df_noise, lim_m= 2.33 , n = 5, m = 3) 
#with open(file_out_path_df+'/df0_median.pkl', 'wb') as writer:
#    pickle.dump(df_median,writer)  
df_median.columns = np.concatenate((np.array(['index']),labels_new))
df_median.to_sql('filtered_med', csv_database0, if_exists='append') 

# In[Noise from database]
##############################################################################
from sqlalchemy import create_engine

csv_database0 = create_engine('sqlite:///'+file_out_path+'/synthetic.db')
col = 'SELECT '
i = 0
for w,r in zip(labels_ws, labels_rg):
    if i == 0:
        col = col + ' azim, ' + w + ', ' + r + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', scan'
    else:
        col = col + ' ' + w + ', ' + r + ','       
    i+=1
selec_noise = col + ' FROM "noise"'
selec_raw = col + ' FROM "raw"'

df_noise = pd.read_sql_query(selec_noise, csv_database0)
df_raw = pd.read_sql_query(selec_raw, csv_database0)
iden_lab = np.array(['azim'])
labels = iden_lab
for i in np.arange(198):
    labels = np.concatenate((labels,np.array(['ws','range_gate']))) 
labels = np.concatenate((labels,np.array(['scan']))) 
df_noise.columns = labels
df_raw.columns = labels

# In[Noisy scans plots and noise visualization]
###############################################################################
scan_t = 2010
vmax = np.max(df_raw.loc[df_raw.scan==scan_t].ws.values)
vmin = np.min(df_raw.loc[df_raw.scan==scan_t].ws.values)
noise_id = (df_noise.loc[df_noise.scan==scan_t].ws.values-
            df_raw.loc[df_noise.scan==scan_t].ws.values) != 0
noisy = df_noise.loc[df_noise.scan==scan_t].ws.values
noisy[~noise_id] = np.nan
non_noisy = df_noise.loc[df_noise.scan==scan_t].ws.values
non_noisy[noise_id] = np.nan

phi_0 = np.where(np.pi/2-phi_1_g<0, 2*np.pi+(np.pi/2-phi_1_g), np.pi/2-phi_1_g)

f=24
fig, ax = plt.subplots(figsize = (9,9))
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im0 = ax.scatter((r_0_g*np.cos(phi_0))[noise_id], (r_0_g*np.sin(phi_0))[noise_id],
                c = noisy[noise_id], cmap = 'Greys')
im = ax.scatter((r_0_g*np.cos(phi_0))[~noise_id], (r_0_g*np.sin(phi_0))[~noise_id], 
                c=non_noisy[~noise_id], cmap = 'jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad = 0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_{LOS},\:$", fontsize=f)
cbar2.ax.set_ylabel("$V_{LOS},\:contaminated$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
fig.tight_layout()

# In[Sensitivity analysis of median filter]
##############################################################################
##############################################################################    
"""Sensitivity analysis median filter"""
##############################################################################
############################################################################## 

# In[Creation of a list of filtering results, very time consuming]
# Noise fraction   
#############################################################
noise_iden = (df_noise.ws.values-df_raw.ws.values) != 0
#############################################################   
n_w = np.array([3, 5, 7, 9, 11, 13])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(3,6,6)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)

noise_det = []
not_noise_det = []
reliable_scan = []

chunk_size = 45
counting = 0

for nw, mw, l in zip(n_w.flatten(), m_w.flatten(), lim.flatten()):
    df_noise_median_i = fl.data_filt_median(df_noise,lim_m=l,lim_g=100,n=nw, m=mw)
    reliable = ~np.isnan(df_noise_median_i.ws.values) & ~noise_iden
    n = np.isnan(df_noise_median_i.ws.values) & noise_iden
    nn = np.isnan(df_noise_median_i.ws.values) & ~noise_iden
    reliable_scan.append([np.sum(reliable[i:i+chunk_size,:])/
                          np.sum(~noise_iden[i:i+chunk_size,:])
                          for i in range(0, n.shape[0], chunk_size)])
    noise_det.append([np.sum(n[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, n.shape[0], chunk_size)])
    not_noise_det.append([np.sum(nn[i:i+chunk_size,:])/
                          np.sum(~noise_iden[i:i+chunk_size,:])
                          for i in range(0, n.shape[0], chunk_size)])
    counting+=1
    print(counting)
# In[Storing of results]
#with open(file_out_path+'/reliable_scan0_long.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan,writer)
#with open(file_out_path+'/noise_det0_long.pkl', 'wb') as writer:
#    pickle.dump(noise_det,writer)      
#with open(file_out_path+'/not_noise_det0_long.pkl', 'wb') as writer:
#    pickle.dump(not_noise_det,writer)
    
with open(file_out_path+'/reliable_scan0.pkl', 'rb') as reader:
    reliable_scan = pickle.load(reader)
with open(file_out_path+'/noise_det0.pkl', 'rb') as reader:
    noise_det = pickle.load(reader)      
with open(file_out_path+'/not_noise_det0.pkl', 'rb') as reader:
    not_noise_det = pickle.load(reader)
# In[2D arrays for sensitivity analysis]    

noise_weight = np.array([np.sum(noise_iden[i:i+chunk_size,:])/
                         len(noise_iden[i:i+chunk_size,:].flatten())
                         for i in range(0, noise_iden.shape[0], chunk_size)])
mean_noise_det = np.reshape(np.array([np.mean(nd) for nd in noise_det]),n_w.shape)   
mean_not_noise_det = np.reshape(np.array([np.mean(nd) for nd in not_noise_det]),n_w.shape)  
mean_reliable = np.reshape(np.array([np.mean(rel) for rel in reliable_scan]),n_w.shape)
mean_tot_w = np.reshape(np.array([np.mean((rel*(1-noise_weight)+nd*noise_weight))
                        for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape) 
# In[Just for the long list, non squared domain]
###############################################################################
    
with open(file_out_path+'/reliable_scan0_long.pkl', 'rb') as reader:
    reliable_scan = pickle.load(reader)
with open(file_out_path+'/noise_det0_long.pkl', 'rb') as reader:
    noise_det = pickle.load(reader)      
with open(file_out_path+'/not_noise_det0_long.pkl', 'rb') as reader:
    not_noise_det = pickle.load(reader)
        
n_w = np.array([3, 5, 7, 9, 11, 13])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(3,6,6)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)
nml = np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]
n_w = np.array([5])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(1,3,4)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)
nml = np.vstack((nml,np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]))
n_w = np.array([3,7,9])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(1,3,4)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)
nml = np.vstack((nml,np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]))

#############################
#param_i = int(dir_mean*180/np.pi),u_mean,ae_i,L_i,G_i,seed_i,scan]
# For L
opt2 = []
a = [0.025, 0.05, .075]
L = [62, 250, 500, 750, 1000]
a, L = np.meshgrid(a, L)

for ai, Li in zip(a.flatten(), L.flatten()):
    optaux = []  
    ind_param = np.where((param[:,2]==ai) & (param[:,3]==Li))[0] 
    ind_param = np.where(param[:,2]>0)[0] 
    noiseweight = np.array([list(noise_weight),]*np.array(noise_det).shape[0])
    mean_noise_det = np.mean(np.array(noise_det)[:,ind_param],axis=1)
    mean_not_noise_det = np.mean(np.array(not_noise_det)[:,ind_param],axis=1)
    mean_reliable = np.mean(np.array(reliable_scan)[:,ind_param],axis=1)
    mean_tot_w = np.mean((np.array(reliable_scan)*(1-noiseweight)+ 
                          np.array(noise_det)*(noiseweight))[:,ind_param],axis=1)

    for n in np.array([3, 5, 7, 9]):
    #    n = np.array([3, 5, 7, 9,])
        n_w = n
        m_w = np.linspace(3,13,80)#np.array([3, 5, 7, 9, 11, 13])
        lim = np.linspace(1,6,80)
        n_w, m_w, lim = np.meshgrid(n_w,m_w, lim) 
        nml0 = np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]
        mean_tot = np.reshape(sp.interpolate.griddata(nml,mean_tot_w, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
        mean_rel = np.reshape(sp.interpolate.griddata(nml,mean_reliable, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
        mean_noise = np.reshape(sp.interpolate.griddata(nml,mean_noise_det, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
        m_w, lim, mean_tot, mean_rel, mean_noise = np.squeeze(m_w), np.squeeze(lim), np.squeeze(mean_tot), np.squeeze(mean_rel), np.squeeze(mean_noise) 
     
        low = np.min(mean_tot)  
        optimal = np.r_[np.c_[lim.flatten(),m_w.flatten()][np.argmax(mean_tot)],n,
                              mean_noise.flatten()[np.argmax(mean_tot)],mean_rel.flatten()[np.argmax(mean_tot)],
                              np.max(mean_tot)]
    #    opt.append(np.r_[np.max(mean_tot),optimal])
        optaux.append(np.r_[np.max(mean_tot),optimal])
    opt2.append(np.array(optaux)[np.argmax(np.array(optaux)[:,0]),:])


################################
    
#mean_noise_det = np.array([np.mean(nd) for nd in noise_det])  
#mean_not_noise_det = np.array([np.mean(nd) for nd in not_noise_det]) 
#mean_reliable = np.array([np.mean(rel) for rel in reliable_scan])
#mean_tot_w = np.array([np.mean((rel*(1-noise_weight)+nd*noise_weight))
#                        for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))])      
    
low =.45
# Figures for nw fixed to 5
#opt = []
for n in np.array([3, 5, 7, 9]):
#    n = np.array([3, 5, 7, 9,])
    n_w = n
    m_w = np.linspace(3,13,80)#np.array([3, 5, 7, 9, 11, 13])
    lim = np.linspace(1,6,80)
    n_w, m_w, lim = np.meshgrid(n_w,m_w, lim) 
    nml0 = np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]
    mean_tot = np.reshape(sp.interpolate.griddata(nml,mean_tot_w, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    mean_rel = np.reshape(sp.interpolate.griddata(nml,mean_reliable, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    mean_noise = np.reshape(sp.interpolate.griddata(nml,mean_noise_det, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    m_w, lim, mean_tot, mean_rel, mean_noise = np.squeeze(m_w), np.squeeze(lim), np.squeeze(mean_tot), np.squeeze(mean_rel), np.squeeze(mean_noise) 
 
    low = np.min(mean_tot)  
    optimal = np.r_[np.c_[lim.flatten(),m_w.flatten()][np.argmax(mean_tot)],n,
                          mean_noise.flatten()[np.argmax(mean_tot)],mean_rel.flatten()[np.argmax(mean_tot)],
                          np.max(mean_tot)]
#    opt.append(np.r_[np.max(mean_tot),optimal])
    opt2.append(np.r_[np.max(mean_tot),optimal])
    fig, ax = plt.subplots(figsize=(8,8))
    lev = np.linspace(low,np.max(mean_tot),21)
    im = ax.contourf(lim,m_w,mean_tot,lev,cmap='jet')
    ax.scatter(optimal[0], optimal[1],s=20, c= 'white')
#    ax.set_title('$Radial\:window\:length,\:n_r\:=\:'+'%.0f' %np.unique(n_w.flatten())[0]+'$', fontsize=16)
    ax.set_xlabel('$\Delta V_{LOS,\:threshold}\:m/s$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{tot}$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(low,np.max(mean_tot),5))#np.r_[lev[0],lev[1::10],lev[-1]])
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()  

    low = np.min(mean_rel)
    fig, ax = plt.subplots(figsize=(8,8))
    lev = np.linspace(low,np.max(mean_rel),21)
    im = ax.contourf(lim,m_w,mean_rel,lev,cmap='jet')
    ax.scatter(optimal[0], optimal[1],s=20, c= 'white')
#    ax.set_title('$Radial\:window\:length,\:n_r\:=\:'+'%.0f' %np.unique(n_w.flatten())[0]+'$', fontsize=16)
    ax.set_xlabel('$\Delta V_{LOS,\:threshold}\:m/s$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{rec}$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(low,np.max(mean_rel),5))#np.r_[lev[0],lev[1::10],lev[-1]])
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()  

    low = np.min(mean_noise)
    fig, ax = plt.subplots(figsize=(8,8))
    lev = np.linspace(low,np.max(mean_noise),21)
    im = ax.contourf(lim,m_w,mean_noise,lev,cmap='jet')
    ax.scatter(optimal[0], optimal[1],s=20, c= 'white')
#    ax.set_title('$Radial\:window\:length,\:n_r\:=\:'+'%.0f' %np.unique(n_w.flatten())[0]+'$', fontsize=16)
    ax.set_xlabel('$\Delta V_{LOS,\:threshold}\:m/s$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{noise}$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(low,np.max(mean_noise),5))#np.r_[lev[0],lev[1::10],lev[-1]])
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()  

# In[1D graph]    
l = lim.flatten()
nw = n_w.flatten()/l
mw = m_w.flatten()/l
tot = (mean_tot).flatten()
noise = (mean_noise).flatten()
rec = (mean_rel).flatten()
#################################
def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)
##################################Figure eta curve
indexnw = np.argsort(nw)
indexmw = np.argsort(mw)
indexnmw = np.argsort(nw*mw*l)

totm = mov_ave(tot[indexnmw], 20)
noisem = mov_ave(noise[indexnmw], 20)
recm = mov_ave(rec[indexnmw], 20)
off = -totm.shape[0]+tot[indexnmw].shape[0] 
############ Some figures

for l in range(len(np.unique(lim.flatten()))):
    fig, ax = plt.subplots()
    lev = np.linspace(np.min(mean_tot_w[:,:,l]),np.max(mean_tot_w[:,:,l]),31)
    im = ax.contourf(n_w[:,:,l],m_w[:,:,l],(mean_tot_w[:,:,l]),lev,cmap='jet')
    ax.set_title('$\Delta V_{LOS,\:threshold}\:=\:'+'%.2f' %np.mean(lim[:,:,l])+'\:m/s$', fontsize=16)
    ax.set_xlabel('$Radial\:window\:length,\: n_r$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{tot}$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(lev[1::8])
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()  

for n in range(len(np.unique(n_w.flatten()))):
    fig, ax = plt.subplots()
    lev = np.linspace(np.min(mean_tot_w),np.max(mean_tot_w),31)
    im = ax.contourf(lim[n,:,:],m_w[:,n,:],(mean_tot_w[n,:,:]),lev,cmap='jet')
    im = ax.contourf(lim[n,:,:],m_w[:,n,:],(mean_tot_w[0,:,:]/mean_tot_w[1,:,:]),lev,cmap='jet')
    ax.set_title('$Radial\:window\:length,\:n_r\:=\:'+'%.0f' %np.unique(n_w.flatten())[n]+'$', fontsize=16)
    ax.set_xlabel('$\Delta V_{LOS,\:threshold}\:m/s$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{tot}$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(lev[1::8])
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()

# In[Figures sensitivity analysis and noise fraction]


fig, ax = plt.subplots(figsize = (8,8))
ax.hist(noise_weight,bins=50,histtype='step',lw=2, color = 'k')
ax.set_xlabel('$f_{noise}$', fontsize=16)
ax.set_ylabel('$Counts$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=24,verticalalignment='top')
fig.tight_layout()
        
fig, ax = plt.subplots(figsize=(8,8))
ax.plot((nw*mw*l)[indexnmw][off:],totm, label = '$\eta_{tot}$', c = 'k', lw=2)
ax.scatter((nw*mw*l)[indexnmw],tot[indexnmw],s = 2, c = 'k', alpha = .2)
ax.plot((nw*mw*l)[indexnmw][off:],noisem, label = '$\eta_{noise}$', c = 'r', lw=2)
ax.scatter((nw*mw*l)[indexnmw], noise[indexnmw],s = 2, c = 'r', alpha = .2)
ax.plot((nw*mw*l)[indexnmw][off:],recm, label = '$\eta_{rec}$', c = 'b', lw=2)
ax.scatter((nw*mw*l)[indexnmw],rec[indexnmw],s = 2, c = 'b', alpha = .2)

ax.set_xscale('log')
ax.set_xlabel('$n_rn_\phi/\Delta V_{LOS,threshold}$',fontsize = 16)
ax.set_ylabel('$\eta$',fontsize = 16)
ax.legend(fontsize = 16)
ax.tick_params(labelsize = 16)
ax.set_xlim(2,100)
ax.set_ylim(.4, 1)
fig.tight_layout()    

nml_opt = optimal[1]*optimal[2]/optimal[0]
ax.scatter(nml_opt,optimal[-1],s = 100, c = 'grey',edgecolors = 'k', alpha = 1,lw = 4)
ax.scatter(nml_opt,optimal[3],s = 100, c ='grey',edgecolors = 'r', alpha = 1,lw = 4)
ax.scatter(nml_opt,optimal[4],s = 100, c ='grey',edgecolors = 'b', alpha = 1,lw = 4)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,verticalalignment='top')

##############################################################################
##############################################################################    
"""End of Sensitivity analysis median filter"""
# In[End of Sensitivity analysis median filter]
##############################################################################
############################################################################## 
# InLoading dataframes]

with open(file_out_path_df+'/df0_noise.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)
with open(file_out_path_df+'/df0_raw.pkl', 'rb') as reader:
    df_raw = pickle.load(reader)
with open(file_out_path_df+'/df0_median.pkl', 'rb') as reader:
    df_median = pickle.load(reader)
with open(file_out_path_df+'/df0_fil2.pkl', 'rb') as reader:
    df_fil = pickle.load(reader)    
  
chunk_size = 45    
noise_iden = (df_noise.ws.values-df_raw.ws.values) != 0  
noise_weight = np.array([np.sum(noise_iden[i:i+chunk_size,:])/
                         len(noise_iden[i:i+chunk_size,:].flatten())
                         for i in range(0, noise_iden.shape[0], chunk_size)])  


noise_det_clust = np.isnan(df_fil.ws.values) & noise_iden
recov_rel_clust = ~np.isnan(df_fil.ws.values) & ~noise_iden
noise_det_median = np.isnan(df_median.ws.values) & noise_iden
recov_rel_median = ~np.isnan(df_median.ws.values) & ~noise_iden


noise_iden_clust = np.array([np.sum(noise_det_clust[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, noise_det_clust.shape[0], chunk_size)])
        
relia_iden_clust = np.array([np.sum(recov_rel_clust[i:i+chunk_size,:])/
                      np.sum(~noise_iden[i:i+chunk_size,:])
                      for i in range(0, recov_rel_clust.shape[0], chunk_size)])

noise_iden_median = np.array([np.sum(noise_det_median[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, noise_det_clust.shape[0], chunk_size)])
        
relia_iden_median = np.array([np.sum(recov_rel_median[i:i+chunk_size,:])/
                      np.sum(~noise_iden[i:i+chunk_size,:])
                      for i in range(0, recov_rel_clust.shape[0], chunk_size)])

tot_clust = noise_weight*noise_iden_clust + (1-noise_weight)*relia_iden_clust
tot_median = noise_weight*noise_iden_median + (1-noise_weight)*relia_iden_median

# In[Histograms]
########################################################################################
f = 24
Density = False
n_bin = 100
fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[noise_iden_clust,noise_iden_median]),
                   np.max(np.r_[noise_iden_clust,noise_iden_median]),n_bin)
ax.hist(noise_iden_clust,bins=bins, histtype = 'step',
                            label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(noise_iden_median,bins=bins, histtype = 'step',
                    label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{noise}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()    

fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[relia_iden_clust,relia_iden_median]),
                   np.max(np.r_[relia_iden_clust,relia_iden_median]),n_bin)
ax.hist(relia_iden_clust,bins=bins, histtype = 'step', label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(relia_iden_median,bins=bins, histtype = 'step', label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{rec}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()    

fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[tot_clust,tot_median]),
                   np.max(np.r_[tot_clust,tot_median]),n_bin)
ax.hist(tot_clust,bins=bins, histtype = 'step', label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(tot_median,bins=bins, histtype = 'step', label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{tot}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()    
    
# In[] 
####################### Loading results
          

with open('mask_filter_r_ws_3.pkl', 'rb') as reader:
    mask0 = pickle.load(reader)
with open('mask_filter_r_ws_CNR_3.pkl', 'rb') as reader:
    mask1 = pickle.load(reader)
with open('mask_filter_r_ws_dvdr_3.pkl', 'rb') as reader:
    mask2 = pickle.load(reader)
with open('mask_filter_r_ws_CNR_dvdr_3.pkl', 'rb') as reader:
    mask3 = pickle.load(reader)
with open('mask_filter_r_azim_ws_3.pkl', 'rb') as reader:
    mask4 = pickle.load(reader)
with open('mask_filter_r_azim_dvdr_3.pkl', 'rb') as reader:
    mask5 = pickle.load(reader)  
with open('mask_filter_r_azim_ws_dvdr_3.pkl', 'rb') as reader:
    mask6 = pickle.load(reader) 
    
#######################Filtering of dataFrame, different parameters
    
r_0 = np.linspace(150,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(150,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)
    
#df_noise0 = df_noise.copy() 
#df_noise1 = df_noise.copy() 
#df_noise2 = df_noise.copy() 
#df_noise3 = df_noise.copy() 
#df_noise4 = df_noise.copy() 
#df_noise5 = df_noise.copy() 
df_noise6 = df_noise.copy()    

#df_noise0.ws = df_noise0.ws.mask(mask0)
#df_noise1.ws = df_noise1.ws.mask(mask1)
#df_noise2.ws = df_noise2.ws.mask(mask2)
#df_noise3.ws = df_noise3.ws.mask(mask3)
#df_noise4.ws = df_noise4.ws.mask(mask4)
#df_noise5.ws = df_noise5.ws.mask(mask5)
df_noise6.ws = df_noise6.ws.mask(mask6)

############################### Median filter, parameter exploration and analysis on stats

#n_w = np.arange(2,10,1)
#m_w = np.arange(2,10,1)
#lim = np.linspace(1,8,8)

n_w = np.array([3, 5, 7])
m_w = np.array([3, 5, 7])
lim = np.linspace(1,8,4)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)

ind_noise_array = np.vstack(tuple(ind_noise))

noise_det = []
not_noise_det = []
reliable_scan = []

chunk_size = 45
counting = 512
for nw, mw, l in zip(n_w.flatten()[512-counting:], m_w.flatten()[512-counting:], lim.flatten()[512-counting:]):
    df_noise_median_i = fl.data_filt_median(df_noise,lim_m=l,lim_g=100,n=nw, m=mw)
    reliable = ~np.isnan(df_noise_median_i.ws.values) & ~ind_noise_array
    n = np.isnan(df_noise_median_i.ws.values) & ind_noise_array
#    nn = np.isnan(df_noise_median_i.ws.values) & ~ind_noise_array
    reliable_scan.append([np.sum(reliable[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
    noise_det.append([np.sum(n[i:i+chunk_size,:])/np.sum(ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
#    not_noise_det.append([np.sum(nn[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
    counting-=1
    print(counting)
############################## Clustering filter stats
#noise2 = np.isnan(df_noise2.ws.values) & ind_noise_array
#noise_removed2 = np.array([np.sum(noise2[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
#noise5 = np.isnan(df_noise5.ws.values) & ind_noise_array
#noise_removed5 = np.array([np.sum(noise5[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
noise6 = np.isnan(df_noise6.ws.values) & ind_noise_array
noise_removed6 = np.array([np.sum(noise6[i:i+chunk_size,:])/np.sum(ind_noise_array[i:i+chunk_size,:]) for i in range(0, df_noise6.ws.values.shape[0], chunk_size)])   

#reliable2 = ~np.isnan(df_noise2.ws.values) & ~ind_noise_array 
#reliable_scan2 = np.array([np.sum(reliable2[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
#reliable5 = ~np.isnan(df_noise5.ws.values) & ~ind_noise_array 
#reliable_scan5 = np.array([np.sum(reliable5[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
reliable6 = ~np.isnan(df_noise6.ws.values) & ~ind_noise_array 
reliable_scan6 = np.array([np.sum(reliable6[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, df_noise6.ws.values.shape[0], chunk_size)])   

#with open('reliable_scan6.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan6,writer)
#with open('reliable_scan2.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan2,writer)  
#with open('reliable_scan_sensitivity_median_nm.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan,writer)
#with open('noise_det_sensitivity_median_nm.pkl', 'wb') as writer:
#    pickle.dump(noise_det,writer)
#with open('not_noise_det_sensitivity_median.pkl', 'wb') as writer:
#    pickle.dump(not_noise_det,writer)

with open('reliable_scan_sensitivity_median_nm.pkl', 'rb') as reader:
    reliable_scan = pickle.load(reader)
    
with open('noise_det_sensitivity_median_nm.pkl', 'rb') as reader:
    noise_det = pickle.load(reader)

noise_weight = np.array([np.sum(ind_noise_array[i:i+chunk_size,:])/len(ind_noise_array[i:i+chunk_size,:].flatten()) for i in range(0, ind_noise_array.shape[0], chunk_size)])

mean_noise_det = np.reshape(np.array([np.mean(nd) for nd in noise_det]),n_w.shape)   
#mean_not_noise_det = np.reshape(np.array([np.mean(nd) for nd in not_noise_det]),n_w.shape)  
mean_reliable = np.reshape(np.array([np.mean(rel) for rel in reliable_scan]),n_w.shape)   

std_noise_det = np.reshape(np.array([np.std(nd) for nd in noise_det]),n_w.shape)     
std_reliable = np.reshape(np.array([np.std(rel) for rel in reliable_scan]),n_w.shape)   

std_tot = np.reshape(np.array([np.std(.5*(rel+nd)) for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape)

mean_tot_w = np.reshape(np.array([np.mean((rel*(1-noise_weight)+nd*noise_weight)) for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape)
    
#mean_tot_6 = np.mean(.5*(reliable_scan6[:-1] + noise_removed6[:-1]))
#std_tot_6 = np.std(.5*(reliable_scan6[:-1] + noise_removed6[:-1]))

mean_tot_6_w = np.mean((reliable_scan6[:-1]*(1-noise_weight[:-1]) + noise_removed6[:-1]*noise_weight[:-1]))


############ Some figures

l = 2

plt.figure()
plt.contourf(n_w[:,:,l],m_w[:,:,l],mean_noise_det[:,:,l]/np.mean(noise_removed6[:-1]),30,cmap='jet')
plt.colorbar()

for l in range(len(np.linspace(1,8,8))):
    fig, ax = plt.subplots()
    im = ax.contourf(n_w[:,:,l],m_w[:,:,l],(mean_tot_w[:,:,l]),30,cmap='jet')
    ax.set_title('$\Delta V_{LOS,\:threshold}\:=\:'+str(np.mean(lim[:,:,l]))+'\:m/s$', fontsize=16)
    ax.set_xlabel('$Radial\:window\:length,\: n_r$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{tot}$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()

l = lim.flatten()
nw = n_w.flatten()/l
mw = m_w.flatten()/l
tot = (mean_tot_w).flatten()
noise = (mean_noise_det).flatten()
rec = (mean_reliable).flatten()
#################################
def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)
##################################Figure eta curve
indexnw = np.argsort(nw)
indexmw = np.argsort(mw)
indexnmw = np.argsort(nw*mw*l)

totm = mov_ave(tot[indexnmw], 5)
noisem = mov_ave(noise[indexnmw], 5)
recm = mov_ave(rec[indexnmw], 5)
off = -totm.shape[0]+tot[indexnmw].shape[0]

fig, ax = plt.subplots()
ax.plot((nw*mw*l)[indexnmw][off:],totm, label = '$\eta_{tot}$', c = 'k', lw=2)

ax.scatter((nw*mw*l)[indexnmw],tot[indexnmw],s = 2, c = 'k', alpha = .2)

ax.plot((nw*mw*l)[indexnmw][off:],noisem, label = '$\eta_{noise}$', c = 'r', lw=2)

ax.scatter((nw*mw*l)[indexnmw], noise[indexnmw],s = 2, c = 'r', alpha = .2)

ax.plot((nw*mw*l)[indexnmw][off:],recm, label = '$\eta_{rec}$', c = 'b', lw=2)

ax.scatter((nw*mw*l)[indexnmw],rec[indexnmw],s = 2, c = 'b', alpha = .2)

ax.set_xscale('log')
ax.set_xlabel('$n_rn_\phi/\Delta V_{LOS,threshold}$',fontsize = 16)
ax.set_ylabel('$\eta$',fontsize = 16)
ax.legend(fontsize = 16)
ax.tick_params(labelsize = 16)
ax.set_xlim(1,100)
fig.tight_layout()

plt.figure()
plt.scatter((nw*mw)[indexnmw],tot[indexnmw])
plt.xscale('log')

plt.figure()
plt.scatter((nw*mw)[indexnmw],noise[indexnmw])
plt.xscale('log')

plt.figure()
plt.scatter((nw*mw)[indexnmw],rec[indexnmw])
plt.xscale('log')

plt.plot(mw[indexmw],tot[indexmw])
plt.plot((nw*mw*l)[indexnmw],tot[indexnmw])

plt.scatter(nw, mw, c = tot)

plt.figure()
plt.contourf(n_w,lim,mean_reliable/np.mean(reliable_scan6[:-1]),30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,std_tot/std_tot_6,30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,(mean_tot/mean_tot_6),30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,(mean_tot/mean_tot_6)*(std_tot_6/std_tot),30,cmap='jet')
plt.colorbar()

############################################### Figures

fig, ax = plt.subplots()
ax.hist(noise_weight,bins=30,histtype='step',lw=2, color = 'k')
ax.set_xlabel('$f_{noise}$', fontsize=16)
ax.set_ylabel('$Counts$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=24,verticalalignment='top')
fig.tight_layout()


indexes = np.argsort((mean_tot_w/mean_tot_6_w).flatten())[::-1]
plt.figure()
plt.plot((mean_tot_w/mean_tot_6_w).flatten()[indexes])
plt.title('meantot')
plt.figure()
plt.plot((m_w*n_w/lim**2).flatten()[indexes]/(np.max((m_w*n_w/lim))))
plt.title('mw*nw/lim')
plt.figure()
plt.plot(mean_noise_det.flatten()[indexes])
plt.title('noisedet')
plt.figure()
plt.plot(mean_reliable.flatten()[indexes])
plt.title('noisedet')


plt.figure()
plt.plot((m_w/lim).flatten()[indexes]/(np.max((m_w/lim))))
plt.title('mw/lim')

plt.figure()
plt.plot((n_w/lim).flatten()[indexes]/(np.max((n_w/lim))))
plt.title('nw/lim')

#
#plt.figure()
#plt.plot((noise_det/np.mean(noise_removed6)).flatten()[indexes][:500])
#plt.title('noisew')
#plt.figure()
#plt.plot((reliable_scan/np.mean(reliable_scan6)).flatten()[indexes][:500])
#plt.title('relw')
#plt.figure()
#plt.plot((noise_weight).flatten()[indexes][:500])
#plt.title('noise')
#plt.figure()
#plt.plot((lim).flatten()[indexes][:500])
#plt.title('lim')
#plt.figure()
#plt.plot((m_w).flatten()[indexes][:500])
#plt.title('mw')
#plt.figure()
#plt.plot((n_w).flatten()[indexes][:500])
#plt.title('nw')
#plt.figure()
#plt.plot((param[:,0]).flatten()[indexes][:500])
#plt.title('ae')
#plt.figure()
#plt.plot((param[:,1]).flatten()[indexes][:500])
#plt.title('L')
#plt.figure()
#plt.plot((param[:,2]).flatten()[indexes][:500])
#plt.title('G')
#plt.figure()
#plt.plot((param[:,3]).flatten()[indexes][:500])
#plt.title('U')
#plt.figure()
#plt.plot((param[:,4]).flatten()[indexes][:500])
#plt.title('dir')



plt.figure()
plt.plot((((m_w*n_w/lim).flatten())/param[:,0])[indexes][:500])
plt.title('mw')

for i in range(5):
    i = 3    
    f = 20
    indr = reliable_scan6[:-1]>.78
    fig, ax = plt.subplots()
    ax.hist(noise_removed6[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(noise_det[indexes[i]][:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.6,.75),fontsize = f)
    ax.tick_params(axis='both', which='major', labelsize = f)
    ax.set_xlabel('$\eta_{noise}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.3,1.])
#    ax.set_ylim([.0,140])
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(reliable_scan6[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(reliable_scan[indexes[i]][:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.18,.65),fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=f)
    ax.set_xlabel('$\eta_{recov}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.85,1.])
#    ax.set_ylim([.0,180])
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(reliable_scan6[:-1][indr]*(1-noise_weight[:-1][indr])+noise_removed6[:-1][indr]*noise_weight[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(reliable_scan[indexes[i]][:-1]*(1-noise_weight[:-1])+noise_det[indexes[i]][:-1]*noise_weight[:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.18,.65),fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=f)
    ax.set_xlabel('$\eta_{tot}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.7,1.])
#    ax.set_ylim([.0,130])
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()


fig, ax = plt.subplots()
#plt.hist(noise_removed2[:-1],bins=30,alpha=.5,label = 'Clustering filter0')
##plt.hist(noise_removed5[:-1],bins=30,alpha=.5,label='5'+str(np.mean(noise_removed5)))
ax.hist(.5*(noise_removed6[:-1]+reliable_scan6[:-1]),bins=30, histtype = 'step', label = 'Clustering filter', lw = 2)
ax.hist(.5*(np.array(noise_det[19][:-1])+np.array(reliable_scan[19][:-1])),bins=30, histtype = 'step', label = 'Median filter', lw = 2)
fig.legend(loc = (.6,.75),fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('$(\eta_{recov}+\eta_{noise})/2$',fontsize = 16)
ax.set_ylabel('$Counts$',fontsize = 16)



###############################
# Noise figures

noise1 = np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.5,azim_frac=.3, tot = 'yes'),r_0_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.7,azim_frac=.6, tot = 'yes'),r_0_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.9,azim_frac=.9, tot = 'yes'),r_0_g.shape)

ind = noise1==0

a = np.max(noise1)
c = np.min(noise1)
if (a-c) > 0:
    b = 1
    d = -1   
    m = (b - d) / (a - c)
noise1 = (m * (noise1 - c)) + d
noise1 = noise1*35.0

noise1[ind] = np.nan

f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
plt.contourf(r_0_g,phi_0_g, noise1, 50,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')


#############################################

with open('noise_removed2.pkl', 'wb') as writer:
    pickle.dump(noise_removed2,writer)
with open('not_noise_removed2.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed2,writer)    
with open('noise_removed5.pkl', 'wb') as writer:
    pickle.dump(noise_removed5,writer)
with open('not_noise_removed5.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed5,writer)  
with open('noise_removed6.pkl', 'wb') as writer:
    pickle.dump(noise_removed6,writer)
with open('not_noise_removed6.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed6,writer)    
with open('noise_removed_med.pkl', 'wb') as writer:
    pickle.dump(noise_removed_med,writer)
with open('not_noise_removed_med.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed_med,writer)

tot_noise = np.array([np.sum(ind_noise[i]) for i in range(len(ind_noise[:-1]))])

tot_not_noise = np.array([np.sum(~ind_noise[i]) for i in range(len(ind_noise[:-1]))])

noise_diff = np.array([(clust-med)/med for med, clust in zip(noise_removed_med[:-1],noise_removed6[:-1])])

not_noise_diff = np.array([(clust-med)/med for med, clust in zip(not_noise_removed_med[:-1],not_noise_removed6[:-1])])

plt.figure()
plt.hist(noise_diff,bins=30,alpha=.5,label=str(np.mean(noise_diff)))
plt.legend()

plt.figure()
plt.hist(not_noise_diff,bins=30,alpha=.5,label=str(np.mean(not_noise_diff)))
plt.legend()

 # In[]
p = ws3_w_df.elev.unique()
r = np.array(ws3_w_df.iloc[(ws3_w_df.elev==
                               min(p)).nonzero()[0][0]].range_gate)
r_g, p_g = np.meshgrid(r, np.radians(p)) # meshgrid

scan_n = 1

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(377):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df.ws.loc[df.scan==scan_n].values,100,cmap='jet')
    plt.pause(.001)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(375):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_int.ws.loc[scan_n].values,100,cmap='jet')
    plt.pause(.001)
           

im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df.ws.loc[df.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_median.ws.loc[df.scan==scan_n].values,np.linspace(-2,13,100),cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),ws3_w_df.ws.loc[ws3_w_df.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


# In[]

############# Re noise
#n = 0
#for dir_mean in Dir:
#    for u_mean in utot:
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
#            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            if vlos0_file_name in onlyfiles:
#                noise0 = np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.5,azim_frac=.3),r_0_g.shape)
#                noise0 = noise0+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.7,azim_frac=.6),r_0_g.shape)
#                noise0 = noise0+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.9,azim_frac=.9),r_0_g.shape)
#                #normalize noise
#                #isolate areas without noise
#                ind_no_noise = (noise0 == 0)
#                a = np.max(noise0)
#                c = np.min(noise0)
#                if (a-c) > 0:
#                    b = 1
#                    d = -1   
#                    m = (b - d) / (a - c)
#                    noise0 = (m * (noise0 - c)) + d
#                noise0[ind_no_noise] = 0.0
#                noise0 = noise0*35.0
#                vlos0 = np.reshape(np.fromfile(vlos0_file_name, dtype=np.float32),r_0_g.shape)
#                vlos0_noise = vlos0 + noise0
#                (vlos0_noise.flatten()).astype(np.float32).tofile('noise0_'+vlos0_file_name)
#                n = n+1
#                print(n)


####################New data frame updated noise, rutine for noisy dataframe creation, should be a function
#with open('df_vlos_noise.pkl', 'rb') as reader:
#    df_vlos0_noise = pickle.load(reader)
#df_vlos0_noise['scan'] = df_vlos0_noise.groupby('azim').cumcount()
## reindex
#df_vlos0_noise.reset_index(inplace=True)
#scan = 0
#v_in = []
#m = np.arange(3,597,3)
#azim_unique = phi_0_g[:,0]*180/np.pi
#df_noise = pd.DataFrame(columns=df_vlos0_noise.columns[1:])
#aux_df_noise = np.zeros((len(azim_unique),len(df_vlos0_noise.columns[1:])))
#for dir_mean in Dir:
#    for u_mean in utot:
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
#            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            if vlos0_file_name in onlyfiles:
#                n = 1
#                aux_df_noise[:,0] = azim_unique
#                v_in.append(vlos0_file_name)
#                ind = df_vlos0_noise_med.scan == scan
#                dataws = np.reshape(np.fromfile('noise0_'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
#                dataCNR = df_vlos0_noise_med.CNR.loc[ind].values
#                datar = df_vlos0_noise_med.range_gate.loc[ind].values
#                for i in range(dataCNR.shape[1]):
#                    aux_df_noise[:,n:n+3] = np.c_[datar[:,i],dataws[:,i],dataCNR[:,i]]
#                    n = n+3
#                df_noise = pd.concat([df_noise, pd.DataFrame(data=aux_df_noise,
#                           index = df_vlos0_noise_med.index[ind],columns = df_vlos0_noise.columns[1:])])    
#                scan+=1
#                print(scan)
#df_noise['scan'] = df_noise.groupby('azim').cumcount()

################## saving the synthetic dataframe

#with open('df_syn_fin.pkl', 'wb') as writer:
#    pickle.dump(df_noise,writer) 

# In[]
######################### Identifying noise ####################################
#ae = [0.025, 0.05, 0.075]
#L = [125,250,500,750]
#G = [0,1,2,2.5,3.5]
#seed = np.arange(1,10)
#ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
#utot = np.linspace(15,25,5)
#Dir = np.linspace(90,270,5)*np.pi/180
#onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]
#ind_noise = []
#param = []
#sigma = []
#for dir_mean in Dir:
#    for u_mean in utot:
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
#            vlos0_file = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            #vlos1_file = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            vlos0_noise_file = 'noise0_'+vlos0_file
#            #vlos1_noise_file = 'noise1_'+vlos1_file
#            if vlos0_file in onlyfiles:
#                #v_in.append(vlos0_file) 
#                param.append(np.array([ae_i,L_i,G_i,seed_i,u_mean,int(dir_mean*180/np.pi)]))
#                vlos0 = np.fromfile(vlos0_file, dtype=np.float32)
#                sigma.append(np.std(vlos0))
#                vlos0_noise = np.fromfile(vlos0_noise_file, dtype=np.float32)
#                diff = vlos0-vlos0_noise
#                ind_noise.append(np.reshape(diff != 0,r_0_g.shape))
#param = np.array(param)
###############################################################################


## In[Figures for CNR]   
#                
#scan = 10
#ind_scan = df_noise.scan == scan
#
#f=16
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_noise.CNR.loc[ind_scan].values, 50,cmap='jet')
##ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
##ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
#
#cbar.ax.tick_params(labelsize=f)
#ax.tick_params(labelsize=f)
#
#cbar.ax.set_ylabel("$CNR, simulated$", fontsize=f)
#ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
#ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
#ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
#
#scan = 1000
#ind_scan = df.scan == scan
#
#f=16
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.CNR.loc[ind_scan].values, 50,cmap='jet')
##ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
##ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
#
#cbar.ax.tick_params(labelsize=f)
#ax.tick_params(labelsize=f)
#
#cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
#ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
#ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
#
#f=16
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.ws.loc[ind_scan].values, np.linspace(-8,8,50),cmap='jet')
##ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
##ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
#
#cbar.ax.tick_params(labelsize=f)
#ax.tick_params(labelsize=f)
#
#cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
#ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
#ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
#
#
#f=16
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_noise.ws.loc[ind_scan].values, 50,cmap='jet')
##ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
##ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
#
#cbar.ax.tick_params(labelsize=f)
#ax.tick_params(labelsize=f)
#
#cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
#ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
#ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
#ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
#
#############################
#
##plt.scatter((r_0_g*np.cos(phi_0_g))[i_noise],(r_0_g*np.sin(phi_0_g))[i_noise])
##plt.colorbar()
##
##plt.figure()
##plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise_median.ws.loc[ind_scan ].values, 30, cmap='jet')
##plt.colorbar()

#######################################################################################################

################## DBSCAN filter, commented are not good, just for comparison
#with open('df_syn_fin.pkl', 'rb') as reader:
#    df_noise = pickle.load(reader)     
#    
#t_step = 3 #5
#ind=np.unique(df_noise.scan.values)%t_step==0
#times= np.unique(np.append(np.unique(df_noise.scan.values)[ind],df_noise.scan.values[-1])) 
##mask0=pd.DataFrame()
##mask1=pd.DataFrame()
##mask2=pd.DataFrame()
##mask3=pd.DataFrame()
##mask4=pd.DataFrame()
##mask5=pd.DataFrame()
#mask6=pd.DataFrame()
#
#for i in range(len(times)-1):
#       print(times[i])
#       loc = (df_noise.scan>=times[i]) & (df_noise.scan<times[i+1])
##       mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)]) 
##       mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)]) 
##       mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
##       mask3 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat3)]) 
##       mask4 = pd.concat([mask4,fl.data_filt_DBSCAN(df_noise.loc[loc],feat4)]) 
##       mask5 = pd.concat([mask5,fl.data_filt_DBSCAN(df_noise.loc[loc],feat5)])
#       mask6 = pd.concat([mask6,fl.data_filt_DBSCAN(df_noise.loc[loc],feat6,epsCNR=True)])        
#       if i == range(len(times)-1):
#          loc = df_noise.scan == times[i+1]
##          mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)]) 
##          mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)]) 
##          mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
##          mask3 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat3)]) 
##          mask4 = pd.concat([mask4,fl.data_filt_DBSCAN(df_noise.loc[loc],feat4)]) 
##          mask5 = pd.concat([mask5,fl.data_filt_DBSCAN(df_noise.loc[loc],feat5)])
#          mask6 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat6,epsCNR=True)])           

################### 5 scans steps
          
#with open('mask_filter_r_ws_5.pkl', 'wb') as writer:
#    pickle.dump(mask0,writer)
#with open('mask_filter_r_ws_CNR_5.pkl', 'wb') as writer:
#    pickle.dump(mask1,writer)
#with open('mask_filter_r_ws_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask2,writer)
#with open('mask_filter_r_ws_CNR_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask3,writer)
#   
#with open('mask_filter_r_azim_ws_5.pkl', 'wb') as writer:
#    pickle.dump(mask4,writer)
#with open('mask_filter_r_azim_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask5,writer)  
#with open('mask_filter_r_azim_ws_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask6,writer) 
     
#######################3 scans steps
          
#with open('mask_filter_r_ws_3.pkl', 'wb') as writer:
#    pickle.dump(mask0,writer)
#with open('mask_filter_r_ws_CNR_3.pkl', 'wb') as writer:
#    pickle.dump(mask1,writer)
#with open('mask_filter_r_ws_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask2,writer)
#with open('mask_filter_r_ws_CNR_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask3,writer)
#with open('mask_filter_r_azim_ws_3.pkl', 'wb') as writer:
#    pickle.dump(mask4,writer)
#with open('mask_filter_r_azim_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask5,writer)  
#with open('mask_filter_r_azim_ws_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask6,writer)   

