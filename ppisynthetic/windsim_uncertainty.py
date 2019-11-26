# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:24:19 2018

@author: lalc
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import tkinter as tkint
import tkinter.filedialog
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from os import listdir
from os.path import isfile, join

import ppiscanprocess.spectraconstruction as sc
import ppiscanprocess.windfieldrec as wr
import spectralfitting.spectralfitting as sf

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


import scipy.spatial.qhull as qhull

from scipy.spatial import Delaunay

from numba import jit


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

# In[]

def num_lidar(r,phi,U,V,x,y,d,rg=35,deg=np.pi/180,n=10,m=10,kernel='epanechnikov',corr=True):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate
    
    phi_loc, r_loc = np.meshgrid(np.linspace(-deg,deg,m),np.linspace(-rg/2,rg/2,n))
    
    x_prime = x-d[0]
    y_prime = y-d[1]  
    if (len(x.shape)==1) & (len(y.shape)==1):
        x_prime, y_prime = np.meshgrid(x_prime,y_prime)
    if len(U.shape)>1 :
        U = U.flatten()
        V = V.flatten()
    # Transform to polar
    
    phi_prime = np.arctan2(y_prime.flatten(),x_prime.flatten()) 
    r_prime = np.sqrt(x_prime.flatten()**2+y_prime.flatten()**2)
      
    tree_lid = KDTree(np.c_[r.flatten(),phi.flatten()],metric='manhattan')
    
    ind = tree_lid.query(np.c_[r_prime,phi_prime],return_distance=False)
    
    V_LOS = np.zeros(len(r.flatten()))
    
    index_phi = (phi_prime>np.min(phi.flatten())-deg) & (phi_prime<np.max(phi.flatten())+deg)
    index_r = (r_prime>np.min(r.flatten())-35/2) & (r_prime<np.max(r.flatten())+35/2)
    n_index=[]
    
    for i in range(len(r.flatten())):        
        index = ind.flatten() == i
        index = ((index) & (index_phi)) & (index_r)
        V_L = np.cos(phi_prime[index])*U[index]+np.sin(phi_prime[index])*V[index]
        if kernel == 'gaussian':
            w = lambda h: sp.stats.norm.pdf(h, loc=0, scale=1)
        elif kernel == 'epanechnikov':
            w = lambda h: .75*(1-(h)**2)
        elif kernel == 'triangle':
            w = lambda h: 1-np.abs(h)
        else:
            w = lambda h: .75*(1-(h)**2)
        n_index.append(np.sum(index))
        
        # interpolation on beams
        
        if n_index[i]>10:    
            
            dr = -r_prime[index]+r.flatten()[i]
            dphi = -phi_prime[index]+phi.flatten()[i]
            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest',fill_value = 0.0)
            if corr:
                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
                V_L_TI = np.repeat([V_L_TI],n,axis=0)
                rg_corr = rg/(1+V_L_TI)
            else:
                rg_corr= rg            
            h = 2*r_loc/rg_corr
            wg = w(h)
            V_LOS[i] = np.nansum(np.nansum(V_L_g*wg,axis=0)/np.nansum(wg,axis=0))/m
            print(i,n_index[i],V_LOS[i],np.sum(np.isnan(V_L_g)),np.sum(np.isnan(wg)))
            
        if (n_index[i]<=10) & (n_index[i]>0):    
            dr = -r_prime[index]+r.flatten()[i]
            dphi = -phi_prime[index]+phi.flatten()[i]
            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest')
            if corr:
                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
                V_L_TI = np.repeat([V_L_TI],n,axis=0)
                rg_corr = rg/(1+V_L_TI)
            else:
                rg_corr= rg
            h = 2*r_loc/rg_corr
            wg = w(h)
            #simple average over azimuth
            V_LOS[i] = np.sum(np.sum(V_L_g*wg,axis=0)/np.sum(wg,axis=0))/m
        if n_index[i]==0:
            V_LOS[i] = np.nan
        
    return (np.reshape(-V_LOS,r.shape),np.array(n_index))

# In[]
    
def num_lidar_rot(r,phi,U_in,V_in,x,y,d,rg=35,deg=np.pi/180,n=31,m=31,kernel='epanechnikov',method='cubic'):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate
        
    x_prime = x.flatten()-d[0]
    y_prime = y.flatten()-d[1]  
    U = U_in.flatten()
    V = V_in.flatten()
    # Transform to polar
    
    phi_prime = np.arctan2(y_prime,x_prime) 
    r_prime = np.sqrt(x_prime**2+y_prime**2)
  
    index_phi = (phi_prime>np.min(phi.flatten())-deg) & (phi_prime<np.max(phi.flatten())+deg)
    index_r = (r_prime>np.min(r.flatten())-rg/2) & (r_prime<np.max(r.flatten())+rg/2)
        
    index = (index_phi) & (index_r)
    
    phi_prime = phi_prime[index]
    r_prime = r_prime[index]
    U = U[index]
    V = V[index]

    r_unique = np.unique(r)
    phi_unique = np.unique(phi)
    
    delta_r = np.min(np.diff(r_unique))
    delta_phi = np.min(np.diff(phi_unique))
    
    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+delta_r/2,len(r_unique)*(n-1)+1)
    h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r    
    
    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+delta_phi/2, len(phi_unique)*(m-1)+1)    
    r_grid, phi_grid = np.meshgrid(r_refine,phi_refine)
    
    # Epanechnikov
    w = .75*(1-h**2) 
    w = np.reshape(np.repeat(w,phi_grid.shape[0]),(phi_grid.T).shape).T
    print(1)
    U_int = sp.interpolate.griddata(np.c_[r_prime, phi_prime], U,
            (r_grid.flatten(), phi_grid.flatten()), method=method,fill_value = np.nan,rescale=True)  
    print(2)
    V_int = sp.interpolate.griddata(np.c_[r_prime, phi_prime], V,
            (r_grid.flatten(), phi_grid.flatten()), method=method,fill_value = np.nan,rescale=True)
    print(3)
    V_L = np.cos(phi_grid.flatten())*U_int+np.sin(phi_grid.flatten())*V_int
    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
    V_L = np.reshape(V_L,r_grid.shape)
    print(V_L.shape)
    
    norm = np.sum(w[0,:(n-1)])

    V_L = (V_L[:,:-1]*w[:,:-1]/norm).T

    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
    
    V_L = np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1).T
    
    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
    
    w_p = np.ones(V_L.shape)/(m-1)   
    V_L = -(V_L[:-1,:]*w_p[:-1,:]) 
    
    print(V_L.shape)
     
    return np.flip(np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1),axis=0)

# In[]
#def del_triang(r,phi,x,y,d):
#    
#    x_prime = x.flatten()-d[0]
#    y_prime = y.flatten()-d[1]  
#
#    # Transform to polar
#    
#    phi_prime = np.arctan2(y_prime,x_prime) 
#    r_prime = np.sqrt(x_prime**2+y_prime**2)
#    
#    index_phi = (phi_prime>np.min(phi.flatten())-deg) & (phi_prime<np.max(phi.flatten())+deg)
#    index_r = (r_prime>np.min(r.flatten())-rg/2) & (r_prime<np.max(r.flatten())+rg/2)
#        
#    index = (index_phi) & (index_r)
#    
#    phi_prime = phi_prime[index]
#    r_prime = r_prime[index]
#    
#    return Delaunay(np.c_[])
    
# In[]

def interp_weights(xy, uv, d = 2):
    print('triangulation...')
    tri = qhull.Delaunay(xy)
    plt.triplot(xy[:,0],xy[:,1],tri.simplices)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interp_weights2(uv, tri, d = 2):
    print('triangulation...')
    #tri = qhull.Delaunay(xy)
    #plt.triplot(xy[:,0],xy[:,1],tri.simplices)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def early_weights(r,phi,x,y,d,n=21,m=51):
    
    x_prime = x.flatten()-d[0]
    y_prime = y.flatten()-d[1] 
    
    r_unique = np.unique(r)
    phi_unique = np.unique(phi)
    
    delta_r = np.min(np.diff(r_unique))
    delta_phi = np.min(np.diff(phi_unique))
    
    phi_prime = np.arctan2(y_prime,x_prime) 
    r_prime = np.sqrt(x_prime**2+y_prime**2)
    
    index_phi = (phi_prime>np.min(phi.flatten())-delta_phi) & (phi_prime<np.max(phi.flatten())+delta_phi)
    index_r = (r_prime>np.min(r.flatten())-delta_r) & (r_prime<np.max(r.flatten())+delta_r)
        
    index = (index_phi) & (index_r)
    
    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+delta_r/2,len(r_unique)*(n-1)+1)      
    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+delta_phi/2, len(phi_unique)*(m-1)+1) 
    
    r_grid, phi_grid = np.meshgrid(r_refine,phi_refine)
    
    xy = np.c_[x_prime[index],y_prime[index]]    
    uv = np.c_[r_grid.flatten()*np.cos(phi_grid.flatten()),r_grid.flatten()*np.sin(phi_grid.flatten())]  
    
    vtx, wts = interp_weights(xy, uv, d = 2)    
    h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r     
    w = .75*(1-h**2) 
    w = np.reshape(np.repeat(w,phi_grid.shape[0]),(phi_grid.T).shape).T
    norm = np.sum(w[0,:(n-1)])
    w = w/norm
    s_ref = np.sin(phi_grid.flatten())
    c_ref = np.cos(phi_grid.flatten())
    shapes = np.array([r_grid.shape[0], r_grid.shape[1], n, m])
        
    return (vtx, wts, w, c_ref, s_ref, shapes)

def early_weights2(r, phi, dir_mean , tri, d, center, n=21, m=51):
    gamma = (2*np.pi-dir_mean) 
    r_unique = np.unique(r)
    phi_unique = np.unique(phi)
    delta_r = np.min(np.diff(r_unique))
    delta_phi = np.min(np.diff(phi_unique))
    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+
                           delta_r/2,len(r_unique)*(n-1)+1)      
    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+
                             delta_phi/2, len(phi_unique)*(m-1)+1)
    r_t_refine, phi_t_refine = np.meshgrid(r_refine,phi_refine)    
    
    #LOS angles
        
    s_ref = np.sin(phi_t_refine-gamma)
    c_ref = np.cos(phi_t_refine-gamma)
    
    r_t_refine,phi_t_refine = wr.translationpolargrid((r_t_refine, phi_t_refine),d)
#    r_t_refine, phi_t_refine = np.meshgrid(r_t_refine,phi_t_refine)
    x_t_refine, y_t_refine = r_t_refine*np.cos(phi_t_refine), r_t_refine*np.sin(phi_t_refine)

# Rotation and translation
    
    
    x_trans = -(center)*np.sin(gamma)
    y_trans = (center)*(1-np.cos(gamma))
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
    Xx = np.array(np.c_[x_t_refine.flatten(), y_t_refine.flatten(),
                                    np.ones(len(y_t_refine.flatten()))]).T
    Xx = np.dot(T1,np.dot(R,Xx))
    uv = Xx[:2,:].T
    vtx, wts = interp_weights2(uv, tri, d = 2)
   
    h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r     
    w = .75*(1-h**2) 
    w = np.reshape(np.repeat(w,phi_t_refine.shape[0]),(phi_t_refine.T).shape).T
    norm = np.sum(w[0,:(n-1)])
    w = w/norm

    shapes = np.array([phi_t_refine.shape[0], phi_t_refine.shape[1], n, m])
        
    return (vtx, wts, w, c_ref, s_ref, shapes)

# In[]
tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)
N_x = 2048
N_y = 2048
x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)
grid = np.meshgrid(x,y)

xtrans = 0
ytrans = y[0]/2
       
ae = [.025,0.05, 0.075]
L = [125,250,500,750,1000,1250,1500,2000]
G = [2,2.5,3,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

utot = np.linspace(15,25,20)
Dir = np.linspace(0,330,12)*np.pi/180

#utot,Dir = np.meshgrid(utot,Dir)

#Dir = Dir.flatten()
#utot = utot.flatten()
#ind = np.argsort(Dir)
#utot = utot[ind]
#Dir = Dir[ind]

i = 2

dir_mean = Dir[2]

#gamma = 0#-(2*np.pi-dir_mean)
#
#S11 = np.cos(gamma)
#S12 = np.sin(gamma)
#T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
#T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
#R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
#T = np.dot(np.dot(T1,R),T2)
##vel = np.array(np.c_[U.flatten(),V.flatten()]).T
##vel = np.dot(T,vel)
#
##vel = np.array(np.c_[U_prime.flatten(),V_prime.flatten()]).T
##vel = np.dot(R[:-1,:-1],vel)
#Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[1].flatten()))]).T
#Xx = np.dot(T,Xx)
##U_rot = np.reshape(vel[0,:],(N_x,N_y))
##V_rot = np.reshape(vel[1,:],(N_x,N_y))
#xx = np.reshape(Xx[0,:],(N_x,N_y))
#yy = np.reshape(Xx[1,:],(N_x,N_y))
#
#r, phi = r_s_g, np.pi-phi_s_g,
#n=21
#m=51    
#x_prime = xx.flatten()-d[0]/2
#y_prime = yy.flatten()-d[1]/2 
#
#r_unique = np.unique(r)
#phi_unique = np.unique(phi)
#
#delta_r = np.min(np.diff(r_unique))
#delta_phi = np.min(np.diff(phi_unique))
#
#phi_prime = np.arctan2(y_prime,x_prime) 
#r_prime = np.sqrt(x_prime**2+y_prime**2)
#
#index_phi = (phi_prime>np.min(phi.flatten())-delta_phi) & (phi_prime<np.max(phi.flatten())+delta_phi)
#index_r = (r_prime>np.min(r.flatten())-delta_r) & (r_prime<np.max(r.flatten())+delta_r)
#    
#index = (index_phi) & (index_r)
#
#r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+delta_r/2,len(r_unique)*(n-1)+1)      
#phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+delta_phi/2, len(phi_unique)*(m-1)+1) 
#
#r_grid, phi_grid = np.meshgrid(r_refine,phi_refine)

#xy = np.c_[x_prime[index],y_prime[index]]    
#uv = np.c_[r_grid.flatten()*np.cos(phi_grid.flatten()),r_grid.flatten()*np.sin(phi_grid.flatten())] 

#plt.scatter(grid[0].flatten(),grid[1].flatten())
#
#plt.scatter(xx.flatten(),yy.flatten())
#
#plt.scatter(xy[:,0]+d[0]/2,xy[:,1]+d[1]/2)
#
#plt.scatter(uv[:,0]+d[0]/2,uv[:,1]+d[1]/2)
#
#plt.scatter((r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten())

#tri_try = Delaunay(xy, qhull_options = "QJ")
#
#points = np.arange(0,len(uv),100000)
#
#simplex = np.zeros(uv.shape[0])
#
#for i in range(len(points)-1):
#    print(i)
#    simplex[points[i]:points[i+1]] = tri_try.find_simplex(uv[points[i]:points[i+1]])
#    
#vertex = xy[tri_try.simplices]    


#r_s_t,phi_s_t = wr.translationpolargrid((r_s_g, np.pi-phi_s_g),-d/2)
#r_v_t,phi_v_t = wr.translationpolargrid((r_v_g, np.pi-phi_v_g),d/2)
#
#x_s_t, y_s_t = (r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()
#x_v_t, y_v_t = (r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()

#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
#ax.use_sticky_edges = False
#ax.margins(0.07)
#
#for dir_mean in np.unique(Dir):
#    #dir_mean = np.unique(Dir)[4]
#    ax.cla()
#    ax.plot([x.min(),x.min()],[y.min(),y.max()],color='grey',alpha=.5)
#    ax.plot([x.min(),x.max()],[y.max(),y.max()],color='grey',alpha=.5)
#    ax.plot([x.max(),x.max()],[y.max(),y.min()],color='grey',alpha=.5)
#    ax.plot([x.max(),x.min()],[y.min(),y.min()],color='grey',alpha=.5)
#    gamma = (2*np.pi-dir_mean)
#    x_trans = -(y[0]/2)*np.sin(gamma)
#    y_trans = (y[0]/2)*(1-np.cos(gamma))
#    S11 = np.cos(gamma)
#    S12 = np.sin(gamma)
#    T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
#    #T2 = np.array([[1,0,-x_trans], [0,1, -y_trans], [0, 0, 1]])
#    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
#    #T = np.dot(R,T1)
#    
#    Xxs = np.array(np.c_[x_s_t, y_s_t,np.ones(len(x_s_t))]).T
#    Xxs = np.dot(T1,np.dot(R,Xxs))
#    
#    Xxv = np.array(np.c_[x_v_t, y_v_t,np.ones(len(x_v_t))]).T
#    Xxv = np.dot(T1,np.dot(R,Xxv))
#    
#    ax.plot(Xxv[0,:],Xxv[1,:],'.',color='k',alpha=0.5)
#    ax.plot(Xxs[0,:],Xxs[1,:],'.',color='b',alpha=0.5)
#    
#    plt.pause(1)

#plt.triplot(xy[:,0]+d[0]/2,xy[:,1]+d[1]/2,triangles = tri_try.simplices)
#
#    
#vtx, wts = interp_weights(xy, uv, d = 2)


tri_try = Delaunay(np.c_[grid[0].flatten(),grid[1].flatten()], qhull_options = "QJ")

dir_mean = np.unique(Dir)[2]

gamma = (2*np.pi-dir_mean)
x_trans = -(y[0]/2)*np.sin(gamma)
y_trans = (y[0]/2)*(1-np.cos(gamma))
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
T2 = np.array([[1,0,-x_trans], [0,1, -y_trans], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(R,T1)

Xxs = np.array(np.c_[x_s_t, y_s_t,np.ones(len(x_s_t))]).T
Xxs = np.dot(T1,np.dot(R,Xxs))

Xxv = np.array(np.c_[x_v_t, y_v_t,np.ones(len(x_v_t))]).T
Xxv = np.dot(T1,np.dot(R,Xxv))

uv_v = Xxv[:2,:].T
xy = np.c_[grid[0].flatten(),grid[1].flatten()]

vtx, wts = interp_weights2(xy, uv_v, tri_try, d = 2)





L_i, G_i, ae_i, seed_i = 500, 3.0, 0.05, -6
    
u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T

U_in = u+15.0
V_in = v

gamma = -(2*np.pi-dir_mean)
x_trans = 0
y_trans = y[0]/2
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
T2 = np.array([[1,0,-x_trans], [0,1, -y_trans], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[1].flatten()))]).T
Xx = np.dot(T,Xx)
xx = np.reshape(Xx[0,:],(N_x,N_y))
yy = np.reshape(Xx[1,:],(N_x,N_y))
vel = np.array(np.c_[U_in.flatten(),V_in.flatten()]).T
vel = np.dot(R[:-1,:-1],vel)
U_rot = np.reshape(vel[0,:],(N_x,N_y))
V_rot = np.reshape(vel[1,:],(N_x,N_y))

vtx, wts, w, c_ref, s_ref, shapes = early_weights2(r_s_g, np.pi-phi_s_g, dir_mean , tri_try, -d/2, y[0]/2)

vloss = num_lidar_rot_del(U_in.flatten(),V_in.flatten(),vtx,wts,w,c_ref.flatten(), s_ref.flatten(), shapes)

plt.figure()
plt.contourf(r_s_t*np.cos(phi_s_t),r_s_t*np.sin(phi_s_t),vloss,cmap='jet')
plt.colorbar()

vtx, wts, w, c_ref, s_ref, shapes = early_weights2(r_v_g, np.pi-phi_v_g, dir_mean , tri_try, d/2, y[0]/2)

vlosv = num_lidar_rot_del(U_in.flatten(),V_in.flatten(),vtx,wts,w,c_ref.flatten(), s_ref.flatten(), shapes)

plt.figure()
plt.contourf(r_v_t*np.cos(phi_v_t),r_v_t*np.sin(phi_v_t),vlosv,cmap='jet')
plt.colorbar()


vlosv_int_sq = sp.interpolate.griddata(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()], vlosv.flatten(), (grid[0].flatten(), grid[1].flatten()), method='cubic')
vloss_int_sq = sp.interpolate.griddata(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()], vloss.flatten(), (grid[0].flatten(), grid[1].flatten()), method='cubic')

vlosv_int_sq = np.reshape(vlosv_int_sq,grid[0].shape)
vloss_int_sq = np.reshape(vloss_int_sq,grid[0].shape)

plt.figure()
plt.contourf(grid[0],grid[1],vlosv_int_sq,cmap='jet')
plt.figure()
plt.contourf(grid[0],grid[1],vloss_int_sq,cmap='jet')

r_tri_s = np.sqrt(grid[0]**2 + grid[1]**2)
phi_tri_s = np.arctan2(grid[1],grid[0])

r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)

T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)

T_a = [np.array([[-np.sin(bi),np.sin(ai)],[np.cos(bi),-np.cos(ai)]])/np.sin(ai-bi) for ai,bi in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten())]

vel = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),vlosv_int_sq.flatten(),vloss_int_sq.flatten())])

U_early_s = np.reshape(vel[:,0],grid[0].shape)
V_early_s = np.reshape(vel[:,1],grid[0].shape)

U_early_s = np.zeros(grid[0].shape)
V_early_s = np.zeros(grid[0].shape)


plt.figure()
plt.contourf(x,y,U_in,cmap='jet')

plt.figure()
plt.contourf(xx,yy,np.sqrt(V_rot**2+U_rot**2),100,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(grid[0],grid[1],U_in,100,cmap='jet')
plt.colorbar()

plt.triplot(tri,alpha=.3)
plt.figure()
plt.contourf(grid[0],grid[1],-V_early_s,100,cmap='jet')
plt.colorbar()

index = ~(np.isnan(vlosv_int_sq)|np.isnan(vloss_int_sq))

U, V = jit_dir_rec_rapid(vlosv_int_sq[index],vloss_int_sq[index],phi_tri_v_s[index],phi_tri_s_s[index])

U_early_s[~index] = np.nan
U_early_s[~index] = np.nan

U_early_s[index] = U
U_early_s[index] = V

# In[Rapid reconstruction]
def dir_rec_rapid(V_a,V_b,a,b):
    U = np.zeros(V_a.shape).astype(np.float32)  
    V = np.zeros(V_a.shape).astype(np.float32)  
    i = 0
    for ai,bi,V_a_i,V_b_i in zip(a,b,V_a,V_b):
        #T = np.array([[-np.sin(bi),np.sin(ai)],[np.cos(bi),-np.cos(ai)]])/np.sin(ai-bi)
        #aux = np.dot(T,np.c_[V_a_i,V_b_i].T)
        U[i] = (-np.sin(bi)*V_a +np.sin(ai)*V_b)/np.sin(ai-bi)
        V[i] = (np.cos(bi)*V_a -np.cos(ai)*V_b)
        i+=1
    return (U,V)

jit_dir_rec_rapid = jit(dir_rec_rapid,nopython=True)    


# In[]
def circleratios(tri):
        """
        Returns a measure of the triangulation triangles flatness.

        The ratio of the incircle radius over the circumcircle radius is a
        widely used indicator of a triangle flatness.
        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
        triangles. Circle ratios below 0.01 denote very flat triangles.

        To avoid unduly low values due to a difference of scale between the 2
        axis, the triangular mesh can first be rescaled to fit inside a unit
        square with :attr:`scale_factors` (Only if *rescale* is True, which is
        its default value).

        Parameters
        ----------
        rescale : boolean, optional
            If True, a rescaling will be internally performed (based on
            :attr:`scale_factors`, so that the (unmasked) triangles fit
            exactly inside a unit square mesh. Default is True.

        Returns
        -------
        circle_ratios : masked array
            Ratio of the incircle radius over the
            circumcircle radius, for each 'rescaled' triangle of the
            encapsulated triangulation.
            Values corresponding to masked triangles are masked out.

        """
        # Coords rescaling
#        if rescale:
#            (kx, ky) = self.scale_factors
#        else:
        #(kx, ky) = (1.0, 1.0)
#        pts = np.vstack([self._triangulation.x*kx,
#                         self._triangulation.y*ky]).T
        
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

# In[]   

def num_lidar_rot_del(U_in,V_in,vtx,wts,w,c_ref, s_ref, shapes):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate    
    n = shapes[2]
    m = shapes[3]
    
    U = interpolate(U_in, vtx, wts, fill_value=np.nan)
    V = interpolate(V_in, vtx, wts, fill_value=np.nan)
    
    V_L = c_ref*U+s_ref*V
    
    V_L = np.reshape(V_L, (shapes[0],shapes[1]))
    
    V_L = (V_L[:,:-1]*w[:,:-1]).T
    
    V_L = np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1).T

    w_p = np.ones(V_L.shape)/(m-1)
    
    V_L = -(V_L[:-1,:]*w_p[:-1,:])     
    
    print(U.shape,V.shape,V_L.shape,shapes)
     
    return np.flip(np.nansum(V_L.reshape(-1,(m-1),V_L.shape[-1]),axis=1),axis=0) 

# In[]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim Output dir')
root.destroy()


cwd = os.getcwd()
os.chdir(file_in_path)


# In[Lidar local coordinates]

r_s = np.linspace(150,7000,198)
r_v = np.linspace(150,7000,198)
phi_s = np.linspace(256,344,45)*np.pi/180
phi_v = np.linspace(196,284,45)*np.pi/180
r_s_g, phi_s_g = np.meshgrid(r_s,phi_s)
r_v_g, phi_v_g = np.meshgrid(r_v,phi_v)  

siro = np.array([6322832.3,0])
vara = np.array([6327082.4,0])
d = vara-siro


r_s_t,phi_s_t = wr.translationpolargrid((r_s_g, np.pi-phi_s_g),-d/2)
r_v_t,phi_v_t = wr.translationpolargrid((r_v_g, np.pi-phi_v_g),d/2)

x_max = np.max(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])
x_min = np.min(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])

y_max = np.max(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])
y_min = np.min(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])

L_x = x_max-x_min
L_y = y_max-y_min

# In[Synthetic wind filed generation]

#Sepctral tensor parameters
os.chdir(file_in_path)
ae = .05
L = 2000
G = 3.1
N_x = 1024
N_y = 1024


input_file = 'sim.inp.txt'

file = open(input_file,'w') 
file.write('2\n') 
file.write('2\n') 
file.write('1\n')
file.write('2\n') 
file.write('3\n')  
file.write(str(N_x)+'\n')
file.write(str(N_y)+'\n')
file.write(str(L_x)+'\n')
file.write(str(L_y)+'\n')
file.write('basic\n')
file.write(str(ae)+'\n')
file.write(str(L)+'\n')
file.write(str(G)+'\n')
file.write('-30\n')
file.write('simu\n')
file.write('simv\n')
file.close() 

arg = 'windsimu'+' '+input_file
p=subprocess.run(arg)

u = np.reshape(np.fromfile("simu", dtype=np.float32),(N_x,N_y))
v = np.reshape(np.fromfile("simv", dtype=np.float32),(N_x,N_y))

os.chdir(cwd)

# In[]
U_mean = 15
V_mean = 0
Dir = 270*np.pi/180

U = U_mean + u
V = V_mean + v

x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)

# Rotation

gamma = -(2*np.pi-Dir) # Rotation
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U.flatten(),V.flatten()]).T
vel = np.dot(T,vel)
X = np.array(np.c_[x,y]).T
X = np.dot(T,X)
U_prime = np.reshape(vel[0,:],(N_x,N_y))
V_prime = np.reshape(vel[1,:],(N_x,N_y))

# In[]

tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)
r_s_r = np.linspace(150,7000,198*2)
r_v_r = np.linspace(150,7000,198*2)
phi_s_r = np.linspace(256,344,45*2)*np.pi/180
phi_v_r = np.linspace(196,284,45*2)*np.pi/180
r_s_g_r, phi_s_g_r = np.meshgrid(r_s_r,phi_s_r)
r_v_g_r, phi_v_g_r = np.meshgrid(r_v_r,phi_v_r) 
tree_r,tri_r, wva_r, neighva_r, indexva_r, wsi_r, neighsi_r, indexsi_r = wr.grid_over2((r_v_g_r, np.pi-phi_v_g_r),(r_s_g_r, np.pi-phi_s_g_r),-d)

# In[]

vloss,_ = num_lidar(r_s_g, np.pi-phi_s_g,U_prime,V_prime,x,y,d/2,corr=False)
vlosv,_ = num_lidar(r_v_g, np.pi-phi_v_g,U_prime,V_prime,x,y,-d/2,corr=False)

r_min = np.max(r_s_g[np.isnan(vloss)])
ind_r = int((r_s == r_min).nonzero()[0]+1)

vloss_r = sp.interpolate.RectBivariateSpline(phi_s, r_s[ind_r:], vloss[:,int(ind_r):])(phi_s_r, r_s_r)

r_min = np.max(r_v_g[np.isnan(vlosv)])
ind_r = int((r_v == r_min).nonzero()[0]+1)

vlosv_r = sp.interpolate.RectBivariateSpline(phi_v, r_v[ind_r:], vlosv[:,int(ind_r):])(phi_v_r, r_v_r)

################ Interpolation of raw data

V_prime_t = sp.interpolate.RectBivariateSpline(y,x, V_prime)(tri.y,tri.x,grid=False)

V_prime_tr = sp.interpolate.RectBivariateSpline(y,x, V_prime)(tri_r.y,tri_r.x,grid=False)

U_prime_t = sp.interpolate.RectBivariateSpline(y,x, U_prime)(tri.y,tri.x,grid=False)

U_prime_tr = sp.interpolate.RectBivariateSpline(y,x, U_prime)(tri_r.y,tri_r.x,grid=False)

################################## Reconstruction
df_s = pd.DataFrame(vloss)
df_v = pd.DataFrame(vlosv)

Lidar_s = (df_s,2*np.pi-phi_s_g,wsi,neighsi,indexsi) 
Lidar_v = (df_v,2*np.pi-phi_v_g,wva,neighva,indexva)

U_rec, V_rec= wr.wind_field_rec(Lidar_v, Lidar_s, tree, tri, d)


df_s = pd.DataFrame(vloss_r)
df_v = pd.DataFrame(vlosv_r)

Lidar_s = (df_s,2*np.pi-phi_s_g_r,wsi_r,neighsi_r,indexsi_r) 
Lidar_v = (df_v,2*np.pi-phi_v_g_r,wva_r,neighva_r,indexva_r)

U_rec_r, V_rec_r= wr.wind_field_rec(Lidar_v, Lidar_s, tree_r, tri_r, d)

###############
#Early interpolation
indv = ~np.isnan(vlosv.flatten())
inds = ~np.isnan(vloss.flatten())
xv = (r_v_t*np.cos(phi_v_t)).flatten()[indv]
yv = (r_v_t*np.sin(phi_v_t)).flatten()[indv]
#
xs = (r_s_t*np.cos(phi_s_t)).flatten()[inds]
ys = (r_s_t*np.sin(phi_s_t)).flatten()[inds]
#
#
T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)
#

grd = np.meshgrid(x,y)

mask_s = np.reshape(tri_r.get_trifinder()(grd[0].flatten(),grd[1].flatten()),grd[0].shape) == -1

vlosv_int_sq = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='cubic')
vloss_int_sq = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='cubic')

ind_sq = np.isnan(vlosv_int_sq)
vlosv_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

ind_sq = np.isnan(vloss_int_sq)
vloss_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

vloss_int_sq = np.reshape(vloss_int_sq,grd[0].shape)
vlosv_int_sq = np.reshape(vlosv_int_sq,grd[0].shape)

vloss_int_sq[mask_s] = 0.0
vlosv_int_sq[mask_s] = 0.0

r_tri_s = np.sqrt(grd[0]**2 + grd[1]**2)
phi_tri_s = np.arctan2(grd[1],grd[0])



r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)



vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),-vlosv_int_sq.flatten(),-vloss_int_sq.flatten())])

U_early_s = np.reshape(vel_s[:,0],grd[0].shape)
V_early_s = np.reshape(vel_s[:,1],grd[0].shape)

U_prime[mask_s] = np.nan
V_prime[mask_s] = np.nan


############################

# In[]

ae = [.025,0.05, 0.075]
L = [125,250]#,500,750,1000,1250,1500,2000]
G = [2,2.5,3,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

N_x = 2048
N_y = 2048
i = 30
for ae_i,L_i,G_i,seed_i in zip(ae.flatten()[i:],L.flatten()[i:],G.flatten()[i:],seed.flatten()[i:]):
    print(i,str(L_i)+str(G_i)+str(ae_i)+str(seed_i))
    #Sepctral tensor parameters
    os.chdir(file_in_path)    
    
    input_file = 'sim.inp.txt'
    
    file = open(input_file,'w') 
    file.write('2\n') #fieldDim
    file.write('2\n') #NComp
    file.write('1\n') #u
    file.write('2\n') #v
    file.write('3\n') #w 
    file.write(str(N_x)+'\n')
    file.write(str(N_y)+'\n')
    file.write(str(L_x)+'\n')
    file.write(str(L_y)+'\n')
    file.write('basic\n')
    file.write(str(ae_i)+'\n')
    file.write(str(L_i)+'\n')
    file.write(str(G_i)+'\n')
    file.write(str(seed_i)+'\n')
    file.write('simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)+'\n')
    file.write('simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)+'\n')
    file.close() 
    
    arg = 'windsimu'+' '+input_file
    p=subprocess.run(arg)
    
    i = i+1

####################################################################
# In[]    
# Mask
tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)
N_x = 2048
N_y = 2048
x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)
grid = np.meshgrid(x,y)
#mask_s = np.reshape(tri.get_trifinder()(grid[0].flatten(),grid[1].flatten()),grid[0].shape) == -1


xtrans = 0
ytrans = y[0]/2
       
ae = [.025,0.05, 0.075]
L = [125,250,500,750,1000,1250,1500,2000]
G = [2,2.5,3,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

utot = np.linspace(15,25,20)
Dir = np.linspace(0,330,12)*np.pi/180

utot,Dir = np.meshgrid(utot,Dir)

Dir = Dir.flatten()
utot = utot.flatten()
ind = np.argsort(Dir)
utot = utot[ind]
Dir = Dir[ind]

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)


xv = (r_v_t*np.cos(phi_v_t))
yv = (r_v_t*np.sin(phi_v_t))
#
xs = (r_s_t*np.cos(phi_s_t))
ys = (r_s_t*np.sin(phi_s_t))

mask_v = np.reshape(tri.get_trifinder()(xv.flatten(),yv.flatten()),xv.shape) == -1
mask_s = np.reshape(tri.get_trifinder()(xs.flatten(),ys.flatten()),xs.shape) == -1

mask = np.reshape(tri.get_trifinder()(grid[0].flatten(),grid[1].flatten()),grid[0].shape) == -1

tri_del_v = Delaunay(np.c_[xv[~mask_v],yv[~mask_v]])
tri_del_s = Delaunay(np.c_[xs[~mask_s],ys[~mask_s]])

vlossrr_int = np.ones(grid[0].shape)*np.nan
vlosvrr_int = np.ones(grid[0].shape)*np.nan

r_tri_s = np.sqrt(grid[0]**2 + grid[1]**2)
phi_tri_s = np.arctan2(grid[1],grid[0])

r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)

del r_tri_s
del phi_tri_s

i = 0
dir_last = Dir[0]


# In[Save weights and others as binary first]

for dir_mean in np.unique(Dir)[2:]:
    print(dir_mean*180/np.pi)
    case = str(int(dir_mean*180/np.pi))
    file_names_s = ['vtxs'+case, 'wtss'+case, 'ws'+case, 'c_refs'+case, 's_refs'+case, 'grid_shapes'+case, 'norms'+case]
    file_names_v = ['vtxv'+case, 'wtsv'+case, 'wv'+case, 'c_refv'+case, 's_refv'+case, 'grid_shapev'+case, 'normv'+case]
    # Rotation 
    gamma = -(2*np.pi-dir_mean) # Rotation
    # Components in matrix of coefficients   
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
    T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
    T = np.dot(np.dot(T1,R),T2)
    Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[1].flatten()))]).T
    Xx = np.dot(T,Xx)
    xx = np.reshape(Xx[0,:],(N_x,N_y))
    yy = np.reshape(Xx[1,:],(N_x,N_y))
    
#    vtxs, wtss, ws, c_refs, s_refs, grid_shapes, norms = early_weights(r_s_g, np.pi-phi_s_g,xx,yy,d/2)
    print('first triangulation')
    res_s = early_weights(r_s_g, np.pi-phi_s_g,xx,yy,d/2)
    for j in range(len(res_s)):
        print(j,len(res_s))
        res_s[j].astype(np.float32).tofile(file_names_s[j])
    del res_s
#    vtxv, wtsv, wv, c_refv, s_refv, grid_shapev, normv = early_weights(r_s_g, np.pi-phi_s_g,xx,yy,-d/2)
    print('second triangulation')
    res_v = early_weights(r_v_g, np.pi-phi_v_g,xx,yy,d/2)
    for j in range(len(res_v)):
        res_v[j].astype(np.float32).tofile(file_names_v[j])
    del res_v


for u_mean, dir_mean in zip(utot, Dir):
    
    print(dir_mean*180/np.pi)
    case = str(int(dir_mean*180/np.pi))
    
    file_names_s = ['vtxs'+case, 'wtss'+case, 'ws'+case, 'c_refs'+case,
                    's_refs'+case, 'grid_shapes'+case, 'norms'+case]
    file_names_v = ['vtxv'+case, 'wtsv'+case, 'wv'+case, 'c_refv'+case,
                    's_refv'+case, 'grid_shapev'+case, 'normv'+case]
    
    if (i==0) | (dir_mean != dir_last):
        i = i+1          
        # Rotation 
        gamma = -(2*np.pi-dir_mean) # Rotation
        # Components in matrix of coefficients   
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
        T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
        R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
        T = np.dot(np.dot(T1,R),T2)
        Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[1].flatten()))]).T
        Xx = np.dot(T,Xx)
        xx = np.reshape(Xx[0,:],(N_x,N_y))
        yy = np.reshape(Xx[1,:],(N_x,N_y))
        vtxs, wtss, ws, c_refs, s_refs, grid_shapes, norms = early_weights(r_s_g, np.pi-phi_s_g,xx,yy,d/2)
        vtxv, wtsv, wv, c_refv, s_refv, grid_shapev, normv = early_weights(r_s_g, np.pi-phi_s_g,xx,yy,-d/2)
    
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten()[0],L.flatten()[0],G.flatten()[0],seed.flatten()[0]):
        
        u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        
        if u_file_name in onlyfiles:
            
            u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
            v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T
            U = u_mean + u
            V = 0 + v
            vel = np.array(np.c_[U.flatten(),V.flatten()]).T
            vel = np.dot(R[:-1,:-1],vel)
            U_rot = np.reshape(vel[0,:],(N_x,N_y))
            V_rot = np.reshape(vel[1,:],(N_x,N_y))
            vlossrr = num_lidar_rot_del(U_rot,V_rot,vtxs, wtss, ws, c_refs, s_refs, grid_shapes, norms)
            vlosvrr = num_lidar_rot_del(U_rot,V_rot,vtxv, wtsv, wv, c_refv, s_refv, grid_shapev, normv)
            #reconstruction
            vlossrr_int[~mask] = sp.interpolate.CloughTocher2DInterpolator(tri_del_s, vlossrr[~mask_s]).__call__(np.c_[grid[0][~mask],grid[1][~mask]])
            vlosvrr_int[~mask] = sp.interpolate.CloughTocher2DInterpolator(tri_del_v, vlosvrr[~mask_s]).__call__(np.c_[grid[0][~mask],grid[1][~mask]])

            vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),vlossrr_int.flatten(),vlossrr_int.flatten())])
            print(1)
            U_early_s = np.reshape(vel_s[:,0],grd[0].shape)
            V_early_s = np.reshape(vel_s[:,1],grd[0].shape)

        



            
u = np.reshape(np.fromfile('simu'+str(1000)+str(3.0)+str(0.05)+str(-8), dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile('simv'+str(1000)+str(3.0)+str(0.05)+str(-8), dtype=np.float32),(N_x,N_y)).T              
utot = np.linspace(15,25,20)
Dir = np.linspace(0,330,12)*np.pi/180    

U_prime = utot[j]+u
V_prime = 0 + v 
      
j = 2        
gamma = -(2*np.pi-Dir[j]) 





S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
#vel = np.array(np.c_[U.flatten(),V.flatten()]).T
#vel = np.dot(T,vel)

vel = np.array(np.c_[U_prime.flatten(),V_prime.flatten()]).T
vel = np.dot(R[:-1,:-1],vel)
Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[1].flatten()))]).T
Xx = np.dot(T,Xx)
U_rot = np.reshape(vel[0,:],(N_x,N_y))
V_rot = np.reshape(vel[1,:],(N_x,N_y))
xx = np.reshape(Xx[0,:],(N_x,N_y))
yy = np.reshape(Xx[1,:],(N_x,N_y))


vloss,_ = num_lidar(r_s_g, np.pi-phi_s_g,U_rot,V_rot,xx,yy,d/2,corr=False)
vlosv,_ = num_lidar(r_v_g, np.pi-phi_v_g,U_rot,V_rot,xx,yy,-d/2,corr=False)

vlossr = num_lidar_rot(r_s_g, np.pi-phi_s_g,U_rot,V_rot,xx,yy,d/2,method='nearest')
vlosvr = num_lidar_rot(r_v_g, np.pi-phi_v_g,U_rot,V_rot,xx,yy,-d/2,method='nearest')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_s_t*np.cos(phi_s_t),r_s_t*np.sin(phi_s_t),vloss,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_v_t*np.cos(phi_v_t),r_v_t*np.sin(phi_v_t),vlosv,cmap='jet')
fig.colorbar(im)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_s_t*np.cos(phi_s_t),r_s_t*np.sin(phi_s_t),vlossr,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_v_t*np.cos(phi_v_t),r_v_t*np.sin(phi_v_t),vlosvr,cmap='jet')
fig.colorbar(im)

#################################

indv = ~np.isnan(vlosv.flatten())
inds = ~np.isnan(vloss.flatten())
xv = (r_v_t*np.cos(phi_v_t)).flatten()[indv]
yv = (r_v_t*np.sin(phi_v_t)).flatten()[indv]
#
xs = (r_s_t*np.cos(phi_s_t)).flatten()[inds]
ys = (r_s_t*np.sin(phi_s_t)).flatten()[inds]
#
#
T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)
#
grd = np.meshgrid(x,y)

mask_s = np.reshape(tri_r.get_trifinder()(grd[0].flatten(),grd[1].flatten()),grd[0].shape) == -1

vlosv_int_sq = sp.interpolate.griddata(np.c_[xv,yv], vlosvr.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='cubic')
vloss_int_sq = sp.interpolate.griddata(np.c_[xs,ys], vlossr.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='cubic')

ind_sq = np.isnan(vlosv_int_sq)
vlosv_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xv,yv], vlosvr.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

ind_sq = np.isnan(vloss_int_sq)
vloss_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xs,ys], vlossr.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

vloss_int_sq = np.reshape(vloss_int_sq,grd[0].shape)
vlosv_int_sq = np.reshape(vlosv_int_sq,grd[0].shape)

vloss_int_sq[mask_s] = 0.0
vlosv_int_sq[mask_s] = 0.0

r_tri_s = np.sqrt(grd[0]**2 + grd[1]**2)
phi_tri_s = np.arctan2(grd[1],grd[0])

r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)

vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),-vlosv_int_sq.flatten(),-vloss_int_sq.flatten())])

U_early_s = np.reshape(vel_s[:,0],grd[0].shape)
V_early_s = np.reshape(vel_s[:,1],grd[0].shape)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

im=ax.contourf(x,y,U_early_s,np.linspace(0,20,50),cmap='jet')
#im=ax.contourf(xx,yy,np.sqrt(U_rot**2+V_rot**2),cmap='jet')
#ax.plot((r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten(),'.',color='b')
#ax.plot((r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten(),'.',color='r')
ax.set_xlim(x[0],x[-1])
ax.set_ylim(y[0],y[-1])

fig.colorbar(im)

mask_s = np.reshape(tri.get_trifinder()(xx.flatten(),yy.flatten()),grd[0].shape) == -1
U_mask = U_rot.copy()
U_mask[mask_s] = 0.0


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

im=ax.contourf(xx,yy,U_mask,np.linspace(0,20,50),cmap='jet')
#im=ax.contourf(xx,yy,np.sqrt(U_rot**2+V_rot**2),cmap='jet')
#ax.plot((r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten(),'.',color='b')
#ax.plot((r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten(),'.',color='r')
ax.set_xlim(x[0],x[-1])
ax.set_ylim(y[0],y[-1])
fig.colorbar(im)

###################################



Xv = np.array(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten(),np.ones(len((r_v_t*np.sin(phi_v_t)).flatten()))]).T
Xv = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
Xv = np.dot(T,Xv)

Xs = np.array(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten(),np.ones(len((r_s_t*np.sin(phi_s_t)).flatten()))]).T
#Xs = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
Xs = np.dot(T,Xs)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

ax.contourf(x,y,U_prime,cmap='jet')
#ax.plot((r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten(),'.',color='b')
#ax.plot((r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten(),'.',color='r')

ax.plot(tri.x,tri.y,'.',color='k')

ax.plot(Xv[0,:],Xv[1,:],'o',color='g')
#ax.plot(Xs[0,:],Xs[1,:],'o',color='k')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(x,y,U_prime,cmap='jet')
ax.plot(tri.x,tri.y,'o',color='k')
for j in range(len(Dir)):
    gamma = -(2*np.pi-Dir[j]) 
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
    T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
    R = np.array([[S11,-S12,0], [S12,S11, 0], [0, 0, 1]])
    T = np.dot(np.dot(T1,R),T2)
    Xv = np.array(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten(),np.ones(len((r_v_t*np.sin(phi_v_t)).flatten()))]).T
    Xs = np.array(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten(),np.ones(len((r_s_t*np.sin(phi_s_t)).flatten()))]).T

    #X = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
    #X = np.dot(T,X)
    Xv = np.dot(T,Xv)
    Xs = np.dot(T,Xs)
    ax.plot(Xv[0,:],Xv[1,:],'.',color='grey',alpha=.5)
    ax.plot(Xs[0,:],Xs[1,:],'.',color='g',alpha=.5)

     

       
grid = np.meshgrid(x,y)

plt.figure()
plt.contourf(grid[0],grid[1],u,cmap='jet')

plt.figure()
plt.contourf(grid[0],grid[1],v,cmap='jet')

dx = np.min(np.diff(x))
dy = np.min(np.diff(y))

n = grid[0].shape[0]
m = grid[1].shape[0]   
# Spectra

fftU = np.fft.fft2(u)
fftV = np.fft.fft2(v)

fftU  = np.fft.fftshift(fftU)
fftV  = np.fft.fftshift(fftV) 

Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m)
#kx = 1/(2*dx)
#ky = 1/(2*dy)   
k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))

Su = sp.integrate.simps(Suu,k2,axis=1)
Sv = sp.integrate.simps(Svv,k2,axis=1)


#fig, ax = plt.subplots()
#im=ax.contourf(k1,k2,np.log10(Suu),np.linspace(0,6,100),cmap='jet')
#ax.set_xlabel('$k_1$', fontsize=18)
#ax.set_ylabel('$k_2$', fontsize=18)
#ax.set_xlim(-.005,0.005)
#ax.set_ylim(-.005,0.005)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax)
#cbar.ax.tick_params(labelsize=18)
#ax.tick_params(labelsize=18)
#cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
#ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')

#plt.figure()
plt.plot(k1,k1*Su)
plt.xscale('log')
plt.yscale('log')
plt.plot(k1,k1*Sv)

u_p = u.copy()
u_p[mask_s] = np.nan

v_p = v.copy()
v_p[mask_s] = np.nan


plt.figure()
plt.contourf(grid[0],grid[1],u,cmap='jet')

plt.figure()
plt.contourf(grid[0],grid[1],v,cmap='jet')


plt.figure()
plt.contourf(grid[0],grid[1],u_p,cmap='jet')

plt.figure()
plt.contourf(grid[0],grid[1],v,cmap='jet')


k1,k2,Su,Sv,Suv = sc.spatial_spec_sq(x,y,u,v,transform = False,shrink = False)
plt.plot(k1,Su,'o')
plt.xscale('log')
#plt.yscale('log')
plt.plot(k1,Sv,'o')

k1,k2,Su,Sv,Suv = sc.spatial_spec_sq(x,y,u_p,v_p,transform = False,shrink = False)
plt.plot(k1,Su,'-')
plt.xscale('log')
#plt.yscale('log')
plt.plot(k1,Sv,'-')

k1,k2,Su,Sv,Suv = sc.spatial_spec_sq(x,y,u_p,v_p,transform = False,shrink = True)
#plt.figure()
plt.plot(k1,Su,'--')
#plt.xscale('log')
#plt.yscale('log')
plt.plot(k1,Sv,'--')


    

# In[]

