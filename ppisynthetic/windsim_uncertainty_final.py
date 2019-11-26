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
import pickle

# In[For figures]

import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Wedge, Circle

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# In[]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim Output dir')
root.destroy()


cwd = os.getcwd()
os.chdir(file_in_path)

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
def wind_sim(ae, L, G, seed, N_x, N_y, file_in_path):  
    cwd = os.getcwd()    
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
    file.write(str(ae)+'\n')
    file.write(str(L)+'\n')
    file.write(str(G)+'\n')
    file.write(str(seed)+'\n')
    file.write('simu'+str(L)+str(G)+str(ae)+str(seed)+'\n')
    file.write('simv'+str(L)+str(G)+str(ae)+str(seed)+'\n')
    file.close()    
    arg = 'windsimu'+' '+input_file
    p=subprocess.run(arg)    
    u = np.reshape(np.fromfile("simu", dtype=np.float32),(N_x,N_y))
    v = np.reshape(np.fromfile("simv", dtype=np.float32),(N_x,N_y))    
    os.chdir(cwd)    
    return (u,v)

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


# In[Rapid reconstruction]

def dir_rec_rapid(V_a,V_b,a,b,shape):
    Sa = np.sin(a)/np.sin(a-b)
    Sb = np.sin(b)/np.sin(a-b)
    Ca = np.cos(a)/np.sin(a-b)
    Cb = np.cos(b)/np.sin(a-b)
    U = (Sb*V_a-Sa*V_b)
    V = (-Cb*V_a+Ca*V_b)
    return (np.reshape(U,shape),np.reshape(V,shape))


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
##################
# In[Noise generation]
# Perlin Noise
def perlin_noise(x,y,scale=30, azim_frac = .3, rad_lim = .1, dr_max = .3, period = 256, tot= False):
    
    n, m = x.shape   
    x = x.flatten()
    y = y.flatten()
    
    GRAD3 = np.array(((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0), 
    	(1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1), 
    	(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
    	(1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1),))
    
    perm = list(range(period))
    perm_right = period - 1
    for i in list(perm):
        j = np.random.randint(0, perm_right)
        perm[i], perm[j] = perm[j], perm[i]
    permutation = np.array(tuple(perm) * 2)
    # Simplex skew constants
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
	# Skew input space to determine which simplex (triangle) we are in
    s = (x + y) * F2
    i = np.floor(x + s)
    j = np.floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t) # "Unskewed" distances from cell origin
    y0 = y - (j - t)   
    ind_loc = x0 > y0   
    i1 = np.ones(x.shape)
    j1 = np.zeros(x.shape)  
    i1[~ind_loc] = 0
    j1[~ind_loc] = 1
    
    i1 = i1.astype(int)
    j1 = i1.astype(int)
		
    x1 = x0 - i1 + G2 # Offsets for middle corner in (x,y) unskewed coords
    y1 = y0 - j1 + G2
    x2 = x0 + G2 * 2.0 - 1.0 # Offsets for last corner in (x,y) unskewed coords
    y2 = y0 + G2 * 2.0 - 1.0

    # Determine hashed gradient indices of the three simplex corners
    perm = permutation
    ii = (i % period).astype(int)
    jj = (j % period).astype(int)
       
    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12

    # Calculate the contribution from the three corners
    noise = np.zeros(x.shape)
    
    tt = 0.5 - x0**2 - y0**2 
    ind_tt = tt > 0
    g = GRAD3[gi0,:]

    noise[ind_tt] = tt[ind_tt]**4 * (g[ind_tt,0] * x0[ind_tt] + g[ind_tt,1] * y0[ind_tt])
    
    tt = 0.5 - x1**2 - y1**2
    ind_tt = tt > 0
    g = GRAD3[gi1,:]
    noise[ind_tt] = noise[ind_tt] + tt[ind_tt]**4 * (g[ind_tt,0] * x1[ind_tt] + g[ind_tt,1] * y1[ind_tt])
    
    tt = 0.5 - x2**2 - y2**2
    ind_tt = tt > 0
    g = GRAD3[gi2,:]
    noise[ind_tt] = noise[ind_tt] + tt[ind_tt]**4 * (g[ind_tt,0] * x2[ind_tt] + g[ind_tt,1] * y2[ind_tt])
    
    if tot=='no':
    # mask definition
        print('something is wrong')
        azim_frac = np.random.uniform(azim_frac,1)
        rad_lim = np.random.uniform(rad_lim,1)
        dr = np.random.uniform(.1,dr_max)
        
        # azimuth positions
        n_pos = int(azim_frac*n)
        pos_azim = np.random.randint(0,n,size = n_pos)
        # radial positions
        # center
        r_mean = int(rad_lim*m)
        r_std = int(dr*m)
        # positions
        pos_rad = np.random.randint(r_mean-r_std,r_mean+r_std,size = n_pos)
        #print(n,m)
        n, m = np.meshgrid(np.arange(m),np.arange(n))
        #print(n.shape)
        ind = np.zeros(n.shape)
        
        for i,nn in enumerate(pos_azim):
            #print(pos_rad[i],r_mean,r_mean-r_std,r_mean+r_std)
            ind_r = n[nn,:]>=pos_rad[i]
            ind[nn,ind_r] = 1
        ind = (ind == 1).flatten()
        noise[~ind] = 0.0
    
        #normalize
        a = np.max(noise)
        c = np.min(noise)
        if (a-c) > 0:
            b = 1
            d = -1   
            m = (b - d) / (a - c)
            noise = (m * (noise - c)) + d
        noise[~ind] = 0.0 
        
    if tot=='yes':
        #normalize
        a = np.max(noise)
        c = np.min(noise)
        if (a-c) > 0:
            b = 1
            d = -1   
            m = (b - d) / (a - c)
            noise = (m * (noise - c)) + d
    
    return noise # scale noise to [-1, 1]

# In[Contaminated scans]

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

ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

utot = np.linspace(15,25,5)
Dir = np.linspace(90,270,5)*np.pi/180

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]
n = 0
for dir_mean in Dir:
    for u_mean in utot:
        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos1_file_name = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            if vlos0_file_name in onlyfiles:
                noise0 = np.reshape(perlin_noise(r_s_g,phi_s_g,period=512,rad_lim=.5,azim_frac=.3),r_s_g.shape)
                noise0 = noise0+np.reshape(perlin_noise(r_s_g,phi_s_g,period=512,rad_lim=.7,azim_frac=.6),r_s_g.shape)
                noise0 = noise0+np.reshape(perlin_noise(r_s_g,phi_s_g,period=512,rad_lim=.9,azim_frac=.9),r_s_g.shape)
                #normalize noise
                a = np.max(noise0)
                c = np.min(noise0)
                if (a-c) > 0:
                    b = 1
                    d = -1   
                    m = (b - d) / (a - c)
                    noise0 = (m * (noise0 - c)) + d
                noise0 = noise0*35.0
                vlos0 = np.reshape(np.fromfile(vlos0_file_name, dtype=np.float32),r_s_g.shape)
                vlos0_noise = vlos0 + noise0
                (vlos0_noise.flatten()).astype(np.float32).tofile('noise0_'+vlos0_file_name)
                n = n+1
                print(n)
            if vlos1_file_name in onlyfiles:
                noise1 = np.reshape(perlin_noise(r_v_g,phi_v_g,period=512,rad_lim=.5,azim_frac=.3),r_v_g.shape)
                noise1 = noise1+np.reshape(perlin_noise(r_v_g,phi_v_g,period=512,rad_lim=.7,azim_frac=.6),r_v_g.shape)
                noise1 = noise1+np.reshape(perlin_noise(r_v_g,phi_v_g,period=512,rad_lim=.9,azim_frac=.9),r_v_g.shape)
                #normalize noise
                a = np.max(noise1)
                c = np.min(noise1)
                if (a-c) > 0:
                    b = 1
                    d = -1   
                    m = (b - d) / (a - c)
                    noise1 = (m * (noise1 - c)) + d
                noise1 = noise1*35.0
                vlos1 = np.reshape(np.fromfile(vlos1_file_name, dtype=np.float32),r_v_g.shape)
                vlos1_noise = vlos1 + noise1
                (vlos1_noise.flatten()).astype(np.float32).tofile('noise1_'+vlos1_file_name)
                
plt.figure()
plt.contourf(r_v_g*np.cos(phi_v_g), r_v_g*np.sin(phi_v_g), vlos1,100, cmap='jet')  
plt.colorbar() 

plt.figure()
plt.contourf(r_v_g*np.cos(phi_v_g), r_v_g*np.sin(phi_v_g), vlos1_noise,100, cmap='jet') 
plt.colorbar()  

plt.figure()
plt.contourf(r_s_g*np.cos(phi_s_g), r_s_g*np.sin(phi_s_g), vlos0,100, cmap='jet')  
plt.colorbar() 

plt.figure()
plt.contourf(r_s_g*np.cos(phi_s_g), r_s_g*np.sin(phi_s_g), vlos0_noise,100, cmap='jet') 
plt.colorbar()  

plt.figure()
noise1 = np.reshape(perlin_noise(r_v_g,phi_v_g,scale=60,period=512,rad_lim=.5,azim_frac=.3),r_v_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_v_g,phi_v_g,scale=60,period=512,rad_lim=.7,azim_frac=.6),r_v_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_v_g,phi_v_g,scale=60,period=512,rad_lim=.9,azim_frac=.99),r_v_g.shape)
#normalize noise
a = np.max(noise1)
c = np.min(noise1)
if (a-c) > 0:
    b = 1
    d = -1   
    m = (b - d) / (a - c)
    noise1 = (m * (noise1 - c)) + d
noise1 = noise1*35.0
vlos1_noise = vlos1 + noise1
plt.contourf(r_v_g*np.cos(phi_v_g), r_v_g*np.sin(phi_v_g), vlos1_noise,100, cmap='jet') 
plt.colorbar()        

# In[]

############################################

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
tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)
ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
N_x = 2048
N_y = 2048
x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)
grid = np.meshgrid(x,y)
xy = np.c_[grid[0].flatten(),grid[1].flatten()]

utot = np.linspace(15,25,5)
Dir = np.linspace(90,270,5)*np.pi/180

r_s = np.linspace(150,7000,198)
r_v = np.linspace(150,7000,198)
phi_s = np.linspace(256,344,45)*np.pi/180
phi_v = np.linspace(196,284,45)*np.pi/180
r_s_g, phi_s_g = np.meshgrid(r_s,phi_s)
r_v_g, phi_v_g = np.meshgrid(r_v,phi_v)

tri_try = Delaunay(np.c_[grid[0].flatten(),grid[1].flatten()], qhull_options = "QJ")

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

r_min=np.min(np.sqrt(tri.x**2+tri.y**2))
d_grid = r_min*2*np.pi/180

n_next = int(2**np.ceil(np.log(L_y/d_grid+1)/np.log(2))) 

x_new = np.linspace(x.min(),x.max(),n_next)
y_new = np.linspace(y.min(),y.max(),n_next)
grid_new = np.meshgrid(x_new,y_new)

r_tri_s = np.sqrt(grid_new[0]**2 + grid_new[1]**2)
phi_tri_s = np.arctan2(grid_new[1],grid_new[0])
r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)

for dir_mean in Dir:
  
    vtx0, wts0, w0, c_ref0, s_ref0, shapes = early_weights2(r_s_g, 
                                                        np.pi-phi_s_g, dir_mean , tri_try, -d/2, y[0]/2)
    vtx1, wts1, w1, c_ref1, s_ref1, shapes = early_weights2(r_v_g,
                                                        np.pi-phi_v_g, dir_mean , tri_try, d/2, y[0]/2)
    
    for u_mean in utot:
        print(dir_mean*180/np.pi,u_mean)
        Urec = []
        Vrec = []

        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            
            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            
            if u_file_name in onlyfiles:
                
                u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
                v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T
                
                U_in = u_mean + u
                V_in = 0 + v
                
                
                vlos0 = num_lidar_rot_del(U_in.flatten(),V_in.flatten(), vtx0,
                                          wts0, w0, c_ref0.flatten(), s_ref0.flatten(), shapes)
                vlos1 = num_lidar_rot_del(U_in.flatten(),V_in.flatten(), vtx1,
                                          wts1, w1, c_ref1.flatten(), s_ref1.flatten(), shapes)
                
                vlos1_int_sq = sp.interpolate.griddata(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()],
                                                             vlos1.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
                vlos0_int_sq = sp.interpolate.griddata(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()],
                                                             vlos0.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
                vlos1_int_sq = np.reshape(vlos1_int_sq,grid_new[0].shape)
                vlos0_int_sq = np.reshape(vlos0_int_sq,grid_new[1].shape)
                
                U,V = dir_rec_rapid(vlos1_int_sq.flatten(),vlos0_int_sq.flatten(),
                                    phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),grid_new[0].shape)
                
                (U.flatten()).astype(np.float32).tofile('U'+str(u_mean)+
                 str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i))
                (V.flatten()).astype(np.float32).tofile('V'+str(u_mean)+
                 str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i))

                
# In[Spectra]
                
resim = False
index = ~(np.isnan(vlos1_int_sq)|np.isnan(vlos0_int_sq))

k_1_r = []
k_2_r = []
Su_u_r = []
Sv_v_r = []
Su_v_r = []

k_1 = []
k_2 = []
Su_u = []
Sv_v = []
Su_v = []

for dir_mean in Dir:
    trical = True
  
    for u_mean in utot:
        print(dir_mean*180/np.pi,u_mean)
        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):                       
            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            U_file_name = 'U'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            V_file_name = 'V'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)            
            if u_file_name in onlyfiles:               
                u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
                v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T
                U_in = u_mean + u
                V_in = 0 + v                
                gamma = (2*np.pi-dir_mean)
                S11 = np.cos(gamma)
                S12 = np.sin(gamma)
                T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
                T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
                R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
                T = np.dot(np.dot(T1,R),T2)
                Xx = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
                Xx = np.dot(T,Xx)
                tri_rot = Delaunay(Xx.T[:,:2], qhull_options = "QJ")               
                mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1                
                mask_rot = np.reshape(mask_rot,grid[0].shape)               
                U_in[mask_rot] = np.nan
                V_in[mask_rot] = np.nan
                
                U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
                V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
                if trical:
                    U, V, mask, mask_int, tri_del = sc.field_rot(x_new, y_new, U, V, tri_calc = True)
                    trical = False
                
                kur,kvr,Sur,Svr,Suvr = sc.spatial_spec_sq(x_new,y_new, U, V, tri_del = tri_del, mask_int = mask_int, tri_calc = False, transform = True)
                
                k_1_r.append(kur)
                k_2_r.append(kvr)
                Su_u_r.append(Sur)
                Sv_v_r.append(Svr)
                Su_v_r.append(Suvr) 
                ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y,U_in,V_in,transform = False, ring=False)
                k_1.append(ku)
                k_2.append(kv)
                Su_u.append(Su)
                Sv_v.append(Sv)
                Su_v.append(Suv)            

utot = 20.0
ae = [0.025, 0.05, 0.075]
L = [125,250,375,500,750]
G = [2.0,2.5,3,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
label = []
label_str = []
for dir_mean in Dir:
  
    u_mean = utot

    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):  
        u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if u_file_name in onlyfiles:             
            label.append([dir_mean,u_mean,ae_i,L_i,G_i,seed_i])
            label_str.append(str(int(dir_mean*180/np.pi))+'Dir-'+str(u_mean)+'Vel-'+str(ae_i)+'ae-'+str(L_i)+'L-'+str(G_i)+'G-'+str(seed_i))
label = np.array(label)
################################################
#with open('S_syn_rot.pkl', 'wb') as V_t:
#     pickle.dump((Su_u,Sv_v,Su_v,k_1,k_2,label),V_t)
#     
#with open('S_syn_rec_rot.pkl', 'wb') as V_t:
#     pickle.dump((Su_u_r,Sv_v_r,Su_v_r,k_1_r,k_2_r),V_t)
   
with open('S_syn_rot_fin.pkl', 'rb') as V_t:
     Su_u,Sv_v,Su_v,k_1,k_2,label = pickle.load(V_t)
     
with open('S_syn_rec_rot_fin.pkl', 'rb') as V_t:
     Su_u_r,Sv_v_r,Su_v_r,k_1_r,k_2_r = pickle.load(V_t)  
     
     
#############################################
with open('S_syn.pkl', 'rb') as V_t:
     Su_u,Sv_v,Su_v,k_1,k_2,label = pickle.load(V_t)
     
with open('S_syn_rec.pkl', 'rb') as V_t:
     Su_u_r,Sv_v_r,Su_v_r,k_1_r,k_2_r = pickle.load(V_t)
######################################
ind_lab = (label[:,0]*180/np.pi==225) & (label[:,1]==20) & (label[:,2]==.05) & (label[:,3]==500) & (label[:,4]==2.5) & (label[:,5]==-1)

u_file_name = 'simu'+str(500)+str(2.5)+str(.05)+str(-1)
v_file_name = 'simv'+str(500)+str(2.5)+str(.05)+str(-1)

U_file_name = 'U'+str(20.0)+str(int(225))+str(500)+str(2.5)+str(.05)+str(-1)
V_file_name = 'V'+str(20.0)+str(int(225))+str(500)+str(2.5)+str(.05)+str(-1) 

U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)

u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T
U_in = 20 + u
V_in = 0 + v 

#################
xtrans = 0
ytrans = y[0]/2
gamma = np.arctan2(np.nanmean(V.flatten()),np.nanmean(U.flatten()))
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
Xx = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
Xx = np.dot(T,Xx)
tri_rot = Delaunay(Xx.T[:,:2], qhull_options = "QJ")     

plt.figure()
plt.contourf(grid[0],grid[1],U_in,np.linspace(14.8,25,10),cmap = 'jet') 
plt.colorbar()
plt.triplot(tri_rot.points[:,0],tri_rot.points[:,1],tri_rot.simplices)

############################################In paper
# Rotation of lidars
f = 16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grid[0],grid[1],U_in,50,cmap='Greys')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$U\:[m/s]$", fontsize=f)
ax.set_ylabel(r'$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel(r'$y\:[m]$', fontsize=f, weight='bold')
#circle = plt.Circle((xtrans,ytrans),ytrans,linestyle=':',linewidth=2.5,color='k',fill=False)
#ax.add_artist(circle)
#wedge = Wedge((d[0]/2, 0), 7000, 196, 284, width=6850, color ='grey',alpha=.25)#, linestyle='--',fill=False,linewidth=1)
#ax.add_artist(wedge)
wedge = Wedge((-d[0]/2, 0), 7000, 256, 344, width=6850, color ='blue',alpha=.25)#, linestyle='--',fill=False,linewidth=1)
ax.add_artist(wedge)
#ax.plot([-d[0]/2,0,d[0]/2],[0,0,0], ':o', color = 'k')

direction = 135*np.pi/180
gamma = (2*np.pi-direction)
x_trans2 = -(ytrans)*np.sin(gamma)
y_trans2 = (ytrans)*(1-np.cos(gamma))
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,x_trans2], [0,1, y_trans2], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
Xx = np.array(np.c_[np.array([-d[0]/2,0,d[0]/2]), np.array([0,0,0]),np.ones(len(np.array([0,0,0])))]).T
Xx = np.dot(T1,np.dot(R,Xx))

#ax.plot(Xx[0,:],Xx[1,:], ':o', color = 'k')

#wedge = Wedge((Xx[0,2], Xx[1,2]), 7000, 196+135, 284+135, width=6850, color ='grey', alpha = 0.25)
#ax.add_artist(wedge)
wedge = Wedge((Xx[0,0], Xx[1,0]), 7000, 256+135, 344+135, width=6850, color ='red', alpha = 0.25)
ax.add_artist(wedge)
fig.tight_layout()

#####################Numerical lidar, figure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1300, 1000)
ax.set_ylim(-5500,-3500)
ax.set_zlim(0, 7)

plt.show()
r = 5000
r0 = r
phi0 = -30
for i in range(3):
    phi_0 = phi0   
    for j in range(4):        
        wedge = Wedge((-4000,-3000), r, phi_0, phi_0+5, width=600, fill = False, edgecolor = 'k', linewidth = 1,alpha=.3)
        ax.add_patch(wedge)
        art3d.pathpatch_2d_to_3d(wedge, z=0, zdir="z")
        phi_0+=5
    r = r-600

wedge = Wedge((-4000,-3000), r0, phi0, phi_0, width= r0-r, facecolor ='lightgrey', edgecolor = 'k', linewidth = 1, zorder=10)    

x_int = np.arange(-2000, 2000, 100)
y_int = np.arange(-6000, -3000, 100)
g = np.meshgrid(x_int, y_int)
coords = list(zip(*(c.flat for c in g)))

lim_x = np.array([np.cos(-20*np.pi/180)*3800,np.cos(-20*np.pi/180)*4400])-4000
lim_y = np.array([np.sin(-20*np.pi/180)*3800,np.sin(-20*np.pi/180)*4400])-3000

line_x = np.linspace(lim_x[0],lim_x[1],51)
line_y = np.linspace(lim_y[0],lim_y[1],51) 

h = 2*(line_x-line_x[int(len(line_x)/2)])/(lim_x[1]-lim_x[0])
w = .75*(1-h**2)
w = 5*w

points = np.vstack([p for p in coords if wedge.contains_point(p, radius=0)])

ax.scatter(points[:,0],points[:,1],np.zeros(points[:,0].shape),'.',c='k',s=3)

ax.grid(False)
ax.set_frame_on(False)

ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# Get rid of the spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# Get rid of the panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000
    a_s = Arrow3D(a_x_s, a_y_s, [0, 0], mutation_scale=20, lw=2, arrowstyle="-|>", linestyle = '--', color="grey",alpha=.5)
    ax.add_artist(a_s)

ax.plot(line_x,line_y,w,c='r',lw=2)

h = 0 
v = []
for k in range(0, len(line_x) - 1):
    x = [line_x[k], line_x[k+1], line_x[k+1], line_x[k]]
    y = [line_y[k], line_y[k+1], line_y[k+1], line_y[k]]
    z = [w[k], w[k+1], h, h]
    v.append(list(zip(x, y, z)))
poly3dCollection = Poly3DCollection(v,facecolor='red',alpha=.1)
ax.add_collection3d(poly3dCollection)

a_x = np.array([np.cos(-20*np.pi/180)*2600,np.cos(-20*np.pi/180)*5600])-4000
a_y = np.array([np.sin(-20*np.pi/180)*2600,np.sin(-20*np.pi/180)*5600])-3000
a = Arrow3D(a_x, a_y, [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
ax.add_artist(a)

rangesx = np.array([np.cos(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-4000
rangesy = np.array([np.sin(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-3000
ax.scatter(rangesx,rangesy,np.zeros(len(rangesx)),'.',c='b',s=30)
fig.tight_layout()
ax.text2D(0.63, 0.8, '$w\:=\:0.75(1-h^2)$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.72, 0.4, '$Laser\:beam$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.2, 0.2, '$Range\:gate$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.22, 0.75, '$Beams\:to\:be\:averaged$', transform=ax.transAxes,fontsize=30)
#ax.text2D(0.15, 0.85, '(a)', transform=ax.transAxes,fontsize=24)
a_w = Arrow3D([rangesx[6],rangesx[4]], [rangesy[6],rangesy[4]], [8, w[25]],
              mutation_scale=20, lw=3, arrowstyle="wedge", color="k")
ax.add_artist(a_w)

a_b = Arrow3D([rangesx[8]+150,rangesx[7]], [rangesy[8]+150,rangesy[7]], [3, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_b)

a_r = Arrow3D([-500,rangesx[4]], [-5200,rangesy[4]], [0, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_r)

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000

    a_f = Arrow3D([a_x[0],a_x_s[0]], [a_y[0],a_y_s[0]],[4, 0],
                  mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
    ax.add_artist(a_f)

##################################
         
mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1                
mask_rot = np.reshape(mask_rot,grid[0].shape)
U_in2 = U_in.copy()
V_in2 = V_in.copy()               
U_in2[mask_rot] = np.nan
V_in2[mask_rot] = np.nan


plt.figure()
plt.contourf(grid[0],grid[1],U_in2,np.linspace(14.8,25,10),cmap = 'jet') 
plt.colorbar()

##############

plt.figure()
plt.contourf(grid[0],grid[1],U_in,np.linspace(14.8,25,10),cmap = 'jet') 
plt.colorbar()

plt.figure()
plt.contourf(grid_new[0],grid_new[1],np.sqrt(U**2+V**2),30, cmap='jet') 
plt.colorbar()

plt.figure()
plt.contourf(grid_new[0],grid_new[1],U,30, cmap='jet') 
plt.colorbar()

plt.figure()
plt.contourf(grid_new[0],grid_new[1],V,30, cmap='jet') 
plt.colorbar()

xtrans = 0
ytrans = y[0]/2

gamma = 2*np.pi-np.arctan2(np.nanmean(V.flatten()),np.nanmean(U.flatten()))#(2*np.pi-225*np.pi/180)
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)

gamma_mask = -225*np.pi/180
S11 = np.cos(gamma_mask)
S12 = np.sin(gamma_mask)
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
R_mask = np.array([[S11,-S12,0], [S12,S11, 0], [0, 0, 1]])
T_mask = np.dot(np.dot(T1,R_mask),T2)
Xx_mask = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
Xx_mask = np.dot(T_mask,Xx_mask)


Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
Xx = np.dot(T,Xx)

xx = np.reshape(Xx[0,:],(N_x,N_y))
yy = np.reshape(Xx[1,:],(N_x,N_y))

vel = np.array(np.c_[U_in.flatten(),V_in.flatten()]).T
vel = np.dot(R[:-1,:-1],vel)
U_rot = np.reshape(vel[0,:],(N_x,N_y)).T
V_rot = np.reshape(vel[1,:],(N_x,N_y)).T

tri_rot = Delaunay(Xx_mask.T[:,:2], qhull_options = "QJ")
#tri_rot = Delaunay(np.c_[tri.x,tri.y], qhull_options = "QJ")
mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1
mask_rot = np.reshape(mask_rot,grid[0].shape)

U_rot[mask_rot] = np.nan
V_rot[mask_rot] = np.nan

plt.figure()
plt.contourf(xx,yy,U_rot,30, cmap='jet') 
plt.colorbar()

plt.figure()
plt.contourf(xx,yy,np.flipud(V_rot),30, cmap='jet') 
plt.colorbar()


plt.figure()
plt.contourf(xx,yy,np.sqrt(U_rot**2+V_rot**2),np.linspace(14.8,25,10), cmap='jet') 
plt.colorbar()


U_in_mask = U_in.copy()
V_in_mask = V_in.copy()

U_in_mask[mask_rot] = np.nan
V_in_mask[mask_rot] = np.nan

plt.figure()
plt.contourf(x,y,U_in_mask,30, cmap='jet') 
plt.colorbar()

plt.figure()
plt.contourf(x,y,V_in_mask,30, cmap='jet') 
plt.colorbar()

U_rot[mask_rot] = np.nan
V_rot[mask_rot] = np.nan


i=300
plt.figure()
plt.plot(k_1[i],k_1[i]*Su_u[i],'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
plt.plot(k_1_r[i],k_1_r[i]*Sv_v_r[i],'.-', label = 'Reconstructed')
plt.legend()

plt.figure()
plt.plot(k_1[i],k_1[i]*Sv_v[i],'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
plt.plot(k_1_r[i],k_1_r[i]*Su_u_r[i],'.-', label = 'Reconstructed')
plt.legend()


indku=ku<10**-2
indkur=kur<10**-2
plt.figure()
plt.plot(k_1[i],k_1[i]*Sv_v[i],'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
plt.plot(k_1_r[i],k_1_r[i]*Su_u_r[i],'.-', label = 'Reconstructed')
plt.legend()

#####################################################
ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y, U_in, V_in, transform = False, ring=False)

ku2,kv2,Su2,Sv2,Suv2 = sc.spatial_spec_sq(x,y, U_in2, V_in2, transform = False, ring=False)

kur,kvr,Sur,Svr,Suvr = sc.spatial_spec_sq(x_new, y_new, U, V, transform = True, ring=False)

plt.figure()
plt.plot(ku2,ku2*Su2,'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
plt.plot(kur,kur*Sur,'.-', label = 'Reconstructed')
plt.legend()

plt.figure()
plt.plot(ku2,ku2*Sv2,'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{vv}(k_1)$',fontsize=18)
plt.plot(kur,kur*Svr,'.-', label = 'Reconstructed')
plt.legend()

indku=ku2<10**-2
indkur=kur<10**-2

plt.figure()
plt.plot(ku2[indku],Sur[indkur]/Su2[indku] ,'.-', label='U')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{vv}(k_1)$',fontsize=18)
plt.plot(kur[indkur],Svr[indkur]/Sv2[indku],'.-', label = 'V')
plt.legend()



#####################################################

indku=k_1[0]<10**-2
indkur=k_1_r[0]<10**-2

k1r = np.array(k_1_r)[:,indkur]
k1  = np.array(k_1)[:,indku]

Suu = np.array(Su_u)[:,indku]
Suur = np.array(Su_u_r)[:,indkur]

Svv = np.array(Sv_v)[:,indku]
Svvr = np.array(Sv_v_r)[:,indkur]

Sh = .5*(Suu + Svv)
Shr = .5*(Suur + Svvr)
#######
#Direction dependance

#############
#U
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[0])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{uu,reconst.}/S_{uu}, 90^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[1])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{uu,reconst.}/S_{uu}, 135^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)


fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{uu,reconst.}/S_{uu}, 180^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[1])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{uu,reconst.}/S_{uu}, 135^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)

#####################
#V

fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[0])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{vv,reconst.}/S_{vv}, 90^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[1])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{vv,reconst.}/S_{vv}, 135^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)

fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{vv,reconst.}/S_{vv}, 180^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[1])# & (label[:,4] == 3.5) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{vv,reconst.}/S_{vv}, 135^{\circ}$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)

##########################################################
#Length scale
#[dir_mean,u_mean,ae_i,L_i,G_i,seed_i]
#############
#U
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[0]) & (label[:,3] == 125) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{uu,reconst.}/S_{uu}, L = 125$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[0]) & (label[:,3] == 750) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{uu,reconst.}/S_{uu}, L = 750$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$90^{\circ}$',fontsize=22)


#####################
#V

fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2]) & (label[:,3] == 125) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{vv,reconst.}/S_{vv}, L = 125$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[2]) & (label[:,3] == 750) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{vv,reconst.}/S_{vv}, L = 750$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$180^{\circ}$',fontsize=22)

##############################
# Gamma
# [dir_mean,u_mean,ae_i,L_i,G_i,seed_i]
#############
#U
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2]) & (label[:,4] == 2) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{uu,reconst.}/S_{uu}, \Gamma = 2$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[2]) & (label[:,4] == 3.5) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{uu,reconst.}/S_{uu}, \Gamma = 3.5$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$180^{\circ}$',fontsize=22)

#####################
#V
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2]) & (label[:,4] == 2) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{vv,reconst.}/S_{vv}, \Gamma = 2$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[2]) & (label[:,4] == 3.5) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{vv,reconst.}/S_{vv}, \Gamma = 3.5$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$180^{\circ}$',fontsize=22)

##############################
# alpha-epsilon
# [dir_mean,u_mean,ae_i,L_i,G_i,seed_i]
#############
#U
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[2]) & (label[:,2] == .025) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{uu,reconst.}/S_{uu}, ae^{2/3} = .025$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[2]) & (label[:,2] == .075) 
H_u_mean = np.mean((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
H_u_std = np.std((Suur[ind_label,:]/Suu[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{uu,reconst.}/S_{uu}, ae^{2/3} = .075$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$180^{\circ}$',fontsize=22)

#####################
#V
fig, ax = plt.subplots()
ind_label = (label[:,0] == Dir[0]) & (label[:,2] == .025) 
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'k',label='$S_{vv,reconst.}/S_{vv}, ae^{2/3} = .025$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="grey",alpha=0.5, edgecolor="")

ind_label = (label[:,0] == Dir[0]) & (label[:,2] == .075)  
H_u_mean = np.mean((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
H_u_std = np.std((Svvr[ind_label,:]/Svv[ind_label,:]),axis=0)
ax.plot(np.mean((k1r[ind_label,:]),axis=0),H_u_mean,lw = 2.5,color = 'r',label='$S_{vv,reconst.}/S_{vv}, ae^{2/3} = .075$')
ax.fill_between(np.mean((k1r[ind_label,:]),axis=0), H_u_mean+H_u_std, H_u_mean-H_u_std, color="r",alpha=0.5, edgecolor="")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$k_1 [m^{-1}]$',fontsize=20,fontweight = 'bold')
ax.legend(fontsize=22)
ax.set_ylim(.1,1.1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('$90^{\circ}$',fontsize=22)

##############################

fig, ax = plt.subplots()
indku=ku<10**-2
indkur=kur<10**-2

for i in range(len(Su_u_r)):
    
    ax.cla()
#    ax.plot(k_1[i],Su_u[i],'.-', label='Original')
#    ax.plot(k_1_r[i],k_1_r[i]*Su_u_r[i],'.-', label = 'Reconstructed')
    ax.plot(k_1[i][indku],Sv_v_r[i][indkur]/Su_u[i][indku],'.-', label='$S_{uu}$')
    ax.plot(k_1[i][indku],Su_u_r[i][indkur]/Sv_v[i][indku],'.-', label = '$S_{vv}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k_1$',fontsize=18)
    ax.set_ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
    ax.set_title(label_str[i])

    ax.legend()
    plt.pause(1)
            
                
ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y,-V_in_mask,U_in_mask,transform = False, ring=False)
kur,kvr,Sur,Svr,Suvr = sc.spatial_spec_sq(x_new,y_new,U,V,transform = True, ring=False)

plt.figure()
plt.plot(ku,ku*Su,'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{uu}(k_1)$',fontsize=18)
plt.plot(kur,kur*Sur,'.-', label = 'Reconstructed')
plt.legend()

plt.figure()
plt.plot(ku,ku*Sv,'.-', label='Original')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$k_1*S_{vv}(k_1)$',fontsize=18)
plt.plot(kur,kur*Svr,'.-', label = 'Reconstructed')
plt.legend()

indku=ku<10**-2
indkur=kur<10**-2
plt.figure()
plt.plot(ku[indku],Sur[indkur]/Su[indku],'.-', label='$S_{uu}$')
plt.plot(ku[indku],Svr[indkur]/Sv[indku],'.-', label = '$S_{vv}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k_1$',fontsize=18)
plt.ylabel('$H$',fontsize=18)
plt.legend()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(y,x,-V_in_mask.T,cmap='jet')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(x_new,y_new,U,cmap='jet')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(y,x,U_in_mask.T,cmap='jet')

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(x_new,y_new,V,cmap='jet')
                


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

# In[Synthetic wind field generation]

ae = [.025,0.05, 0.075]
L = [62,125,250,375,500,750]#,1000,1250,1500,2000]
G = [0,1.0]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

N_x = 2048
N_y = 2048
i = 0
onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
    print(i,str(L_i)+str(G_i)+str(ae_i)+str(seed_i))
    #Sepctral tensor parameters
    os.chdir(file_in_path)  
    
    file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
    if file_name in onlyfiles:
        print('skiping')
    else:   
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

# In[]
    
#def num_lidar_rot(r,phi,U_in,V_in,x,y,d,rg=35,deg=np.pi/180,n=31,m=31,kernel='epanechnikov',method='cubic'):
#    # Translate (x,y) field to lidar origin and transform to polar coordinates
#    # Translate
#        
#    x_prime = x.flatten()-d[0]
#    y_prime = y.flatten()-d[1]  
#    U = U_in.flatten()
#    V = V_in.flatten()
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
#    U = U[index]
#    V = V[index]
#
#    r_unique = np.unique(r)
#    phi_unique = np.unique(phi)
#    
#    delta_r = np.min(np.diff(r_unique))
#    delta_phi = np.min(np.diff(phi_unique))
#    
#    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+delta_r/2,len(r_unique)*(n-1)+1)
#    h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r    
#    
#    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+delta_phi/2, len(phi_unique)*(m-1)+1)    
#    r_grid, phi_grid = np.meshgrid(r_refine,phi_refine)
#    
#    # Epanechnikov
#    w = .75*(1-h**2) 
#    w = np.reshape(np.repeat(w,phi_grid.shape[0]),(phi_grid.T).shape).T
#    print(1)
#    U_int = sp.interpolate.griddata(np.c_[r_prime, phi_prime], U,
#            (r_grid.flatten(), phi_grid.flatten()), method=method,fill_value = np.nan,rescale=True)  
#    print(2)
#    V_int = sp.interpolate.griddata(np.c_[r_prime, phi_prime], V,
#            (r_grid.flatten(), phi_grid.flatten()), method=method,fill_value = np.nan,rescale=True)
#    print(3)
#    V_L = np.cos(phi_grid.flatten())*U_int+np.sin(phi_grid.flatten())*V_int
#    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
#    V_L = np.reshape(V_L,r_grid.shape)
#    print(V_L.shape)
#    
#    norm = np.sum(w[0,:(n-1)])
#
#    V_L = (V_L[:,:-1]*w[:,:-1]/norm).T
#
#    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
#    
#    V_L = np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1).T
#    
#    print(V_L.shape,np.nanmax(V_L),np.nanmin(V_L))
#    
#    w_p = np.ones(V_L.shape)/(m-1)   
#    V_L = -(V_L[:-1,:]*w_p[:-1,:]) 
#    
#    print(V_L.shape)
#     
#    return np.flip(np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1),axis=0)

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

# In[Figures function]

xtrans = 0
ytrans = y[0]/2
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
vel = np.array(np.c_[U_in.flatten(),V_in.flatten()]).T
vel = np.dot(R[:-1,:-1],vel)
U_rot = np.reshape(vel[0,:],(N_x,N_y))
V_rot = np.reshape(vel[1,:],(N_x,N_y))

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(xx,yy,U_rot,100,cmap='jet')
ax.set_xlim(tri.x.min(),tri.x.max())
ax.set_ylim(tri.y.min(),tri.y.max())
                
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(x_new,y_new,U,100,cmap='jet')
ax.set_xlim(tri.x.min(),tri.x.max())
ax.set_ylim(tri.y.min(),tri.y.max())

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(xx,yy,V_rot,100,cmap='jet')
ax.set_xlim(tri.x.min(),tri.x.max())
ax.set_ylim(tri.y.min(),tri.y.max())
                
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.contourf(x_new,y_new,V,100,cmap='jet')
ax.set_xlim(tri.x.min(),tri.x.max())
ax.set_ylim(tri.y.min(),tri.y.max())

#for dir_mean in Dir:
#  
#    for u_mean in utot:
#        print(dir_mean*180/np.pi,u_mean)
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):                       
#            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            U_file_name = 'U'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            V_file_name = 'V'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)            
#            if u_file_name in onlyfiles:               
#                u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
#                v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T
#                U_in = u_mean + u
#                V_in = 0 + v                
#                gamma = (2*np.pi-dir_mean)
#                S11 = np.cos(gamma)
#                S12 = np.sin(gamma)
#                T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
#                T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
#                R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
#                T = np.dot(np.dot(T1,R),T2)
#                Xx = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
#                Xx = np.dot(T,Xx)
#                tri_rot = Delaunay(Xx.T[:,:2], qhull_options = "QJ")               
#                mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1                
#                mask_rot = np.reshape(mask_rot,grid[0].shape)               
#                U_in[mask_rot] = np.nan
#                V_in[mask_rot] = np.nan
#                U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
#                V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
#                kur,kvr,Sur,Svr,Suvr = sc.spatial_spec_sq(x_new,y_new,np.fliplr(U),np.fliplr(V),transform = True, ring=False)
#                k_1_r.append(kur)
#                k_2_r.append(kvr)
#                Su_u_r.append(Sur)
#                Sv_v_r.append(Svr)
#                Su_v_r.append(Suvr) 
#                ku,kv,Su,Sv,Suv = sc.spatial_spec_sq(x,y,U_in,V_in,transform = False, ring=False)
#                k_1.append(ku)
#                k_2.append(kv)
#                Su_u.append(Su)
#                Sv_v.append(Sv)
#                Su_v.append(Suv)  

# In[]

#def num_lidar(r,phi,U,V,x,y,d,rg=35,deg=np.pi/180,n=10,m=10,kernel='epanechnikov',corr=True):
#    # Translate (x,y) field to lidar origin and transform to polar coordinates
#    # Translate
#    
#    phi_loc, r_loc = np.meshgrid(np.linspace(-deg,deg,m),np.linspace(-rg/2,rg/2,n))
#    
#    x_prime = x-d[0]
#    y_prime = y-d[1]  
#    if (len(x.shape)==1) & (len(y.shape)==1):
#        x_prime, y_prime = np.meshgrid(x_prime,y_prime)
#    if len(U.shape)>1 :
#        U = U.flatten()
#        V = V.flatten()
#    # Transform to polar
#    
#    phi_prime = np.arctan2(y_prime.flatten(),x_prime.flatten()) 
#    r_prime = np.sqrt(x_prime.flatten()**2+y_prime.flatten()**2)
#      
#    tree_lid = KDTree(np.c_[r.flatten(),phi.flatten()],metric='manhattan')
#    
#    ind = tree_lid.query(np.c_[r_prime,phi_prime],return_distance=False)
#    
#    V_LOS = np.zeros(len(r.flatten()))
#    
#    index_phi = (phi_prime>np.min(phi.flatten())-deg) & (phi_prime<np.max(phi.flatten())+deg)
#    index_r = (r_prime>np.min(r.flatten())-35/2) & (r_prime<np.max(r.flatten())+35/2)
#    n_index=[]
#    
#    for i in range(len(r.flatten())):        
#        index = ind.flatten() == i
#        index = ((index) & (index_phi)) & (index_r)
#        V_L = np.cos(phi_prime[index])*U[index]+np.sin(phi_prime[index])*V[index]
#        if kernel == 'gaussian':
#            w = lambda h: sp.stats.norm.pdf(h, loc=0, scale=1)
#        elif kernel == 'epanechnikov':
#            w = lambda h: .75*(1-(h)**2)
#        elif kernel == 'triangle':
#            w = lambda h: 1-np.abs(h)
#        else:
#            w = lambda h: .75*(1-(h)**2)
#        n_index.append(np.sum(index))
#        
#        # interpolation on beams
#        
#        if n_index[i]>10:    
#            
#            dr = -r_prime[index]+r.flatten()[i]
#            dphi = -phi_prime[index]+phi.flatten()[i]
#            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest',fill_value = 0.0)
#            if corr:
#                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
#                V_L_TI = np.repeat([V_L_TI],n,axis=0)
#                rg_corr = rg/(1+V_L_TI)
#            else:
#                rg_corr= rg            
#            h = 2*r_loc/rg_corr
#            wg = w(h)
#            V_LOS[i] = np.nansum(np.nansum(V_L_g*wg,axis=0)/np.nansum(wg,axis=0))/m
#            print(i,n_index[i],V_LOS[i],np.sum(np.isnan(V_L_g)),np.sum(np.isnan(wg)))
#            
#        if (n_index[i]<=10) & (n_index[i]>0):    
#            dr = -r_prime[index]+r.flatten()[i]
#            dphi = -phi_prime[index]+phi.flatten()[i]
#            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest')
#            if corr:
#                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
#                V_L_TI = np.repeat([V_L_TI],n,axis=0)
#                rg_corr = rg/(1+V_L_TI)
#            else:
#                rg_corr= rg
#            h = 2*r_loc/rg_corr
#            wg = w(h)
#            #simple average over azimuth
#            V_LOS[i] = np.sum(np.sum(V_L_g*wg,axis=0)/np.sum(wg,axis=0))/m
#        if n_index[i]==0:
#            V_LOS[i] = np.nan
#        
#    return (np.reshape(-V_LOS,r.shape),np.array(n_index))
