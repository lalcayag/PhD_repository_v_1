# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:59:55 2020
Test of wind reconstruction in the different components
@author: lalc
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

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[]
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# In[Some functions]
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

def filterfft(vorti, mask, sigma=20):
    vort = vorti.copy()
    vort[np.isnan(vort)] = np.nanmean(vort)   
    input_ = np.fft.fft2(vort)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.real(np.fft.ifft2(result))
    result[mask] = np.nan
    return result

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

def expand_field(grd,U_list, V_list, tri, dx, dy, dt):  
    dr = np.sqrt(dx**2+dy**2)
    t = np.flip(np.arange(len(U_list))*dt)
    x = np.array([])
    y = np.array([])
    time = np.array([])
    U = np.array([])
    V = np.array([])
    for i in range(len(U_list)-1):
        mask = np.isnan(U_list[i])
        x_new = grd[0][~mask]+U_list[i][~mask]*t[i]
        y_new = grd[1][~mask]+V_list[i][~mask]*t[i]
        out_grid = tri.find_simplex(np.c_[x_new, y_new])
        x = np.r_[x,x_new[out_grid==-1]]
        y = np.r_[y,y_new[out_grid==-1]]
        U = np.r_[U,U_list[i][~mask][out_grid==-1]]
        V = np.r_[V,V_list[i][~mask][out_grid==-1]]
        time = np.r_[time, np.ones(np.sum(~mask))*t[i]]
        
#        x.append(x_new[out_grid==-1])
#        y.append(y_new[out_grid==-1])
#        U.append(U_list[i][~mask][out_grid==-1])
#        V.append(V_list[i][~mask][out_grid==-1])
#        time.append(np.ones(np.sum(out_grid==-1))*t[i])
    mask = np.isnan(U_list[-1])    
    x = np.r_[x,grd[0][~mask]]
    y = np.r_[y,grd[1][~mask]]
    U = np.r_[U, U_list[-1][~mask]]
    V = np.r_[V, V_list[-1][~mask]]
    time = np.r_[time, np.ones(np.sum(~mask))*t[i]]
    print('tree')
#    x = np.array(x).ravel()
#    y = np.array(y).ravel()
#    time = np.array(time).ravel()
#    U = np.array(U).ravel()
#    V = np.array(V).ravel()
    pos = np.c_[x, y]
    print(x,y)
    print(x.shape,y.shape)
    tree = KDTree(pos)
    dist, ind = tree.query(tree.data, k=2, return_distance=True)
    ind_d = dist[:,1]<dr/2   
    ind_t = time[ind[:,1]]<time[ind[:,0]]
    x = x[ind[~(ind_d & ind_t),0]]
    y = y[ind[~(ind_d & ind_t),0]]
    U = U[ind[~(ind_d & ind_t),0]]
    V = V[ind[~(ind_d & ind_t),0]]
    time = time[ind[~(ind_d & ind_t),0]]
    print('triangle')
    tri_exp = Delaunay(np.c_[x,y])
    #grid
    xg = np.arange(np.nanmin(x), np.nanmax(x)+dx, dx)
    yg = np.arange(np.nanmin(y), np.nanmax(y)+dy, dy)
    xg, yg = np.meshgrid(xg, yg)
    print('interpolation')
    Uex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,U)(np.c_[xg.flatten(),yg.flatten()])
    Vex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,V)(np.c_[xg.flatten(),yg.flatten()])    
    return (xg, yg, Uex, Vex)

def areatriangles(tri, delaunay = True):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        tri: ntri, dim+1 indices of triangles or simplexes, as from
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        area: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    #ind = ~np.isnan(z)
    if delaunay:
        xy = tri.points
        triangles = tri.simplices
    else:
        xy = np.c_[tri.x,tri.y]
        triangles = tri.triangles
        
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    #assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    #assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    area = np.zeros( ntri )
    #areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for i, tri in enumerate(triangles):
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area[i] = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area[i] = abs( np.linalg.det( t )) / dimfac  # v slow
        #aux = area * np.nanmean(z[tri],axis=0)
        #if ~np.isnan(aux):
        #    zsum += aux
        #    areasum += area
    return area

# In[]
def expand_field_point(grd,U_list, V_list, point , dt, r, L = 4, tree=True, treetype = 'Kn', grid_i = []):
    dx = grd[0][0,1]-grd[0][0,0]
    dy = grd[1][1,0]-grd[1][0,0]
    dr = np.sqrt(dx**2+dy**2)
    # Time steps
    t = np.flip(np.arange(len(U_list))*dt)     
    # Local mean wind speed  parameters
    ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
    s = r/ds  
    x = np.array([])
    y = np.array([])
    xfin = np.array([])
    yfin = np.array([])
    time = np.array([])
    U = np.array([])
    V = np.array([])      
    if tree:
    ############# Interpolation with trees (linear)
        if treetype == 'Kn':
            neigh = KNeighborsRegressor(n_neighbors =  8, weights='distance',algorithm='auto', leaf_size=80,n_jobs=1)
        if treetype == 'Extra': 
            neigh = ExtraTreesRegressor(n_estimators=5, random_state=0)
        if treetype == 'Decision': 
            neigh = DecisionTreeRegressor(max_features='auto', random_state=0, max_depth = 5000)
        if treetype == 'Bayesian':
             neigh = BayesianRidge(n_iter = 300, lambda_init=1e7, alpha_init=1e8, compute_score = True)#, lambda_init=100, alpha_init=100)
#             neigh.set_params(alpha_init=1, lambda_init=1e-3)                                
        Umeans = [filterfft(u,np.isnan(u),sigma=L*s) for u in U_list]
        Vmeans = [filterfft(u,np.isnan(u),sigma=L*s) for u in V_list]
        
        Ad_x = np.array([u*t_i+grd[0] for u, t_i in zip (Umeans, t)]).flatten()
        Ad_y = np.array([u*t_i+grd[1] for u, t_i in zip (Vmeans, t)]).flatten()
        U_inst = np.array(U_list).flatten()
        V_inst = np.array(V_list).flatten()
        weights = np.array([1-np.ones(grd[0].shape)*t_i/np.max(t) for t_i in t]).flatten()
        weights[weights==0] = np.min(weights[weights>0])
        weights = weights/np.sum(weights)
        if len(grid_i)==0:
            xg = np.arange(np.nanmin(Ad_x), np.nanmax(Ad_x)+dx, dx)
            yg = np.arange(np.nanmin(Ad_y), np.nanmax(Ad_y)+dy, dy)
            xg, yg = np.meshgrid(xg, yg)
        else:
            xg, yg = grid_i[0], grid_i[1]
        ind = ~np.isnan(U_inst)
        print('fit U', )
        if treetype == 'Kn':
            scaler = RobustScaler(quantile_range=(25, 75)).fit(np.c_[Ad_x[ind], Ad_y[ind], U_inst[ind]])
            X = scaler.transform(np.c_[Ad_x[ind], Ad_y[ind], U_inst[ind]])
            neigh.fit(X[:,:2], X[:,2])
        else:
            scaler = RobustScaler(quantile_range=(25, 75)).fit(np.c_[Ad_x[ind], Ad_y[ind], U_inst[ind]])
            X = scaler.transform(np.c_[Ad_x[ind], Ad_y[ind], U_inst[ind]])
            neigh.fit(X[:,:2], X[:,2], sample_weight=weights[ind])  
            #print(neigh.alpha_, neigh.lambda_, neigh.scores_)  
            
        print('predict U')
        Xg = scaler.transform(np.c_[xg.flatten(),yg.flatten(),np.ones(len(yg.flatten()))])
        X = scaler.inverse_transform(np.c_[Xg[:,:2], neigh.predict(Xg[:,:2])])
        xg, yg, Uex = np.reshape(X[:,0],xg.shape), np.reshape(X[:,1],xg.shape), np.reshape(X[:,2],xg.shape)
        
        ind = ~np.isnan(V_inst)
        print('fit V')
        if treetype == 'Kn':
            scaler = RobustScaler(quantile_range=(25, 75)).fit(np.c_[Ad_x[ind], Ad_y[ind], V_inst[ind]])
            X = scaler.transform(np.c_[Ad_x[ind], Ad_y[ind], V_inst[ind]])
            neigh.fit(X[:,:2], X[:,2])
        else:
            scaler = RobustScaler(quantile_range=(25, 75)).fit(np.c_[Ad_x[ind], Ad_y[ind], V_inst[ind]])
            X = scaler.transform(np.c_[Ad_x[ind], Ad_y[ind], V_inst[ind]])
            neigh.fit(X[:,:2], X[:,2], sample_weight=weights[ind])  
            #print(neigh.alpha_, neigh.lambda_, neigh.scores_)  
            
        print('predict U')
        Xg = scaler.transform(np.c_[xg.flatten(),yg.flatten(),np.ones(len(yg.flatten()))])
        X = scaler.inverse_transform(np.c_[Xg[:,:2], neigh.predict(Xg[:,:2])])
        xg, yg, Vex = np.reshape(X[:,0],xg.shape), np.reshape(X[:,1],xg.shape), np.reshape(X[:,2],xg.shape)
    else: 
    ############# Interpolation with triangulation (cubic)
        for i in range(len(U_list)-1):
            Umean = filterfft(U_list[i],np.isnan(U_list[i]),sigma=L*s)
            Vmean = filterfft(V_list[i],np.isnan(V_list[i]),sigma=L*s)
            mask = np.isnan(Umean)
            xnew = grd[0][~mask]+Umean[~mask]*dt
            ynew = grd[1][~mask]+Vmean[~mask]*dt
            for p in point:
                T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
                xy = np.array(np.c_[xnew,ynew,np.ones(len(xnew))]).T
                xy = np.dot(T1,xy).T       
                ind = (xy[:,0]**2 + xy[:,1]**2) <= r**2
                x = np.r_[x, xnew[ind]]
                y = np.r_[y, ynew[ind]]        
                print(xnew.shape, np.sum(ind))
                xfin = np.r_[xfin, xnew[ind]+Umean[~mask][ind]*(t[i]-dt)]
                yfin = np.r_[yfin, ynew[ind]+Vmean[~mask][ind]*(t[i]-dt)]
                U = np.r_[U, U_list[i][~mask][ind]]
                V = np.r_[V, V_list[i][~mask][ind]]
                time = np.r_[time, np.ones(xnew[ind].shape[0])*t[i]]
        mask = np.isnan(U_list[-1])    
        x = np.r_[x,grd[0][~mask]]
        y = np.r_[y,grd[1][~mask]]
        xfin = np.r_[xfin,grd[0][~mask]]
        yfin = np.r_[yfin,grd[1][~mask]]  
        U = np.r_[U, U_list[-1][~mask]]
        V = np.r_[V, V_list[-1][~mask]]
        time = np.r_[time, np.ones(np.sum(~mask))*t[i]]
    #   Tree
        print('tree')       
        pos = np.c_[xfin, yfin]
        tree = KDTree(pos)
        dist, ind = tree.query(tree.data, k=2, return_distance=True)
        ind_d = dist[:,1]<dr/2   
        ind_t = time[ind[:,1]]<time[ind[:,0]]
        xfin = xfin[ind[~(ind_d & ind_t),0]]
        yfin = yfin[ind[~(ind_d & ind_t),0]]
        U = U[ind[~(ind_d & ind_t),0]]
        V = V[ind[~(ind_d & ind_t),0]]
        time = time[ind[~(ind_d & ind_t),0]]   
        print('triangle')
        tri_exp = Delaunay(np.c_[xfin,yfin])
        areas = areatriangles(tri_exp)
        indareas = areas<3*dx*dy
        trian = np.arange(len(areas))
        #grid
        if len(grid_i)==0:
            xg = np.arange(np.nanmin(xfin), np.nanmax(xfin)+dx, dx)
            yg = np.arange(np.nanmin(yfin), np.nanmax(yfin)+dy, dy)
            xg, yg = np.meshgrid(xg, yg)
        else:
            xg, yg = grid_i[0], grid_i[1]        
        ind_tri = np.isin(tri_exp.find_simplex(np.c_[xg.flatten(),yg.flatten()]),trian[indareas])
        print('interpolation')
        Uex = (np.zeros(xg.shape)*np.nan).flatten()
        Vex = (np.zeros(xg.shape)*np.nan).flatten()
        Uex[ind_tri] = sp.interpolate.CloughTocher2DInterpolator(tri_exp,U)(np.c_[xg.flatten()[ind_tri],yg.flatten()[ind_tri]])
        Vex[ind_tri] = sp.interpolate.CloughTocher2DInterpolator(tri_exp,V)(np.c_[xg.flatten()[ind_tri],yg.flatten()[ind_tri]])    
    return (xg, yg, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape))
###########################################
# In[]
def expand_field_point_time(grd, U_list, V_list, point, n = 12, r = 1000,
                            tree=True, treetype = 'Kn', grid_i = [], nx = 100, alpha = [0, 2*np.pi],
                            beta = 1, al = 10, t_scale = 1, probe='circle', Luy = [], time_scan = [],
                            t_int = [], interp_t = False, tri_dom = [], part = 10):
    #Output arrays
    xfin = np.array([])
    yfin = np.array([])
    tfin = np.array([])
    time = np.array([])
    times = np.array([])
    Ufin = np.array([])
    Vfin = np.array([])
    # Input time and coordinates
    date = datetime(1904, 1, 1) 
    dx = grd[0][0,1]-grd[0][0,0]
    dy = grd[1][1,0]-grd[1][0,0]
    dr = np.sqrt(dx**2+dy**2)
    print('means')
    # Local mean wind speed  parameters
    ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2) 
    Umeans = [filterfft(u,np.isnan(u),sigma=L/ds) for u, L in zip(U_list,Luy)]
    Vmeans = [filterfft(u,np.isnan(u),sigma=L/ds) for u, L in zip(V_list,Luy)]    
    if interp_t:
        print('interpolation in time')
        xp = grid_i[0].flatten()
        yp = grid_i[1].flatten()
        points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]        
        # time steps from t_int, first time bins
        t = np.max(time_scan, axis=1)-date
        #t = np.array([(ti).seconds+ (ti).microseconds/1000000 for ti in t])
        t = np.array([(ti).total_seconds() for ti in t])
        t_int_s = t_int-date
        t_int_s = np.array([(ti).total_seconds() for ti in t_int_s]) 
        t_init = t[0]
        t_int_s = t_int_s-t_init
        t = t-t_init
        t_int_s = t_int_s[t_int_s<=t[-1]] 
        print(t_init)
        dt = np.diff(t)
        xg = np.dot(np.array(xp)[:,None],np.ones(t_int_s.shape)[None,:])
        yg = np.dot(np.array(yp)[:,None],np.ones(t_int_s.shape)[None,:])
        tg = np.dot(np.ones(np.array(xp).shape)[:,None], t_int_s[None,:])        
        for i in range(len(U_list)-1):            
            #Advecting the current wind field  
            mask = np.isnan(U_list[i])
            x0 = grd[0][~mask] + Umeans[i][~mask]*dt[i]
            y0 = grd[1][~mask] + Vmeans[i][~mask]*dt[i]         
            ind_tri = tri_dom.find_simplex(np.c_[x0,y0])==-1          
            for j,p in enumerate(points):
                if j == 0:           
                    indy = (y0<p[1]+al*dr) & (y0>p[1]-al*dr)
                else:
                    indy = indy | (y0<p[1]+al*dr) & (y0>p[1]-al*dr)
            ind_tri = ind_tri & indy
            x0 = grd[0][~mask][ind_tri]
            y0 = grd[1][~mask][ind_tri]   
            x0_min = np.min(x0)
            U_min = Umeans[i][~mask][ind_tri][np.argmin(x0)]
            dtmax = np.array([])
            for p in points:
                dtmax = np.r_[dtmax, (p[0]-x0_min)/U_min]
            tmax = np.max(dtmax)+t[i]  
            if np.sum(t>=tmax)>0:
                tmax = t[t>=tmax][0]
            ind_t = (t_int_s>=t[i]) & (t_int_s<=tmax)
            print(np.sum(ind_t), tmax, t[i])
            #Check the end of the current scan within the domain           
            x0 = np.dot(grd[0][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])
            y0 = np.dot(grd[1][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:]) 
            xnew = np.dot(Umeans[i][~mask][ind_tri][:,None], (t_int_s[ind_t]-t[i])[None,:]) + x0
            ynew = np.dot(Vmeans[i][~mask][ind_tri][:,None], (t_int_s[ind_t]-t[i])[None,:])+ y0                              
            for j,p in enumerate(points):
                T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
                xy = np.array(np.c_[xnew.flatten(),ynew.flatten(),np.ones(len(xnew.flatten()))]).T
                xy = np.dot(T1,xy).T
                if probe == 'square':
                    if j == 0:
                        ind = (xy[:,0] <= beta*dr) & (xy[:,1] <= beta*dr)
                    else:
                        ind = ind | (xy[:,0] <= beta*dr) & (xy[:,1] <= beta*dr)
                if probe == 'circle':
                    if j == 0:
                        ind = (xy[:,0]**2 + xy[:,1]**2) <= (beta*dr)**2
                    else:
                        ind = ind | ((xy[:,0]**2 + xy[:,1]**2) <= (beta*dr)**2)              
            ind = np.reshape(ind, xnew.shape)
            xnew = xnew[ind]
            ynew = ynew[ind]                           
            tnew = np.dot(np.ones(Umeans[i][~mask][ind_tri].shape)[:,None], (t_int_s[ind_t])[None,:])[ind]
            Unew = np.dot(U_list[i][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])[ind]
            Vnew = np.dot(V_list[i][~mask][ind_tri][:,None],np.ones(t_int_s[ind_t].shape)[None,:])[ind]
            print(ind.shape,tnew.shape, Unew.shape, Vnew.shape, xnew.shape, ynew.shape)                
            xfin = np.r_[xfin, xnew]
            yfin = np.r_[yfin, ynew]
            tfin = np.r_[tfin, tnew]
            Ufin = np.r_[Ufin, Unew]
            Vfin = np.r_[Vfin, Vnew]
        print('tree')
        if tree:  
            # xg, yg, tg = np.meshgrid(xp,yp,t_int_s)
            Uex = np.zeros(xg.shape)*np.nan
            Vex = np.zeros(xg.shape)*np.nan
            neigh = KNeighborsRegressor(n_neighbors =  26, weights='distance',algorithm='auto', leaf_size=30,n_jobs=1)
            X = np.c_[xfin, yfin, tfin*t_scale]
            Xg = np.c_[xg.flatten(), yg.flatten(), tg.flatten()*t_scale]
            print('fit U')
            neigh.fit(X, Ufin)                
            print('Predict U')
            Uex = neigh.predict(Xg) 
            print('fit V')
            neigh.fit(X, Vfin)
            print('Predict V')
            Vex = neigh.predict(Xg)           
            tout = tg.flatten()
            tout = tout + t_init
            tout = np.array([timedelta(seconds=ti) for ti in tout])
            tout = date+tout
            tout = np.reshape(tout,tg.shape)   
        else:
            from scipy.interpolate import Rbf
            t_part = np.linspace(np.min(tfin), np.max(tfin)+1, part+1)
            # xg, yg, tg = np.meshgrid(xp,yp,t_int_s)
            Uex = np.ones(xg.shape)*np.nan
            Vex = np.ones(xg.shape)*np.nan
            for i in range(len(t_part)-1):
                print(i)
                indt_s = (tfin>=t_part[i]) & (tfin<=t_part[i+1])
                indt_g = (tg>=t_part[i]) & (tg<=t_part[i+1])
                rbfiu = Rbf(xfin[indt_s], yfin[indt_s], tfin[indt_s]*t_scale, Ufin[indt_s], function = 'cubic')
                rbfiv = Rbf(xfin[indt_s], yfin[indt_s], tfin[indt_s]*t_scale, Vfin[indt_s], function = 'cubic')
                print('Interpolation')
                Uex[indt_g] = rbfiu(xg[indt_g] , yg[indt_g] , tg[indt_g]*t_scale)
                Vex[indt_g] = rbfiv(xg[indt_g] , yg[indt_g] , tg[indt_g]*t_scale)   
            tout = tg.flatten()
            tout = tout + t_init
            tout = np.array([timedelta(seconds=ti) for ti in tout])
            tout = date+tout
            tout = np.reshape(tout,tg.shape) 
        return (xg, yg, tg, tout, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape), np.c_[xp,yp])  
    else:
        print('interpolation in space')
        # Time steps for spatial interpolation (one time, many spatial points)
        # t = np.max(time_scan, axis=1)-date
        # t = np.array([(ti).seconds+ (ti).microseconds/1000000 for ti in t])
        # t_int = (t_int-date).seconds + (t_int-date).microseconds/1000000
        
        
        t = np.max(time_scan, axis=1)-date
        t = np.array([(ti).total_seconds() for ti in t])
        t_int = t_int-date
        t_int = t_int.total_seconds()
        t_init = t[0]
        t_int = t_int-t_init
        t = t-t_init

        # bin for interpolation
        ti = np.min(t[t>=t_int])
        te = np.max(t[t<=t_int])
        tbin = [ti, te]
        dt = np.diff(t)   
        print('probes') 
        chunk = 512
        Uarr = np.vstack([Umeans[i] for i in range(len(U_out_u))])
        Umask = ~np.isnan(np.nanmean(Uarr.reshape((int(len(Uarr)/chunk),chunk,chunk)),axis=0))
        xmask = grd[0][Umask]
        ymask = grd[1][Umask]
        center = .5*np.r_[grd[0][0,0]+grd[0][0,-1], grd[1][0,0]+grd[1][-1,0]]  
        R = np.array([])
        for p in point:
            R = np.r_[R, np.sqrt(np.sum((p[:2]-center)**2))]
        p = point[np.argmax(R)]
        # alpha0 = np.arctan2((point[0][:2]-center)[1], (point[0][:2]-center)[0])
        # alpha1 = np.arctan2((point[1][:2]-center)[1], (point[1][:2]-center)[0])
        alpha = np.linspace(alpha[0],alpha[1], n+1)   
        xp = center[0]+np.max(R)*np.cos(alpha)
        yp = center[1]+np.max(R)*np.sin(alpha)
        xp = xp[(yp>np.nanmin(grd[1][Umask]))&(yp<np.nanmax(grd[1][Umask]))]
        yp = yp[(yp>np.nanmin(grd[1][Umask]))&(yp<np.nanmax(grd[1][Umask]))]
        tree_p = KDTree(np.c_[xmask, ymask])
        ri, _ = tree_p.query(np.c_[xp,yp], k = 1, return_distance=True)
        points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]
        print('advection to time t_int')
        Ui = U_list[np.nonzero(t==tbin[0])[0][0]]
        Ue = U_list[np.nonzero(t==tbin[1])[0][0]]
        mask = np.isnan(Ui)
        Ui = Ui[~mask]
        xi = grd[0][~mask]
        yi = grd[1][~mask]
        mask = np.isnan(Ue)
        Ue = Ue[~mask]
        xe = grd[0][~mask]
        ye = grd[1][~mask]
        #
        Vi = V_list[np.nonzero(t==tbin[0])[0][0]]
        Ve = V_list[np.nonzero(t==tbin[1])[0][0]]
        mask = np.isnan(Vi)
        Vi = Vi[~mask]
        mask = np.isnan(Ve)
        Ve = Ve[~mask]
        #
        ti = np.ones(Ui.shape)*tbin[0]
        te = np.ones(Ue.shape)*tbin[1]
        #
        xfin = np.r_[xfin, xi, xe]
        yfin = np.r_[yfin, yi, ye]
        time = np.r_[time, ti, te]
        times = np.r_[times, ti, te]+t_init
        
        Ufin = np.r_[Ufin, Ui, Ue]
        Vfin = np.r_[Vfin, Vi, Ve]                 
        for i in range(len(t[t<tbin[1]])):
            mask = np.isnan(U_list[i])
            xnew = grd[0][~mask]+Umeans[i][~mask]*dt[i]
            ynew = grd[1][~mask]+Vmeans[i][~mask]*dt[i]
            #sampling
            for j,p in enumerate(points):
                T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
                xy = np.array(np.c_[xnew,ynew,np.ones(len(xnew))]).T
                xy = np.dot(T1,xy).T
                if probe == 'square':
                    if j == 0:
                        ind = (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
                    else:
                        ind = ind | (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
                if probe == 'circle':
                    if j == 0:
                          ind = (xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2
                    else:
                          ind = ind | ((xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2) 
            
            xi = xnew[ind]+Umeans[i][~mask][ind]*(t_int-t[i+1])#(tbin[0]-t[i+1])
            yi = ynew[ind]+Vmeans[i][~mask][ind]*(t_int-t[i+1])#(tbin[0]-t[i+1])
            Ui = U_list[i][~mask][ind]
            Vi = V_list[i][~mask][ind]
            ti = np.ones(xi.shape)*tbin[0]
            tis = np.ones(Ui.shape)*(t[i]+t_init)
            
            xfin = np.r_[xfin, xi]#, xe]
            yfin = np.r_[yfin, yi]#, ye]
            time = np.r_[time, ti]#, te]
            times = np.r_[times, tis]#, te]
            Ufin = np.r_[Ufin, Ui]#, Ue]
            Vfin = np.r_[Vfin, Vi]#, Ve]        
        if len(grid_i)==0:
            xg = np.arange(np.nanmin(xfin), np.nanmax(xfin)+dx, dx)
            yg = np.arange(np.nanmin(yfin), np.nanmax(yfin)+dy, dy) 
            xg, yg = np.meshgrid(xg, yg)
        else:
            xg, yg = grid_i[0], grid_i[1]     
        tg = t_int*np.ones(xg.shape)   
        print('envelope')
        x_bin = np.linspace(np.nanmin(xfin), np.nanmax(xfin), nx)   
        indyg = np.zeros(xg.shape, dtype=bool)    
        for i in range(len(x_bin)-1):
            indx = (xfin >= x_bin[i]) & (xfin < x_bin[i+1])
            indxg = (xg >= x_bin[i]) & (xg < x_bin[i+1])
            indyg = (((yg > np.nanmin(yfin[indx])) & (yg < np.nanmax(yfin[indx]))) & indxg) | indyg       
        print('points')
        X = np.c_[xfin, yfin, time]
        Xg = np.c_[xg[indyg],yg[indyg],tg[indyg]]
        plt.figure()
        plt.scatter(xg[indyg],yg[indyg])
        
        print('tree')
        if tree:  
            Uex = np.zeros(xg.shape)*np.nan
            Vex = np.zeros(xg.shape)*np.nan
            tex = np.zeros(xg.shape)*np.nan
            h = 3*dr
            func = lambda x: (1/np.sqrt(2*np.pi)/h)*np.exp(-x**2/(2*h**2))
            neigh = KNeighborsRegressor(n_neighbors =  26, weights=func,algorithm='auto', leaf_size=30,n_jobs=1)
            print('fit U')
            neigh.fit(X, Ufin)                
            print('Predict U')
            Uex[indyg] = neigh.predict(Xg) 
            print('fit V')
            neigh.fit(X, Vfin)
            print('Predict V')
            Vex[indyg] = neigh.predict(Xg)
            print('fit t')
            neigh.fit(X, times)
            print('Predict t')
            tex[indyg] = neigh.predict(Xg)
        else:
            print('triangulation')
            tri_exp = Delaunay(X)
            print('Interpolation')
            Uex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Ufin)(Xg)
            Vex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Vfin)(Xg)               
        return (xg, yg, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape),np.reshape(tex,xg.shape), np.c_[xp,yp])
#######################################################
#######################################################
# In[]
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

def acf(x,dt):
    U = np.mean(x)
    N = len(x)
    T = dt*(N-1)    
    tau = np.arange(0,T+dt,dt)
    x = x-U
    corr = np.array([np.sum(x[k:]*x[:N-k]/(N-k)) for k in range(N)])
    tau = tau[:np.nonzero(corr<=0)[0][0]+1]
    corr = corr[:np.nonzero(corr<=0)[0][0]+1] 
    Tau = np.trapz(corr,tau)
    L = Tau*U
    return (tau, corr/np.var(x), Tau, L) 

def sample(u,t,ti):
    from scipy.interpolate import interp1d
    date = datetime(1904, 1, 1) 
    t_int = t-date
    t_int = np.array([(t).total_seconds() for  t in t_int])
    tint = ti-date
    tint = np.array([(t).total_seconds() for  t in tint])
    f = interp1d(t_int , u, kind='cubic')
    return f(tint)

def detect_peaks(image, size = 0):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)
    if size==0:
        #apply the local maximum filter; all pixel of maximal value 
        #in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood)==image
        local_min = maximum_filter(-image, footprint=neighborhood)==-image
    else:
        #apply the local maximum filter; all pixel of maximal value 
        #in their neighborhood are set to 1
        local_max = maximum_filter(image, size = size)==image
        local_min = maximum_filter(-image, size = size)==-image
    local_max_min = local_max | local_min 
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    #we create the mask of the background
    background = (image==0)
    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max_min ^ eroded_background
    return detected_peaks

###############################################################################
# In[Load data]
###############################################################################
root = tkint.Tk()
file_in_path_0 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir, lidar 0')
root.destroy()
root = tkint.Tk()
file_in_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir, lidar 1')
root.destroy()
root = tkint.Tk()
file_in_path_corr = tkint.filedialog.askdirectory(parent=root,title='Choose an corr 1 dir')
root.destroy()
root = tkint.Tk()
file_out_path_u_field = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy() 

# In[Actual data loading, def of number of scns to reconstruct]
# U_out2, V_out2, grd, s_u2 = joblib.load(file_out_path_u_field+'/U_rec_08_06_2016.pkl')      
# U_out2 = [U_out2[i] for i in range(800)]
# V_out2 = [V_out2[i] for i in range(800)]  

# phase2 = 537
# i = phase2
# chunk = 512
# ch = 14  
 
# U_arr = np.vstack([U_out2[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
# V_arr = np.vstack([V_out2[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
# scan2 = [s_u2[i][0] for i in range(i-int(ch/2),i+int(ch/2)+1)]

###############################################################################
# In[Current recontruction]
######################################
#date and lidar location
date = datetime(1904, 1, 1) 
###############################################################################
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

######################################
#Create engines for database reading
csv_database_r = create_engine('sqlite:///'+file_in_path_corr+'/corr_uv_west_phase2_ind.db')
csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0_phase2.db')
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1_phase2.db')  

##########################################################
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)      

drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r)
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r)
dfL_phase2.dropna(inplace=True)
dfL_phase2.reset_index(inplace=True)
dfL_phase2.drop(['index'],axis=1,inplace=True)
dfL_phase2 = dfL_phase2.drop_duplicates(subset='time', keep='first')

heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
    
L_list_2 = np.hstack([L_smooth(np.array(res_flux_2)[:,i,:]) for i in range(len(heights))])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_2)[:,2,0]])
    
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]

cols = [item for sublist in cols for item in sublist]

stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2

dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2.index = dfL_phase2.index
aux2 = dfL_phase2.time
dfL_phase2.time = pd.to_datetime(dfL_phase2.time)

for i in range(len(t2)-1):
    print(i,t2[i])
    ind = (dfL_phase2.time>=t2[i]) & (dfL_phase2.time<=t2[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values

dfL_phase2.time = aux2
dfL_phase2[cols] = Lph2
dfL_phase2.sort_values(by=['time'], inplace = True)
ind = drel_phase2.time0.isin(dfL_phase2.time)
timein = pd.concat([drel_phase2.time0.loc[ind],drel_phase2.time1.loc[ind],drel_phase2.relscan.loc[ind]],axis = 1)
timein.index = dfL_phase2.index
dfL_phase2[['time0','time1','rel']] = timein
dfL_phase2['date'] = pd.to_datetime(dfL_phase2.time.values)

##############################################
# In[Identify average LuxLuy]
colL = '$L_{flux,103}$'
rel = (dfL_phase2['rel']>.5) & (dfL_phase2[colL].abs()>1000) 

#rel = (dfL_phase2['rel']>.5) & ((dfL_phase2[colL]>100) & (dfL_phase2[colL]<300))

rel = (dfL_phase2['rel']>.5) & ((dfL_phase2[colL]<0) & (dfL_phase2[colL]>-500))

shape = (dfL_phase2['$L_{u,x}$'].div(dfL_phase2['$L_{u,y}$'])>0)&(dfL_phase2['$L_{u,x}$'].div(dfL_phase2['$L_{u,y}$'])<2)

vel = (dfL_phase2['$U_{175}$']<5) & (dfL_phase2['$U_{175}$']>3)

rel = rel & vel & shape 

lux_mean = dfL_phase2['$L_{u,x}$'].loc[rel].mean()
luy_mean = dfL_phase2['$L_{u,y}$'].loc[rel].mean()

alfa = .9
betax = .1
betay = .01
offx = betax*dfL_phase2['$L_{u,x}$'].loc[rel].std()
offy = betay*dfL_phase2['$L_{u,y}$'].loc[rel].std()

lux = (dfL_phase2['$L_{u,x}$']>alfa*(lux_mean+offx)) & (dfL_phase2['$L_{u,x}$']<(2-alfa)*(lux_mean+offx))
luy = (dfL_phase2['$L_{u,y}$']>alfa*(luy_mean+offy)) & (dfL_phase2['$L_{u,y}$']<(2-alfa)*(luy_mean+offy))

luxluy = (lux & luy) 

# In[]
# Pick random scan meeting conditions
pick_rnd = dfL_phase2.loc[luxluy & rel].sample()

pick_rnd_old = pick_rnd

joblib.dump(pick_rnd_old, file_out_path_u_field+'/case_'+pick_rnd_old.name.values[0]+pick_rnd_old.hms.values[0]+'.pkl')

#Load case by date

name = '20160801'
hms = '213103'

pick_rnd = dfL_phase2.loc[(dfL_phase2.name == name) & (dfL_phase2.hms == hms)]



pick_rnd = joblib.load(file_out_path_u_field+'/case_20160801213103.pkl')

# In[]
pick_rnd = pick_rnd2

t0 = pd.to_datetime(pick_rnd.time0.values)
t1 = pd.to_datetime(pick_rnd.time1.values)

minutes = 120

t_init0 = t0 - timedelta(minutes=minutes)
t_end0 = t0 + timedelta(minutes=minutes)
hms_i_0 = t_init0.strftime("%H%M%S").values[0]
hms_e_0 = t_end0.strftime("%H%M%S").values[0]

t_init1 = t1 - timedelta(minutes=minutes)
t_end1 = t1 + timedelta(minutes=minutes)
hms_i_1 = t_init1.strftime("%H%M%S").values[0]
hms_e_1 = t_end1.strftime("%H%M%S").values[0]

dy = t0.strftime("%Y%m%d").values[0]


mask = (dfL_phase2['date'] >= t_init0.strftime("%Y%m%d%H%M%S").values[0]) & (dfL_phase2['date'] <= t_end0.strftime("%Y%m%d%H%M%S").values[0])
name_hms = dfL_phase2[['name','scan','hms']].loc[mask]
#dy = name_hms.name.values[0]

#check spread around 10 min

### Reconstruction of fields for a specific day
######################################
#dy = '20160421'#phase1

t_0_0 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
t_0_0 = t_0_0+timedelta(seconds = int(hms_i_0[4:]))
t_0_0 = t_0_0+timedelta(minutes = int(hms_i_0[2:4]))
t_0_0 = t_0_0+timedelta(hours = int(hms_i_0[:2]))
t_0_0 = str((t_0_0-date).total_seconds())
#
t_1_0 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
t_1_0 = t_1_0+timedelta(seconds = int(hms_e_0[4:]))
t_1_0 = t_1_0+timedelta(minutes = int(hms_e_0[2:4]))
t_1_0 = t_1_0+timedelta(hours = int(hms_e_0[:2]))
t_1_0 = str((t_1_0-date).total_seconds())

t_0_1 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
t_0_1 = t_0_1+timedelta(seconds = int(hms_i_1[4:]))
t_0_1 = t_0_1+timedelta(minutes = int(hms_i_1[2:4]))
t_0_1 = t_0_1+timedelta(hours = int(hms_i_1[:2]))
t_0_1 = str((t_0_1-date).total_seconds())
#
t_1_1 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
t_1_1 = t_1_1+timedelta(seconds = int(hms_e_1[4:]))
t_1_1 = t_1_1+timedelta(minutes = int(hms_e_1[2:4]))
t_1_1 = t_1_1+timedelta(hours = int(hms_e_1[:2]))
t_1_1 = str((t_1_1-date).total_seconds())

#t_0 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
#t_0 = t_0+timedelta(seconds = int(name_hms.hms.values[0][4:]))
#t_0 = t_0+timedelta(minutes = int(name_hms.hms.values[0][2:4]))
#t_0 = t_0+timedelta(hours = int(name_hms.hms.values[0][:2]))
#t_0 = str((t_0-date).total_seconds())
##
#t_1 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
#t_1 = t_1+timedelta(seconds = int(name_hms.hms.values[-1][4:]))
#t_1 = t_1+timedelta(minutes = int(name_hms.hms.values[-1][2:4]))
#t_1 = t_1+timedelta(hours = int(name_hms.hms.values[-1][:2]))
#t_1= str((t_1-date).total_seconds())

#t_0 = str((t_init.to_pydatetime()[0]-date).total_seconds())
#t_1 = str((t_end.to_pydatetime()[0]-date).total_seconds())


#dy = '20160806'#phase2
#
#hms = drel_phase2[['scan0','hms']].loc[drel_phase2['name']==dy].loc[drel_phase2['scan0'].isin(np.squeeze(scan2))]

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
selec_raw = col_raw + ' FROM "table_raw"'

# Reconstruction of chuncks of 1 hour scans?
switch = 0
U_out_c, V_out_c, su_c = [], [], [] 

#t_0 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
#t_0 = t_0+timedelta(seconds = int(hms.hms.values[0][4:]))
#t_0 = t_0+timedelta(minutes = int(hms.hms.values[0][2:4]))
#t_0 = t_0+timedelta(hours = int(hms.hms.values[0][:2]))
#t_0 = str((t_0-date).total_seconds())
#
#t_1 = datetime(int(dy[:4]), int(dy[4:6]), int(dy[6:]))
#t_1 = t_1+timedelta(seconds = int(hms.hms.values[-1][4:]))
#t_1 = t_1+timedelta(minutes = int(hms.hms.values[-1][2:4]))
#t_1 = t_1+timedelta(hours = int(hms.hms.values[-1][:2]))
#t_1= str((t_1-date).total_seconds())


#query_fil = selec_fil+ ' where name = ' + dy + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1 #' and scan <=' + str(hms.scan0.values[-1]) #
#query_raw = selec_raw+ ' where name = ' + dy + ' and stop_time >= ' + t_0 + ' and stop_time <= ' + t_1 #' and scan <=' + str(hms.scan0.values[-1]) #

query_fil_0 = selec_fil+ ' where name = ' + dy + ' and stop_time >= ' + t_0_0 + ' and stop_time <= ' + t_1_0 #' and scan <=' + str(hms.scan0.values[-1]) #
query_raw_0 = selec_raw+ ' where name = ' + dy + ' and stop_time >= ' + t_0_0 + ' and stop_time <= ' + t_1_0 #' and scan <=' + str(hms.scan0.values[-1]) #

query_fil_1 = selec_fil+ ' where name = ' + dy + ' and stop_time >= ' + t_0_1 + ' and stop_time <= ' + t_1_1 #' and scan <=' + str(hms.scan0.values[-1]) #
query_raw_1 = selec_raw+ ' where name = ' + dy + ' and stop_time >= ' + t_0_1 + ' and stop_time <= ' + t_1_1 #' and scan <=' + str(hms.scan0.values[-1]) #

# First database loading
df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ind)
df = pd.read_sql_query(query_raw_0, csv_database_0_ind)
# Retrieving good CNR values from un-filtered scans
for i in range(198):
    ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
    df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None
# df_0.drop(columns = labels_CNR,inplace=True)
df_0.columns = labels_short

df_1 = pd.read_sql_query(query_fil_1, csv_database_1_ind)
df = pd.read_sql_query(query_raw_1, csv_database_1_ind)
for i in range(198):
    ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
    df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None   
#df_1.drop(columns = labels_CNR,inplace=True) 
df_1.columns = labels_short

# # In[Current code] 
# loc0 = np.array([6322832.3,0])
# loc1 = np.array([6327082.4,0])
# d = loc0-loc1 
# switch = 0

loc0 = np.array([0,6322832.3])#-d
loc1 = np.array([0,6327082.4])# d
d = loc1-loc0  
switch = 0
######################################
U_out_u, V_out_u, su_u = [], [], [] 
U_list = []
V_list = [] 
s_syn,t_scan,_,_ = synch_df(df_0,df_1,dtscan=45/2)
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
            u, v, grdu, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = 128)  
            U_out_u.append(u), V_out_u.append(v), su_u.append(s)
#############################################################################################            
            tri_calc = True
            chunk = 128
            x = grdu[0][0,:]
            y = grdu[1][:,0]
            U_arr = np.vstack([u[i] for i in range(len(u))])
            V_arr = np.vstack([v[i] for i in range(len(v))])
            scan = s_syn[:,0]
            U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
            V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
            ur, vr, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 
            for j in range(len(u)):
                print(j)
                U, V = U_rot(grdu, u[j], v[j], gamma = gamma, tri_calc = False, 
                                          tri_del = tri_del, mask_int = mask_int, mask = mask) 
                U_list.append(U)
                V_list.append(V)
            

#############################################################################################  
U_out_u = [item for sublist in U_out_u for item in sublist]
V_out_u = [item for sublist in V_out_u for item in sublist]
su_u    = [item for sublist in su_u for item in sublist]

# U_list = [item for sublist in U_list for item in sublist]
# V_list= [item for sublist in V_list for item in sublist]


# In[MySQL Østerild sonic and lidar data]
osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')

t0 = pd.to_datetime(pick_rnd.time0.values)
t1 = pd.to_datetime(pick_rnd.time1.values)

dy = t0.strftime("%Y%m%d").values[0]

t_init0 = t0 - timedelta(minutes=minutes)
t_end0 = t0 + timedelta(minutes=minutes)
hms_i_0 = t_init0.strftime("%H%M%S").values[0]
hms_e_0 = t_end0.strftime("%H%M%S").values[0]

t_init1 = t1 - timedelta(minutes=minutes)
t_end1 = t1 + timedelta(minutes=minutes)
hms_i_1 = t_init1.strftime("%H%M%S").values[0]
hms_e_1 = t_end1.strftime("%H%M%S").values[0]




if t0>t1:
    stampi = dy+hms_i_1
    stampe = dy+hms_e_0
else:
    stampi = dy+hms_i_0
    stampe = dy+hms_e_1

Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
Sdir_1 = ['Sdir_241m_LMN','Sdir_175m_LMN','Sdir_103m_LMN','Sdir_37m_LMN','Sdir_7m_LMN']
Name = ['Name']
T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
X_1 = ['X_241m_LMN','X_175m_LMN','X_103m_LMN','X_37m_LMN','X_7m_LMN']
Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
Y_1 = ['Y_241m_LMN','Y_175m_LMN','Y_103m_LMN','Y_37m_LMN','Y_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']
P_0 = ['Press_241m_LMS','Press_7m_LMS']
P_1 = ['Press_241m_LMN','Press_7m_LMN']

#for j,stamp in enumerate(t_ph1):
#    print(stamp,j)    
table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
query_name_i = 'select name' + table_20Hz + where + ' limit 1'
where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
query_name_e = 'select name' + table_20Hz + where + ' limit 1'
name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
where = 'where name > ' + name_i + ' and name < ' + name_e
sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_0.fillna(value=pd.np.nan, inplace=True)
sql_query = 'select ' + ", ".join(T_1+X_1+Y_1+Z_1+Sspeed_1+Sdir_1+Name) + table_20Hz +  where
stab_20hz_1 = pd.read_sql_query(sql_query[0],osterild_database)  
stab_20hz_1.fillna(value=pd.np.nan, inplace=True)



# gammax = np.arctan2(np.nanmean(-stab_20hz_0[X_0].values[:,0]),np.nanmean(stab_20hz_0[Y_0].values[:,0]))

# S11 = np.cos(gammax)
# S12 = np.sin(gammax)
# R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
# vel = np.c_[stab_20hz_0[Y_0].values[:,0], stab_20hz_0[X_0].values[:,0]].T
# vel = np.dot(R[:-1,:-1],vel)
# u_s_0 = vel[0,:]#.reshape(stab_20hz_0[X_0].values.shape)
# v_s_0 = vel[1,:]#.reshape(stab_20hz_0[X_0].values.shape)
# vel = np.c_[stab_20hz_1[Y_1].values[:,0], stab_20hz_1[X_1].values[:,0]].T
# vel = np.dot(R[:-1,:-1],vel)
# u_s_1 = vel[0,:]#.reshape(stab_20hz_1[X_1].values.shape)
# v_s_1 = vel[1,:]#.reshape(stab_20hz_1[X_1].values.shape)


# u_s_0 = np.sqrt(stab_20hz_0[X_0].values**2+stab_20hz_0[Y_0].values**2)
# u_s_1 = np.sqrt(stab_20hz_1[X_1].values**2+stab_20hz_1[Y_1].values**2)
# w_s_0 = stab_20hz_0[Z_0].values
# w_s_1 = stab_20hz_1[Z_1].values
# h, xm = np.meshgrid(heights, np.arange(u_s_0.shape[0])*np.mean(u_s_0)/20)

# plt.figure()
# plt.contourf(xm, h, u_s_0,cmap='jet')
# plt.colorbar()
# plt.figure()
# plt.contourf(xm, h, u_s_1,cmap='jet')
# plt.colorbar()
# plt.figure()
# plt.contourf(xm, h, w_s_0,cmap='jet')
# plt.colorbar()
# plt.figure()
# plt.contourf(xm, h, w_s_1,cmap='jet')
# plt.colorbar() 

# In[Whole field preparations]
tri_calc = True
chunk = 128
x = grdu[0][0,:]
y = grdu[1][:,0]
U_arr = np.vstack([U_out_u[i] for i in range(len(U_out_u))])
V_arr = np.vstack([V_out_u[i] for i in range(len(V_out_u))])
scan = s_syn[:,0]
U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 


L2 = np.abs(grdu[0][0,0]-grdu[0][0,-1])
ds2 = np.sqrt(np.diff(grdu[0][0,:])[0]**2+np.diff(grdu[1][:,0])[0]**2)

s_i2 = np.min(dfL_phase2.loc[np.isin(dfL_phase2.scan.values,s_syn[:,0])][['$L_{u,x}$','$L_{u,y}$']].values,axis=1)/ds2

L_s2 = np.min(dfL_phase2.loc[np.isin(dfL_phase2.scan.values,s_syn[:,0])][['$L_{u,x}$','$L_{u,y}$']].values,axis=1)

S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
dr0 = np.array(np.r_[-d/2,np.ones(1)]).T
dr0 = np.dot(T,dr0)  
dr1 = np.array(np.r_[d/2,np.ones(1)]).T
dr1 = np.dot(T,dr1) 

U_list = []
V_list = []    
for j in range(len(U_out_u)):
    print(j)
    U, V = U_rot(grdu, U_out_u[j], V_out_u[j], gamma = gamma, tri_calc = False, 
                              tri_del = tri_del, mask_int = mask_int, mask = mask) 
    U_list.append(U)
    V_list.append(V)
dx = grdu[0][0,1]-grdu[0][0,0]
dy = grdu[1][1,0]-grdu[1][0,0]
dt = 45    
r = 3000
Luy = np.min(dfL_phase2.loc[np.isin(dfL_phase2.scan.values,s_syn[:,0])][['$L_{u,x}$','$L_{u,y}$']].values,axis=1)                 
Lux = np.max(dfL_phase2.loc[np.isin(dfL_phase2.scan.values,s_syn[:,0])][['$L_{u,x}$','$L_{u,y}$']].values,axis=1)

# In[]
df1 = pd.read_sql_query(query_raw_0, csv_database_1_ind)
df0 = pd.read_sql_query(query_raw_0, csv_database_0_ind)

df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ind)
# Retrieving good CNR values from un-filtered scans
for i in range(198):
    ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
    df_0['ws_'+str(i)].loc[ind] = df0['ws_'+str(i)].loc[ind]
df = None
# df_0.drop(columns = labels_CNR,inplace=True)
df_0.columns = labels_short



scan = df_0.scan.unique()[46]
ws = df_0.loc[df_0.scan==scan].ws
rad = np.unique(df_0.loc[df_0.scan==scan].range_gate.values)
azim = df_0.loc[df_0.scan==scan].azim.unique()
ws_raw = df0.loc[df_0.scan==scan].values

rad, azim = np.meshgrid(rad,azim)
xr, yr = rad*np.cos((90-azim)*np.pi/180), rad*np.sin((90-azim)*np.pi/180)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(xr, yr, ws_raw,np.linspace(-22,-6,10), cmap='jet')
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
fig0.colorbar(im0)


# In[wind fields]
# file_pre = '/case_'+pick_rnd.name.values[0]+pick_rnd.hms.values[0]
# os.mkdir(file_out_path_u_field+file_pre)
fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
newdate = []


for j in range(len(U_list)): 
    new_date = t_scan[j,0].strftime("%Y-%m-%d %H:%M:%S")
    newdate.append(new_date[:11]+ '\:'+new_date[11:])


meanU = []
meanV = []
for j in range(len(U_list)):
    ax0.cla()    
    meanU.append(np.nanmean(U_list[j]))
    meanV.append(np.nanmean(V_list[j]))
    ax0.set_title('$'+str(newdate[j])+'$', fontsize = 20) 
    # ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
    # ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
    im0 = ax0.contourf(grdu[0], grdu[1], U_list[j], np.linspace(5,15,10), cmap='jet')
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

    fig0.tight_layout()
    fig0.savefig(file_out_path_u_field+file_pre+'/'+str(j)+'.png')
    plt.pause(.01)

# In[Spectra]

t0 = pd.to_datetime(pick_rnd.time0.values)
dy = t0.strftime("%Y%m%d").values[0]
hms = drel_phase2[['scan0','hms']].loc[drel_phase2['name']==dy].loc[drel_phase2['scan0'].isin(np.squeeze(s_syn[:,0]))]
df_corr_phase2 = pd.read_sql_query("select * from 'corr' where name = '"+dy+"' and hms >= '" +                                   
                                   hms.hms.min() + "' and hms <= '" +  hms.hms.max()+"'",csv_database_r)
reslist = [df_corr_phase2[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[df_corr_phase2.scan == s].values for s in s_syn[:,0]]

tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist]
eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist]
ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist]
rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist]
ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist]

resarray = np.vstack([r for r in reslist])

tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)

N = 512
taumax = np.nanmax(np.abs(tau_arr))
taui = np.linspace(0,taumax,int(N/2)+1)
taui = np.r_[-np.flip(taui[1:]),taui]
etamax = np.nanmax(np.abs(eta_arr))
etai = np.linspace(0,etamax,int(N/2)+1)
etai = np.r_[-np.flip(etai[1:]),etai]
taui, etai = np.meshgrid(taui,etai)

rui = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ru_list)])
rvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, rv_list)])
ruvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ruv_list)])

ru_mean1 = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
rv_mean1 = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
ruv_mean1 = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)

susv = np.sqrt(np.nanmax(np.abs(ru_mean1)))*np.sqrt(np.nanmax(np.abs(rv_mean1)))
su2 = np.nanmax(np.abs(ru_mean1))
sv2 = np.nanmax(np.abs(rv_mean1))


Luym, Luxm = sc.integral_lenght_scale(ru_mean1,taui[0,:], etai[:,0])

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(etai,taui, ru_mean1/su2,np.linspace(np.nanmin(ru_mean1/su2),np.nanmax(ru_mean1/su2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-3500, 3200,'(a)',fontsize=30,color='w')
ax1.set_xlim(-4000,4000)
ax1.set_ylim(-4000,4000)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{uu}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(etai[:,0], ru_mean1[:,256].T/np.nanmax(ru_mean1),'k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(taui[0,:], ru_mean1[256,:].T/np.nanmax(ru_mean1),'r',label='$\\rho_{u,u}(0,\\eta)$')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# ax.set_xlim(-4000,4000)
fig.tight_layout()

k1r,k2r,Suur,Svvr, Suvr = sc.spectra_fft((taui,etai),ru_mean1,rv_mean1,ruv_mean1)

sc.plot_log2D((k1r,k2r), np.abs(Svvr.T), label_S = "$\log_{10}{Suu}$", C = 10**-3, minS = 0, nl = 30)



fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
# ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(k1r,k2r,np.log10(np.abs(Suur.T)),30, cmap='jet')
plt.colorbar(im1)


fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
# ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(k1r,k2r,np.log10(np.imag(Suur)),10, cmap='jet')
plt.colorbar(im1)


# In[]
t,e,ru, rv, ruv = sc.spectra_fft(np.meshgrid(k1r,k2r),Suur,Svvr, Suvr,inv=True)

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(e,t,ru/su2, np.linspace(np.nanmin(ru_mean1/su2),np.nanmax(ru_mean1/su2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-3500, 3200,'(a)',fontsize=30,color='w')
ax1.set_xlim(-4000,4000)
ax1.set_ylim(-4000,4000)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{uu}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(e, ru[:,255].T/np.nanmax(ru),'k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(t, ru[255,:].T/np.nanmax(ru),'r',label='$\\rho_{u,u}(0,\\eta)$')
ax.plot(etai[:,0], ru_mean1[:,256].T/np.nanmax(ru_mean1),'b',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(taui[0,:], ru_mean1[256,:].T/np.nanmax(ru_mean1),'g',label='$\\rho_{u,u}(0,\\eta)$')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# ax.set_xlim(-4000,4000)
fig.tight_layout()


sc.plot_log2D((k1r,k2r), Suur, label_S = "$\log_{10}{Suu}$", C = 10**-4)

k1, k2,Suu,Svv,Suv = [], [], [], [], []
for j in range(len(U_list)):
    print(j)
    k1_int, k2_int,Suu_int,Svv_int,Suv_int = sc.spatial_spec_sq(x,y,U_list[j],
                                                                V_list[j], tri_calc = False, transform = False, shrink = True, ring=False, plot = False, one_dim = False)
    k1.append(k1_int)
    k2.append(k2_int)
    Suu.append(Suu_int)
    Svv.append(Svv_int)
    Suv.append(Suv_int)
kgrid = np.meshgrid(k1[4], k2[4])
chunk = 424
S_arr = np.vstack([s for s in Suu[4:]])
S_ave =  np.nanmean(S_arr.reshape((int(len(S_arr)/chunk),chunk,chunk)),axis=0)

sc.plot_log2D(kgrid, S_ave, label_S = "$\log_{10}{Suu}$", C = 10**-3, minS = 0, nl = 30)



# In[Divergence's spectra]

tau_out = []
eta_out = []    
r_u_out = []
r_c_out = []
r_uc_out = []

for j in range(len(U_list)):
    
    dudy, dudx = np.gradient(U_list[j], grdu[1][:,0], grdu[0][0,:]) 
    dvdy, dvdx = np.gradient(V_list[j], grdu[1][:,0], grdu[0][0,:])    
    contsh = dudx + dvdy
    mask_c = np.isnan(contsh)
    U = U_list[j].copy()
    U[mask_c] = np.nan
    tau,eta,r_u,r_c,r_uc,_,_,_,_ = sc.spatial_autocorr_sq(grdu, U, contsh,
                                                          transform = False,
                                                          transform_r = False,
                                                          e_lim=.1,refine=32)
    tau_out.append(tau.astype(np.float32))
    eta_out.append(eta.astype(np.float32)) 
    r_u_out.append(r_u.astype(np.float32))
    r_c_out.append(r_c.astype(np.float32))
    r_uc_out.append(r_uc.astype(np.float32))


N = 512
taumax = np.max(np.array([np.max(np.abs(t)) for t in tau_out]))#np.nanmax(np.abs(np.array(tau_out)))
taui = np.linspace(0,taumax,int(N/2)+1)
taui = np.r_[-np.flip(taui[1:]),taui]
etamax = np.max(np.array([np.max(np.abs(t)) for t in eta_out]))#np.nanmax(np.abs(np.array(eta_out)))
etai = np.linspace(0,etamax,int(N/2)+1)
etai = np.r_[-np.flip(etai[1:]),etai]
taui, etai = np.meshgrid(taui,etai)

rui = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_u_out)])
rci = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_c_out)])
ruci = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_uc_out)])

rhoui = np.vstack([sc.autocorr_interp_sq(r/np.nanmax(r),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_u_out)])
rhoci = np.vstack([sc.autocorr_interp_sq(r/np.nanmax(r),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_c_out)])
rhouci = np.vstack([sc.autocorr_interp_sq(r/np.sqrt(np.nanmax(ru)*np.nanmax(rc)),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,ru,rc,r in zip(tau_out, eta_out, r_u_out, r_c_out, r_uc_out)])


ru_mean = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
rc_mean = np.nanmean(rci.reshape((int(len(rci)/(N+1)),N+1,N+1)),axis=0)
ruc_mean = np.nanmean(ruci.reshape((int(len(ruci)/(N+1)),N+1,N+1)),axis=0)

susv = np.sqrt(np.nanmax(np.abs(ru_mean)))*np.sqrt(np.nanmax(np.abs(rc_mean)))
sc2 = np.nanmax(np.abs(rc_mean))

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(etai,taui, rc_mean/sc2,np.linspace(np.nanmin(rc_mean/sc2),np.nanmax(rc_mean/sc2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-1800, 1500,'(a)',fontsize=30,color='w')
ax1.set_xlim(-2500,2500)
ax1.set_ylim(-2500,2500)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{cc}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(etai[:,0], rc_mean[:,256].T/np.nanmax(rc_mean),'k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(taui[0,:], rc_mean[256,:].T/np.nanmax(rc_mean),'r',label='$\\rho_{u,u}(0,\\eta)$')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.set_xlim(-3000,3000)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# ax.set_xlim(-4000,4000)
fig.tight_layout()

k1r,k2r,Suur,Sccr, Sucr = sc.spectra_fft((taui,etai),ru_mean,rc_mean,ruc_mean)

sc.plot_log2D((k1r,k2r), np.abs(Sccr), label_S = "$\log_{10}{S_{div}}$", C = 10**-3, nl = 30, minS=-2.5)

fig1, ax1 = plt.subplots(figsize=(8, 8))
fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(k1r,k2r,np.log(np.abs(Svvr.T)),30, cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$k1\:[rad/m]$', fontsize=24)
ax1.set_ylabel('$k2\:[rad/m]$', fontsize=24)
ax1.set_xlim(-.05,.05)
ax1.set_ylim(-.05,.05)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\log_{10}{S_{vv}}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()



fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
# ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(k1r,k2r,np.imag(Suur),10, cmap='jet')
plt.colorbar(im1)

t, e, ru, rc, ruc = sc.spectra_fft(np.meshgrid(k1r,k2r),Suur,Sccr, Sucr,inv=True)

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)
im1 = ax1.contourf(e,t,rc/sc2, np.linspace(np.nanmin(rc/sc2),np.nanmax(rc/sc2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-3500, 3200,'(a)',fontsize=30,color='w')
ax1.set_xlim(-4000,4000)
ax1.set_ylim(-4000,4000)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{cc}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(e, rc[:,255].T/np.nanmax(rc),'k',label='$\\rho_{c,c}(\\tau,0)$')
ax.plot(t, rc[255,:].T/np.nanmax(rc),'r',label='$\\rho_{c,c}(0,\\eta)$')
ax.plot(etai[:,0], rc_mean[:,256].T/np.nanmax(rc_mean),'b',label='$\\rho_{c,c}(\\tau,0)$')
ax.plot(taui[0,:], rc_mean[256,:].T/np.nanmax(rc_mean),'g',label='$\\rho_{c,c}(0,\\eta)$')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{c,c}$',fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# ax.set_xlim(-4000,4000)
fig.tight_layout()


   
# In[]    
# ##############################################################################
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# # Enforce the margins, and enlarge them to give room for the vectors.
# ax.use_sticky_edges = False
# ax.margins(0.07)
# levels=np.linspace(10,21,20)
# _,_,U = sc.shrink(grd,U_out[0])
# grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[0])
# grd_shr = (grd_shr_x,grd_shr_y)

# #im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=levels,cmap='rainbow')
# im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
# divider = make_axes_locatable(ax)
# cb = fig.colorbar(im)
# cb.ax.set_ylabel("Wind speed [m/s]")
# #Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
# ax.set_xlabel(r'West-East [m]', fontsize=12, weight='bold')
# ax.set_ylabel(r'North-South [m]', fontsize=12, weight='bold') 

# def update(scan_n):   
#     ax.cla()      
#     from datetime import datetime, timedelta
#     date = datetime(1904,1,1) # January 1st, 1904 at midnight   
#     delta = timedelta(seconds = df_0.loc[df_0.scan==scan_n+9000].stop_time.max())   
#     newdate = date + delta    
#     ax.set_title(str(newdate))  
    
#     _,_,U = sc.shrink(grd,U_out[scan_n])
#     grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[scan_n])
#     grd_shr = (grd_shr_x,grd_shr_y)
#     im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
    
#     #check if there is more than one axes

    
#     return ax

# if __name__ == '__main__':
#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 200ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(650, 850), interval=500)
#     #if len(sys.argv) > 1 and sys.argv[1] == 'save':
#     anim.save('U_night.gif', dpi=80, writer='imagemagick')

##############################################################################
# In[]
import os

file_pre = '/case_'+pick_rnd.name.values[0]+pick_rnd.hms.values[0]
os.mkdir(file_out_path_u_field+file_pre)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)



umin = 3#np.nanmin(np.array(U_list))*0
umax = np.nanmax(np.array(U_list))  

a = .3
lx, ly = .1, .2
    
for j in range(len(U_list)):
    ax0.cla() 
    newdate = t_scan[j,0].strftime("%Y-%m-%d %H:%M:%S")
    newdate = newdate[:11]+ '\:'+newdate[11:]
    ax0.set_title('$'+str(newdate)+'$', fontsize = 20) 
    ax0.set_xlim(-7000, 0)
    ax0.set_ylim(-4000, 4000)
    # ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
    # ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
#     dudy, dudx = np.gradient(U_list[j], grdu[1][:,0], grdu[0][0,:]) 
#     dvdy, dvdx = np.gradient(V_list[j], grdu[1][:,0], grdu[0][0,:])    
#     contsh = dudx + dvdy
    
#     detected_peaks = detect_peaks(contsh)
# ##############################################################################
#     #     ind_peaks_pos = (img > .5*S_M*slim_peak) & detected_peaks
#     #     ind_peaks_neg = (img < .5*S_m*slim_peak) & detected_peaks
#     #     ind_peaks = ind_peaks_pos | ind_peaks_neg
#     #     max_vort = .5*S_M*s
#     #     min_vort = .5*S_m*s
#     #     peaks = img[ind_peaks]
#     #     levels = [min_vort,max_vort]
#     #     contours_min = measure.find_contours(img.T, min_vort)
#     #     contours_max = measure.find_contours(img.T, max_vort)
#     #     xc, yc = x[ind_peaks], y[ind_peaks]
#     #     im = ax.contourf(grd[0], grd[1], img, 5*np.linspace(S_m, S_M, 20), cmap='jet')
#     # #    im = ax.contourf(grd[0], grd[1], U, cmap='jet')
#     # #    im = ax.contourf(grd[0], grd[1], V, cmap='jet')
#     # #    #im = ax.contourf(grd[0], grd[1], dudxj+dvdyj, cmap='jet')
#     #     ax.set_title(str(su[j][0]))
#     #     # are peaks in the contours       
#     #     struct = []
#     #     strength = []
#     #     size = []
#     #     for contour in [contours_min, contours_max]:
#     #         for c in contour:
#     #             isin = False
#     #             cx, cy = c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0] 
#     #             if (len(cx)>=5): 
#     #                 pind = np.isnan(cx)|np.isnan(cy)
#     #                 p = geometry.Polygon([[xi, yi] for xi,yi in zip(cx[~pind],cy[~pind])])
#     #                 sgt = []
#     #                 for xi,yi,pk in zip(xc, yc, peaks):                          
#     #                     xy_point=geometry.Point(xi, yi)                        
#     #                     if p.contains(xy_point):
#     #                         isin = True
#     #                         sgt.append(np.c_[xi,yi,pk])                            
#     #                 if isin:
#     #                     struct.append(np.c_[cx[~pind],cy[~pind]])  
#     #                     size.append(np.c_[np.abs(np.max(cx[~pind])-np.min(cx[~pind])),np.abs(np.max(cy[~pind])-np.min(cy[~pind]))])
#     #                     plt.plot(cx[~pind],cy[~pind], 'k', lw= 3)
    
# ##############################################################################
       
#     grd_shr_x, grd_shr_y, U = sc.shrink(grdu, U_list[j])
#     grdus = (grd_shr_x, grd_shr_y)
    
#     grd_shr_x, grd_shr_y, contsh = sc.shrink(grdu, contsh)
#     grduc = (grd_shr_x, grd_shr_y)    
#     detected_peaks = detect_peaks(contsh, size = 10)   
#     contsh= filterfft(contsh,np.isnan(contsh),sigma=[ly*np.nanmean(Luy)/dy, lx*np.nanmean(Lux)/dx])    

    im0 = ax0.contourf(grdu[0], grdu[1], U_list[j], np.linspace(umin,umax,20), cmap='jet')
    
    # ax0.scatter(grduc[0][detected_peaks], grduc[1][detected_peaks])
    ax0.tick_params(axis='both', which='major', labelsize=24)
    ax0.set_xlabel('$x_1\:[m]$', fontsize = 24, weight='bold')
    ax0.set_ylabel('$x_2\:[m]$', fontsize = 24, weight='bold')
    
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
    plt.pause(.5) 
    
    
    # divider = make_axes_locatable(ax0)
    # cax= divider.append_axes("right", size="5%", pad=0.05)
    # cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
    # cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
    # cb.ax.tick_params(labelsize=24)
    # ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
    # fig0.tight_layout()
    # fig0.tight_layout()
    # plt.pause(.5)   
# In[]    
############################################################################

# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# # Enforce the margins, and enlarge them to give room for the vectors.
# ax.use_sticky_edges = False
# ax.margins(0.07)
# levels=np.linspace(10,21,20)
# _,_,U = sc.shrink(grd,U_out[0])
# grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[0])
# grd_shr = (grd_shr_x,grd_shr_y)

# #im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=levels,cmap='rainbow')
# im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
# divider = make_axes_locatable(ax)
# cb = fig.colorbar(im)
# cb.ax.set_ylabel("Wind speed [m/s]")
# #Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
# ax.set_xlabel(r'West-East [m]', fontsize=12, weight='bold')
# ax.set_ylabel(r'North-South [m]', fontsize=12, weight='bold') 


# def update(scan_n):   
#     ax.cla()   
    
#     from datetime import datetime, timedelta
    
#     date = datetime(1904,1,1) # January 1st, 1904 at midnight   
#     delta = timedelta(seconds = df_0.loc[df_0.scan==scan_n+9000].stop_time.max())   
#     newdate = date + delta
    
#     ax.set_title(str(newdate))  
    
#     _,_,U = sc.shrink(grd,U_out[scan_n])
#     grd_shr_x,grd_shr_y,V = sc.shrink(grd,V_out[scan_n])
#     grd_shr = (grd_shr_x,grd_shr_y)
#     im=ax.contourf(grd_shr[1].T,grd_shr[0].T,np.sqrt(U**2+V**2).T,levels,cmap='jet')
    
#     #check if there is more than one axes
#     if len(fig.axes) > 1: 
#         # if so, then the last axes must be the colorbar.
#         # we get its extent
#         pts = fig.axes[-1].get_position().get_points()
#         # and its label
#         label = fig.axes[-1].get_ylabel()
#         # and then remove the axes
#         fig.axes[-1].remove()
#         # then we draw a new axes a the extents of the old one
#         divider = make_axes_locatable(ax)
#         cax= divider.append_axes("right", size="5%", pad=0.05)
#         cb = fig.colorbar(im, cax=cax)
#         cb.ax.set_ylabel("Wind speed [m/s]")
#         ax.set_xlabel(r'West-East [m]', fontsize=12, weight='bold')
#         ax.set_ylabel(r'North-South [m]', fontsize=12, weight='bold') 
#         # unfortunately the aspect is different between the initial call to colorbar 
#         #   without cax argument. Try to reset it (but still it's somehow different)
#         #cbar.ax.set_aspect(20)
#     else:
#         fig.colorbar(im)
    
#     return ax

# if __name__ == '__main__':
#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 200ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(650, 850), interval=500)
#     #if len(sys.argv) > 1 and sys.argv[1] == 'save':
#     anim.save('U_night.gif', dpi=80, writer='imagemagick')


                 
# In[]

sync,time_s, _, _ = synch_df(df_0,df_1,dtscan=45/2)
ts = np.max(time_s, axis=1)
#Triangulation of the domain
S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]]) 
xtrans = (x[0]+x[-1])/2
ytrans = (y[0]+y[-1])/2
T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
Xx = np.array(np.c_[tri.x, tri.y, np.ones(tri.x.shape)]).T
Xx = np.dot(T,Xx)  
tri_dom = Delaunay(np.c_[Xx[0,:],Xx[1,:]])  
xp, yp = np.array([dr0[0], dr1[0]]), np.array([dr0[1], dr1[1]])  
ts1 = np.array([s[:4]+'-'+s[4:6]+'-'+ s[6:8]+' '+s[8:10]+':'+s[10:] for s in stab_20hz_1.Name.values[::len(stab_20hz_1.Name.values)-1]])
ts1 = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M') for s in ts1])
ts1[-1] = ts1[-1] + timedelta(seconds=10*60-1/20)
ts1 = np.unique(ts1)
ts1 = pd.date_range(ts1[0], ts1[-1], freq='.05S') 

####################################
# Pointwise interpolation
xtg, ytg, ttg, tout, Uext, Vext, _ = expand_field_point_time(grdu, U_list, V_list, [dr0, dr1], tree=True,
                            grid_i = (xp, yp), beta = 1, al = 5, Luy = Lux, time_scan = time_s,
                            t_int = ts1, interp_t = True, tri_dom = tri_dom, t_scale = 1e6)
####################################
figu0,axu0 = plt.subplots()
figv0,axv0 = plt.subplots()
figu1,axu1 = plt.subplots()
figv1,axv1 = plt.subplots()

col = 'r'
h = 1
dxy = np.sqrt(dx**2+dy**2) 
fs = np.mean(Uext.flatten())/dxy
fa , fb = 0, fs/100
S11 = np.cos(gamma)
S12 = np.sin(gamma)
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])

ux0 = stab_20hz_0[Y_0].values[:,h]
uy0 = stab_20hz_0[X_0].values[:,h]
uz0 = stab_20hz_0[Z_0].values[:,h]
vel0 = np.array(np.c_[ux0, uy0]).T
vel0 = np.dot(R[:-1,:-1],vel0)
ux0 = vel0[0,:]
uy0 = vel0[1,:]
filtu0,H_w,h_n,ff = kaiser_filter(ux0, 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
filtv0,H_w,h_n,ff = kaiser_filter(uy0, 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
filtw0,H_w,h_n,ff = kaiser_filter(uz0, 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
ux1 = stab_20hz_1[Y_1].values[:,h]
uy1 = stab_20hz_1[X_1].values[:,h]
uz1 = stab_20hz_1[Z_1].values[:,h]
vel1 = np.array(np.c_[ux1, uy1]).T
vel1 = np.dot(R[:-1,:-1],vel1)
ux1 = vel1[0,:]
uy1 = vel1[1,:]
filtu1,H_w,h_n,ff = kaiser_filter(ux1, 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
filtv1,H_w,h_n,ff = kaiser_filter(uy1, 20, fa, fb, width=.001, A_p = .01, A_s = 80)  

filtu1g,H_w,h_n,ff = kaiser_filter(Uext[1,:], 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
filtv1g,H_w,h_n,ff = kaiser_filter(Vext[1,:], 20, fa, fb, width=.001, A_p = .01, A_s = 80) 

filtu0g,H_w,h_n,ff = kaiser_filter(Uext[0,:], 20, fa, fb, width=.001, A_p = .01, A_s = 80)  
filtv0g,H_w,h_n,ff = kaiser_filter(Vext[0,:], 20, fa, fb, width=.001, A_p = .01, A_s = 80) 

filtw1,H_w,h_n,ff = kaiser_filter(uz1, 20, fa, fb, width=.001, A_p = .01, A_s = 80) 

ts0 = np.array([s[:4]+'-'+s[4:6]+'-'+ s[6:8]+' '+s[8:10]+':'+s[10:] for s in stab_20hz_0.Name.values[::len(stab_20hz_0.Name.values)-1]])
ts0 = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M') for s in ts0])
ts0[-1] = ts0[-1] + timedelta(seconds=10*60-1/20)
ts0 = np.unique(ts0)
ts0 = pd.date_range(ts0[0], ts0[-1], freq='.05S')
filtu0 = sample(filtu0,ts0,tout[0,:])
filtv0 = sample(filtv0,ts0,tout[0,:])
filtw0 = sample(filtw0,ts0,tout[0,:])

filtu1 = sample(filtu1,ts1,tout[1,:])
filtv1 = sample(filtv1,ts1,tout[1,:])
filtw1 = sample(filtw1,ts1,tout[1,:])


figw1,axw1 = plt.subplots()
axw1.plot(tout[1,:],filtw0,'.-', label=r'$w\:sonic,\:175m$')
axw2 = axw1.twinx()
axw2.plot(tout[1,:],filtu0,'.-', color = 'k', label=r'$u\:sonic,\:175m$')
axw2.plot(tout[1,:]-timedelta(seconds=45),filtu0g,'.-',lw=3,color = col, label=r'$w\:scans,\:175m$')
axw1.set_xlabel(r'$time$', fontsize=20)
axw1.set_ylabel(r'$w\:[m/s]$', fontsize=20)
axw2.set_ylabel(r'$u\:[m/s]$', fontsize=20)
axw1.tick_params(axis='both', which='major', labelsize=20)
axw2.tick_params(axis='both', which='major', labelsize=20)
figw1.legend(loc = 'upper center',fontsize=20)
figw1.tight_layout()

fw1,a1 = plt.subplots()
a2 = a1.twinx()
for hh in range(len(heights)-2):
    uz1 = stab_20hz_0[Z_0].values[:,hh]
    filtw1,_,_,_ = kaiser_filter(uz1, 20, fa, fb, width=.001, A_p = .01, A_s = 80) 
    filtw1 = sample(filtw1,ts0,tout[1,:])
    a1.plot(tout[1,:],filtw1,'.-', label=r'$w\:sonic,\:' + str(heights[hh])+'m$')
    a1.set_xlabel(r'$time$', fontsize=20)
    a1.set_ylabel(r'$w\:[m/s]$', fontsize=20)
fw1.legend(loc = 'upper center',fontsize=20)
# filtc,_,_,_ = kaiser_filter(contgs[~np.isnan(contgs)], fs/10, fa, fb, width=.001, A_p = .01, A_s = 80) 
# filtc = filtc[int(len(filtc)/2-len(tgs[~np.isnan(contgs)])/2):int(len(filtc)/2-len(tgs[~np.isnan(contgs)])/2)*2]
a2.plot(tgs[~np.isnan(contgs)]-timedelta(seconds=45),contgs[~np.isnan(contgs)], 'o', color='red')
a2.tick_params(axis='both', which='major', labelsize=20)
a2.set_ylabel(r'$div_h\:[1/s]$', fontsize=20)

axu0.plot(tout[0,:],Uext[0,:], '.-', lw=3,color = col)
axu0.plot(tout[0,:],filtu0, '.-', label=r'$From\:sonic\:at\:175m$')
axu0.set_xlabel(r'$time\:[s]$',fontsize=20)
axu0.set_ylabel(r'$Wind\:speed\:[m/s]$',fontsize=20)
axv0.plot(tout[0,:],Vext[0,:],'.-',lw=3,color = col)
axv0.plot(tout[0,:],filtv0,'.-', label=r'$From\:sonic\:at\:175m$')
axv0.set_xlabel(r'$time\:[s]$',fontsize=20)
axv0.set_ylabel(r'$Wind\:speed\:[m/s]$',fontsize=20)
axu1.plot(tout[1,:],Uext[1,:],'.-',lw=3,color = col)
axu1.plot(tout[1,:],filtu1,'.-', label=r'$From\:sonic\:at\:175m$')

axu1.set_xlabel(r'$time\:[s]$',fontsize=20)
axu1.set_ylabel(r'$Wind\:speed\:[m/s]$',fontsize=20)
axv1.plot(tout[1,:],Vext[1,:],'.-',lw=3,color = col)
axv1.plot(tout[1,:],filtv1,'.-', label=r'$From\:sonic\:at\:175m$')
axv1.set_xlabel(r'$time\:[s]$',fontsize=20)
axv1.set_ylabel(r'$Wind\:speed\:[m/s]$',fontsize=20)



######################################

xg, yg, Uex_t, Vex_t,t_ext, points = expand_field_point_time(grdu, U_list, V_list,
                                                   [dr0, dr1], Luy = 3*Lux, nx=1000,
                                                   n = 36, alpha = [0, 2*np.pi],
                                                   time_scan = time_s, t_int = tout[0,-1],
                                                   beta = 2, tri_dom = tri_dom)

dx = grdu[0][0,1]-grdu[0][0,0]
dy = grdu[1][1,0]-grdu[1][0,0]
lx, ly = .1, .1
Lx, Ly = 1, 2
dudy, dudx = np.gradient(filterfft(Uex_t,np.isnan(Uex_t), [ly*np.nanmean(Luy)/dy, lx*np.nanmean(Lux)/dx]), yg[:,0], xg[0,:]) 
dvdy, dvdx = np.gradient(filterfft(Vex_t,np.isnan(Vex_t), [ly*np.nanmean(Luy)/dy, lx*np.nanmean(Lux)/dx]), yg[:,0], xg[0,:])  
contsh = dudx + dvdy
vort = dvdx-dudy
ums = filterfft(Uex_t,np.isnan(Uex_t),[Ly*np.nanmean(Luy)/dy, Lx*np.nanmean(Lux)/dx])
vms = filterfft(Vex_t,np.isnan(Vex_t),[Ly*np.nanmean(Luy)/dy, Lx*np.nanmean(Lux)/dx])
a = .12  
b = 1
                                                  
fig, ax = plt.subplots(2,1,figsize=(8, 8))
ax[0].set_aspect('equal')
ax[0].use_sticky_edges = False
ax[0].margins(0.07)
ax[1].set_aspect('equal')
ax[1].use_sticky_edges = False
ax[1].margins(0.07)
im0 = ax[0].contourf(xg, yg, contsh, 50,cmap='bwr',norm=MidpointNormalize(midpoint=0,
                                                                      vmin = np.nanmin(contsh)*b,
                                                                      vmax=np.nanmax(contsh)*b))   
ax[0].contour(xg, yg, Uex_t-ums, levels = [np.nanmin(Uex_t-ums)*a], colors = 'k', linewidths = 2, linestyles = 'solid') 
im1 = ax[1].contourf(xg,yg,Uex_t-ums,50,cmap='bwr', norm=MidpointNormalize(midpoint=0,
                                                                      vmin = np.nanmin(Uex_t-ums),
                                                                      vmax=np.nanmax(Uex_t-ums)))
ax[1].scatter(dr1[0],dr1[1], marker = '+', color = 'red', s = 300)
ax[1].scatter(dr0[0],dr0[1], marker = '+', color = 'red', s = 300)


ax[0].set_xlabel(r'$x_1\:[m]$',fontsize=20)
ax[0].set_ylabel(r'$x_2\:[m]$',fontsize=20)
ax[1].set_xlabel(r'$x_1\:[m]$',fontsize=20)
ax[1].set_ylabel(r'$x_2\:[m]$',fontsize=20)

divider = make_axes_locatable(ax[0])
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm2))
cb.ax.set_ylabel(r"$div_h\:[1/s]$",fontsize=24)

divider = make_axes_locatable(ax[1])
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$u_1\:[m/s]$",fontsize=24)

########################################################
dudyh, dudxh = np.gradient(Uex_t, yg[:,0], xg[0,:]) 
dvdyh, dvdxh = np.gradient(Vex_t, yg[:,0], xg[0,:])  
contshh = dudxh + dvdyh
h, bins, patch = plt.hist(contshh[~np.isnan(contshh)],bins=100, alpha =.5, density=True, label=r'$Advected$')
contshs = []
for j in range(len(U_list)):

    dudys, dudxs = np.gradient(U_list[j], grdu[1][:,0], grdu[0][0,:]) 
    dvdys, dvdxs = np.gradient(V_list[j], grdu[1][:,0], grdu[0][0,:])  
    contshs.append((dudxs + dvdys).flatten())

contshs = np.array(contshs)
plt.hist(contshs[~np.isnan(contshs)],bins=bins, alpha =.5, density=True, label=r'$Scans$')

plt.xlim(-.15,.15)
plt.xlabel(r'$div_h$', fontsize=20)
plt.legend(fontsize=20)

div = np.linspace(-3,3,80)
pdf_adv = sp.stats.gaussian_kde(contshh[~np.isnan(contshh)]/np.std(contshh[~np.isnan(contshh)]))
yadv = pdf_adv(div)
pdf_scans = sp.stats.gaussian_kde(contshs[~np.isnan(contshs)]/np.std(contshs[~np.isnan(contshs)]))
yscans = pdf_scans(div)

plt.figure()
plt.plot(div,yadv, lw=2, color = 'r', label=r'$Advected$')
plt.plot(div,yscans, lw=2, color = 'k', label=r'$Scans$')
#plt.yscale('log')
plt.xlabel(r'$div_h/\sigma$', fontsize=20)
plt.legend(fontsize=20)
plt.plot(div, np.exp(-.5*div**2)/np.sqrt(2*np.pi),'--b', lw = 2)

########################################################

tgs = t_ext[np.argmin(np.abs(yg[:,0]-dr0[1])),:]
contgs = contsh[np.argmin(np.abs(yg[:,0]-dr0[1])),:]
contgs = contgs[~np.isnan(tgs)]
Ugs = Uex_t[np.argmin(np.abs(yg[:,0]-dr0[1])),:]
Ugs = Ugs[~np.isnan(tgs)]
Vgs = Vex_t[np.argmin(np.abs(yg[:,0]-dr0[1])),:]
Vgs = Vgs[~np.isnan(tgs)]
tgs = tgs[~np.isnan(tgs)]
Ugs = Ugs[np.argsort(tgs)]
contgs = contgs[np.argsort(tgs)]
Vgs = Vgs[np.argsort(tgs)]
tgs = tgs[np.argsort(tgs)]
tgs = np.array([timedelta(seconds=ti) for ti in tgs])
tgs = tgs+date
plt.figure()
plt.plot(tgs,Ugs)
plt.plot(tout[1,:],Uext[1,:],'.-',lw=3,color = col)
plt.plot(tout[1,:],filtu1,'.-', label=r'$From\:sonic\:at\:175m$')

plt.figure()
plt.plot(tgs,Vgs)
plt.plot(tout[1,:],Vext[1,:],'.-',lw=3,color = col)
plt.plot(tout[1,:],filtv1,'.-', label=r'$From\:sonic\:at\:175m$')


# fig.colorbar(im0,ax=ax[0])
# fig.colorbar(im1,ax=ax[1])

# Interpolate line 
# Southern mast at 200m
ymast = yg.flatten()[np.argmin(np.abs(yg.flatten()-dr1[1]))]
line = np.c_[np.unique(xg.flatten()), (dr1[1]-0)*np.ones(len(np.unique(xg.flatten())))]

#Uline = sp.interpolate.interp2d(xg[0,:], yg[:,0], Uex_t)(line[:,0], dr0[1])

Uex_t_0 = Uex_t
Uex_t_0[np.isnan(Uex_t_0)] = 0
Vex_t_0 = Vex_t
Vex_t_0[np.isnan(Vex_t_0)] = 0
Uline = sp.interpolate.RectBivariateSpline(yg[:,0], xg[0,:],Uex_t_0)(line[:,1], line[:,0], grid=False)
Vline = sp.interpolate.RectBivariateSpline(yg[:,0], xg[0,:],Vex_t_0)(line[:,1], line[:,0], grid=False)
indline = Uline < 10
Uline[indline] = np.nan
Vline[indline] = np.nan
plt.figure()
plt.plot(xm[:,0], u_s_0, alpha=.5)
#plt.plot(xm[:,1], u_s_1[::-1,1], alpha=.5)
plt.plot(line[:,0],Uline, 'or')

plt.figure()
plt.plot(xm[:,1], u_s_0[:,1], alpha=.5)
#plt.plot(xm[:,1], u_s_1[::-1,1], alpha=.5)
plt.plot(line[:,0],Vline, 'or')


ax[1].scatter(line[:,0], line[:,1], s= 10)
plt.figure()
plt.plot(line[:,0],Vline)



# In[]
# Autocorrelation

# # In[]
# j = 18

# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# for j in range(len(U_out_u)):
#     ax0.cla()
#     im0 = ax0.contourf(grdu[0], grdu[1], U_out_u[j], np.linspace(np.nanmin(U_out_u[j]),np.nanmax(U_out_u[j]),10), cmap='jet')
#     ax0.contour(grdu[0], grdu[1], V_out_u[j]-np.nanmean(V_out_u[j]), levels = [-1], colors = 'k', linewidths = 1.5)
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
#     ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
# #    divider = make_axes_locatable(ax0)
# #    cax= divider.append_axes("right", size="5%", pad=0.05)
# #    cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# #    cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
# #    cb.ax.tick_params(labelsize=24)
# #    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
# #    fig0.tight_layout()
# #    fig0.tight_layout()
#     plt.pause(.5)

# fig1, ax1 = plt.subplots(figsize=(8, 8))
# ax1.set_aspect('equal')
# ax1.use_sticky_edges = False
# ax1.margins(0.07)
# for j in range(len(U_out_c)):
#     ax1.cla()
#     im1 = ax1.contourf(grdc[0], grdc[1], V_out_c[j], np.linspace(np.nanmin(V_out_c[j]),np.nanmax(V_out_c[j]),10), cmap='jet')
# #    ax1.contour(grdc[0], grdc[1], V_out_c[j]-np.nanmean(V_out_c[j]), levels = [-1], colors = 'k', linewidths = 1.5)
#     ax1.tick_params(axis='both', which='major', labelsize=24)
#     ax1.set_xlabel('$Easting\:[m]$', fontsize=24)
#     ax1.set_ylabel('$Northing\:[m]$', fontsize=24)
#     plt.pause(.5)
# divider = make_axes_locatable(ax1)
# cax= divider.append_axes("right", size="5%", pad=0.05)
# cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
# cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
# cb.ax.tick_params(labelsize=24)
# ax1.text(-6000, 2500,'(a)',fontsize=30,color='k')
# fig1.tight_layout()
# fig1.tight_layout()

# #########################################################################

# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# for j in range(len(U_out_u)):
#     ax0.cla()
#     im0 = ax0.contourf(grdu[0], grdu[1], V_out_u[j], np.linspace(np.nanmin(V_out_u[j]),np.nanmax(V_out_u[j]),10), cmap='jet')
#     ax0.contour(grdu[0], grdu[1], U_out_u[j]-np.nanmean(U_out_u[j]), levels = [-1], colors = 'k', linewidths = 1.5)
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
#     ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
#     plt.pause(.5)

# divider = make_axes_locatable(ax0)
# cax= divider.append_axes("right", size="5%", pad=0.05)
# cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
# cb.ax.tick_params(labelsize=24)
# ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
# fig0.tight_layout()
# fig0.tight_layout()

# #########################################################

# L2 = np.abs(grdu[0][0,0]-grdu[0][0,-1])
# ds2 = np.sqrt(np.diff(grdu[0][0,:])[0]**2+np.diff(grdu[1][:,0])[0]**2)
# s_i2 = [np.min(dfL_phase2.loc[(dfL_phase2.scan==s)][['$L_{u,x}$','$L_{u,y}$']].values)/ds2 for s in s_syn[:,0]]
# L_s2 = np.squeeze(np.array([dfL_phase2.loc[(dfL_phase2.scan==s)][['$L_{u,x}$','$L_{u,y}$']].values for s in s_syn[:,0]]))

# #indu = (ush<6) | (vsh<-3.5)
# #ush[indu] = np.nan
# #vsh[indu] = np.nan
# a=.2
# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# minU = np.nanmin(np.array(U_out_u))
# maxU = np.nanmax(np.array(U_out_u))*.8
# for j in range(len(U_out_u)):
    
#     ax0.cla()
    
#     dudy, dudx = np.gradient(U_out_u[j], grdu[1][:,0], grdu[0][0,:]) 
#     dvdy, dvdx = np.gradient(V_out_u[j], grdu[1][:,0], grdu[0][0,:])    
#     contsh = dudx + dvdy
# #    vortsh = dvdx - dudy 
#     #sd = np.nanmedian(np.array(s_i2))
#     contsh= filterfft(contsh,np.isnan(contsh),sigma=s_i2[j]/4)
# #    vortsh= filterfft(vortsh,np.isnan(vortsh),sigma=sd/4)
    
#     im0 = ax0.contourf(grdu[0], grdu[1], U_out_u[j], np.linspace(minU,maxU,10), cmap='jet')
# #    im0 = ax0.contourf(grdu[0], grdu[1], V_out_u[j], 10, cmap='jet')

# #    im0 = ax0.quiver(grdu[0], grdu[1], U_out_u[j]-np.nanmean(U_out_u[j]), V_out_u[j]-np.nanmean(V_out_u[j]), scale=150, alpha=.3)
    
# #    im0 = ax0.contourf(grdu[0], grdu[1], contsh, 10, cmap='jet')
#     ax0.contour(grdu[0], grdu[1], contsh, levels = [np.nanmin(contsh)*a], colors = 'k', linewidths = 2, linestyles = 'solid')
#     ax0.contour(grdu[0], grdu[1], contsh, levels = [np.nanmax(contsh)*a], colors = 'r', linewidths = 2, linestyles = 'solid')
# #    ax0.contour(grdc[0], grdc[1], U_out_u[j]-np.nanmean(U_out_u[j]), levels = [-1], colors = 'k', linewidths = 1.5)
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
#     ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
#     plt.pause(1)

# divider = make_axes_locatable(ax0)
# cax= divider.append_axes("right", size="5%", pad=0.05)
# cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
# cb.ax.tick_params(labelsize=24)
# ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
# fig0.tight_layout()
# fig0.tight_layout()

# # In[]

# tri_calc = True
# chunk = 512

# x = grdu[0][0,:]
# y = grdu[1][:,0]

# # Mean wind speed and
# U_arr = np.vstack([U_out_u[i] for i in range(len(U_out_u))])
# V_arr = np.vstack([V_out_u[i] for i in range(len(V_out_u))])
# scan = s_syn[:,0]


# U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
# V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)

# u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc)

# u, v = U_rot(grdu, U, V, gamma = gamma, tri_calc = False, 
#                           tri_del = tri_del, mask_int = mask_int, mask = mask)

# In[autocorrelation]

tau_out = []
eta_out = []    
r_u_out = []
r_c_out = []
r_uc_out = []



for j in range(len(U_list)):
 

    tau,eta,r_u,r_c,r_uc,_,_,_,_ = sc.spatial_autocorr_sq(grdu,U,contsh,
                                                          transform = False,
                                                          transform_r = False,
                                                          e_lim=.1,refine=32)
    tau_out.append(tau.astype(np.float32))
    eta_out.append(eta.astype(np.float32)) 
    r_u_out.append(r_u.astype(np.float32))
    r_c_out.append(r_c.astype(np.float32))
    r_uc_out.append(r_uc.astype(np.float32))


# N = 512
# taumax = np.nanmax(np.abs(np.array(tau_out)))
# taui = np.linspace(0,taumax,int(N/2)+1)
# taui = np.r_[-np.flip(taui[1:]),taui]
# etamax = np.nanmax(np.abs(np.array(eta_out)))
# etai = np.linspace(0,etamax,int(N/2)+1)
# etai = np.r_[-np.flip(etai[1:]),etai]
# taui, etai = np.meshgrid(taui,etai)

# rui = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_u_out)])
# rci = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_c_out)])
# ruci = np.vstack([sc.autocorr_interp_sq(r,e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_uc_out)])

# rhoui = np.vstack([sc.autocorr_interp_sq(r/np.nanmax(r),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_u_out)])
# rhoci = np.vstack([sc.autocorr_interp_sq(r/np.nanmax(r),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_out, eta_out, r_c_out)])
# rhouci = np.vstack([sc.autocorr_interp_sq(r/np.sqrt(np.nanmax(ru)*np.nanmax(rc)),e,t,tau_lin = taui, eta_lin = etai)[2] for t,e,ru,rc,r in zip(tau_out, eta_out, r_u_out, r_c_out, r_uc_out)])


# ru_mean = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
# rc_mean = np.nanmean(rci.reshape((int(len(rci)/(N+1)),N+1,N+1)),axis=0)
# ruc_mean = np.nanmean(ruci.reshape((int(len(ruci)/(N+1)),N+1,N+1)),axis=0)

# rhou_mean = np.nanmean(rhoui.reshape((int(len(rhoui)/(N+1)),N+1,N+1)),axis=0)
# rhoc_mean = np.nanmean(rhoci.reshape((int(len(rhoci)/(N+1)),N+1,N+1)),axis=0)
# rhouc_mean = np.nanmean(rhouci.reshape((int(len(rhouci)/(N+1)),N+1,N+1)),axis=0)

# ru_std = np.nanstd(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
# rc_std = np.nanstd(rci.reshape((int(len(rci)/(N+1)),N+1,N+1)),axis=0)
# ruc_std = np.nanstd(ruci.reshape((int(len(ruci)/(N+1)),N+1,N+1)),axis=0)

# rhou_std = np.nanstd(rhoui.reshape((int(len(rhoui)/(N+1)),N+1,N+1)),axis=0)
# rhoc_std = np.nanstd(rhoci.reshape((int(len(rhoci)/(N+1)),N+1,N+1)),axis=0)
# rhouc_std = np.nanstd(rhouci.reshape((int(len(rhouci)/(N+1)),N+1,N+1)),axis=0)


# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# im0 = ax0.contourf(tau_out[j], eta_out[j], r_uc_out[j]/np.sqrt(np.nanmax(r_u_out[j])*np.nanmax(r_c_out[j])), 10, cmap='jet')
# #im0 = ax0.contourf(tau_out[0], eta_out[0], r_u_out[0]/np.nanmax(r_u_out[0]), 10, cmap='jet')
# #im0 = ax0.contourf(tau_out[j], eta_out[j], r_c_out[j]/np.nanmax(r_c_out[j]), 10, cmap='jet')
# divider = make_axes_locatable(ax0)
# cax= divider.append_axes("right", size="5%", pad=0.05)
# cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# cb.ax.set_ylabel(r"$\rho$",fontsize=24)
# cb.ax.tick_params(labelsize=24)


# fig, ax = plt.subplots(figsize=(8, 8))
# #fig.set_size_inches(8,8)
# ax.use_sticky_edges = False
# ax.margins(0.07)
# #ax.plot(tau_out[0][0,:], r_uc_out[0][67,:],'k',label='$\\rho_{u,c}(\\tau,0)$')
# #ax.plot(eta_out[0][:,0], r_uc_out[0][:,79],'r',label='$\\rho_{u,c}(0,\\eta)$')
# ax.plot(taui[0,:], ruci[0][256,:],'k',label='$\\rho_{u,c}(\\tau,0)$')
# ax.plot(etai[:,0], ruci[0][:,256],'r',label='$\\rho_{u,c}(0,\\eta)$')
# ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
# ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
# ax.legend(fontsize=24)
# ax.tick_params(axis='both', which='major', labelsize=24)
# ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# ax.set_xlim(-4000,4000)
# fig.tight_layout()




# #ax0.set_xlim(-1000,1000)
# #ax0.set_ylim(-1000,1000)

# fig, ax = plt.subplots(figsize=(8, 8))
# #fig.set_size_inches(8,8)
# ax.use_sticky_edges = False
# ax.margins(0.07)
# ax.plot(taui[0,:], rhouc_std[256,:],'-r',label='$\\rho_{u,u}(\\tau,0)$')#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--r',label='$\\rho_{u,u}(0,\\eta)$')
# ax.plot(etai[:,0], rhouc_std[:,256],'-k',label='$\\rho_{u,u}(0,\\eta)$')#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.legend(fontsize=24)
# ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
# ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
# ax.tick_params(axis='both', which='major', labelsize=24)
# ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# fig.tight_layout()
# ax.plot([0,0],[-.05,.06],'--k')
# ax.plot(etai[:,0], rhouc_mean[:,256]+rhouc_std[:,256],'--k',label='$\\rho_{u,u}(0,\\eta)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(etai[:,0], rhouc_mean[:,256]-rhouc_std[:,256],'--k',label='$\\rho_{u,u}(0,\\eta)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(taui[0,:], rhouc_mean[256,:]+rhouc_std[256,:],'--r',label='$\\rho_{u,u}(\\tau,0)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(taui[0,:], rhouc_mean[256,:]-rhouc_std[256,:],'--r',label='$\\rho_{u,u}(\\tau,0)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')


# fig, ax = plt.subplots(figsize=(8, 8))
# #fig.set_size_inches(8,8)
# ax.use_sticky_edges = False
# ax.margins(0.07)
# ax.plot(taui[0,:], rhouc_mean[256,:],'-r',label='$\\rho_{u,u}(\\tau,0)$')#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--r',label='$\\rho_{u,u}(0,\\eta)$')
# ax.plot(etai[:,0], rhouc_mean[:,256],'-k',label='$\\rho_{u,u}(0,\\eta)$')#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.legend(fontsize=24)
# ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
# ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
# ax.tick_params(axis='both', which='major', labelsize=24)
# ax.text(-3700, .95,'(a)',fontsize=30,color='k')
# fig.tight_layout()
# ax.plot([0,0],[-.05,.06],'--k')
# ax.plot(etai[:,0], rhouc_mean[:,256]+rhouc_std[:,256],'--k',label='$\\rho_{u,u}(0,\\eta)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(etai[:,0], rhouc_mean[:,256]-rhouc_std[:,256],'--k',label='$\\rho_{u,u}(0,\\eta)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(taui[0,:], rhouc_mean[256,:]+rhouc_std[256,:],'--r',label='$\\rho_{u,u}(\\tau,0)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')
# ax.plot(taui[0,:], rhouc_mean[256,:]-rhouc_std[256,:],'--r',label='$\\rho_{u,u}(\\tau,0)$',alpha=.2)#/np.sqrt(np.nanmax(ru_mean)*np.nanmax(rc_mean)),'--k',label='$\\rho_{u,u}(\\tau,0)$')

# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# for j in range(len(tau_out)):
#     ax0.cla()
#     #im0 = ax0.contourf(tau_out[0], eta_out[0], r_u_out[0]/np.nanmax(r_u_out[0]), 10, cmap='jet')
#     #im0 = ax0.contourf(tau_out[j], eta_out[j], r_c_out[j]/np.nanmax(r_c_out[j]), 10, cmap='jet')
#     im0 = ax0.contourf(tau_out[j], eta_out[j], r_uc_out[j]/np.sqrt(np.nanmax(r_u_out[j])*np.nanmax(r_c_out[j])), 10, cmap='jet')
#     ax0.set_xlim(-500,500)
#     ax0.set_ylim(-500,500)
#     plt.pause(1)

# divider = make_axes_locatable(ax0)
# cax= divider.append_axes("right", size="5%", pad=0.05)
# cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# cb.ax.set_ylabel(r"$\rho$",fontsize=24)
# cb.ax.tick_params(labelsize=24)









# In[MySQL Østerild sonic and lidar data]
# t1 = pd.to_datetime(pick_rnd.time1.values)

# minutes = 1*60

# t_init1 = t1 - timedelta(minutes=minutes)
# t_end1 = t1 + timedelta(minutes=minutes)
# hms_i_1 = t_init1.strftime("%H%M%S").values[0]
# hms_e_1 = t_end1.strftime("%H%M%S").values[0]
# dyi = t_init1.strftime("%Y%m%d").values[0]
# dye = t_end1.strftime("%Y%m%d").values[0]


# t_0_1 = datetime(int(dyi[:4]), int(dyi[4:6]), int(dyi[6:]))
# t_0_1 = t_0_1+timedelta(seconds = int(hms_i_1[4:]))
# t_0_1 = t_0_1+timedelta(minutes = int(hms_i_1[2:4]))
# t_0_1 = t_0_1+timedelta(hours = int(hms_i_1[:2]))
# t_0_1 = str((t_0_1-date).total_seconds())
# #
# t_1_1 = datetime(int(dye[:4]), int(dye[4:6]), int(dye[6:]))
# t_1_1 = t_1_1+timedelta(seconds = int(hms_e_1[4:]))
# t_1_1 = t_1_1+timedelta(minutes = int(hms_e_1[2:4]))
# t_1_1 = t_1_1+timedelta(hours = int(hms_e_1[:2]))
# t_1_1 = str((t_1_1-date).total_seconds())


# osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')

# #pick_rnd = dfL_phase2.loc[luxluy & rel].sample()
# t0 = pd.to_datetime(pick_rnd.time0.values)
# t1 = pd.to_datetime(pick_rnd.time1.values)

# dy = t0.strftime("%Y%m%d").values[0]

# t_init0 = t0 - timedelta(minutes=minutes)
# t_end0 = t0 + timedelta(minutes=minutes)
# hms_i_0 = t_init0.strftime("%H%M%S").values[0]
# hms_e_0 = t_end0.strftime("%H%M%S").values[0]

# t_init1 = t1 - timedelta(minutes=minutes)
# t_end1 = t1 + timedelta(minutes=minutes)
# hms_i_1 = t_init1.strftime("%H%M%S").values[0]
# hms_e_1 = t_end1.strftime("%H%M%S").values[0]

# stampi = dyi+hms_i_1
# stampe = dye+hms_e_1


# Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
# Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
# Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
# Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
# Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
# Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
# Sdir_1 = ['Sdir_241m_LMN','Sdir_175m_LMN','Sdir_103m_LMN','Sdir_37m_LMN','Sdir_7m_LMN']
# Name = ['Name']
# T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
# T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
# X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
# X_1 = ['X_241m_LMN','X_175m_LMN','X_103m_LMN','X_37m_LMN','X_7m_LMN']
# Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
# Y_1 = ['Y_241m_LMN','Y_175m_LMN','Y_103m_LMN','Y_37m_LMN','Y_7m_LMN']
# Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
# Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']
# P_0 = ['Press_241m_LMS','Press_7m_LMS']
# P_1 = ['Press_241m_LMN','Press_7m_LMN']

# #for j,stamp in enumerate(t_ph1):
# #    print(stamp,j)    
# table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
# where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stampi[:-2] + ')'# & name <= (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2]
# query_name_i = 'select name' + table_20Hz + where + ' limit 1'
# where = 'where name = (select min(name)' + table_20Hz + 'where name > ' + stampe[:-2] + ')'
# query_name_e = 'select name' + table_20Hz + where + ' limit 1'
# name_i = pd.read_sql_query(query_name_i,osterild_database).values[0]
# name_e = pd.read_sql_query(query_name_e,osterild_database).values[0] 
# where = 'where name > ' + name_i + ' and name < ' + name_e
# sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Name) + table_20Hz +  where
# stab_20hz_0 = pd.read_sql_query(sql_query[0],osterild_database)  
# stab_20hz_0.fillna(value=pd.np.nan, inplace=True)
# sql_query = 'select ' + ", ".join(T_1+X_1+Y_1+Z_1+Sspeed_1+Sdir_1+Name) + table_20Hz +  where
# stab_20hz_1 = pd.read_sql_query(sql_query[0],osterild_database)  
# stab_20hz_1.fillna(value=pd.np.nan, inplace=True)


# gammax = np.arctan2(np.nanmean(stab_20hz_1[Y_1].values[:,0]),np.nanmean(stab_20hz_1[X_1].values[:,0]))

# S11 = np.cos(gamma)
# S12 = np.sin(gamma)
# R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
# vel = np.c_[stab_20hz_0[X_0].values[:,0], stab_20hz_0[Y_0].values[:,0]].T
# vel = np.dot(R[:-1,:-1],vel)
# u_s_0 = vel[0,:]#.reshape(stab_20hz_0[X_0].values.shape)
# v_s_0 = vel[1,:]#.reshape(stab_20hz_0[X_0].values.shape)
# vel = np.c_[stab_20hz_1[X_1].values[:,0], stab_20hz_1[Y_1].values[:,0]].T
# vel = np.dot(R[:-1,:-1],vel)
# u_s_1 = vel[0,:]#.reshape(stab_20hz_1[X_1].values.shape)
# v_s_1 = vel[1,:]#.reshape(stab_20hz_1[X_1].values.shape)


# # u_s_0 = np.sqrt(stab_20hz_0[X_0].values**2+stab_20hz_0[Y_0].values**2)
# # u_s_1 = np.sqrt(stab_20hz_1[X_1].values**2+stab_20hz_1[Y_1].values**2)
# w_s_0 = stab_20hz_0[Z_0].values
# w_s_1 = stab_20hz_1[Z_1].values
# h, xm = np.meshgrid(heights, np.arange(u_s_0.shape[0])*np.mean(u_s_0)/20)



# ### labels
# iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
# labels = iden_lab
# labels_new = iden_lab
# #Labels for range gates and speed
# labels_mask = []
# labels_ws = []
# labels_rg = []
# labels_CNR = []
# for i in np.arange(3):
#     vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
#     mask_lab = np.array(['ws_mask'+str(i)])
#     labels_new = np.concatenate((labels_new,vel_lab))
#     labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
#     labels_mask = np.concatenate((labels_mask,mask_lab))
#     labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
#     labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
#     labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
# labels_new = np.concatenate((labels_new,np.array(['scan'])))     

# ######################################

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
# selec_raw = col + ' FROM "table_raw"'

# query_raw_1 = selec_raw+ ' where name >= ' + dyi + ' and name <='+ dyi + ' and stop_time >= ' + t_0_1 + ' and stop_time <= ' + t_1_1 #' and scan <=' + str(hms.scan0.values[-1]) #

# df_1 = pd.read_sql_query(query_raw_1, csv_database_1_ind)

# g = np.mean(stab_20hz_1[Sdir_1].values)

# gs = df_1.azim.unique()[np.argmin(np.abs(df_1.azim.unique()-g))]

# ws = df_1['ws_1'].loc[df_1['azim']==gs].values
# t = df_1['stop_time'].loc[df_1['azim']==gs].values
# ts = [s[:4]+'-'+s[4:6]+'-'+ s[6:8]+' '+s[8:10]+':'+s[10:] for s in stab_20hz_1.Name.values]
# ts = [datetime.strptime(s, '%Y-%m-%d %H:%M')-date for s in ts]
# ts = np.array([int(s.days*24*3600+s.seconds+s.microseconds/1000000) for s in ts]).astype(float)

# ts = np.linspace(ts[0],ts[-1],len(ts))

# plt.figure()
# #plt.plot(np.arange(len(stab_20hz_1.Sspeed_241m_LMN.values))/20, stab_20hz_1.Sspeed_241m_LMN.values)
# plt.plot(ts-t[0], stab_20hz_1.Sspeed_241m_LMN.values, label=r'$From\:sonic\:at\:175m$')
# plt.plot(t-t[0],-ws, color = 'r', marker = 'o', markersize = 10, label=r'$From\:lidar\:at\:200m$')
# plt.xlabel(r'$time\:[s]$',fontsize=20)
# plt.ylabel(r'$Wind\:speed\:[m/s]$',fontsize=20)
# plt.legend(fontize = 20)

###########################################
###########################################
# In[time instead]
# xg, yg, Uex, Vex = expand_field_point(grdu, U_list, V_list, [dr0, dr1] , dt, r, L = 4, tree=True, grid_i = [])
# mask_t = np.isnan(Uex) 
# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# plt.contourf(xg,yg,np.reshape(Uex,xg.shape), np.linspace(10,20,10),cmap='jet')   
# ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
# ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')  
# ax0.set_xlim(-6500, np.nanmax(xg)+1000)
# ax0.set_ylim(-3500, 3500)

# ts0 = np.array([s[:4]+'-'+s[4:6]+'-'+ s[6:8]+' '+s[8:10]+':'+s[10:] for s in stab_20hz_0.Name.values[::len(stab_20hz_0.Name.values)-1]])
# ts0 = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M') for s in ts0])
# ts0[-1] = ts0[-1] + timedelta(seconds=10*60-1/20)
# ts0 = np.unique(ts0)
# ts0 = pd.date_range(ts0[0], ts0[-1], freq='.5S')

# # 
# # ts0 = np.array([int(s.days*24*3600+s.seconds+s.microseconds/1000000) for s in ts0]).astype(float)
# # ts0 = np.linspace(ts0[0],ts0[-1],len(ts0))
# # ts1 = [s[:4]+'-'+s[4:6]+'-'+ s[6:8]+' '+s[8:10]+':'+s[10:]+':'+ str(int(i/20 % 60)) +
# #        str(np.round((i/20 % 60) - int(i/20 % 60),decimals=3))[1:]  for i, s in enumerate(stab_20hz_1.Name.values)]
# # ts1 = [datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f') for s in ts1]
# # ts0 = np.array([int(s.days*24*3600+s.seconds+s.microseconds/1000000) for s in ts0]).astype(float)
# # ts0 = np.linspace(ts0[0],ts0[-1],len(ts0))

# xtg0, ytg0, ttg0, tou0t, Uext0, Vext0, _ = expand_field_point_time(grdu, U_list, V_list, [dr0, dr1], tree=False,
#                             grid_i = dr0, beta = 1, al = 20, Luy = 2*Lux, time_scan = time_s,
#                             t_int = ts0, interp_t = True, tri_dom = tri_dom, part=10, t_scale = 100)


#########################################################################################


# def expand_field_point_time(grd, U_list, V_list, point, n = 5, dt = 45, r = 1000,
#                             L = 4, tree=True, treetype = 'Kn', grid_i = [],
#                             nx = 100, beta = 1, probe='circle', Luy = []):
#     dx = grd[0][0,1]-grd[0][0,0]
#     dy = grd[1][1,0]-grd[1][0,0]
#     dr = np.sqrt(dx**2+dy**2)
#     # Time steps
#     t = np.flip(np.arange(len(U_list))*dt)     
#     # Local mean wind speed  parameters
#     ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
# #    s = /ds  
#     xfin = np.array([])
#     yfin = np.array([])
#     time = np.array([])
#     Ufin = np.array([])
#     Vfin = np.array([]) 
#     print('means')  
#     Umeans = [filterfft(u,np.isnan(u),sigma=L/ds) for u, L in zip(U_list,Luy)]
#     Vmeans = [filterfft(u,np.isnan(u),sigma=L/ds) for u, L in zip(V_list,Luy)]
#     print('probes') 
#     chunk = 512
#     Uarr = np.vstack([Umeans[i] for i in range(len(U_out_u))])
#     Umask = ~np.isnan(np.nanmean(Uarr.reshape((int(len(Uarr)/chunk),chunk,chunk)),axis=0))
#     xmask = grd[0][Umask]
#     ymask = grd[1][Umask]
#     center = .5*np.r_[grd[0][0,0]+grd[0][0,-1], grd[1][0,0]+grd[1][-1,0]]    
#     R = np.sqrt(np.sum((point[0][:2]-center)**2)) 
#     alpha0 = np.arctan2((point[0][:2]-center)[1], (point[0][:2]-center)[0])
#     alpha1 = np.arctan2((point[1][:2]-center)[1], (point[1][:2]-center)[0])
#     alpha = np.linspace(alpha0, alpha1, n)   
#     xp = center[0]+R*np.cos(alpha)
#     yp = center[1]+R*np.sin(alpha)
#     # indx = (xp > center[0]) #!!
#     # indy = (yp > np.nanmin(ymask)) & (yp < np.nanmax(ymask))
#     # xp = xp[indx&indy]
#     # yp = yp[indx&indy]
#     tree_p = KDTree(np.c_[xmask, ymask])
#     ri, _ = tree_p.query(np.c_[xp,yp], k = 1, return_distance=True)
#     points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]
#     print('advection')
#     for i in range(len(U_list)-1):
#         mask = np.isnan(U_list[i])
#         xnew = grd[0][~mask]+Umeans[i][~mask]*dt
#         ynew = grd[1][~mask]+Vmeans[i][~mask]*dt
#         for j,p in enumerate(points):
#             T1 = np.array([[1,0,-p[0]], [0,1, -p[1]], [0, 0, 1]])            
#             xy = np.array(np.c_[xnew,ynew,np.ones(len(xnew))]).T
#             xy = np.dot(T1,xy).T
#             if probe == 'square':
#                 if j == 0:
#                     ind = (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
#                 else:
#                     ind = ind | (xy[:,0] <= beta*ri[j]) & (xy[:,1] <= beta*ri[j])
#             if probe == 'circle':
#                 if j == 0:
#                      ind = (xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2
#                 else:
#                      ind = ind | ((xy[:,0]**2 + xy[:,1]**2) <= (beta*ri[j])**2)
#         print(xnew.shape, np.sum(ind), ri[j])
#         xfin = np.r_[xfin, xnew[ind]+Umeans[i][~mask][ind]*(t[i]-dt)]
#         yfin = np.r_[yfin, ynew[ind]+Vmeans[i][~mask][ind]*(t[i]-dt)]
#         Ufin = np.r_[Ufin, U_list[i][~mask][ind]]
#         Vfin = np.r_[Vfin, V_list[i][~mask][ind]]
#         time = np.r_[time, np.ones(xnew[ind].shape[0])*t[i]]
#     mask = np.isnan(U_list[-1])    
#     xfin = np.r_[xfin,grd[0][~mask]]
#     yfin = np.r_[yfin,grd[1][~mask]]  
#     Ufin = np.r_[Ufin, U_list[-1][~mask]]
#     Vfin = np.r_[Vfin, V_list[-1][~mask]]
#     time = np.r_[time, np.ones(np.sum(~mask))*t[i]]
#     print('tree')
#     pos = np.c_[xfin, yfin]
#     tree = KDTree(pos)
#     dist, ind = tree.query(tree.data, k = 10, return_distance=True)
#     ind_d = dist<dr/2 
#     ind_t = np.less_equal(time[ind],np.tile(np.median(time[ind],axis=1),(time[ind].shape[1],1)).T)
#     xfin = xfin[np.unique(ind[(ind_d & ind_t)].flatten())] 
#     yfin = yfin[np.unique(ind[(ind_d & ind_t)].flatten())]
#     Ufin = Ufin[np.unique(ind[(ind_d & ind_t)].flatten())]
#     Vfin = Vfin[np.unique(ind[(ind_d & ind_t)].flatten())]
#     time = time[np.unique(ind[(ind_d & ind_t)].flatten())]         
#     if len(grid_i)==0:
#         xg = np.arange(np.nanmin(xfin), np.nanmax(xfin)+dx, dx)
#         yg = np.arange(np.nanmin(yfin), np.nanmax(yfin)+dy, dy) 
#         xg, yg = np.meshgrid(xg, yg)
#     else:
#         xg, yg = grid_i[0], grid_i[1]     
#     tg = np.tile(np.linspace(t[-1], t[0], xg.shape[1]),(yg.shape[0],1))    
#     print('envelope')
#     x_bin = np.linspace(np.nanmin(xfin), np.nanmax(xfin), nx)   
#     indyg = np.zeros(xg.shape, dtype=bool)    
#     for i in range(len(x_bin)-1):
#         indx = (xfin >= x_bin[i]) & (xfin < x_bin[i+1])
#         indxg = (xg >= x_bin[i]) & (xg < x_bin[i+1])
#         indyg = (((yg > np.nanmin(yfin[indx])) & (yg < np.nanmax(yfin[indx]))) & indxg) | indyg       
           

#     X = np.c_[xfin, yfin, time]
#     Xg = np.c_[xg[indyg],yg[indyg],tg[indyg]]

#     if tree:  
#         Uex = np.zeros(xg.shape)*np.nan
#         Vex = np.zeros(xg.shape)*np.nan
#         neigh = KNeighborsRegressor(n_neighbors =  26, weights='distance',algorithm='auto', leaf_size=30,n_jobs=1)
#         print('fit U')
#         neigh.fit(X, Ufin)                
#         print('Predict U')
#         Uex[indyg] = neigh.predict(Xg) 
#         print('fit V')
#         neigh.fit(X, Vfin)
#         print('Predict V')
#         Vex[indyg] = neigh.predict(Xg)
#     else:
#         print('triangulation')
#         tri_exp = Delaunay(X)
#         print('Interpolation')
#         Uex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Ufin)(Xg)
#         Vex = sp.interpolate.CloughTocher2DInterpolator(tri_exp,Vfin)(Xg)
           
#     return (xg, yg, np.reshape(Uex,xg.shape), np.reshape(Vex,xg.shape), np.c_[xp,yp])

# In[]

# fig0, ax0 = plt.subplots(figsize=(8, 8))
# ax0.set_aspect('equal')
# ax0.use_sticky_edges = False
# ax0.margins(0.07)
# a = .1

# for j in range(len(U_out_u)):
#     ax0.cla()    
#     U, V = U_rot(grdu, U_out_u[j], V_out_u[j], gamma = gamma, tri_calc = False, 
#                               tri_del = tri_del, mask_int = mask_int, mask = mask)     
#     ax0.scatter(dr0[0], dr0[1], marker='+', color = 'r')
#     ax0.scatter(dr1[0], dr1[1], marker='+', color = 'r')    
#     dudy, dudx = np.gradient(U, grdu[1][:,0], grdu[0][0,:]) 
#     dvdy, dvdx = np.gradient(V, grdu[1][:,0], grdu[0][0,:])    
#     contsh = dudx + dvdy
#     contsh= filterfft(contsh,np.isnan(contsh),sigma=s_i2[j]/4)
# #    im0 = ax0.contourf(grdu[0], grdu[1], U_out_u[j], np.linspace(np.nanmin(U_out_u[j]),np.nanmax(U_out_u[j]),10), cmap='jet')
#     im0 = ax0.contourf(grdu[0], grdu[1], U, np.linspace(np.nanmin(U),np.nanmax(U),10), cmap='jet')
# #    ax0.contour(grdu[0], grdu[1], contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 2, linestyles = 'solid')
# #    ax0.contour(grdu[0], grdu[1], V_out_u[j]-np.nanmean(V_out_u[j]), levels = [-1], colors = 'k', linewidths = 1.5)
#     ax0.tick_params(axis='both', which='major', labelsize=24)
#     ax0.set_xlabel('$x_1\:[m]$', fontsize=24)
#     ax0.set_ylabel('$x_2\:[m]$', fontsize=24)
# #    divider = make_axes_locatable(ax0)
# #    cax= divider.append_axes("right", size="5%", pad=0.05)
# #    cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
# #    cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
# #    cb.ax.tick_params(labelsize=24)
# #    ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
# #    fig0.tight_layout()
# #    fig0.tight_layout()
#     plt.pause(.5)

# # In[Whole field]
# tri_calc = True
# chunk = 512
# x = grdu[0][0,:]
# y = grdu[1][:,0]
# U_arr = np.vstack([U_out_u[i] for i in range(len(U_out_u))])
# V_arr = np.vstack([V_out_u[i] for i in range(len(V_out_u))])
# scan = s_syn[:,0]
# U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)
# V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)
# u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc) 

