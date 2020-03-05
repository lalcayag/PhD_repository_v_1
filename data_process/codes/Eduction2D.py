# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:16:46 2020
Eduction scheme
@author: lalc
"""

import numpy as np
import scipy as sp
import pandas as pd
import scipy.ndimage as nd
import os
from os import listdir
from os.path import isfile, join, getsize, abspath
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import pickle
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
#import spectralfitting.spectralfitting as sf
from sqlalchemy import create_engine
from scipy.spatial import Delaunay
from datetime import datetime, timedelta

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage import measure

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

from scipy import ndimage

# In[Functions]
# Stability

def mov_con(x,N):
    return np.convolve(x, np.ones((N,))/N,mode='same') 

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
def datetimeDF(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

####################

# Vorticity and continuity
def vort_cont(U,V,grd):
    #print(U.shape,grd[0].shape,grd[0][0,:].shape)
    du_dy, du_dx = np.gradient(U, grd[0][0,:], grd[1][:,0])#!!!!!!!!!!!!!!!!!!
    dv_dy, dv_dx = np.gradient(V, grd[0][0,:], grd[1][:,0])#!!!!!!!!!!!!!!!!!!  
    vort = dv_dx - du_dy
    cont = du_dx + dv_dy
    return (vort, cont)

# Edge and peak detection
def edge_detec(img, sigma=2):
    ind = np.isnan(img)
    img[ind] = 0
    LoG = nd.gaussian_laplace(img , sigma)
    thres = np.absolute(LoG[ind]).mean() * 0.75
    output = sp.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y-1:y+2, x-1:x+2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
    return output

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)
    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    local_min = maximum_filter(-image, footprint=neighborhood)==-image
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

def detect_struct(vort_f_rot, u_f_rot, grd, m = 2):#!!!!!!!!!!!!!!!!!

    detected_peaks = detect_peaks(vort_f_rot)
    dudy, dudx = np.gradient(u_f_rot, grd[0][0,:], grd[1][:,0]) 
    ind_peaks = (np.abs(vort_f_rot)> np.nanmax(np.abs(dudy)*m)) & detected_peaks
    peaks = vort_f_rot[ind_peaks]
    pos_peaks = np.c_[grd[0][ind_peaks], grd[1][ind_peaks]]
    
    slim_peak = 2.1
    slim_cont = 1.7
    S_M = np.nanmax(dudy)
    S_m = np.nanmin(dudy)
    
    
    return

                
# Geometric
def shrink(grid,U):
    patch = ~np.isnan(U)
    ind_patch_x = np.sum(patch,axis=1) != 0
    ind_patch_y = np.sum(patch,axis=0) != 0
#    if np.sum(ind_patch_x) > np.sum(ind_patch_y):
#        ind_patch_y = ind_patch_x
#    elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
#        ind_patch_x = ind_patch_y        
    n = np.sum(ind_patch_x)
    m = np.sum(ind_patch_y)          
    ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
    ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
    U = np.reshape(U[ind_patch_grd],(n,m))
    grid_x = np.reshape(grid[0][ind_patch_grd],(n,m))
    grid_y = np.reshape(grid[1][ind_patch_grd],(n,m))
    return (grid_x,grid_y,U) 

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
    xtrans = 0
    ytrans = y[0]/2
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
            simp_grid = -np.ones(grd[1].flatten().shape)
            simp_grid[mask_int] = tri_del.find_simplex(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
            mask_simp = np.isin(simp_grid,np.array(range(0,len(tri_del.simplices)))[ind_simp])     
            U_tesa = np.zeros(U.shape)*np.nan
            U_tesa[mask] = sp.interpolate.NearestNDInterpolator(tri_del.points[~ind_nan,:],U[mask][~ind_nan])(Xx.T[mask,0],Xx.T[mask,1])
            U[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del,U_tesa[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_simp])
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
            simp_grid = -np.ones(grd[1].flatten().shape)
            simp_grid[mask_int] = tri_del.find_simplex(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
            mask_simp = np.isin(simp_grid,np.array(range(0,len(tri_del.simplices)))[ind_simp])     
            V_tesa = np.zeros(V.shape)*np.nan
            V_tesa[mask] = sp.interpolate.NearestNDInterpolator(tri_del.points[~ind_nan,:],V[mask][~ind_nan])(Xx.T[mask,0],Xx.T[mask,1])
            V[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del,V_tesa[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_simp])
            V[~mask_simp] = np.nan
            V = np.reshape(V, grid[0].shape)
    return (U, V)

# Wind/Vorticity field smoothing
def filterfft(vorti, mask, sigma=20):
    vort = vorti.copy()
    vort[np.isnan(vort)] = np.nanmean(vort)
    
    input_ = np.fft.fft2(vort)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.real(np.fft.ifft2(result))
    result[mask] = np.nan
    return result

# Synchronized scans detection
def synch_df(df0,df1,dtscan=45):
    s0 = df0.scan.unique()
    s1 = df1.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    dt = [(t0[i]-t1[j]).total_seconds() for i in range(len(t0)) for j in range(len(t1))]
    s = np.array([[s_0,s_1] for s_0 in s0 for s_1 in s1])
    ind_synch = np.abs(dt)<dtscan
    if np.sum(ind_synch)==0:
       sync = []
       off0 = s0[-1]
       off1 = s1[-1]
    else:
       sync = s[ind_synch,:]
       off0 = sync[-1,0]+1
       off1 = sync[-1,1]+1
    return (sync,off0*45,off1*45)

def gamma_10m(df_0,df_1,U,V,su):
    t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in np.array(su)[:,0]])
    t1 = np.array([date+timedelta(seconds = df_1.loc[df_1.scan==s].stop_time.min()) for s in np.array(su)[:,1]])  
    tmin, tmax = np.min(np.r_[t0,t1]), np.max(np.r_[t0,t1])+timedelta(minutes = 10)
    t_10m = pd.date_range(tmin, tmax,freq='10T')#.strftime('%Y%m%d%H%M)
    gamma = []
    indt = [np.where((t0>t_10m[i]) & (t0<t_10m[i+1]))[0] for i in range(len(t_10m)-1)]  
    U_mean = [np.nanmedian(np.array([U[ii0] for ii0 in i0])) for i0 in indt]   
    V_mean = [np.nanmedian(np.array([V[ii0] for ii0 in i0])) for i0 in indt]     
    gamma = [np.arctan2(vi,ui) for vi,ui in zip(V_mean,U_mean)]
    gamma_out = np.zeros(len(U))
    for i in range(len(gamma)):
        gamma_out[indt[i]] = gamma[i]
    return gamma_out 


# In[Data loading for reconstructed fields]
### Path of file's directory

root = tkint.Tk()
file_in_path_0 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()
root = tkint.Tk()
file_in_path_1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_in_path_corr1 = tkint.filedialog.askdirectory(parent=root,title='Choose an corr dir')
root.destroy()

root = tkint.Tk()
file_in_path_corr2 = tkint.filedialog.askdirectory(parent=root,title='Choose an corr dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

root = tkint.Tk()
file_out_path_u_field = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()    

### Reconstructed fields
#Phase 1
U_out, V_out, grd, su = joblib.load(file_out_path_u_field+'/U_rec_20160421.pkl')  
#Phase 2
with open(file_out_path_u_field+'/U_rec_08_06_2016.pkl', 'rb') as field:
     U_out, V_out, grd, su = pickle.load(field)
     
#with open(file_out_path_u_field+'/vort_rec_08_06_2016.pkl', 'rb') as field:
#     vort_m, cont_m = pickle.load(field) 

U_out = [U_out[i] for i in range(800)]
V_out = [V_out[i] for i in range(800)]
     
#######################################################
# In[stability conditions]   
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)   

csv_database_r2 = create_engine('sqlite:///'+file_in_path_corr2+'/corr_uv_west_phase2_ind.db')
days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r2).values
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)
heights = [241, 175, 103, 37, 7]
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

ax = dfL_phase2.loc[dfL_phase2['$L_{flux,103}$'].abs()<3000].plot(x = 'time', y='$L_{flux,103}$',style='o')

ax = dfL_phase2.loc[(dfL_phase2.name=='20160804')|(dfL_phase2.name=='20160805')|
        (dfL_phase2.name=='20160806')].plot(x = 'time', y='$L_{flux,103}$',style='o')

ax = dfL_phase2.loc[(dfL_phase2.name=='20160806')].plot(x = 'time', y='$L_{flux,103}$',style='o')

########################################################   
# In[]

########################################################

with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph1.pkl', 'rb') as reader:
    res_flux_1 = pickle.load(reader)   

csv_database_r1 = create_engine('sqlite:///'+file_in_path_corr1+'/corr_uv_west_phase1_ind.db')
days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r1).values
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
heights = [241, 175, 103, 37, 7]
L_list_1 = np.hstack([L_smooth(np.array(res_flux_1)[:,i,:]) for i in range(len(heights))])
t1 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_1)[:,2,0]])
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]
cols = [item for sublist in cols for item in sublist]
stab_phase1_df = pd.DataFrame(columns = cols, data = L_list_1)
stab_phase1_df['time'] = t1
dfL_phase1[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))
Lph1 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))
Lph1.index = dfL_phase1.index
aux1 = dfL_phase1.time
dfL_phase1.time = pd.to_datetime(dfL_phase1.time)
for i in range(len(t1)-1):
    print(i,t1[i])
    ind = (dfL_phase1.time>=t1[i]) & (dfL_phase1.time<=t1[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase1_df[cols].loc[stab_phase1_df.time==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
dfL_phase1.time = aux1
dfL_phase1[cols] = Lph1
dfL_phase1.sort_values(by=['time'], inplace = True)

ax = dfL_phase1.loc[dfL_phase1['$L_{flux,103}$'].abs()>=500].plot(y='$L_{flux,103}$',style='o')

#ax = dfL_phase1.loc[(dfL_phase1.name=='20160804')|(dfL_phase1.name=='20160805')|
#        (dfL_phase1.name=='20160806')].plot(x = 'time', y='$L_{flux,103}$',style='o')

ax = dfL_phase1.loc[(dfL_phase1.name=='20160421')].plot(x = 'hms', y='$L_{flux,103}$',style='o',figsize=(20,20), ylim=(-3000,3000))

########################################################     

# In[Fields reconstruction (if not reconsturcted before and loaded in the previous section)]
     
### labels
iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
labels_CNR = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan'])))     

### Reconstruction of fields for a specific day

######################################
dy = '20160421'#phase1'20160804' #### DAY!!!!!!!!!!!!!!
######################################

date = datetime(1904, 1, 1) 
loc0 = np.array([6322832.3,0])
loc1 = np.array([6327082.4,0])
d = loc0-loc1  
  
csv_database_r1 = create_engine('sqlite:///'+file_in_path_corr+'/corr_uv_west_phase1_ind.db')
csv_database_r2 = create_engine('sqlite:///'+file_in_path_corr+'/corr_uv_west_phase2_ind.db')

csv_database_0_ind = create_engine('sqlite:///'+file_in_path_0+'/raw_filt_0_phase1.db')
csv_database_1_ind = create_engine('sqlite:///'+file_in_path_1+'/raw_filt_1_phase1.db')     

labels_short = np.array([ 'stop_time', 'azim'])
for w,r in zip(labels_ws,labels_rg):
    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate'])))
labels_short = np.concatenate((labels_short,np.array(['scan'])))   
lim = [-8,-24]
i=0
col = 'SELECT '
col_raw = 'SELECT '
for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
        col_raw = col_raw + w + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
        col_raw = col_raw + ' ' + w
    else:
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' 
        col_raw = col_raw + ' ' + w + ', '
    i+=1

selec_fil = col + ' FROM "table_fil"'
selec_raw = col_raw + ' FROM "table_raw"'

#init = pd.read_sql_query('select name, hms, scan from "L" where scan = (select max(scan) from "L")', csv_database_r2)
#n_i, h_i, scan_i = init.name.values[0], init.hms.values[0], init.scan.values[0]
chunk_scan = int(13*6)

days0 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_0_ind).values
days1 = pd.read_sql_query('select distinct name from "table_fil"', csv_database_1_ind).values
days_old = pd.read_sql_query('select distinct name from "L"', csv_database_r1).values
days_old = np.squeeze(days_old)
days0 = np.squeeze(days0)
days1 = np.squeeze(days1)
days = np.unique(np.r_[days0[np.isin(days0,days1)],days0[np.isin(days1,days0)]])
days = days[~np.isin(days,days_old)]
switch = 0
U_out, V_out, su = [], [], [] 

query_fil = selec_fil+ ' where name = '+dy
query_raw = selec_raw+ ' where name = '+dy

df_0 = pd.read_sql_query(query_fil, csv_database_0_ind)
df = pd.read_sql_query(query_raw, csv_database_0_ind)
for i in range(198):
    ind = (df_0['CNR_'+str(i)]<lim[0])&(df_0['CNR_'+str(i)]>lim[1])
    df_0['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
df = None
df_0.drop(columns = labels_CNR,inplace=True)
df_0.columns = labels_short
day_r = pd.to_datetime(pd.read_sql_query('select time from L where name = '+
                                          dy,csv_database_r1).time.values).strftime("%H%M%S").values  
                                      
#if len(day_r)>0: 
s0 = df_0.scan.unique()
t0 = np.array([date+timedelta(seconds = df_0.loc[df_0.scan==s].stop_time.min()) for s in s0])                                         
ind = np.isin(pd.to_datetime(t0).strftime("%H%M%S").values,day_r)
s0 = s0[ind]
ind_df0 = df_0.scan.isin(s0)
df_0 = df_0.loc[ind_df0]
if ~(np.sum(ind)==0):
    df_1 = pd.read_sql_query(query_fil, csv_database_1_ind)
    df = pd.read_sql_query(query_raw, csv_database_1_ind)
    for i in range(198):
        ind = (df_1['CNR_'+str(i)]<lim[0])&(df_1['CNR_'+str(i)]>lim[1])
        df_1['ws_'+str(i)].loc[ind] = df['ws_'+str(i)].loc[ind]
    df = None   
    df_1.drop(columns = labels_CNR,inplace=True) 
    df_1.columns = labels_short
    
    #Synchronous?    
    s_syn,_,_ = synch_df(df_0,df_1,dtscan=45/2)
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
        t_1h = pd.date_range(tmin, tmax,freq='1H')#.strftime('%Y%m%d%H%M)
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
                    r_0, phi_0 = np.meshgrid(r0, np.pi-np.radians(phi0)) # meshgrid
                    r_1, phi_1 = np.meshgrid(r0, np.pi-np.radians(phi1)) # meshgrid                
                    tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
                    switch = 1 
                u, v, grd, s = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d, N_grid = 512)  
                U_out.append(u), V_out.append(v), su.append(s)
                
U_out = [item for sublist in U_out for item in sublist]
V_out = [item for sublist in V_out for item in sublist]
su    = [item for sublist in su for item in sublist]

filename = file_out_path_u_field+'/U_rec_'+dy+'.pkl'
joblib.dump((U_out, V_out, grd, su), filename)  

with open(file_out_path_u_field+'/U_rec_'+dy+'.pkl', 'wb') as field:
      pickle.dump((U_out, V_out, grd, su),field)

#### Vorticity

#vort_m = []
#cont_m = []
#for i in range(len(U_out)):
#    print(i)
#    vort, cont =  vort_cont(U_out[i], V_out[i], grd)    
#    vort_m.append(vort)
#    cont_m.append(cont)

############### Cutting off some scans ###################
    
#vort_m = [vort_m[i] for i in range(800)]
#cont_m = [cont_m[i] for i in range(800)]


# In[Vorticity smoothing]

############### Spatial smoothing ###################
# Phase 2
L = np.abs(grd[0][0,0]-grd[0][0,-1])
ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
s_i = [np.min(dfL_phase2.loc[(dfL_phase2.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)/ds for i in range(800)]
L_s = np.squeeze(np.array([dfL_phase2.loc[(dfL_phase2.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values for i in range(800)]))

# Pahse 1

L = np.abs(grd[0][0,0]-grd[0][0,-1])
ds = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
s_i = [np.min(dfL_phase1.loc[(dfL_phase1.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)/ds for i in range(800)]
L_s = np.squeeze(np.array([dfL_phase1.loc[(dfL_phase1.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values for i in range(800)]))

vort_spatial_smooth = [filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s_i[i]/4) for i in range(800)]

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for j in range(len(vort_spatial_smooth)):   
    ax.cla()
    im = ax.contourf(grd[0], grd[1], vort_m[j], np.linspace(-.05,.05,20), cmap='jet') 
    plt.pause(.3)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0], grd[1], vort_m[0],20,cmap='jet') 


############### Temporal smoothing

for i,u,v,w in range(len(U_out[1:-1])):
    
    U = np.nanmean(U_out[i+1])
    V = np.nanmean(V_out[i+1])
    Us = np.sqrt(U**2 + V**2)
    T1 = int(.1*L/Us)
    indt = range(i+1-T1,i+1-T1+1)
    s = (np.min(dfL_phase2.loc[(dfL_phase2.scan==su[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)/ds)
    vort_smooth_aux = [filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s) for i in indt]
    
    vort_smooth_aux = [np.nanmean(np.array([U[ii0] for ii0 in i0])) for i0 in indt]
    

# In[]


import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

fname = 'test.png'
neighborhood_size = 5
threshold = 1500

data = scipy.misc.imread(fname)

data_max = filters.maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = filters.minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

plt.imshow(data)
#plt.savefig('data.png', bbox_inches = 'tight')

plt.autoscale(False)
plt.plot(xy[:, 1], xy[:, 0], 'ro')
#plt.savefig('result.png', bbox_inches = 'tight')


