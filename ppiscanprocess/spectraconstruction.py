# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:27:46 2018

Module for 2D autocorrelation and spectra from horizontal wind field  
measurements. The structure expected is a triangulation from
scattered positions.

Autocorrelation is calculated in terms 

To do:
    
- Lanczos interpolaton on rectangular grid (usampling)

@author: lalc
"""
# In[Packages used]
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer

from matplotlib import ticker    
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable  
                
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay

# In[Autocorrelation for non-structured grid, brute force for non-interpolated wind field]  

def spatial_autocorr_sq(grid,U,V,step=2):
    """
    Function to estimate autocorrelation in cartesian components of wind
    velocity, U and V. The 2D spatial autocorrelation is calculated in a 
    squared and structured grid, and represent the correlation of points
    displaced a distance tau and eta in x and y, respectively.

    Input:
    -----
        grid     - Grid of cartesian coordinates.
        
        U,V      - Arrays with cartesian components of wind speed.

        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.  
                   
    """  
    # Squared grid of spatial increments  
    patch = ~np.isnan(U)
    ind_patch_x = np.sum(patch,axis=1) != 0
    ind_patch_y = np.sum(patch,axis=0) != 0
    if np.sum(ind_patch_x) > np.sum(ind_patch_y):
        ind_patch_y = ind_patch_x
    elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
        ind_patch_x = ind_patch_y
    ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
    ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]        
    n_tau_max = np.sum(ind_patch_x)
    n_eta_max = np.sum(ind_patch_y)
    
    n_tau_next = int(2**np.ceil(np.log(n_tau_max)/np.log(2))) 
    n_eta_next = int(2**np.ceil(np.log(n_eta_max)/np.log(2)))
    
    dx = np.diff(grid[0][0,:])[0]
    dy = np.diff(grid[1][:,0])[0]
    
#    n_tau = np.arange(-n_tau_next,n_tau_next,step)
    
    n_tau = np.cumsum((np.logspace(0,np.log(n_tau_next)/np.log(10),32)).astype(int))
    n_tau = n_tau[n_tau<n_tau_next]
    n_tau = np.r_[-np.flip(n_tau),0,n_tau]
    
    n_eta = np.cumsum((np.logspace(0,np.log(n_eta_next)/np.log(10),32)).astype(int))
    n_eta = n_eta[n_eta<n_eta_next]
    n_eta = np.r_[-np.flip(n_eta),0,n_eta]
    
    tau = n_tau *dx
#    n_eta = np.arange(-n_eta_next,n_eta_next,step)
    eta = n_eta *dy

    
    # De-meaning of U and V. The mean U and V in the whole scan is used
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    U[np.isnan(U)] = 0.0
    V[np.isnan(V)] = 0.0

    tau,eta = np.meshgrid(tau,eta)
    n_tau,n_eta = np.meshgrid(n_tau,n_eta)

    # Autocorrelation is calculated just for non-empty scans 
    if len(U[~np.isnan(U)])>0:
        # autocorr() function over the grid tau and eta.
        r_u = [autocorr_sq(U,t,e) for t, e in zip(n_tau.flatten(),n_eta.flatten())]
        r_v = [autocorr_sq(V,t,e) for t, e in zip(n_tau.flatten(),n_eta.flatten())]
        r_uv = [crosscorr_sq(U,V,t,e) for t, e in zip(n_tau.flatten(),n_eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(tau,eta,r_u,r_v,r_uv)

def autocorr_sq(U,n_tau,n_eta):
    # Displacement
    # n_tau in x (columns)
    # n_eta in y (rows)
    print(n_tau,n_eta)
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = U[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = U[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = U[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = U[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = U[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = U[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = U[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = U[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = U
    # Autocorrelation is the off-diagonal value of the correlation matrix.
    
    r = np.corrcoef(U.flatten(),U_del.flatten(),rowvar=False)[0,1]
    return r

def crosscorr_sq(U,V,n_tau,n_eta):
    # Displacement
    # n_tau in x (columns)
    # n_eta in y (rows)
    print(n_tau,n_eta)
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = V[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = V[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = V[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = V[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = V[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = V[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = V[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = V[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = U
    # Autocorrelation is the off-diagonal value of the correlation matrix.
    r = np.corrcoef(U.flatten(),U_del.flatten(),rowvar=False)[0,1]
    return r


def spatial_autocorr(tri,U,V,N,alpha):
    """
    Function to estimate autocorrelation in cartesian components of wind
    velocity, U and V. The 2D spatial autocorrelation is calculated in a 
    squared and structured grid, and represent the correlation of points
    displaced a distance tau and eta in x and y, respectively.

    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U,V      - Arrays with cartesian components of wind speed.
        
        N        - Number of points in the autocorrelation's squared grid.
        
        alpha    - Fraction of the spatial domain that will act as the limit 
                   for tau and eta increments. 
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.  
                   
    """
    
    # Squared grid of spatial increments
    tau = np.linspace(-alpha*(np.max(tri.x)-np.min(tri.x)),alpha*(np.max(tri.x)-np.min(tri.x)),N)
    eta = np.linspace(-alpha*(np.max(tri.y)-np.min(tri.y)),alpha*(np.max(tri.y)-np.min(tri.y)),N)
    tau,eta = np.meshgrid(tau,eta)
    # De-meaning of U and V. The mean U and V in the whole scan is used
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    # Interpolator object to estimate U and V fields when translated by
    # (tau,eta)
    U_int = LinearTriInterpolator(tri, U)
    V_int = LinearTriInterpolator(tri, V)
    # Autocorrelation is calculated just for non-empty scans 
    if len(U[~np.isnan(U)])>0:
        # autocorr() function over the grid tau and eta.
        r_u = [autocorr(tri,U,U_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
        r_v = [autocorr(tri,V,V_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(r_u,r_v)

def autocorr(tri,U,Uint,tau,eta):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U        - Arrays with a cartesian component wind speed.
        
        U_int    - Linear interpolator object.
        
        tau      - Increment in x coordinate. 
        
        eta      - Increment in y coordinate.
        
    Output:
    ------
        r        - Autocorrelation function value.  
                   
    """ 
    # Only un-structured grid with valid wind speed
    ind = ~np.isnan(U)
    # Interpolation of U for a translation of the grid by (tau,eta)
    U_delta = Uint(tri.x[ind]+tau,tri.y[ind]+eta)
    # Autocorrelation on valid data in the original unstructured grid and the
    # displaced one.
    if len(U_delta.data[~U_delta.mask]) == 0:
        r = np.nan
    else:
        # Autocorrelation is the off-diagonal value of the correlation matrix.
        r = np.corrcoef(U_delta.data[~U_delta.mask],U[ind][~U_delta.mask],
                        rowvar=False)[0,1]
    return r


# In[Autocorrelation for non-structured grid, using FFT for interpolated
#                                                         (or not) wind field] 
    
def spatial_autocorr_fft(tri,U,V,N_grid=512,auto=False,transform = False, tree = None, interp = 'Lanczos'):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
                   
        N_grid     - Squared, structured grid resolution to apply FFT.
        
        U, V     - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.               
    """   
    if transform:
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        # Wind direction
        gamma = np.arctan2(V_mean,U_mean)
        # Components in matrix of coefficients
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        T = np.array([[S11,S12], [-S12,S11]])
        vel = np.array(np.c_[U,V]).T
        vel = np.dot(T,vel)
        X = np.array(np.c_[tri.x,tri.y]).T
        X = np.dot(T,X)
        U = vel[0,:]
        V = vel[1,:]
        tri = Triangulation(X[0,:],X[1,:])
        mask=TriAnalyzer(tri).get_flat_tri_mask(.05)
        tri=Triangulation(tri.x,tri.y,triangles=tri.triangles[~mask])
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
    else:
        # Demeaning
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        
    grid = np.meshgrid(np.linspace(np.min(tri.x),
           np.max(tri.x),N_grid),np.linspace(np.min(tri.y),
                 np.max(tri.y),N_grid))   
    
    U = U-U_mean
    V = V-V_mean
    
    # Interpolated values of wind field to a squared structured grid
                   
    if interp == 'cubic':     
        U_int= CubicTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= CubicTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
    else:
#        U_int= lanczos_int_sq(grid,tree,U)
#        V_int= lanczos_int_sq(grid,tree,V) 
        U_int= LinearTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= LinearTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
    

               
    #zero padding
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    #plt.triplot(tri)
    #plt.contourf(U_int,cmap='jet')
    fftU = np.fft.fft2(U_int)
    fftV = np.fft.fft2(V_int)
    if auto:

        # Autocorrelation
        r_u = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftU)**2)))/len(U_int.flatten())
        r_v = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftV)**2)))/len(U_int.flatten())
        r_uv = np.real(np.fft.fftshift(np.fft.ifft2(np.real(fftU*np.conj(fftV)))))/len(U_int.flatten())
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]   
    # Spectra
    fftU  = np.fft.fftshift(fftU)
    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    Suu = 2*(np.abs(fftU)**2)*(dx*dy)/(n*m)
    Svv = 2*(np.abs(fftV)**2)*(dx*dy)/(n*m)
    Suv = 2*np.real(fftUV)*(dx*dy)/(n*m)
#    kx = 1/(2*dx)
#    ky = 1/(2*dy)   
#    k1 = kx*np.linspace(-1,1,len(Suu))
#    k2 = ky*np.linspace(-1,1,len(Suu))
    
    k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
    k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))
    if auto:
        return(r_u,r_v,r_uv,Suu,Svv,Suv,k1,k2)
    else:
        return(Suu,Svv,Suv,k1,k2)
 
# In[Ring average of spectra]
def spectra_average(S_image,k,bins,angle_bin = 30,stat=False):
    """
    S_r = spectra_average(S_image,k,bins)
    
    A function to reduce 2D Spectra to a radial cross-section.
    
    INPUT:
    ------
        S_image   - 2D Spectra array.
        
        k         - Tuple containing (k1_max,k2_max), wavenumber axis
                    limits
        bins      - Number of bins per decade.
        
        angle_bin - Sectors to determine spectra alignment
        
        stat      - Bin statistics output
        
     OUTPUT:
     -------
      S_r - a data structure containing the following
                   statistics, computed across each annulus:
          .k      - horizontal wavenumber k**2 = k1**2 + k2**2
          .S      - mean of the Spectra in the annulus
          .std    - std. dev. of S in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
#    import numpy as np

    class Spectra_r:
        """Empty object container.
        """
        def __init__(self): 
            self.S = None
            #self.std = None
            #self.median = None
            #self.numel = None
            #self.max = None
            #self.min = None
            #self.k1 = None
            #self.k2 = None
    
    #---------------------
    # Set up input parameters
    #---------------------
    S_image = np.array(S_image)
    npix, npiy = S_image.shape       
        
    k1 = k[0]#*np.linspace(-1,1,npix)
    k2 = k[1]#*np.linspace(-1,1,npiy)
    k1, k2 = np.meshgrid(k1,k2)
    # Polar coordiantes (complex phase space)
    r = np.absolute(k1+1j*k2)
    
    # Ordered 1 dimensinal arrays
    #ind = np.argsort(r.flatten())
    
#    r_sorted = r.flatten()[ind]
#    Si_sorted = S_image.flatten()[ind]
#    r_log = np.log(r_sorted)
#    r_log10 = np.log10(r_sorted)
#    decades = len(np.unique(r_log10.astype(int))) 
#    bin_tot = decades*bins
#    r_bin10 = np.linspace(np.min(r_log10.astype(int))-1,np.max(r_log10.astype(int)),bin_tot+1)
#    mat = np.array([r_log10/rb<1  for rb in r_bin10[1:]])
#    # bin number array
#    r_n_bin = np.sum(mat,axis=0)
#    # Find all pixels that fall within each radial bin.
#    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
#    bin_ind = np.where(delta_bin)[0] # location of changes in bin
#    nr = bin_ind[1:] - bin_ind[:-1]  # number of elements per bin 
#    r_log = r_n_bin*np.max(r_log)/bin_tot
#    bin_centers= np.exp(0.5*(r_log[bin_ind[1:]]+r_log[bin_ind[:-1]]))
#    
#    # Cumulative sum to 2D spectra to find sum per bin
#    csSim = np.cumsum(Si_sorted, dtype=float)
#    tbin = csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]
#    
    
#    r_sorted = r.flatten()#[ind]
    Si_sorted = S_image.flatten()#[ind]
#    r_log = np.log(r_sorted)
    r_log10 = np.log10(r.flatten())
    decades = len(np.unique(r_log10.astype(int)))
    
    bin_tot = decades*bins
    r_bin10 = np.linspace(np.min(r_log10.astype(int))-1,np.max(r_log10.astype(int)),bin_tot+1)
    
#    mat = np.array([r_log10/rb<1  for rb in r_bin10[1:]])
   
#    r_n_bin = np.sum(mat,axis=0)
#    
#    # Find all pixels that fall within each radial bin.
#    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
#    
#    bin_ind = np.r_[0,np.where(delta_bin)[0]+1,len(r_n_bin)]# location of changes in bin
#    nr = bin_ind[1:] - bin_ind[:-1]
#    bin_ind = np.r_[np.where(delta_bin)[0],len(r_n_bin)-1]
#    csSim = np.cumsum(Si_sorted, dtype=float)
#    tbin = np.r_[csSim[bin_ind[0]],csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]]
    
#    bin_centers = np.sqrt(10**r_bin10[r_n_bin[bin_ind]]*10**r_bin10[r_n_bin[bin_ind]+1])
    
    S_ring = np.zeros(Si_sorted.shape)#tbin/nr
    
#    rlog10 = r_bin10[r_n_bin[bin_ind]]
    
#    for i in range(len(rlog10)-1):
#        
#        ind0 = (r_log10>rlog10[i]) & (r_log10<rlog10[i+1])
#        print(i,tbin[i]/nr[i],np.sum(ind0),rlog10[i],rlog10[i+1])
#        S_ring[ind0] = tbin[i]/nr[i]
                
    for i in range(len(r_bin10)-1):
        ind0 = (r_log10>r_bin10[i]) & (r_log10<r_bin10[i+1])
        S_ring[ind0] = np.sum(S_image.flatten()[ind0])/np.sum(ind0)   
    
#    S_ring[ind] = S_ring

    
    
#    # Same for angles and orientation
#    k1 = k[0]*np.linspace(-1,1,npix)
#    k2 = k[1]*np.linspace(-1,1,npiy/2)
#    k1, k2 = np.meshgrid(k1,k2)
#    phi = np.angle(k1+1j*k2)
#
#    ind_p = np.argsort(phi.flatten())
#    
#    Si_sorted_p = S_image.flatten()[ind_p]
#    phi_sorted = phi.flatten()[ind_p]
#    phi_int = (phi_sorted*180/np.pi/angle_bin).astype(int)
#    delta_bin = phi_int[1:] - phi_int[:-1]
#    phiind = np.r_[np.where(delta_bin)[0],len(delta_bin)] # location of changes in sector
#    nphi = phiind[1:] - phiind[:-1] 
#    # The other half
#    # Cumulative sum to figure out sums for each radius bin
#    csSim_p = np.cumsum(Si_sorted_p, dtype=float)
#    tbin_p = csSim_p[phiind[1:]] - csSim_p[phiind[:-1]]

    # Initialization of the data
    S_r = Spectra_r()
    S_r.S = np.reshape(S_ring,S_image.shape)
#    S_r.S_p = tbin_p / nphi
#    S_r.k1 = k1
#    S_r.k2 = k2
#    S_r.phi = 0.5*(phi_int[phiind[1:]]+phi_int[phiind[:-1]])*angle_bin
    # optional?
    if stat==True:
        S_stat = np.array([[np.std(Si_sorted[r_n_bin==r]), 
                            np.median(Si_sorted[r_n_bin==r]),
                            np.max(Si_sorted[r_n_bin==r]),
                            np.min(Si_sorted[r_n_bin==r])]
                            for r in np.unique(r_n_bin)])        
        S_r.std = S_stat[:,0]
        S_r.median = S_stat[:,1]
        S_r.max = S_stat[:,2]
        S_r.min = S_stat[:,3]
    
    return S_r

# In[Just spectra]
def spectra_fft(tri,grid,U,V):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
                   
        grid     - Squared, structured grid to apply FFT.
        
        U, V     - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.               
    """
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]
    
    U_int = np.reshape(LinearTriInterpolator(tri, U)(grid[0].flatten(),
                                grid[1].flatten()).data,grid[0].shape)    
    V_int = np.reshape(LinearTriInterpolator(tri, V)(grid[0].flatten(),
                                grid[1].flatten()).data,grid[1].shape)
    #zero padding
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    fftU = np.fft.fftshift(np.fft.fft2(U_int))
    fftV = np.fft.fftshift(np.fft.fft2(V_int))
    fftUV = fftU*np.conj(fftV)
    Suu = 2*(np.abs(fftU)**2)/(n*m*dx*dy)
    Svv = 2*(np.abs(fftV)**2)/(n*m*dx*dy)
    Suv = 2*np.real(fftUV)/(n*m*dx*dy) 
    kx = 1/(2*dx)
    ky = 1/(2*dy) 
    k1 = kx*np.linspace(-1,1,len(Suu))
    k2 = ky*np.linspace(-1,1,len(Suu))
    return(Suu,Svv,Suv,k1,k2)

# In[]
def field_rot(x, y, U, V, gamma = [], grid = [], tri_calc = False, tri_del = []):
    
    if not gamma:
        
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
  
    if tri_calc:
        if not grid:
            grid = np.meshgrid(x,y)       
        xtrans = 0
        ytrans = y[0]/2
        T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
        T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
        T = np.dot(np.dot(T1,R),T2)
        Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
        Xx = np.dot(T,Xx)   
        tri_del = Delaunay(np.c_[Xx[0,:][mask],Xx[1,:][mask]])
        mask_int = ~(tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()]) == -1)  
        
    return (U, V, mask, mask_int, tri_del)    
    
# In[]
def spatial_spec_sq(x,y,U_in,V_in, tri_del = [], mask_int = [], tri_calc = True, transform = False, shrink = False, ring=False, plot = False, bins=20):

    U = U_in.copy()
    V = V_in.copy()
    n_p = np.sum(np.isnan(U))
    grid = np.meshgrid(x,y)
    n, m = grid[0].shape
    dx = np.min(np.abs(np.diff(x)))
    dy = np.min(np.abs(np.diff(y)))
    a_eff = (n*m-n_p)*dx*dy
    
#######################################################################
 # Rotation of U, V
    if transform:
        
        if tri_calc:
            
            U, V, mask,mask_int, tri_del = field_rot(x, y, U, V, gamma = [], grid = grid,
                                            tri_calc = tri_calc)
        else:
            U, V, mask, _, _ = field_rot(x, y, U, V, gamma = [], grid = grid,
                                            tri_calc = tri_calc)
         
        print('reducing da domain')  
            
        print('interp U ',type(tri_del))
        U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        U[~mask_int] = np.nan
        print('interp V ',type(tri_del))
        V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        V[~mask_int] = np.nan
        print('interp ends')
        
#        print('interp U ',type(tri_del))
#        U[mask_int] = sp.interpolate.LinearNDInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
#        U[~mask_int] = np.nan
#        print('interp V ',type(tri_del))
#        V[mask_int] = sp.interpolate.LinearNDInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
#        V[~mask_int] = np.nan
#        print('interp ends')
        
        n_p = np.sum(np.isnan(U))
        a_eff = (n*m-n_p)*dx*dy        
    
        U = np.reshape(U,grid[0].shape)
        V = np.reshape(V,grid[0].shape)
#      

    
#######################################################################
## Rotation of U, V
#    if transform:
#        
#        U_mean = np.nanmean(U.flatten())
#        V_mean = np.nanmean(V.flatten())
#        # Wind direction
#        gamma = np.arctan2(V_mean,U_mean)
#        xtrans = 0
#        ytrans = y[0]/2
#        # Components in matrix of coefficients
#        S11 = np.cos(gamma)
#        S12 = np.sin(gamma)
#        T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
#        T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
#        R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
#        T = np.dot(np.dot(T1,R),T2)
#
#         
#        if not tri_del:
#            
#            Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
#            Xx = np.dot(T,Xx)
#            tri_del = Delaunay(np.c_[Xx[0,:],Xx[1,:]])
#            
#        vel = np.array(np.c_[U.flatten(),V.flatten()]).T
#        vel = np.dot(R[:-1,:-1],vel)
#        U = vel[0,:]
#        V = vel[1,:]
#        
#        ind_nan = ~np.isnan(U)
#               
##        mask = np.isnan(U)
##        DX_new = np.max(Xx[0,:])-np.min(Xx[0,:])
##        DY_new = np.max(Xx[1,:])-np.min(Xx[1,:])
##        DX = np.max(x)-np.min(x)
##        DY = np.max(y)-np.min(y)
##        n = int(np.ceil(DX_new*(n-1)/DX+1))
##        m = int(np.ceil(DY_new*(m-1)/DY+1))
#        
##        x_new = np.linspace(np.min(Xx[0,:]),np.max(Xx[0,:]),n)
##        y_new = np.linspace(np.min(Xx[1,:]),np.max(Xx[1,:]),m)
#        
##        dx = np.min(np.diff(x_new))
##        dy = np.min(np.diff(y_new))
#        
#        
##        grid_new = np.meshgrid(x_new,y_new)
#        
#        U = sp.interpolate.griddata(Xx.T[ind_nan,:2], U[ind_nan], (grid[0].flatten(),grid[1].flatten()), method='cubic')   
#        V = sp.interpolate.griddata(Xx.T[ind_nan,:2], V[ind_nan], (grid[0].flatten(),grid[1].flatten()), method='cubic') 
#                
#        n_p = np.sum(np.isnan(U))
#        a_eff = (n*m-n_p)*dx*dy        
#        
##        plt.figure()
##        #plt.scatter(grid[0].flatten(),grid[1].flatten())
##        plt.scatter(Xx.T[:,0],Xx.T[:,1])
##        plt.scatter(grid_new[0].flatten(),grid_new[1].flatten())
#        
#        U = np.reshape(U,grid[0].shape)
#        V = np.reshape(V,grid[0].shape)
##        

######################################################################
    
    # Shrink and square
    if shrink:
        patch = ~np.isnan(U)
        ind_patch_x = np.sum(patch,axis=1) != 0
        ind_patch_y = np.sum(patch,axis=0) != 0
        if np.sum(ind_patch_x) > np.sum(ind_patch_y):
            ind_patch_y = ind_patch_x
        elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
            ind_patch_x = ind_patch_y        
        n = np.sum(ind_patch_x)
        m = np.sum(ind_patch_y)          
        ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
        ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
        U = np.reshape(U[ind_patch_grd],(n,m))
        V = np.reshape(V[ind_patch_grd],(n,m))
        grid[0] = np.reshape(grid[0][ind_patch_grd],(n,m))
        grid[1] = np.reshape(grid[1][ind_patch_grd],(n,m))        
#    else:
#        n = grid[0].shape[0]
#        m = grid[1].shape[0]   
        
    k1_int = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
    k2_int = np.fft.fftshift((np.fft.fftfreq(m, d=dy))) 
    k_int_grd = np.meshgrid(k1_int,k2_int)

#    k1l, k2l = k1, k2
#    k1, k2 = np.meshgrid(k1,k2)
#    k1p, k2p = k1, k2

    U_mean = np.nanmean(U.flatten())
    V_mean = np.nanmean(V.flatten())
    
    U_t = U-U_mean
    V_t = V-V_mean
    
    U_t[np.isnan(U_t)] = 0.0
    V_t[np.isnan(V_t)] = 0.0

    # Spectra
    fftU = np.fft.fft2(U_t)
    fftV = np.fft.fft2(V_t)
    fftU  = np.fft.fftshift(fftU)
    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    
    Suu_int = 2*(np.abs(fftU)**2)*(dx*dy)**2/a_eff
    Svv_int = 2*(np.abs(fftV)**2)*(dx*dy)**2/a_eff
    Suv_int = 2*np.real(fftUV)*(dx*dy)**2/a_eff

       
###############################################################################
## Rotation of fuu    
#    if transform:
#        
#        U_mean = np.nanmean(U.flatten())
#        V_mean = np.nanmean(V.flatten())
#        # Wind direction
#        gamma = np.arctan2(V_mean,U_mean)
#        # Components in matrix of coefficients
#        S11 = np.cos(gamma)
#        S12 = np.sin(gamma)
#        T = np.array([[S11,S12], [-S12,S11]])
#        
#        K = np.array(np.c_[k_int_grd[0].flatten(),k_int_grd[1].flatten()]).T
#        K = np.dot(T,K)
#        
#        k1 = np.reshape(K[0,:],k_int_grd[0].shape)
#        k2 = np.reshape(K[1,:],k_int_grd[0].shape)
#        
#        Frot = np.array(np.c_[fftU.flatten(),fftV.flatten()]).T
#        Frot = np.dot(T,Frot)
#        fftU = np.reshape(Frot[0,:],k1.shape)
#        fftV = np.reshape(Frot[1,:],k2.shape)
#                
##        dk = np.min([np.max(k1p.flatten()),np.max(k2p.flatten())])/np.sqrt(2)
##        
##        ind_k1 = (k1l>-dk) & (k1l<dk)
##        ind_k2 = (k2l>-dk) & (k2l<dk) 
#        
##        k1_int = k1l[ind_k1]
##        k2_int = k2l[ind_k2]
##        k_int_grd = np.meshgrid(k1_int,k2_int)
#        
#        #k_1 = np.max(k1_int)
#        #k_2 = np.max(k2_int)
#        
#        #dx = 1/(2*k_1)
#        #dy = 1/(2*k_2)
#        
##        Dk_1 = np.max(np.diff(k1_int))
##        Dk_2 = np.max(np.diff(k2_int))
##        
##        n = 2*k_1/Dk_1
##        m = 2*k_2/Dk_2
##    
##        a_eff = 1/Dk_1/Dk_2
##    
##        mask = (k1<=dk) & (k1>=-dk)
#    
#        fftU_real = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              np.real(fftU.flatten()), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='cubic')
#        fftV_real = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              np.real(fftV.flatten()), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='cubic')
#        
#        fftU_imag = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              np.imag(fftU.flatten()), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='cubic')
#        fftV_imag = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              np.imag(fftV.flatten()), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='cubic')
#        
#        fftU = fftU_real+1j*fftU_imag
#        fftV = fftV_real+1j*fftV_imag 
#        
#        fftU = np.reshape(fftU,k_int_grd[0].shape)
#        fftV = np.reshape(fftV,k_int_grd[0].shape)
#        
#        fftUV = fftU*np.conj(fftV) 
#        
##        
#        Suu_int = 2*(np.abs(fftU)**2)*(dx*dy)**2/a_eff
#        Svv_int = 2*(np.abs(fftV)**2)*(dx*dy)**2/a_eff
#        Suv_int = 2*np.real(fftUV)*(dx*dy)**2/a_eff
#           
#        mask = np.isnan(Suu_int)
#        print(np.sum(mask),np.sum(~mask),np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        
#        Suu_int = Suu_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        Svv_int = Svv_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        Suv_int = Suv_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        
#        Suu_int[mask] = 0.0
#        Svv_int[mask] = 0.0
#        Suv_int[mask] = 0.0
    
###############################################################################
## Rotation of Suu        
#    if transform:
#        
#        U_mean = np.nanmean(U.flatten())
#        V_mean = np.nanmean(V.flatten())
#        # Wind direction
#        gamma = np.arctan2(V_mean,U_mean)
#        # Components in matrix of coefficients
#        S11 = np.cos(gamma)
#        S12 = np.sin(gamma)
#        T = np.array([[S11,S12], [-S12,S11]])
#        
#        K = np.array(np.c_[k_int_grd[0].flatten(),k_int_grd[1].flatten()]).T
#        K = np.dot(T,K)
#        
#        k1 = np.reshape(K[0,:],k_int_grd[0].shape)
#        k2 = np.reshape(K[1,:],k_int_grd[0].shape)
#        
##        dx_prime = np.abs(DX[0])
##        dy_prime = np.abs(DX[1])
#        
##        print(dx_prime,dy_prime,dx,dy)
#        
##        corr = 1#(dx_prime**2*dy_prime**2)/(dx**2*dy**2)
#        c = np.cos(gamma)
#        s = np.sin(gamma)
#        
#        Suu_prime = (c**2*Suu_int+s**2*Svv_int+s*c*(Suv_int+np.conj(Suv_int)))
#        Svv_prime = (s**2*Suu_int+c**2*Svv_int-s*c*(Suv_int+np.conj(Suv_int)))
#        Suv_prime = (-s*c*(Suu_int+Svv_int)+c**2*Suv_int-s**2*np.conj(Suv_int))
#              
##        dk = np.min([np.max(k1p.flatten()),np.max(k2p.flatten())])/np.sqrt(2)
##        
##        ind_k1 = (k1l>-dk) & (k1l<dk)
##        ind_k2 = (k2l>-dk) & (k2l<dk) 
##        
##        k1_int = k1l[ind_k1]
##        k2_int = k2l[ind_k2]
##        k_int_grd = np.meshgrid(k1_int,k2_int)
##    
##        mask = (k1<dk*1.1) & (k1>-dk*1.1)
#        
##        
##        Suu_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
##              Suu_prime[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
##              method='cubic')  
##        Svv_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
##              Svv_prime[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
##              method='cubic')    
##        Suv_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
##              Suv_prime[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
##              method='cubic')
#        
#        Suu_int = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              Suu_prime.flatten(), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='linear',rescale=True,fill_value=0.0)
#        Svv_int = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              Svv_prime.flatten(), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='linear',rescale=True,fill_value=0.0)   
#        Suv_int = sp.interpolate.griddata(np.c_[k1.flatten(),k2.flatten()],
#              Suv_prime.flatten(), (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
#              method='linear',rescale=True,fill_value=0.0)
#        
#        Suu_int = np.reshape(Suu_int,k_int_grd[0].shape)
#        Svv_int = np.reshape(Svv_int,k_int_grd[0].shape)
#        Suv_int = np.reshape(Suv_int,k_int_grd[0].shape)
#        
#        print(np.sum(Suu_int<0.0),np.sum(Suu_int>0.0))
#        
#        Suu_int[Suu_int<0.0] = 0.0
#        Svv_int[Svv_int<0.0] = 0.0
#        
##        Suu_int[Suu_int>np.max(Suu_prime)] = np.max(Suu_prime)
##        Svv_int[Svv_int>np.max(Svv_prime)] = np.max(Svv_prime)
#
#
#
##        mask = np.isnan(Suu_int)
##        print(np.sum(mask),np.sum(~mask),np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        
##        Suu_int = Suu_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
##        Svv_int = Svv_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
##        Suv_int = Suv_int/(np.sum(~mask)/(np.sum(~mask)+np.sum(mask)))
#        
##        Suu_int[mask] = 0.0
##        Svv_int[mask] = 0.0
##        Suv_int[mask] = 0.0
#        
#        ###############################################
#        
##    else:
##       
##        Suu_int = 2*(np.abs(fftU)**2)*(dx*dy)/(n*m-n_p)
##        Svv_int = 2*(np.abs(fftV)**2)*(dx*dy)/(n*m-n_p) 
##        Suv_int = 2*np.real(fftUV)*(dx*dy)/(n*m-n_p)
#        
##        Suu_int = Suu
##        Svv_int = Svv
##        Suv_int = Suv
##        k1_int = k1l
##        k2_int = k2l
##        k_int_grd = np.meshgrid(k1_int,k2_int)
      
#####################################################################################################
    if plot:
        C = 10**-4
        k_log_1 = np.sign(k_int_grd[0])*np.log10(1+np.abs(k_int_grd[0])/C)#np.log10(np.abs(k_int_grd[0]))   
        k_log_2 = np.sign(k_int_grd[1])*np.log10(1+np.abs(k_int_grd[1])/C)#np.log10(np.abs(k_int_grd[1]))
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        im=ax.contourf(k_log_1,k_log_2,np.log10(Suu_int),np.linspace(0,7,100),cmap='jet')
        ax.set_xlabel('$k_1$', fontsize=18)
        ax.set_ylabel('$k_2$', fontsize=18)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=18)
        
        fig.canvas.draw()
        
        xticks = np.max(ax.get_xlim())
        xticks = np.sign(xticks)*C*(10**(np.abs(xticks))-1)
        xticks = np.sign(xticks)*np.log10(np.abs(xticks))
        xticks = np.arange(np.ceil(np.log10(C)),np.ceil(xticks))
        xticks1 = 10**xticks    
        xticks = np.r_[xticks[::-1],-np.inf,xticks]
        xticks1 = np.r_[-xticks1[::-1],0,xticks1]
        xticks1 = np.sign(xticks1)*np.log10(1+np.abs(xticks1)/C);
        
        yticks = np.max(ax.get_ylim())
        yticks = np.sign(yticks)*C*(10**(np.abs(yticks))-1)
        yticks = np.sign(yticks)*np.log10(np.abs(yticks))
        yticks = np.arange(np.ceil(np.log10(C)),np.ceil(yticks))
        yticks1 = 10**yticks
        yticks = np.r_[yticks[::-1],-np.inf,yticks]
        yticks1 = np.r_[-yticks1[::-1],0,yticks1]
        yticks1 = np.sign(yticks1)*np.log10(1+np.abs(yticks1)/C);
           
        xticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(xticks,xticks1)]
        yticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(yticks,yticks1)]
        
        xticklabels_old =  [item.get_text() for item in ax.get_xticklabels()]
        
        #print(xticklabels,xticklabels_old)
        
        ax.set_xticks(xticks1)
        #print(xticks1)
        ax.set_xlim(-2,2)
        ax.set_yticks(yticks1)
        ax.set_ylim(-2,2)
        
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        
        ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
        
        ####################
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        im=ax.contourf(k_log_1,k_log_2,np.log10(Svv_int),np.linspace(0,7,100),cmap='jet')
        ax.set_xlabel('$k_1$', fontsize=18)
        ax.set_ylabel('$k_2$', fontsize=18)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=18)
        
        fig.canvas.draw()
        
        xticks = np.max(ax.get_xlim())
        xticks = np.sign(xticks)*C*(10**(np.abs(xticks))-1)
        xticks = np.sign(xticks)*np.log10(np.abs(xticks))
        xticks = np.arange(np.ceil(np.log10(C)),np.ceil(xticks))
        xticks1 = 10**xticks    
        xticks = np.r_[xticks[::-1],-np.inf,xticks]
        xticks1 = np.r_[-xticks1[::-1],0,xticks1]
        xticks1 = np.sign(xticks1)*np.log10(1+np.abs(xticks1)/C);
        
        yticks = np.max(ax.get_ylim())
        yticks = np.sign(yticks)*C*(10**(np.abs(yticks))-1)
        yticks = np.sign(yticks)*np.log10(np.abs(yticks))
        yticks = np.arange(np.ceil(np.log10(C)),np.ceil(yticks))
        yticks1 = 10**yticks
        yticks = np.r_[yticks[::-1],-np.inf,yticks]
        yticks1 = np.r_[-yticks1[::-1],0,yticks1]
        yticks1 = np.sign(yticks1)*np.log10(1+np.abs(yticks1)/C);
           
        xticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(xticks,xticks1)]
        yticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(yticks,yticks1)]
        
        xticklabels_old =  [item.get_text() for item in ax.get_xticklabels()]
        
        #print(xticklabels,xticklabels_old)
        
        ax.set_xticks(xticks1)
        #print(xticks1)
        ax.set_xlim(-2,2)
        ax.set_yticks(yticks1)
        ax.set_ylim(-2,2)
        
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        
        ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel("$\log_{10}{S_{vv}}$", fontsize=18)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
        
    #######################
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        im=ax.contourf(k_log_1,k_log_2,np.log10(-Suv_int),np.linspace(-7,7,100),cmap='jet')
        ax.set_xlabel('$k_1$', fontsize=18)
        ax.set_ylabel('$k_2$', fontsize=18)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=18)
        
        fig.canvas.draw()
        
        xticks = np.max(ax.get_xlim())
        xticks = np.sign(xticks)*C*(10**(np.abs(xticks))-1)
        xticks = np.sign(xticks)*np.log10(np.abs(xticks))
        xticks = np.arange(np.ceil(np.log10(C)),np.ceil(xticks))
        xticks1 = 10**xticks    
        xticks = np.r_[xticks[::-1],-np.inf,xticks]
        xticks1 = np.r_[-xticks1[::-1],0,xticks1]
        xticks1 = np.sign(xticks1)*np.log10(1+np.abs(xticks1)/C);
        
        yticks = np.max(ax.get_ylim())
        yticks = np.sign(yticks)*C*(10**(np.abs(yticks))-1)
        yticks = np.sign(yticks)*np.log10(np.abs(yticks))
        yticks = np.arange(np.ceil(np.log10(C)),np.ceil(yticks))
        yticks1 = 10**yticks
        yticks = np.r_[yticks[::-1],-np.inf,yticks]
        yticks1 = np.r_[-yticks1[::-1],0,yticks1]
        yticks1 = np.sign(yticks1)*np.log10(1+np.abs(yticks1)/C);
           
        xticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(xticks,xticks1)]
        yticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(yticks,yticks1)]
        
        xticklabels_old =  [item.get_text() for item in ax.get_xticklabels()]
        
        #print(xticklabels,xticklabels_old)
        
        ax.set_xticks(xticks1)
        #print(xticks1)
        ax.set_xlim(-2,2)
        ax.set_yticks(yticks1)
        ax.set_ylim(-2,2)
        
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        
        ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel("$\log_{10}{S_{uv}}$", fontsize=18)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
        
    ####################
        
#    fig, ax = plt.subplots()
#    im=ax.contourf(k1_int,k2_int,np.log10(Suu_int),np.linspace(0,7,50),cmap='jet')
#    ax.set_xlabel('$k_1$', fontsize=18)
#    ax.set_ylabel('$k_2$', fontsize=18)
#    #ax.set_xlim(-.005,0.005)
#    #ax.set_ylim(-.005,0.005)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=18)
#    ax.tick_params(labelsize=18)
#    cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
#    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
    
    if ring:
        Su=spectra_average(Suu_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #ku = Su.k        
        Suu_int = Su.S    
        Sv=spectra_average(Svv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kv = Sv.k
        Svv_int = Sv.S
        Suv=spectra_average(Suv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kuv = Suv.k
        Suv_int = Suv.S
        
#    else:
#    fig, ax = plt.subplots()
#    im=ax.contourf(k1_int,k2_int,np.log10(Suu_int),np.linspace(0,7,300),cmap='jet')
#    ax.set_xlabel('$k_1$', fontsize=18)
#    ax.set_ylabel('$k_2$', fontsize=18)
#    ax.set_xlim(-.005,0.005)
#    ax.set_ylim(-.005,0.005)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=18)
#    ax.tick_params(labelsize=18)
#    cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
#    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
    
    Su_u = sp.integrate.simps(Suu_int.T,k2_int,axis=1)[k1_int>0]
    Sv_v = sp.integrate.simps(Svv_int.T,k2_int,axis=1)[k1_int>0]
    Su_v = sp.integrate.simps(Suv_int.T,k2_int,axis=1)[k1_int>0]
    
    print((k2_int>0).shape,k2_int.shape,(k1_int>0).shape,k1_int.shape)
    
    return (k1_int[k1_int>0],k2_int[k2_int>0],Su_u,Sv_v,Su_v)  

# In[]
    
def avetriangles(xy,z,tri):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    ind = ~np.isnan(z)
    xy = xy[ind,:]
    z = z[ind]
    triangles = Triangulation(xy[:,0],xy[:,1]).triangles
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        aux = area * np.nanmean(z[tri],axis=0)
        if ~np.isnan(aux):
            zsum += aux
            areasum += area
    return zsum/areasum
     
# In[]   
def upsample2 (x, k):
  """
  Upsample the signal to the new points using a sinc kernel. The
  interpolation is done using a matrix multiplication.
  Requires a lot of memory, but is fast.
  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  output:
  y     the interpolated signal at points xp
  """
  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")
  nn = n * k
  [T, Ts]  = np.mgrid[1:n:nn*1j, 1:n:n*1j]
  TT = Ts - T
  del T, Ts
  y = np.sinc(TT).dot (x.reshape(n, 1))
  return y.squeeze()   

# In[Plots]
def plot_log2D(k_int_grd, S, label_S = "$\log_{10}{S}$", C = 10**-4):
    
    k_log_1 = np.sign(k_int_grd[0])*np.log10(1+np.abs(k_int_grd[0])/C)#np.log10(np.abs(k_int_grd[0]))   
    k_log_2 = np.sign(k_int_grd[1])*np.log10(1+np.abs(k_int_grd[1])/C)#np.log10(np.abs(k_int_grd[1]))
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    im=ax.contourf(k_log_1,k_log_2,np.log10(S),np.linspace(-1,6,100),cmap='jet')
    ax.set_xlabel('$k_1$', fontsize=18)
    ax.set_ylabel('$k_2$', fontsize=18)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    
    fig.canvas.draw()
    
    xticks = np.max(ax.get_xlim())
    xticks = np.sign(xticks)*C*(10**(np.abs(xticks))-1)
    xticks = np.sign(xticks)*np.log10(np.abs(xticks))
    xticks = np.arange(np.ceil(np.log10(C)),np.ceil(xticks))
    xticks1 = 10**xticks    
    xticks = np.r_[xticks[::-1],-np.inf,xticks]
    xticks1 = np.r_[-xticks1[::-1],0,xticks1]
    xticks1 = np.sign(xticks1)*np.log10(1+np.abs(xticks1)/C);
    
    yticks = np.max(ax.get_ylim())
    yticks = np.sign(yticks)*C*(10**(np.abs(yticks))-1)
    yticks = np.sign(yticks)*np.log10(np.abs(yticks))
    yticks = np.arange(np.ceil(np.log10(C)),np.ceil(yticks))
    yticks1 = 10**yticks
    yticks = np.r_[yticks[::-1],-np.inf,yticks]
    yticks1 = np.r_[-yticks1[::-1],0,yticks1]
    yticks1 = np.sign(yticks1)*np.log10(1+np.abs(yticks1)/C);
       
    xticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(xticks,xticks1)]
    yticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(yticks,yticks1)]
    
    xticklabels_old =  [item.get_text() for item in ax.get_xticklabels()]
    
    #print(xticklabels,xticklabels_old)
    
    ax.set_xticks(xticks1)
    #print(xticks1)
    ax.set_xlim(-2,2)
    ax.set_yticks(yticks1)
    ax.set_ylim(-2,2)
    
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    
    ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(label_S, fontsize=18)
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
    return []  
# In[Lanczos polar]  
def lanczos_kernel(r,r_1 = 1.22,r_2 = 2.233,a=1):
    kernel = lambda r: 2*sp.special.jv(1,a*np.pi*r)/r/np.pi/a
    kernel_w = kernel(r)*kernel(r*r_1/r_2)
    kernel_w[np.abs(r)>=r_2] = 0.0
    return kernel_w
    
def lanczos_int_sq(grid,tree,U,a=1):    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    X = grid[0].flatten()
    Y = grid[1].flatten()
    tree_grid = KDTree(np.c_[X,Y])
    d, n  = tree.query(tree_grid.data, k=40, return_distance = True)
    d=d/np.sqrt(dx*dy)
    S = np.sum(lanczos_kernel(d)*U[n],axis=1)
    S = np.reshape(S,grid[0].shape)
    return S
