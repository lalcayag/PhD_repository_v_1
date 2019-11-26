# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:04:18 2019

@author: lalc
"""

import os, sys, datetime, string
import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid#, NetCDFFile
from pylab import *
#import laplaceFilter
#import mpl_util

__author__   = 'Trond Kristiansen'
__email__    = 'trond.kristiansen (at) imr.no'
__created__  = datetime.datetime(2008, 8, 15)
__modified__ = datetime.datetime(2009, 7, 21)
__version__  = "1.0"
__status__   = "Development"

def findSubsetIndices(min_lat,max_lat,min_lon,max_lon,lats,lons):
    """Array to store the results returned from the function"""
    res=np.zeros((4),dtype=np.float64)
    minLon=min_lon; maxLon=max_lon
    distances1 = []; distances2 = []
    indices=[]; index=1
    for point in lats:
        s1 = max_lat-point # (vector subtract)
        s2 = min_lat-point # (vector subtract)
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index-1))
        index=index+1
        distances1.sort()
        distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])
    distances1 = []; distances2 = []; index=1
    
    for point in lons:
        s1 = maxLon-point # (vector subtract)
        s2 = minLon-point # (vector subtract)
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index-1))
        index=index+1
        distances1.sort()
        distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])
    
    """ Save final product: max_lat_indices,min_lat_indices,max_lon_indices,min_lon_indices"""
    minJ=indices[1][2]
    maxJ=indices[0][2]
    minI=indices[3][2]
    maxI=indices[2][2]
    res[0]=minI; res[1]=maxI; res[2]=minJ; res[3]=maxJ;
    return res

def makeMap(lonStart,lonEnd,latStart,latEnd,name,stLon,stLat):
    plt.figure(figsize=(8,8))
    etopo1name='ETOPO1_Ice_g_gmt4.grd'
    etopo1 = Dataset(etopo1name,'r')
    etopo2name='ETOPO1_Bed_g_gmt4.grd'
    etopo2 = Dataset(etopo2name,'r')
    
    lons = etopo1.variables["x"][:]
    lats = etopo1.variables["y"][:]
    
    dg = np.max([latEnd-latStart,lonEnd-lonStart])
    
    res = findSubsetIndices(latStart-dg,latEnd+dg,lonStart-dg,lonEnd+dg,lats,lons).astype(int)
    lon,lat=np.meshgrid(lons[res[0]:res[1]],lats[res[2]:res[3]])
    print ("Extracted data for area %s : (%s,%s) to (%s,%s)"%(name,lon.min(),lat.min(),lon.max(),lat.max()))
    bathy = etopo1.variables["z"][int(res[2]):int(res[3]),int(res[0]):int(res[1])]
    bathySmoothed = laplace_filter(bathy,M=None)
    
    oro = etopo2.variables["z"][int(res[2]):int(res[3]),int(res[0]):int(res[1])]
    oroSmoothed = laplace_filter(oro,M=None)
    
#    levels = [-6000,-5000,-3000, -2000, -1500, -1000,-500, -400, -300, -250,
#            -200, -150, -100, -75, -65, -50, -35, -25, -15, -10, -5, 0]
    

    
    levels_sea = np.unique(np.r_[-np.flip(np.exp(np.linspace(np.log(1),np.log(100),30))).astype(int),0])#
    levels_ground = np.unique(np.r_[0,np.exp(np.linspace(np.log(1),np.log(100),30)).astype(int)])
    
    if (lonStart < 0) & (lonEnd < 0):
        lon_0= - (abs(lonEnd)+abs(lonStart))/2.0
    else:
        lon_0=(abs(lonEnd)+abs(lonStart))/2.0
    mp = Basemap(llcrnrlat=latStart,urcrnrlat=latEnd,\
    llcrnrlon=lonStart,urcrnrlon=lonEnd,\
#    rsphere=(6378137.00,6356752.3142),\
    resolution='f',area_thresh=1000.,projection='merc',\
    lat_1=latStart,lon_0=lon_0)
    x, y = mp(lon,lat)
    mp.drawcoastlines()
#    mp.drawcountries()
#    mp.fillcontinents(color='grey')
    mp.drawmeridians(np.arange(lons.min(),lons.max(),10),labels=[0,0,0,1])
    mp.drawparallels(np.arange(lats.min(),lats.max(),4),labels=[1,0,0,0])
    #map.bluemarble()    
    CS1 = mp.contourf(x,y,bathySmoothed,levels_sea,
    cmap = LevelColormap(levels_sea,cmap=cm.Blues_r),
    extend='upper',
    alpha=1.0,
    origin='lower')
    
    CS2 = mp.contourf(x,y,oroSmoothed,levels_ground,
    cmap = LevelColormap(levels_ground,cmap=cm.gist_earth),
    extend='upper',
    alpha=1.0,
    origin='lower')
#    
    CS1.axis='tight'
    CS2.axis='tight'
    """Plot the station as a position dot on the map"""
    xpt,ypt = mp(stLon,stLat)
    mp.plot([xpt],[ypt],'ro', markersize=10)
    plt.text(xpt+100000,ypt+100000,name)
    plt.title('Area %s'%(name))
#    plotfile='figures/map_'+str(name)+'.pdf'
#    plt.savefig(plotfile,dpi=150,orientation='portrait')
    plt.show()
        
def laplace_X(F,M):
    """1D Laplace Filter in X-direction (axis=1)"""
    jmax, imax = F.shape
    # Add strips of land
    F2 = np.zeros((jmax, imax+2), dtype=F.dtype)
    F2[:, 1:-1] = F
    M2 = np.zeros((jmax, imax+2), dtype=M.dtype)
    M2[:, 1:-1] = M
    MS = M2[:, 2:] + M2[:, :-2]
    FS = F2[:, 2:]*M2[:, 2:] + F2[:, :-2]*M2[:, :-2]
    return np.where(M > 0.5, (1-0.25*MS)*F + 0.25*FS, F)

def laplace_Y(F,M):
    """1D Laplace Filter in Y-direction (axis=1)"""
    jmax, imax = F.shape
    # Add strips of land
    F2 = np.zeros((jmax+2, imax), dtype=F.dtype)
    F2[1:-1, :] = F
    M2 = np.zeros((jmax+2, imax), dtype=M.dtype)
    M2[1:-1, :] = M
    MS = M2[2:, :] + M2[:-2, :]
    FS = F2[2:, :]*M2[2:, :] + F2[:-2, :]*M2[:-2, :]
    return np.where(M > 0.5, (1-0.25*MS)*F + 0.25*FS, F)

def laplace_filter(F, M=None):
    if M == None:
        M = np.ones_like(F)
    
    return 0.5*(laplace_X(laplace_Y(F, M), M) +
    laplace_Y(laplace_X(F, M), M))    
    
import matplotlib
from numpy import ma
import pylab as pl
"""A set of utility functions for matplotlib"""

# ------------------
# Plot land mask
# ------------------

def landmask(M, color='0.8'):

   # Make a constant colormap, default = grey
   constmap = pl.matplotlib.colors.ListedColormap([color])

   jmax, imax = M.shape
   # X and Y give the grid cell boundaries,
   # one more than number of grid cells + 1
   # half integers (grid cell centers are integers)
   X = -0.5 + pl.arange(imax+1)
   Y = -0.5 + pl.arange(jmax+1)

   # Draw the mask by pcolor
   M = ma.masked_where(M > 0, M)
   pl.pcolor(X, Y, M, shading='flat', cmap=constmap)

# -------------
# Colormap
# -------------

def LevelColormap(levels, cmap=None):
    """Make a colormap based on an increasing sequence of levels"""
    
    # Start with an existing colormap
    if cmap == None:
        cmap = pl.get_cmap()

    # Spread the colours maximally
    nlev = len(levels)
    S = pl.arange(nlev, dtype='float')/(nlev-1)
    A = cmap(S)

    # Normalize the levels to interval [0,1]
    levels = pl.array(levels, dtype='float')
    L = (levels-levels[0])/(levels[-1]-levels[0])

    # Make the colour dictionary
    R = [(L[i], A[i,0], A[i,0]) for i in range(nlev)]
    G = [(L[i], A[i,1], A[i,1]) for i in range(nlev)]
    B = [(L[i], A[i,2], A[i,2]) for i in range(nlev)]
    cdict = dict(red=tuple(R),green=tuple(G),blue=tuple(B))

    # Use 
    return matplotlib.colors.LinearSegmentedColormap(
        '%s_levels' % cmap.name, cdict, 256)    



