# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:02:04 2019

@author: lalc
"""
import os
os.environ['PROJ_LIB'] = r'C:\Users\lalc\AppData\Local\Continuum\anaconda3\Library\share'

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from mpl_toolkits.basemap import pyproj
import numpy as np
from matplotlib.patches import Wedge, Circle
from matplotlib.ticker import EngFormatter, StrMethodFormatter

# In[]

fig, ax = plt.subplots(1,2,figsize=(8, 6))
#ax.set_aspect('equal')
ax[0].use_sticky_edges = False
ax[0].margins(0.07)
ax[0].text(0.05, 0.95, '(a)', color='black', transform=ax[0].transAxes, fontsize=24,
        verticalalignment='top')

ax[1].use_sticky_edges = False
ax[1].margins(0.07)
ax[1].text(0.05, 0.95, '(b)', color='white', transform=ax[1].transAxes, fontsize=24,
        verticalalignment='top')

utm = 'epsg:32632'
tmer = 'epsg:5520'
web = 'epsg:3857'
geo = 'epsg:4326'

#inProj = Proj(init=web)
#outProj = Proj(init=tmer)
#########################################################
#########################################################
#
#x1, y1 = 988595.319505, 7777932.818994
#x2, y2 = 988279.718181, 7770711.311313
#x1dm, y1dm = 988751.179383, 7773664.885859

x1, y1 = 988595.319505, 7777932.818994
x2, y2 = 988279.718181, 7770711.311313
x1dm, y1dm = 988751.179383, 7773664.885859

scale = 40
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = .8*dy
x0, y0 = x2-dx, y2-dy
x3, y3 = x1+dx, y1+dy

inProj = Proj(init=web)
lon, lat = inProj([x0, x3], [y0, y3], inverse=True)
x1dm, y1dm= inProj(x1dm, y1dm, inverse=True)

#################################################
def format_funcx(value, tick_number):
    # find number of multiples of pi/2
    inProj = Proj(init = 'epsg:3857')
    deg,_ = inProj(x0+value, y0, inverse=True)
    return u"${0:.1f} ^\circ$".format(deg)

def format_funcy(value, tick_number):
    # find number of multiples of pi/2
    inProj = Proj(init = 'epsg:3857')
    _, deg = inProj(x0, y0+value, inverse=True)
    return u"${0:.1f} ^\circ$".format(deg)
###############################################


#x1, y1 = transform(Proj(init='epsg:3857'),Proj(init=geo), x1+dx, y1+dy)
#x2, y2 = transform(Proj(init='epsg:3857'),Proj(init=geo), x2-dx, y2-dy)
#x1dm, y1dm = transform(Proj(init='epsg:3857'),Proj(init=geo), x1dm, y1dm)

mymap = Basemap(llcrnrlon = lon[0], llcrnrlat=lat[0], urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax[0],epsg=3857)

mymap.ax = ax[0]
mymap.shadedrelief()
mymap.drawcoastlines()
ptmid = mymap.plot(x1dm, y1dm, 'o',markersize = 40, markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 6, latlon=True)
mymap.arcgisimage(service = 'World_Physical_Map', xpixels = 3000, verbose= True)


x1d, y1d = inProj(x1, y1, inverse=True)
x1dm, y1dm = inProj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = inProj(x2, y2, inverse=True)

my = ax[0].get_ylim()[1]
mx = ax[0].get_xlim()[1]
c1, c2 = mx/np.diff(lon), my/np.diff(lat)

ax[0].xaxis.set_major_locator(plt.MultipleLocator(mx/4))
ax[0].xaxis.set_minor_locator(plt.MultipleLocator(mx/8))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_funcx))

ax[0].yaxis.set_major_locator(plt.MultipleLocator(my/4))
ax[0].yaxis.set_minor_locator(plt.MultipleLocator(my/8))
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(format_funcy))

ax[0].set_xlabel('$Longitude$', fontsize=24)
ax[0].set_ylabel('$Latitude$', fontsize=24)

#########################################################
#########################################################
x1, y1 = 988595.319505, 7777932.818994
x2, y2 = 988279.718181, 7770711.311313
x1dm, y1dm = 988751.179383, 7773664.885859
scale = 1.3
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = 2*dy
x0, y0 = x2-dx, y2-dy
x3, y3 = x1+dx, y1+dy

inProj = Proj(init=web)
lon, lat = inProj([x0, x3], [y0, y3], inverse=True)
x1dm, y1dm= inProj(x1dm, y1dm, inverse=True)

mymap = Basemap(llcrnrlon = lon[0], llcrnrlat=lat[0], urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax[1],epsg=3857)

mymap.ax = ax[1]
mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)

x3d, y3d = inProj(x3, y3, inverse=True)

x1d, y1d = inProj(x1, y1, inverse=True)
x1dm, y1dm = inProj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = inProj(x2, y2, inverse=True)
pt1 = mymap.plot([x1d,x2d], [y1d,y2d], 'ro', markersize=8, latlon=True, label = '$Meteorological\:masts$')

fig.set_size_inches(16,8)

my = ax[1].get_ylim()[1]
mx = ax[1].get_xlim()[1]
c1, c2 = mx/x3d, my/y3d

#################################################
def format_funcx(value, tick_number):
    # find number of multiples of pi/2
#    inProj = Proj(init = 'epsg:3857')
#    deg,_ = inProj(x0+value, y0, inverse=True)
#    return u"${0:.2f} ^\circ\:E$".format(deg)
    c = 4300/7163.933783133118  
    return u"${0:.1f}$".format(c*(value-mx/2)/1000)

def format_funcy(value, tick_number):
    # find number of multiples of pi/2
#    inProj = Proj(init = 'epsg:3857')
#    _, deg = inProj(x0, y0+value, inverse=True)
#    return u"${0:.2f} ^\circ\:N$".format(deg)
    c = 4300/7163.933783133118    
    return u"${0:.1f}$".format(c*(value-my/2)/1000)

###############################################


#c1, c2 = (14500)*scale/x1d, (14758)*scale/y1d



x1dm, y1dm = 14758*scale, 7221*scale+7222 #14443 
x2dm, y2dm = 14443*scale, 7221*scale
R = 1.1*7000*np.sqrt((x1dm-x2dm)**2+(y1dm-y2dm)**2)/4300
wedge = Wedge((x2dm, y2dm), R, 106, 196, width=R, color ='r',alpha = 0.25)
ax[1].add_artist(wedge)
wedge = Wedge((x1dm, y1dm), R, 166, 256, width=R, color ='b', alpha = 0.25)
ax[1].add_artist(wedge)
wedge = Wedge((x1dm, y1dm), R, 106+180, 196+180, width=R, color ='b',alpha=.25)
ax[1].add_artist(wedge)
wedge = Wedge((x2dm, y2dm), R, 166+180, 256+180, width=R, color ='r', alpha = 0.25)
ax[1].add_artist(wedge)

ax[1].xaxis.set_major_locator(plt.MultipleLocator(mx/5))
ax[1].xaxis.set_minor_locator(plt.MultipleLocator(mx/10))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(format_funcx))

ax[1].yaxis.set_major_locator(plt.MultipleLocator(my/5))
ax[1].yaxis.set_minor_locator(plt.MultipleLocator(my/10))
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(format_funcy))

#s = ax[1].legend(fontsize=22, framealpha = 0.2)
#
#for t in s.get_texts():
#    t.set_color("white")

ax[0].tick_params(axis='both', which='major', labelsize=24)
ax[1].tick_params(axis='both', which='major', labelsize=24)

ax[1].set_xlabel('$Easting\:[km]$', fontsize=24)
ax[1].set_ylabel('$Northing\:[km]$', fontsize=24)

fig.tight_layout()

ax[0].grid(color='Grey', linestyle='-')


############################################################
############################################################
############################################################
# In[]
############################################################
############################################################
############################################################

transform(Proj(inProj),Proj(init='epsg:4326'), 989706.642236, 7775327.391225)

57.067910
8.883683

x1, y1 = 492768.8, 6322832.3
x2, y2 = 492768.7, 6327082.4
scale = 50
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = .5*dy

myproj = Proj("+proj=tmerc +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)


lat = [57.067910-3, 57.067910+2]
lon = [8.883683-3, 8.883683+4]
x1dm, y1dm =  8.883683, 57.067910

ax[0].set_xticks(np.linspace(lon[0]+.1, lon[1]-.1, 4))
ax[0].set_yticks(np.linspace(lat[0]+.1, lat[1]-.1, 4))

ax[0].set_xlim(lon[0], lon[1])
ax[0].set_ylim(lat[0], lat[1])

x1d, y1d = myproj(x1, y1, inverse=True)
x1dm, y1dm = myproj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

mymap = Basemap(projection = 'tmerc', llcrnrlon = lon[0],llcrnrlat=lat[0],urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax[0], epsg=3857)

mymap.ax = ax[0]
ptmid = mymap.plot(x1dm, y1dm, 'o',markersize=40, markerfacecolor= 'none', markeredgecolor= 'red', latlon=True)
mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)

scale = 1.5
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = 2*dy

myproj = Proj("+proj=tmerc +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)

lat = [57.048533-2/35, 57.086582+2/35]
lon = [8.880641-4/35, 8.880392+5/35]

x1d, y1d = myproj(x1, y1, inverse=True)
x1dm, y1dm = myproj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

x1d, y1d, x2d, y2d = 57.048533, 8.880641, 57.086582, 8.880392

mymap = Basemap(projection = 'tmerc', llcrnrlon = lon[0],llcrnrlat=lat[0],urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax[1], epsg=5520)

mymap.ax = ax[1]
pt0 = mymap.plot(x1d, y1d, 'o', markersize=15, markerfacecolor= 'red', markeredgecolor= 'red', latlon=True, label = 'Meteo. masts and WindSanner location')
pt1 = mymap.plot(x2d, y2d, 'ro', markersize=15, latlon=True, label = 'Meteo. masts and WindSanner location')
mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)

fig.set_size_inches(20,8)

ax[1].set_xticks(np.linspace(lon[0]+.01, lon[1]-.01, 4))
ax[1].set_yticks(np.linspace(lat[0]+.01, lat[1]-.01, 4))

ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=16)


ax[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ N"))
ax[0].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ E"))
ax[1].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ N"))
ax[1].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ E"))


myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lonw, latw = myproj([x1,x1+7000], [y1,y1], inverse=True)

ax[1].legend(fontsize=22)

wedge = Wedge((x1d, y1d), np.diff(lonw), 106, 196, width=np.diff(lonw), color ='r',alpha=.25)
ax[1].add_artist(wedge)
wedge = Wedge((x2d, y2d), np.diff(lonw), 166, 256, width=np.diff(lonw), color ='b', alpha = 0.25)
ax[1].add_artist(wedge)
wedge = Wedge((x2d, y2d), np.diff(lonw), 106+180, 196+180, width=np.diff(lonw), color ='b',alpha=.25)
ax[1].add_artist(wedge)
wedge = Wedge((x1d, y1d), np.diff(lonw), 166+180, 256+180, width=np.diff(lonw), color ='r', alpha = 0.25)
ax[1].add_artist(wedge)
fig.tight_layout()
plt.savefig(r'C:\Users\lalc\Documents\Old Documents folder\Publications\Filtering using a clustering algorithm\Figures\balcony')

# lons, lats, xs, ys = mymap.makegrid(200, 200, returnxy=True)
# gc = pyproj.Geod(a=mymap.rmajor, b=mymap.rminor)
# distances1 = np.zeros(lons.size)
# distances2 = np.zeros(lons.size)
# for k, (lo, la) in enumerate(zip(lons.flatten(), lats.flatten())):
#     _, _, distances1[k] = gc.inv(x1d, y1d, lo, la)
#     _, _, distances2[k] = gc.inv(x2d, y2d, lo, la)
#    
# distances1 = distances1.reshape(200, 200)  # In km.
# distances2 = distances2.reshape(200, 200)  # In km.

# Plot perimeters of equal distance.
# levels = [1000]  # [50, 100, 150]
# cs1 = mymap.contour(xs, ys, distances1, levels, colors='k')
# cs2 = mymap.contour(xs, ys, distances2, levels, colors='k')
#
#mymap.arcgisimage(service='World_Shaded_Relief', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='World_Topo_Map', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='Ocean_Basemap', xpixels = 3000, verbose= True)





#http://server.arcgisonline.com/arcgis/rest/services


myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
scale = 2
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = dy
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)
x1d, y1d = myproj(x1, y1, inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

lat_start, lat_end = lat[0], lat[1]
lon_start, lon_end = lon[0], lon[1]

names=['Østerild']

makeMap(lon_start, lon_end, lat_start, lat_end, names, [x1d,x2d], [y1d,y2d])


import sys
sys.path.append("/Users/lalc/dev/ext-libs")

import elevation

from elevation import cli
cli.selfcheck()

import elevation
import os
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling

bounds = np.array([lon[0], lat[0], lon[1], lat[1]])

bounds = [-90.000672,	45.998852,	-87.998534,	46.999431]

west, south, east, north = bounds
west, south, east, north = bounds  = west - .05, south - .05, east + .05, north + .05
dem_path = '\\Iron_River_DEM.tif'
output = os.getcwd() + dem_path
elevation.clip(bounds=bounds, output=output, product='SRTM3')





