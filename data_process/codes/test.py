# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:56:11 2020

@author: lalc
"""

import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import data    
from matplotlib.path import Path 
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from shapely.geometry import LineString
from shapely import geometry

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter().set_scientific(True)

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

# lena = sp.misc.lena() this function was deprecated in version 0.17
output_m = []

for i in range(len(output_m),len(vort_spatial_smooth)):   
    print(i)
    img = filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s_i[i]/2) # use a standard image from skimage instead
    ind = np.isnan(img)
    img[ind] = 0
    LoG = nd.gaussian_laplace(img , 2)
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
    output_m.append(output)


fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for i in range(63,len(output_m)):       
    ax.cla()
    im = ax.contour(output_m[i])
    plt.pause(.5)
    
plt.show()

plt.figure()
plt.imshow(img)
plt.show()


plt.figure()
plt.imshow(LoG)
plt.show()


# In[]

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

#for some reason I had to reshape. Numpy ignored the shape header.
#paws_data = np.loadtxt("paws.txt").reshape(4,11,14)

#getting a list of images
#paws = [p.squeeze() for p in np.vsplit(paws_data,4)]


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

def coherent_detection(vort, U, V, grd, c = 2, alpha = .15):
    # Mean wind  and mean shear
    ch = len(U) 
    U = np.vstack(U)
    V = np.vstack(V)
    chunk = np.min(U.shape)
    U =  U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)).sum(axis=0)/ch
    V =  V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)).sum(axis=0)/ch
    u, v = U_rot(grd,U,V, tri_calc = True)
    imgU = filterfft(u,np.isnan(u),sigma=s_i[i]*2)
    dudy, dudx = np.gradient(imgU, grd[0][0,:], grd[1][:,0]) 
    # Vorticity smoothing
    chv = len(vort)
    img = vort_array.reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/chv
    detected_peaks = detect_peaks(img)
    ind_peaks = (np.abs(img)> np.nanmax(np.abs(dudy)*c)) & detected_peaks
    
    peaks = img[ind_peaks]
    
    cont = []
    
    for i,p in enumerate(peaks):
        cont = alpha*p
        min_vort = np.nanmin(img[ind_peaks])

    #smoothing
    
     
    return contour

def coherent_param():
    
    return coh_par

i = 180
chunk = 512
ch = 28
#vort_array = np.vstack([filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s_i[i]) for i in range(i-1,i+2)])
#img = vort_array.reshape((int(len(vort_array)/chunk),chunk,chunk)).sum(axis=0)/ch

img = filterfft(vort_m[i],np.isnan(vort_m[i]),sigma=s_i[i]/2) 
# Mean wind speed and
U_arr = np.vstack([U_out[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
V_arr = np.vstack([V_out[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
U =  np.nanmedian(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)
V =  np.nanmedian(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)
u, v, img_rot  = U_rot(grd, U, V, img, tri_calc = True)
imgU = filterfft(u,np.isnan(u),sigma=s_i[i]/2)
imgV = filterfft(u,np.isnan(v),sigma=s_i[i]/2)

Uf = filterfft(U,np.isnan(U),sigma=s_i[i]/2)
Vf = filterfft(V,np.isnan(V),sigma=s_i[i]/2)

g = np.arctan2(V,U)*180/np.pi
gf = filterfft(g,np.isnan(g),sigma=s_i[i]/2)


# Peak detection
detected_peaks = detect_peaks(img_rot)
dudy, dudx = np.gradient(imgU, grd[0][0,:], grd[1][:,0]) 
dvdy, dvdx = np.gradient(imgV, grd[0][0,:], grd[1][:,0]) 

#cont = edge_detec(img, sigma=2)

urot, vrot, _ = U_rot(grd, U_out[i], V_out[i], img, tri_calc = True)

x = grd[0][0,:]
y = grd[1][:,0]
tri_calc = True
U, V, mask, mask_int, tri_del = field_rot(x, y, U_out[i], V_out[i], grid = grd,
                                    tri_calc = tri_calc)
aux = img.flatten()
mask_C = ~np.isnan(aux)

aux[mask] = sp.interpolate.NearestNDInterpolator(np.c_[grd[0].flatten(),grd[1].flatten()][mask_C],
                                                     aux[mask_C])(np.c_[grd[0].flatten(),grd[1].flatten()][mask])

aux[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, aux[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
aux[~mask_int] = np.nan

img = np.reshape(aux,grd[0].shape)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],urot,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],imgU,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],dvdy,cmap='jet')
fig.colorbar(im)



fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],img_rot,cmap='jet')

slim_peak = 2.0
slim_cont = 1.5

S_M = np.nanmax(dudy)
S_m = np.nanmin(dudy)

ind_peaks_pos = (img_rot > S_M*slim_peak) & detected_peaks
ind_peaks_neg = (img_rot < S_m*slim_peak) & detected_peaks

ind_peaks = (np.abs(img_rot)> np.nanmax(S_M*slim_peak)) & detected_peaks
ind_peaks = ind_peaks_pos | ind_peaks_neg

max_vort = S_M*slim_cont #np.nanmax(img_rot[ind_peaks])
min_vort = S_m*slim_cont #np.nanmin(img_rot[ind_peaks])
peaks = img_rot[ind_peaks]
#contours = ax.contour(grd[0],grd[1], img_rot, levels = [alpha*np.sort(peaks)[-1]], colors='black',linewidths = 3) 
contours = ax.contour(grd[0],grd[1], img_rot, levels = [min_vort,max_vort], colors='black',linewidths = 3) 
#contours = ax.contour(grd[0],grd[1], cont, colors='black',alpha=.8,linewidths = 1) 
fig.colorbar(im)
ind_large = np.argsort(np.abs(img)[ind_peaks])[-10:]
ax.scatter(grd[0][ind_peaks][np.argsort(peaks)],grd[1][ind_peaks][np.argsort(peaks)],marker='+',s=100,color='r')
#ax.scatter(grd[0][ind_peaks][ind_large],grd[1][ind_peaks][ind_large],s=100,color='k')

plt.figure()
plt.plot(np.sort(np.abs(img_rot[detected_peaks])))
#plt.plot(np.sort(np.abs(img_rot[ind_peaks])))
plt.plot([0,len(np.sort(np.abs(img_rot[detected_peaks])))-1],[np.nanmax(np.abs(dudy))*2,np.nanmax(np.abs(dudy))*2], '--r')

dat0= contours.allsegs[0]
for d in dat0:
    ax.plot(d[:,0],d[:,1],'k',lw=4)

dat0= contours.allsegs[0]
for d in dat0:
    ax.plot(d[:,0],d[:,1],'--k',lw=4)


###############################################################
    
from skimage import measure
from shapely.geometry import LineString
from shapely import geometry


x, y = np.meshgrid(np.linspace(-np.pi,np.pi,100), np.linspace(-np.pi,np.pi,100))
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
contours = measure.find_contours(r, 0.2)
n,m = x.shape
sx = (x[0,-1]-x[0,0])/(n-1)
sy = (y[-1,0]-y[0,0])/(m-1)
cont_points = []
poly = []

fig, ax = plt.subplots()
ax.contour(x,y,r.T)
x_range= [-np.pi,np.pi]
y_range= [-np.pi,np.pi]
N_points = 100
x_rand = -np.pi+(2*np.pi)*np.random.rand(N_points)
y_rand = -np.pi+(2*np.pi)*np.random.rand(N_points)
counts = 0
plt.scatter(x_rand,y_rand)

for c in contours:
    cx, cy = c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0]  
    cont_points.append(np.c_[cx, cy])
    poly.append(geometry.Polygon([[xi, yi] for xi,yi in zip(cx,cy)]))
#    ax.plot(c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0], 'r', lw=2)

    

for p, c in zip(poly, contours):
    i = 2
    p = poly[i]
    c = contours[i]
    x_in = []
    y_in = []
    for xi,yi in zip(x_rand,y_rand):
        print(xi,yi)
        xy_point=geometry.Point(xi, yi)
        if p.contains(xy_point):
            counts+=1
            x_in.append(xi)
            y_in.append(yi)
    x_in = np.array(x_in)
    y_in = np.array(y_in) 
    plt.scatter(x_in,y_in, color = 'r')  
    ax.plot(c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0], 'r', lw=2)

##############################################################################

#Utest, Vtest, masktest, mask_inttest, tri_del, gamma, Xx = field_rot(x, y, U_out[j], V_out[j], gamma = gamma, tri_calc = False, tri_del = tri_del)
#
#ind_nan = np.isnan(Utest[mask])
#vertex_nan = np.array(range(0,len(Utest[mask])))[ind_nan]
#simplices_n = np.array(range(0,len(tri_del.simplices)))
#simplices_n = np.c_[simplices_n, simplices_n, simplices_n]
#ind_simp = np.isin(tri_del.simplices.flatten(), vertex_nan)
#ind_simp = np.unique(simplices_n.flatten()[ind_simp])
#ind_simp = ~np.isin(np.array(range(0,len(tri_del.simplices))), ind_simp)
#Uint = np.zeros(grd[1].flatten().shape)*np.nan
#ind_simp = ind_simp & ~(circleratios(tri_del)<.05)
#simp_grid = -np.ones(grd[1].flatten().shape)
#simp_grid[mask_int] = tri_del.find_simplex(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
#mask_simp = np.isin(simp_grid,np.array(range(0,len(tri_del.simplices)))[ind_simp])
#
#U_tesa = np.zeros(Utest.shape)*np.nan
#U_tesa[mask] = sp.interpolate.NearestNDInterpolator(tri_del.points[~ind_nan,:],Utest[mask][~ind_nan])(Xx.T[mask,0],Xx.T[mask,1])
#Uint = np.zeros(Utest.shape)*np.nan
#Uint[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del,U_tesa[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_simp])
#
##Uint[mask_simp] = sp.interpolate.CloughTocher2DInterpolator(tri_del.points[~ind_nan,:],
##                                                            Utest[mask][~ind_nan])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_simp])
#
#plt.triplot(tri_del.points[:,0], tri_del.points[:,1],triangles=tri_del.simplices, color = 'r')
#plt.triplot(tri_del.points[:,0], tri_del.points[:,1],triangles=tri_del.simplices[ind_simp], color = 'k')
#plt.scatter(tri_del.points[:,0], tri_del.points[:,1], color = 'k')
#plt.scatter(tri_del.points[vertex_nan,0], tri_del.points[vertex_nan,1], s = 100, color = 'red')
#plt.scatter(grd[0].flatten()[mask_simp],grd[1].flatten()[mask_simp], s = 100, color = 'green')
#
#Uint=np.zeros(Utest.shape)*np.nan
#tri = matplotlib.tri.Triangulation(tri_del.points[:,0], tri_del.points[:,1], triangles=tri_del.simplices[ind_simp])#, mask=~ind_simp)
#Uint[mask_int] = matplotlib.tri.CubicTriInterpolator(tri, Utest[mask], kind='min_E')(grid[0].flatten()[mask_int],grid[1].flatten()[mask_int])
#
#plt.triplot(tri_del.points[:,0], tri_del.points[:,1],triangles=tri_del.simplices[simp])

###########################################################
    
####################################################################
# In[]
### Reconstructed fields
#Phase 1
U_out1, V_out1, grd, s_u1 = joblib.load(file_out_path_u_field+'/U_rec_20160421.pkl')  
U_out1 = [U_out1[i] for i in range(800)]
V_out1 = [V_out1[i] for i in range(800)]
#Phase 2
#with open(file_out_path_u_field+'/U_rec_08_06_2016.pkl', 'rb') as field:
#     U_out2, V_out2, _, su2 = pickle.load(field)
U_out2, V_out2, grd, s_u2 = joblib.load(file_out_path_u_field+'/U_rec_08_06_2016.pkl')      
U_out2 = [U_out2[i] for i in range(800)]
V_out2 = [V_out2[i] for i in range(800)]     
 
phase1 = 512
phase2 = 537  

#phase1 = 512
phase2 = 512 

# In[] 
i = phase1
chunk = 512
ch = 1

tri_calc = True

x = grd[0][0,:]
y = grd[1][:,0]

## Mean wind speed and
#U_arr = np.vstack([filterfft(U_out[i],np.isnan(U_out[i]),sigma=s_i[i]) for i in range(i-int(ch/2),i+int(ch/2)+1)])
#V_arr = np.vstack([filterfft(V_out[i],np.isnan(V_out[i]),sigma=s_i[i]) for i in range(i-int(ch/2),i+int(ch/2)+1)])
#scan = np.vstack([su[i][0] for i in range(i-int(ch/2),i+int(ch/2)+1)])

# Mean wind speed and
U_arr = np.vstack([U_out1[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
V_arr = np.vstack([V_out1[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
scan1 = [s_u1[i][0] for i in range(i-int(ch/2),i+int(ch/2)+1)]


U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)

u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc)
u1, v1 = U_rot(grd, U, V, gamma = gamma, tri_calc = False, 
                          tri_del = tri_del, mask_int = mask_int, mask = mask)

aux = np.sqrt(U_arr**2+V_arr**2).reshape((int(len(U_arr)/chunk),chunk,chunk))
Umm =  [np.nanmin(aux), np.nanmax(aux)]#/len(U_arr)

# In[] 
i = phase2
chunk = 512
ch = 1

tri_calc = True

x = grd[0][0,:]
y = grd[1][:,0]

## Mean wind speed and
#U_arr = np.vstack([filterfft(U_out[i],np.isnan(U_out[i]),sigma=s_i[i]) for i in range(i-int(ch/2),i+int(ch/2)+1)])
#V_arr = np.vstack([filterfft(V_out[i],np.isnan(V_out[i]),sigma=s_i[i]) for i in range(i-int(ch/2),i+int(ch/2)+1)])
#scan = np.vstack([su[i][0] for i in range(i-int(ch/2),i+int(ch/2)+1)])

# Mean wind speed and
U_arr = np.vstack([U_out2[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
V_arr = np.vstack([V_out2[i] for i in range(i-int(ch/2),i+int(ch/2)+1)])
scan2 = [s_u2[i][0] for i in range(i-int(ch/2),i+int(ch/2)+1)]


U =  np.nanmean(U_arr.reshape((int(len(U_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)
V =  np.nanmean(V_arr.reshape((int(len(V_arr)/chunk),chunk,chunk)),axis=0)#/len(U_arr)

u, v, mask, mask_int, tri_del, gamma, Xx = field_rot(x, y, U, V, tri_calc = tri_calc)
u2, v2 = U_rot(grd, U, V, gamma = gamma, tri_calc = False, 
                          tri_del = tri_del, mask_int = mask_int, mask = mask)

aux = np.sqrt(U_arr**2+V_arr**2).reshape((int(len(U_arr)/chunk),chunk,chunk))
Umm =  [np.nanmin(aux), np.nanmax(aux)]#/len(U_arr)

########################################################################################################
# In[]

def autocorr_interp_sq(r, eta, tau, N = [], tau_lin = [], eta_lin = []):
    if (len(eta_lin) == 0) | (len(eta_lin) == 0):
        if len(N) == 0:
            N = 2**(int(np.ceil(np.log(np.max([tau.shape[1],eta.shape[0]]))/np.log(2)))+3)
        tau_lin = np.linspace(np.min(tau.flatten()),np.max(tau.flatten()),N)
        eta_lin = np.linspace(np.min(eta.flatten()),np.max(eta.flatten()),N)
        tau_lin, eta_lin = np.meshgrid(tau_lin,eta_lin)
    ind = ~np.isnan(r.flatten())
    tri_tau = Delaunay(np.c_[tau.flatten()[ind],eta.flatten()[ind]])   
    r_int = sp.interpolate.CloughTocher2DInterpolator(tri_tau, r.flatten()[ind])(np.c_[tau_lin.flatten(),eta_lin.flatten()])
    r_int[np.isnan(r_int)] = 0.0
    return (tau_lin,eta_lin,np.reshape(r_int,tau_lin.shape))

csv_database_r1 = create_engine('sqlite:///'+file_in_path_corr1+'/corr_uv_west_phase1_ind.db')
csv_database_r2 = create_engine('sqlite:///'+file_in_path_corr2+'/corr_uv_west_phase2_ind.db')

drel_phase1 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r1)
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r2)



L1 = dfL_phase1[['$L_{u,x}$', '$L_{u,y}$']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))].mean()
L2 = dfL_phase2[['$L_{u,x}$', '$L_{u,y}$']].loc[dfL_phase2['name']=='20160806'].loc[dfL_phase2['scan'].isin(np.squeeze(scan2))]


L_1 = dfL_phase1[['$L_{flux,103}$']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))]
L_2 = dfL_phase2[['$L_{flux,103}$']].loc[dfL_phase2['name']=='20160806'].loc[dfL_phase2['scan'].isin(np.squeeze(scan2))]

scl = dfL_phase2[['scan']].loc[(dfL_phase2['$L_{u,x}$']>800) &(dfL_phase2['$L_{u,x}$']<900)].loc[dfL_phase2['name']=='20160806']


hms = drel_phase1[['scan0','hms']].loc[drel_phase1['name']=='20160421'].loc[drel_phase1['scan0'].isin(np.squeeze(scan1))]
df_corr_phase1 = pd.read_sql_query("select * from 'corr' where name = '20160421' and hms >= '" +                                   
                                   hms.hms.min() + "' and hms <= '" +  hms.hms.max()+"'",csv_database_r1)


hms = drel_phase2[['scan0','hms']].loc[drel_phase2['name']=='20160806'].loc[drel_phase2['scan0'].isin(np.squeeze(scan2))]
df_corr_phase2 = pd.read_sql_query("select * from 'corr' where name = '20160806' and hms >= '" +                                   
                                   hms.hms.min() + "' and hms <= '" +  hms.hms.max()+"'",csv_database_r2)

##############################################################################
# In[Phase 1]
i = phase1
s1 = [s_u1[j][0] for j in range(i-int(ch/2),i+int(ch/2)+1)]
reslist1 = [df_corr_phase1[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[df_corr_phase1.scan == s].values for s in s1]

tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist1]
eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist1]
ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist1]
rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist1]
ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist1]

resarray = np.vstack([r for r in reslist1])

tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)

N = 512
taumax = np.nanmax(np.abs(tau_arr))
taui = np.linspace(0,taumax,int(N/2)+1)
taui = np.r_[-np.flip(taui[1:]),taui]
etamax = np.nanmax(np.abs(eta_arr))
etai = np.linspace(0,etamax,int(N/2)+1)
etai = np.r_[-np.flip(etai[1:]),etai]
taui, etai = np.meshgrid(taui,etai)

rui = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ru_list)])
rvi = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, rv_list)])
ruvi = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ruv_list)])

ru_mean1 = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
rv_mean1 = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
ruv_mean1 = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)


susv = np.sqrt(np.nanmax(np.abs(ru_mean1)))*np.sqrt(np.nanmax(np.abs(rv_mean1)))
su2 = np.nanmax(np.abs(ru_mean1))
sv2 = np.nanmax(np.abs(rv_mean1))

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
ax.set_xlim(-4000,4000)
fig.tight_layout()

Luy1,Lux1 = integral_lenght_scale(ru_mean1,taui[0,:],etai[:,0])

#grid_x,grid_y,ush = shrink(grd,u)
#grid_x,grid_y,vsh = shrink(grd,v)
#
#grid_x = grid_x+np.nanmin(grid_x)
#grid_y = grid_y-np.nanmin(grid_y)/2

#i = 530
#ch=14
#for j in range(i-int(ch/2),i+int(ch/2)+1):
#    ax0.cla()
#    ax0.set_title(str(j))
#    im0 = ax0.contourf(grd[0], grd[1], V_out2[j], np.linspace(7,17,10), cmap='jet')
#    plt.pause(.3)

# In[Phase 2]
i = phase2
s2 = [s_u2[j][0] for j in range(i-int(ch/2),i+int(ch/2)+1)]
reslist2 = [df_corr_phase2[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[df_corr_phase2.scan == s].values for s in s2]

tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist2]
eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist2]
ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist2]
rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist2]
ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist2]

resarray = np.vstack([r for r in reslist2])

tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)

N = 512
taumax = np.nanmax(np.abs(tau_arr))
taui = np.linspace(0,taumax,int(N/2)+1)
taui = np.r_[-np.flip(taui[1:]),taui]
etamax = np.nanmax(np.abs(eta_arr))
etai = np.linspace(0,etamax,int(N/2)+1)
etai = np.r_[-np.flip(etai[1:]),etai]
taui, etai = np.meshgrid(taui,etai)

rui = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ru_list)])
rvi = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, rv_list)])
ruvi = np.vstack([autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2] for t,e,r in zip(tau_list, eta_list, ruv_list)])

ru_mean2 = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
rv_mean2 = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
ruv_mean2 = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)

ru_std2 = np.nanstd(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
rv_std2 = np.nanstd(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
ruv_std2 = np.nanstd(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)

susv = np.sqrt(np.nanmax(np.abs(ru_mean2)))*np.sqrt(np.nanmax(np.abs(rv_mean2)))
su2 = np.nanmax(np.abs(ru_mean2))
sv2 = np.nanmax(np.abs(rv_mean2))

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)

im1 = ax1.contourf(etai,taui, ru_mean2,np.linspace(np.nanmin(ru_mean2),np.nanmax(ru_mean2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-3500, 3200,'(b)',fontsize=30,color='w')
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
ax.plot(etai[:,0], ru_mean2[:,256].T,'k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(taui[0,:], ru_mean2[256,:].T,'r',label='$\\rho_{u,u}(0,\\eta)$')
ax.legend(fontsize=24)
ax.plot(etai[:,0], (ru_mean2[:,256]+ru_std2[:,256]).T,'--k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(etai[:,0], (ru_mean2[:,256]-ru_std2[:,256]).T,'--k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(etai[:,0], (ru_mean2[256,:]+ru_std2[256,:]).T,'--r')
ax.plot(etai[:,0], (ru_mean2[256,:]-ru_std2[256,:]).T,'--r')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(b)',fontsize=30,color='k')
ax.set_xlim(-4000,4000)
fig.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(etai[:,0], (ru_std2[256,:]).T,'--k',label='$\\rho_{u,u}(\\tau,0)$')

ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
#ax.text(-3700, .95,'(b)',fontsize=30,color='k')
ax.set_xlim(-4000,4000)
fig.tight_layout()



def integral_lenght_scale(r_u,tau,eta):
    zero_axis_eta = np.where(eta==0)[0]
    zero_axis_tau = np.where(tau==0)[0]
    rx = np.squeeze(r_u[zero_axis_eta,:])
    ry = np.squeeze(r_u[:,zero_axis_tau])
    zero_crossings_x = np.where(np.diff(np.sign(rx)))[0]
    zero_crossings_y = np.where(np.diff(np.sign(ry)))[0]
    if len(zero_crossings_x) == 0:
        zero_crossings_x = 0
        tau_zero = tau[zero_crossings_x]
    else:
        tau_zero = tau[zero_crossings_x][np.argsort(np.abs(tau[zero_crossings_x]-tau[tau==0]))[0]]    
    if len(zero_crossings_y) == 0:
        zero_crossings_y = 0
        eta_zero = eta[zero_crossings_y]
    else:
#        print(np.sort(np.abs(eta[zero_crossings_y]-eta[eta==0])),eta[zero_crossings_y],np.argsort(np.abs(eta[zero_crossings_y]-eta[eta==0])))
        eta_zero = eta[zero_crossings_y][np.argsort(np.abs(eta[zero_crossings_y]-eta[eta==0]))[0]]
    ind_tau = np.abs(tau) > np.abs(tau_zero)
    ind_eta = np.abs(eta) > np.abs(eta_zero) 
    plt.figure()
    plt.plot(tau[~ind_tau],rx[~ind_tau]/np.nanmax(rx))
    plt.figure()
    plt.plot(eta[~ind_eta],ry[~ind_eta]/np.nanmax(ry))
    Lx = np.trapz(rx[~ind_tau]/np.nanmax(rx),tau[~ind_tau])/2
    Ly = np.trapz(ry[~ind_eta]/np.nanmax(ry),eta[~ind_eta])/2
    return (Lx,Ly)

Luy2,Lux2 = integral_lenght_scale(ru_mean2,taui[0,:],etai[:,0])


#for j in range(i-int(ch*2/2),i+int(ch*2/2)+1):
#    ax0.cla()
#    im0 = ax0.contourf(grd[0], grd[1], V_out[j], 10, cmap='jet')
#    plt.pause(.3)

# In[Phase1]
# Velocity plots
# Phase 1
L1 = np.abs(grd[0][0,0]-grd[0][0,-1])
ds1 = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
s_i1 = [np.min(dfL_phase1.loc[(dfL_phase1.scan==s_u1[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)/ds1 for i in range(800)]
L_s1 = np.squeeze(np.array([dfL_phase1.loc[(dfL_phase1.scan==s_u1[i][0])][['$L_{u,x}$','$L_{u,y}$']].values for i in range(800)]))
#############################
def fmts(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
#############################
a = .1

grid_x,grid_y,ush = shrink(grd,u1)
grid_x,grid_y,vsh = shrink(grd,v1)

indu = (ush<4) | (vsh<-3.5)
ush[indu] = np.nan
vsh[indu] = np.nan

dudy, dudx = np.gradient(ush, grid_y[:,0], grid_x[0,:]) 
dvdy, dvdx = np.gradient(vsh, grid_y[:,0], grid_x[0,:])    
contsh = dudx + dvdy
sd = np.nanmedian(np.array([s_i1[i] for i in range(i-int(ch/2),i+int(ch/2)+1)]))
contsh= filterfft(contsh,np.isnan(contsh),sigma=sd/2)

grid_x = grid_x+np.nanmin(grid_x)
grid_y = grid_y-np.nanmin(grid_y)/2

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grid_x, grid_y, ush, np.linspace(5.5, np.nanmax(ush),10), cmap='jet')
ax0.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1.5)
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax0)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
cb.ax.tick_params(labelsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.tight_layout()
fig0.tight_layout()
ax0.set_xlim(-6200,0)
ax0.set_ylim(-3500,3500)

#fig1, ax1 = plt.subplots(figsize=(8, 8))
#ax1.set_aspect('equal')
#ax1.use_sticky_edges = False
#ax1.margins(0.07)
#im1 = ax1.contourf(grid_x, grid_y, vsh, np.linspace(np.nanmin(vsh), np.nanmax(vsh),10), cmap='jet')
#ax1.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
#ax1.tick_params(axis='both', which='major', labelsize=24)
#ax1.set_xlabel('$Easting\:[m]$', fontsize=24)
#ax1.set_ylabel('$Northing\:[m]$', fontsize=24)
#divider = make_axes_locatable(ax1)
#cax= divider.append_axes("right", size="5%", pad=0.05)
#cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
#cb.ax.set_ylabel(r"$V\:[m/s]$",fontsize=24)
#cb.ax.tick_params(labelsize=24)
#ax1.text(-6000, 2500,'(a)',fontsize=30,color='k')
#fig1.tight_layout()
#fig1.tight_layout()
#ax1.set_xlim(-6200,0)
#ax1.set_ylim(-3500,3500)


fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_aspect('equal')
ax2.use_sticky_edges = False
ax2.margins(0.07)
im2 = ax2.contourf(grid_x, grid_y, contsh, np.linspace(np.nanmin(contsh), np.nanmax(contsh),100), cmap='jet')
ax2.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.set_xlabel('$Easting\:[m]$', fontsize=24)
ax2.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax2)
cax= divider.append_axes("right", size="5%", pad=0.05)
cbformat = ticker.ScalarFormatter()   # create the formatter
cbformat.set_powerlimits((-6,-3)) 
cb = fig2.colorbar(im2, cax = cax, format = cbformat)#ticker.FuncFormatter(fmts))
cb.ax.set_ylabel(r"$\frac{\partial U_1}{\partial x_1}+\frac{\partial U_2}{\partial x_2}\:[1/s]$",fontsize=24)
cb.ax.tick_params(labelsize=20)
ax2.text(-6000, 2500,'(b)',fontsize=30,color='k')
fig2.tight_layout()
fig2.tight_layout()
ax2.set_xlim(-6200,0)
ax2.set_ylim(-3500,3500)

# In[Phase2]
# Phase 2
i = phase2
L2 = np.abs(grd[0][0,0]-grd[0][0,-1])
ds2 = np.sqrt(np.diff(grd[0][0,:])[0]**2+np.diff(grd[1][:,0])[0]**2)
s_i2 = [np.min(dfL_phase2.loc[(dfL_phase2.scan==s_u2[i][0])][['$L_{u,x}$','$L_{u,y}$']].values)/ds2 for i in range(i-int(ch/2),i+int(ch/2)+1)]
L_s2 = np.squeeze(np.array([dfL_phase2.loc[(dfL_phase2.scan==s_u2[i][0])][['$L_{u,x}$','$L_{u,y}$']].values for i in range(i-int(ch/2),i+int(ch/2)+1)]))

grid_x,grid_y,ush = shrink(grd,u2)
grid_x,grid_y,vsh = shrink(grd,v2)

#indu = (ush<6) | (vsh<-3.5)
#ush[indu] = np.nan
#vsh[indu] = np.nan

dudy, dudx = np.gradient(ush, grid_y[:,0], grid_x[0,:]) 
dvdy, dvdx = np.gradient(vsh, grid_y[:,0], grid_x[0,:])    
contsh = dudx + dvdy
vortsh = dvdx - dudy 
sd = np.nanmedian(np.array(s_i2))
contsh= filterfft(contsh,np.isnan(contsh),sigma=sd/2)
vortsh= filterfft(vortsh,np.isnan(vortsh),sigma=sd/4)
#contsh = contsh/np.nanmax(contsh)

grid_x = grid_x+np.nanmin(grid_x)
grid_y = grid_y-np.nanmin(grid_y)/2

grid_x,grid_y,ush = shrink(grd,u2)
grid_x,grid_y,vsh = shrink(grd,v2)

grid_x = grid_x+np.nanmin(grid_x)
grid_y = grid_y-np.nanmin(grid_y)/2

a = .1

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_aspect('equal')
ax0.use_sticky_edges = False
ax0.margins(0.07)
im0 = ax0.contourf(grid_x, grid_y, ush, np.linspace(np.nanmin(ush), np.nanmax(ush),10), cmap='jet')
ax0.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
ax0.tick_params(axis='both', which='major', labelsize=24)
ax0.set_xlabel('$Easting\:[m]$', fontsize=24)
ax0.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax0)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig0.colorbar(im0, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$U_1\:[m/s]$",fontsize=24)
cb.ax.tick_params(labelsize=24)
ax0.text(-6000, 2500,'(a)',fontsize=30,color='k')
fig0.tight_layout()
fig0.tight_layout()
ax0.set_xlim(-6200,0)
ax0.set_ylim(-3500,3500)

#fig1, ax1 = plt.subplots(figsize=(8, 8))
#ax1.set_aspect('equal')
#ax1.use_sticky_edges = False
#ax1.margins(0.07)
#im1 = ax1.contourf(grid_x, grid_y, vsh,  np.linspace(np.nanmin(vsh), np.nanmax(vsh),10), cmap='jet')
#ax1.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
#ax1.tick_params(axis='both', which='major', labelsize=24)
#ax1.set_xlabel('$Easting\:[m]$', fontsize=24)
#ax1.set_ylabel('$Northing\:[m]$', fontsize=24)
#divider = make_axes_locatable(ax1)
#cax= divider.append_axes("right", size="5%", pad=0.05)
#cb = fig1.colorbar(im1, cax=cax, format = ticker.FuncFormatter(fm))
#cb.ax.set_ylabel(r"$V\:[m/s]$",fontsize=24)
#cb.ax.tick_params(labelsize=24)
#ax1.text(-6000, 2500,'(b)',fontsize=30,color='k')
#fig1.tight_layout()
#fig1.tight_layout()
#ax1.set_xlim(-6200,0)
#ax1.set_ylim(-3500,3500)

fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_aspect('equal')
ax2.use_sticky_edges = False
ax2.margins(0.07)
im2 = ax2.contourf(grid_x, grid_y, contsh, np.linspace(np.nanmin(contsh), np.nanmax(contsh),100), cmap='jet')
ax2.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.set_xlabel('$Easting\:[m]$', fontsize=24)
ax2.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax2)
cax= divider.append_axes("right", size="5%", pad=0.05)
cbformat = ticker.ScalarFormatter()   # create the formatter
cbformat.set_powerlimits((-6,-3)) 
cb = fig2.colorbar(im2, cax = cax, format = cbformat)#t
#cb = fig2.colorbar(im2, cax=cax,format = ticker.FuncFormatter(fmts))
cb.ax.set_ylabel(r"$\frac{\partial U_1}{\partial x_1}\:+\:\frac{\partial U_2}{\partial x_2}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
ax2.text(-6000, 2500,'(b)',fontsize=30,color='k')
fig2.tight_layout()
fig2.tight_layout()
ax2.set_xlim(-6200,0)
ax2.set_ylim(-3500,3500)


fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.set_aspect('equal')
ax3.use_sticky_edges = False
ax3.margins(0.07)
im3 = ax3.contour(grid_x, grid_y, vortsh, np.linspace(-4e-2, 4e-2,10), cmap='jet')
ax3.contour(grid_x, grid_y, contsh, levels = [np.nanmax(contsh)*a], colors = 'k', linewidths = 1)
ax3.tick_params(axis='both', which='major', labelsize=24)
ax3.set_xlabel('$Easting\:[m]$', fontsize=24)
ax3.set_ylabel('$Northing\:[m]$', fontsize=24)
divider = make_axes_locatable(ax3)
cax= divider.append_axes("right", size="5%", pad=0.05)
cbformat = ticker.ScalarFormatter()   # create the formatter
cbformat.set_powerlimits((-6,-3)) 
cb = fig3.colorbar(im3, cax = cax, format = cbformat)#t
cb.ax.set_ylabel(r"$\omega\:[1/s]$",fontsize=24)
cb.ax.tick_params(labelsize=24)
ax2.text(-6000, 2500,'(b)',fontsize=30,color='k')
fig3.tight_layout()
fig3.tight_layout()
ax3.set_xlim(-6200,0)
ax3.set_ylim(-3500,3500)


######################################################################################################
# In[]
susv = np.sqrt(np.nanmax(np.abs(ru_mean)))*np.sqrt(np.nanmax(np.abs(rv_mean)))
su2 = np.nanmax(np.abs(ru_mean))
sv2 = np.nanmax(np.abs(rv_mean))

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)

im1 = ax1.contourf(etai,taui, ru_mean/su2,np.linspace(np.nanmin(ru_mean/su2),np.nanmax(ru_mean/su2),10), cmap='jet')
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

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)

im1 = ax1.contourf(etai,taui, rv_mean/sv2,np.linspace(np.nanmin(rv_mean/sv2),np.nanmax(rv_mean/sv2),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-1700, 1500,'(a)',fontsize=30,color='w')
ax1.set_xlim(-2000,2000)
ax1.set_ylim(-2000,2000)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{vv}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

fig1, ax1 = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax1.set_aspect('equal')
ax1.use_sticky_edges = False
ax1.margins(0.07)

im1 = ax1.contourf(etai,taui, ruv_mean/susv,np.linspace(np.nanmin(ruv_mean/susv),np.nanmax(ruv_mean/susv),10), cmap='jet')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlabel('$\\tau\:[m]$', fontsize=24)
ax1.set_ylabel('$\eta\:[m]$', fontsize=24)
ax1.text(-900, 800,'(a)',fontsize=30,color='k')
ax1.set_xlim(-1000,1000)
ax1.set_ylim(-1000,1000)
divider = make_axes_locatable(ax1)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig1.colorbar(im1, cax=cax,format=ticker.FuncFormatter(fm))
cb.ax.set_ylabel(r"$\rho_{uv}$",fontsize=24)
cb.ax.tick_params(labelsize=24)
fig1.tight_layout()

#######################################################################################################
# In[]
for j in range(i-int(ch/2),i+int(ch/2)+1):
    
    ax0.cla()
    ax1.cla()
    U, V = U_rot(grd, U_out[j], V_out[j], gamma = gamma, tri_calc = False, 
                              tri_del = tri_del, mask_int = mask_int, mask = mask)
    
    im0 = ax0.contourf(grd[1][:,0], grd[0][0,:], U, 10, cmap='jet')
    plt.pause(1)
#    fig0.colorbar(im0)
    res = df_corr_phase2[['tau', 'eta', 'r_u', 'r_v', 'scan']].loc[df_corr_phase2.scan == su[j][0]].values
    tau, eta, ru, rv, scan = np.split(res,res.shape[1],axis=1)
    
    
    taui, etai, rui = autocorr_interp_sq(ru,tau,eta,tau_lin = taui, eta_lin = etai)
    taui, etai, rvi = autocorr_interp_sq(rv,tau,eta,tau_lin = taui, eta_lin = etai)
    im1 = ax1.contourf(etai,taui, rui/np.nanmax(rui),np.linspace(-.3,1,10), cmap='jet')
    ax1.set_xlim(-4000,4000)
    ax1.set_ylim(-4000,4000)
#    fig1.colorbar(im1)
    plt.pause(1)

fig, ax = plt.subplots(figsize=(8, 8))
#fig.set_size_inches(8,8)
ax.use_sticky_edges = False
ax.margins(0.07)
ax.plot(etai[:,0], ru_mean[:,256].T/np.nanmax(ru_mean),'k',label='$\\rho_{u,u}(\\tau,0)$')
ax.plot(taui[0,:], ru_mean[256,:].T/np.nanmax(ru_mean),'r',label='$\\rho_{u,u}(\\eta,0)$')
ax.set_xlabel('$\\tau,\:\\eta\:[m]$',fontsize=24)
ax.set_ylabel('$\\rho_{u,u}$',fontsize=24)
ax.legend(fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.text(-3700, .95,'(b)',fontsize=30,color='k')
ax.set_xlim(-4000,4000)

###########################################################################################################
# In[]
sd = np.nanmedian(np.array([s_i[i] for i in range(i-int(ch/2),i+int(ch/2)+1)]))

imgU = filterfft(u,np.isnan(u),sigma=sd/8)
imgV = filterfft(v,np.isnan(v),sigma=sd/8)

dudy, dudx = np.gradient(imgU, grd[0][0,:], grd[1][:,0]) 
dvdy, dvdx = np.gradient(imgV, grd[0][0,:], grd[1][:,0])     

x, y = grd
n,m = x.shape
sx = (x[0,-1]-x[0,0])/(n-1)
sy = (y[-1,0]-y[0,0])/(m-1)

slim_peak = 2#!!!!!!!!!!!
slim_cont = 1#!!!!!!!!!!!
S_M = np.nanmean(dudy+dvdx)+np.nanstd(dudy+dvdx)#.5*np.nanmax(dudy+dvdx)
S_m = np.nanmean(dudy+dvdx)-np.nanstd(dudy+dvdx)#.5*np.nanmin(dudy+dvdx)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)



slim_cont = np.linspace(.5,2,5)

super_struct1 = []
strength_struct1 = []
size_struct1 = []
for s in slim_cont:
    super_struct0 = []
    strength_struct0 = []
    size_struct0 = []
    for j in range(i-int(ch/2),i+int(ch/2)+1):
        print(j)
        ax.cla()
        U, V = U_rot(grd, U_out[j], V_out[j], gamma = gamma, tri_calc = False, 
                              tri_del = tri_del, mask_int = mask_int, mask = mask)
        vort, _ =  vort_cont(U, V, grd)
        img = filterfft(vort,np.isnan(vort),sigma=sd/2)
    #    
        imgUj = filterfft(U,np.isnan(U),sigma=s_i[i]/8)
        imgVj = filterfft(V,np.isnan(V),sigma=s_i[i]/8)
    
        dudyj, dudxj = np.gradient(imgUj, grd[0][0,:], grd[1][:,0]) 
        dvdyj, dvdxj = np.gradient(imgVj, grd[0][0,:], grd[1][:,0])     
        
        detected_peaks = detect_peaks(img)
        ind_peaks_pos = (img > .5*S_M*slim_peak) & detected_peaks
        ind_peaks_neg = (img < .5*S_m*slim_peak) & detected_peaks
        ind_peaks = ind_peaks_pos | ind_peaks_neg
        max_vort = .5*S_M*s
        min_vort = .5*S_m*s
        peaks = img[ind_peaks]
        levels = [min_vort,max_vort]
        contours_min = measure.find_contours(img.T, min_vort)
        contours_max = measure.find_contours(img.T, max_vort)
        xc, yc = x[ind_peaks], y[ind_peaks]
        im = ax.contourf(grd[0], grd[1], img, 5*np.linspace(S_m, S_M, 20), cmap='jet')
    #    im = ax.contourf(grd[0], grd[1], U, cmap='jet')
    #    im = ax.contourf(grd[0], grd[1], V, cmap='jet')
    #    #im = ax.contourf(grd[0], grd[1], dudxj+dvdyj, cmap='jet')
        ax.set_title(str(su[j][0]))
        # are peaks in the contours       
        struct = []
        strength = []
        size = []
        for contour in [contours_min, contours_max]:
            for c in contour:
                isin = False
                cx, cy = c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0] 
                if (len(cx)>=5): 
                    pind = np.isnan(cx)|np.isnan(cy)
                    p = geometry.Polygon([[xi, yi] for xi,yi in zip(cx[~pind],cy[~pind])])
                    sgt = []
                    for xi,yi,pk in zip(xc, yc, peaks):                          
                        xy_point=geometry.Point(xi, yi)                        
                        if p.contains(xy_point):
                            isin = True
                            sgt.append(np.c_[xi,yi,pk])                            
                    if isin:
                        struct.append(np.c_[cx[~pind],cy[~pind]])  
                        size.append(np.c_[np.abs(np.max(cx[~pind])-np.min(cx[~pind])),np.abs(np.max(cy[~pind])-np.min(cy[~pind]))])
                        plt.plot(cx[~pind],cy[~pind], 'k', lw= 3)
                        strength.append(sgt)
    
        ax.scatter(grd[0][ind_peaks][np.argsort(peaks)],grd[1][ind_peaks][np.argsort(peaks)],marker='+',s=100,color='r')
        plt.pause(.3)
        
        super_struct0.append(struct)
        size_struct0.append(size)
        strength_struct0.append(strength)
    super_struct1.append(super_struct0)
    size_struct1.append(size_struct0)
    strength_struct1.append(strength_struct0)

lst = np.array([[len(super_struct1[i][j]) for j in range(len(super_struct1[i]))] for i in range(len(super_struct1))])
plt.figure()
[plt.plot(slim_cont,lst[:,i]) for i in range(lst.shape[1])]

xp, yp = np.array(strength_struct1)[i,0][0][:,3]

for i in range(np.array(strength_struct1).shape[0]):
    l = np.array(strength_struct1)[i,0]
    




#u, v, img_rot  = U_rot(grd, U, V, img, tri_calc = True)

##############################################################################

plt.hist((dudy+dvdx).flatten(),bins = 500)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],imgU,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],dudy,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],dvdx,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],dudy+dvdx,cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],dudx+dvdy,np.linspace(-.01,.01,10),cmap='jet')
fig.colorbar(im)

#############################################################################




# Peak detection
detected_peaks = detect_peaks(img_rot)
dudy, dudx = np.gradient(imgU, grd[0][0,:], grd[1][:,0]) 

Uf = filterfft(U,np.isnan(U),sigma=s_i[i]/2)
Vf = filterfft(V,np.isnan(V),sigma=s_i[i]/2)    

slim_peak = 2.3#!!!!!!!!!!!
slim_cont = 1.1#!!!!!!!!!!!

S_M = np.nanmax(dudy)
S_m = np.nanmin(dudy)
ind_peaks_pos = (img_rot > S_M*slim_peak) & detected_peaks
ind_peaks_neg = (img_rot < S_m*slim_peak) & detected_peaks
ind_peaks = ind_peaks_pos | ind_peaks_neg
max_vort = S_M*slim_cont
min_vort = S_m*slim_cont
peaks = img_rot[ind_peaks]
levels = [min_vort,max_vort]
contours_min = measure.find_contours(img_rot.T, min_vort)
contours_max = measure.find_contours(img_rot.T, max_vort)

x, y = grd
n,m = x.shape
sx = (x[0,-1]-x[0,0])/(n-1)
sy = (y[-1,0]-y[0,0])/(m-1)

xc, yc = x[ind_peaks], y[ind_peaks]

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],img_rot,cmap='jet')
# are peaks in the contours
struct = []
for contour in [contours_min, contours_max]:
    for c in contour:
        cx, cy = c[:,0]*sx+x[0,0],c[:,1]*sy+y[0,0] 
        if (len(cx)>=5): 
            pind = np.isnan(cx)|np.isnan(cy)
            p = geometry.Polygon([[xi, yi] for xi,yi in zip(cx[~pind],cy[~pind])])
            for xi,yi in zip(xc, yc):             
                xy_point=geometry.Point(xi, yi)
                if p.contains(xy_point):
                    struct.append(np.c_[cx[~pind],cy[~pind]])     
                    plt.plot(cx[~pind],cy[~pind], 'k', lw= 3)
ax.scatter(grd[0][ind_peaks][np.argsort(peaks)],grd[1][ind_peaks][np.argsort(peaks)],marker='+',s=100,color='r')
            
            
            
fig.colorbar(im)
ax.scatter(grd[0][ind_peaks][np.argsort(peaks)],grd[1][ind_peaks][np.argsort(peaks)],marker='+',s=100,color='r')


poly.is_ring

xy = poly.exterior.coords.xy












