# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:57:20 2020
Filter for scanning lidars
@author: lalc
"""

import numpy as np
import pandas as pd
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import ppiscanprocess.windfieldrec as wr

# In[]

rp = 75
dp = 35
R = 7000
Dtheta = 2*np.pi/180
k = lambda k1,k2: np.sqrt(k1**2+k2**2)

azim0 = ((90-np.arange(256,346,2)) % 360)*np.pi/180
azim1 = ((90-np.arange(196,286,2)) % 360)*np.pi/180
r = np.arange(105,7035,35)

r0, azim0 =  np.meshgrid(r,azim0)
r1, azim1 =  np.meshgrid(r,azim1)

loc0 = np.array([0,6322832.3])#-d
loc1 = np.array([0,6327082.4])# d
d = loc1-loc0  

r_prime0, azim_prime0 = wr.translationpolargrid((r0,azim0),d/2)
r_prime1, azim_prime1 = wr.translationpolargrid((r1,azim1),-d/2)

x0,y0 = r_prime0*np.cos(azim_prime0), r_prime0*np.sin(azim_prime0)
x1,y1 = r_prime1*np.cos(azim_prime1), r_prime1*np.sin(azim_prime1)

r_hat0 = np.array([x0/r_prime0,y0/r_prime0])
r_hat1 = np.array([x1/r_prime1,y1/r_prime1])

x = np.linspace(np.min(np.r_[x0.flatten(),x1.flatten()]),np.max(np.r_[x0.flatten(),x1.flatten()]),100)
y = np.linspace(np.min(np.r_[y0.flatten(),y1.flatten()]),np.max(np.r_[y0.flatten(),y1.flatten()]),100)

x, y = np.meshgrid(x,y)

rhat0 = np.zeros((x.flatten().shape[0],2))*np.nan
rhat1 = np.zeros((x.flatten().shape[0],2))*np.nan

rhat0[:,0] = sp.interpolate.griddata(np.c_[x0.flatten(),y0.flatten()],r_hat0[0,:,:].flatten(),
                                     np.c_[x.flatten(),y.flatten()], method='linear')
rhat0[:,1] = sp.interpolate.griddata(np.c_[x0.flatten(),y0.flatten()],r_hat0[1,:,:].flatten(),
                                     np.c_[x.flatten(),y.flatten()], method='linear')
rhat1[:,0] = sp.interpolate.griddata(np.c_[x1.flatten(),y1.flatten()],r_hat1[0,:,:].flatten(),
                                     np.c_[x.flatten(),y.flatten()], method='linear')
rhat1[:,1] = sp.interpolate.griddata(np.c_[x1.flatten(),y1.flatten()],r_hat1[1,:,:].flatten(),
                                     np.c_[x.flatten(),y.flatten()], method='linear')

rhat = .5*(rhat0+rhat1)


# Need to include the dependance on position

plt.figure()
plt.contourf(x,y,rhat[:,0].reshape(x.shape),cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(x,y,rhat[:,1].reshape(x.shape),cmap='jet')
plt.colorbar()

k_prime = lambda
H = lambda k1,k2: 