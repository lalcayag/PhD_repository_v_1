# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:53:14 2020
Runge-Kutta Spectra with buoyancy
@author: lalc
"""
import numpy as np
import pandas as pd
import scipy as sp
import sympy
import ppiscanprocess.spectra_construction as sc
# In[Symbols for equations]
A = sympy.Matrix(4, 4, lambda i,j:sympy.var('a%d%d' % (i+1,j+1)))
f_a  = np.array(Msy*A).flatten()

# In[Initial conditions]
#Isotropic spectra
def vKarman(alpha,eps,L,k):
    return alpha*eps**(2/3)*L**(5/3)*(k*L)**4/(1+(k*L)**2)**(17/6)
#Isotropic spectra temperature
def S_prime(alpha,eps,eta,beta,L,k):
    E = vKarman(alpha,eps,L,k)
    return beta*eta*(1+(k*L)**2)*E/(k*L)**2
#Initial conditions for dZi
def dZ0(k1,k2,k3,alpha,eps,eta,beta,L):
    k = np.sqrt(k1**2+k2**2+k3**2)
    S = S_prime(alpha,eps,eta,beta,L,k)
    E = vKarman(alpha,eps,L,k)
    dz0 = np.array([[0,k3,-k2,0],[-k3,0,k1,0],[k2,0,-k1,0],[0,0,0,(k**2*S/E)**.5]])*(E/(4*np.pi*k**4))**.5
    return dz0

# In[Solving with ode int]
def dadb(a,b,params):
    a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44 = a
    k1,k2,k3,Ri = params
    deriv = [a31*(2*k1**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - 1) - a41*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       a32*(2*k1**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - 1) - a42*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       a33*(2*k1**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - 1) - a43*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       a34*(2*k1**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - 1) - a44*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       2*a31*k1*k2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - a41*k2*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       2*a32*k1*k2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - a42*k2*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       2*a33*k1*k2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - a43*k2*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       2*a34*k1*k2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) - a44*k2*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0),
       2*a31*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + a41*(-(-b*k1 + k3)**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + 1),
       2*a32*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + a42*(-(-b*k1 + k3)**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + 1),
       2*a33*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + a43*(-(-b*k1 + k3)**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + 1),
       2*a34*k1*(-b*k1 + k3)*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + a44*(-(-b*k1 + k3)**2*(k1**2 + k2**2 + (-b*k1 + k3)**2)**(-1.0) + 1),
       Ri*a31, Ri*a32, Ri*a33, Ri*a34]
    return deriv
def Amatrix(gamma_par,L,Ri,ks):
    k1m,k2m,k3m = ks
    km = (k1m**2+k2m**2+k3m**2)**.5
    hyp = sp.special.hyp2f1(1/3,17/6,4/3,-(km*L**(-2)))
    Beta = gamma_par/( (km*L**(2/3))* hyp**.5)
    k30 = k3m + Beta*k1m
    bm = np.linspace(0, Beta, 10)
    params = [k1m,k2m,k30,-.2]
    y0 = np.eye(4).flatten()
    psoln = sp.integrate.odeint(dadb, y0, bm, args=(params,))
    An = psoln[-1,:].reshape((4,4))
    return (An,k30)
def phi_k(alpha,eps,gamma_par,L,eta,beta,Ri,k):
    k1,k2,k3 = k
    A, k30 = Amatrix(gamma_par,L,Ri,[k1,k2,k3])
    dz0 = dZ0(k1,k2,k30,alpha,eps,eta,beta,L)
    PHI0 = dz0.dot(dz0.T)
    PHI = (A.dot(PHI0)).dot(A.T)
    return (PHI, PHI0, k30)   
def PHI(alpha,eps,gamma_par,L,eta,beta,Ri,kmesh):
    k1,k2,k3 = kmesh
    phi0 = np.zeros((4,4,len(k1.flatten())))
    phi = np.zeros((4,4,len(k1.flatten())))
    ks = np.c_[k1.flatten(),k2.flatten(),k3.flatten()]
    ks0 = np.c_[k1.flatten(),k2.flatten(),k3.flatten()]
    for i in range(ks.shape[0]):
        print(i)
        PHI,PHI0, k30 = phi_k(alpha,eps,gamma_par,L,eta,beta,Ri,ks[i,:])
        phi0[:,:,i] = PHI0
        phi[:,:,i] = PHI
        ks0[i,2] = k30
    return (phi0,phi,ks0,ks)

# In[]
gamma_par = 2.5
L = 35
alpha = 1.7
eps = (0.0635/1.7)**(3/2)
beta = .8/alpha
eta = 0*.15
Ri = 0

k1 = 10**np.linspace(-4,1,10)
k2 = 10**np.linspace(-4,1,10)
k3 = 10**np.linspace(-4,1,10)

k1 = np.r_[-np.flip(k1),k1]
k2 = np.r_[-np.flip(k2),k2]
k3 = np.r_[-np.flip(k3),k3]

kmesh = np.meshgrid(k1,k2,k3)
phi0,phi,ks0,ks = PHI(alpha,eps,gamma_par,L,eta,beta,Ri,kmesh)

kmesh = np.meshgrid(k1,k2,k3)

phi011 = np.reshape(phi0[1,1,:],kmesh[0].shape)
phi11 = np.reshape(phi[1,1,:],kmesh[0].shape)

phi033 = np.reshape(phi0[3,3,:],kmesh[0].shape)
phi33 = np.reshape(phi[3,3,:],kmesh[0].shape)

sc.plot_log2D((kmesh[0][:,:,10],kmesh[1][:,:,10]),phi011[:,:,10] , label_S = "$\log_{10}{S}$", 
              C =10**-2,fig_num='a',nl=30, minS = -1.5)

sc.plot_log2D((kmesh[0][:,:,10],kmesh[1][:,:,10]),phi11[:,:,10] , label_S = "$\log_{10}{S}$", 
              C =10**-2,fig_num='a',nl=30, minS = -1.5)

# In[R-K in multidimensions]

# Functions



dZsy= sympy.Matrix(3, 1, lambda i,j:sympy.var('dZ%d' % (i+1)))
Msy = sympy.Matrix(3, 1, lambda i,j:(-sympy.KroneckerDelta(i,0)+2*ksy[0]*ksy[i]/(ksy.dot(ksy)))*dZsy[2])


x = sympy.symbols("x")
y = sympy.Function("y")
f = y(x)**2 + x
f_np = sympy.lambdify((y(x), x), f)
y0 = 0
xp = np.linspace(0, 1.9, 100)
yp = sp.integrate.odeint(f_np, y0, xp)
xm = np.linspace(0, -5, 100)
ym = sp.integrate.odeint(f_np, y0, xm)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_direction_field(x, y(x), f, ax=ax)
ax.plot(xm, ym, 'b', lw=2)
ax.plot(xp, yp, 'r', lw=2)


A = sympy.MatrixSymbol('A', 3, 3)
M = sympy.MatrixSymbol('M', 3, 3)
beta = sympy.symbols("beta")


def rk4(f, x0, y0, x1, n):
    vx = [0] * (n + 1)
    vy = [0] * (n + 1)
    h = (x1 - x0) / float(n)
    vx[0] = x = x0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        vx[i] = x = x0 + i * h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vx, vy
 
def f(x, y):
    return x * sqrt(y)
vx, vy = rk4(f, 0, 1, 10, 100)
for x, y in list(zip(vx, vy))[::10]:
    print("%4.1f %10.5f %+12.4e" % (x, y, y - (4 + x * x)**2 / 16))

def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
	    )( dt * f( t + dt/2, y + dy2/2 ) )
	    )( dt * f( t + dt/2, y + dy1/2 ) )
	    )( dt * f( t       , y         ) )
 
def theory(t): return np.cos(t**2)#(t**2 + 4)**2 /16
 
from math import sqrt
dy = RK4(lambda t, y:t * sqrt(y))
 
t, y, dt = 1, 1., .1
ti,yi = [], []
while t <= 10:
    if abs(round(t) - t) < 1e-5:
        print("y(%2.1f)\t= %4.6f \t error: %4.6g" % ( t, y, abs(y - theory(t))))
    t, y = t + dt, y + dy( t, y, dt )
    ti.append(t)
    yi.append(y)
plt.figure();plt.plot(ti,yi)    


import math



def fa1(x):
    return 0.9*(1 - x[1]*x[1])*x[0] - x[1] + math.sin(x[2])

def fb1(x):
    return x[0]

def fc1(x):
    return 0.5

def VDP1(t,xi):
    f = [fa1, fb1, fc1]
    x = xi
    hs = .05
    for i in range(t):
        x=rKN(x, f, 3, hs)
    return x

t=np.arange(20000)
xi = [1, 1, 0]
x = np.zeros((len(t),3))
for i,ti in enumerate(t):
    x[i,:] = VDP1(1,xi)
    xi = x[i,:]
    
plt.figure()
plt.plot(x[:,-1],x[:,0])    
    











