# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:33:01 2024

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

# import seaborn as sns
# color_names = ["windows blue", "red", "amber", "faded green"]
# colors = sns.xkcd_palette(color_names)
# sns.set_style("white")
# sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
###############################################################################
# %% Effective model
###############################################################################

# %%
#functions
def environment(xx,yy,M=40):
    """
    a simple Gaussian diffusion 2-D environement
    """
#    M = 20   #max concentration
    sig2 = 60  #width of Gaussian
    target = np.array([50,50])   #target position
    NaCl = M*np.exp(-((xx-target[0])**2+(yy-target[1])**2)/2/sig2**2) ### Gaussian bump
#    NaCl = max(0,M*((xx-target[0])**2+(yy-target[1])**2)**0.5) ### linear sharp distance
    return NaCl+np.random.randn()*0. 

def steering(vv,alpha_s,dcp,K):
    """
    slow continuous steering angle change
    """
    # dth_s = alpha_s*dcp*np.abs(vv) + np.random.randn()*K
    dth_s = np.random.vonmises(alpha_s*dcp*np.abs(vv), K, 1)
    return dth_s

def Pirouette(vv,alpha_p,lamb0):
    """
    Frequency of the random turning process
    """
    # lambda_p = lamb0 + alpha_p*vv
    # lambda_p = min(1,max(lambda_p,0))
    th = 0.
    lambda_p = 0.1/(1+np.exp(alpha_p*(vv-th))) + lamb0
    return lambda_p

def turn_angle(vv,alpha_g,gamma0):
    """
    Biased turning angle after Pirouette
    """
    gammaB = gamma0 + alpha_g*vv
    gammaA = np.max(1 - gammaB,0)
    sigma = 10  #np.pi #/12
    dth_b = gammaA*(np.random.rand()*2*np.pi-np.pi)*0 + np.random.vonmises(np.pi, sigma, 1)
    #gammaB*(sigma*np.random.randn()-np.pi) #opposite direction
    #f_theta = gammaA/(np.pi*2) + gammaB/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(th-np.pi)/(2*sigma**2))
    return dth_b

def dc_measure(dxy,xx,yy):
    """
    perpendicular concentration measure
    """
    perp_dir = np.array([-dxy[1], dxy[0]])  #perpdendicular direction
    perp_dir = perp_dir/np.linalg.norm(perp_dir)*1 #unit norm vector
    perp_dC = environment(xx+perp_dir[0], yy+perp_dir[1]) - environment(xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

def ang2dis(x,y,th):
    e1 = np.array([1,0])
    vec = np.array([x,y])
    theta = math.acos(np.clip(np.dot(vec,e1)/np.linalg.norm(vec)/np.linalg.norm(e1), -1, 1)) #current orienation relative to (1,0)
    v = vv + vs*np.random.randn()
    dd = np.array([v*np.sin(th), v*np.cos(th)])  #displacement
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,s), (-s, c)))  #rotation matrix, changing coordinates
    dxy = np.dot(R,dd)
    return dxy

#simple switch
def sigmoid(x,w,t):
    ss = 1/(1+np.exp(-w*x-t))
    return ss

def compute_dwell_time(binary_sequence, value):
    dwell_times = []
    current_dwell_time = 0
    
    for bit in binary_sequence:
        if bit == value:
            current_dwell_time += 1
        elif current_dwell_time > 0:
            dwell_times.append(current_dwell_time)
            current_dwell_time = 0
    
    # Append the last dwell time if the sequence ends with the specified value
    if current_dwell_time > 0:
        dwell_times.append(current_dwell_time)
    
    return dwell_times

# %% functionize
def worm_track(init, couple, mode='staPAW',T=300):
    ### params
    dt = 0.1
    # T = 300
    lt = int(T/dt)
    alpha = 4
    beta = 1.5
    gamma = .2
    tau = 1.
    proj = np.array([.1,.2])*1
    lamb0 = 0.0
    gamma0 = 0.5
    alpha_p, alpha_s, alpha_g = .4, -.01, .001
    dxy = np.random.randn(2)
    vv,vs = 2.2, .2 
    K = 20
    if mode=='staPAW':
        J = np.array([[.07, -2.],[-2, 10]])*couple#2.5  
    elif mode=='dPAW':
        J = np.array([[.07, -2.],[-2, 10]])*0
    noise = 10.
    env_noise = 0 
    
    ### initialization
    Ft,St = np.zeros(lt),np.zeros(lt) #fast and slow part of AWC
    Vs = np.zeros((2,lt))
    Cs = np.zeros(lt)
    ths = np.zeros(lt)
    prt = np.zeros(lt)
    XY = np.random.randn(2,lt)
    XY[:,0] = init*1 #np.array([100,0])
    
    ### dynamics
    for tt in range(lt-1):
        ### sensor dynamics
        Ft[tt+1] = Ft[tt] + dt*(alpha*Cs[tt] - beta*Ft[tt])  # fast dynamics
        St[tt+1] = St[tt] + dt*gamma*(Ft[tt] - St[tt])  # slow dynamics
        Is = Ft[tt] - St[tt]
        ###neural dynamics
        Vs[:,tt+1] = Vs[:,tt] + dt/tau*( -Vs[:,tt] + proj*Is + J @ sigmoid(Vs[:,tt],1,0) ) \
        + np.random.randn(2)*np.sqrt(dt)*noise
        
        if mode =='dPAW':
        ###behavior
            lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
            if lambda_p>=np.random.rand():
                dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
            else:
                dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
                dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
        
        elif mode=='staPAW':
        ### staPAW-like
            if Vs[0,tt+1]>Vs[1,tt+1]:
                lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
                if lambda_p>=np.random.rand():
                    dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
                else:
                    dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
                    dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
            elif Vs[0,tt+1]<=Vs[1,tt+1]:
                dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
                dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
        ###
        
        ths[tt+1] = ths[tt]+dth
        ###environment
        dxy = ang2dis(XY[0,tt],XY[1,tt],ths[tt+1])
        XY[:,tt+1] = XY[:,tt] + dxy*dt
        Cs[tt+1] = environment(XY[0,tt+1],XY[1,tt+1]) + np.random.randn()*env_noise
        prt[tt+1] = dth
        
        ## checking breaks
        if np.sum((XY[:,tt+1]-np.array([50,50]))**2)<eps_target**2:
            break
    
    XY = XY[:,:tt]
    Vs = Vs[:,:tt]
    Cs = Cs[:tt]
    return XY, Vs, Cs

# %% exp tracks
eps_target = 6
x0s = np.array([[0.1,0.1], [100,0], [0,100], [100,100]])
plt.figure()
y, x = np.meshgrid(np.linspace(-10, 110, 100), np.linspace(-10, 110, 100))
plt.imshow(environment(x,y),origin='lower',extent = [-10,110,-10,110])

for ii in range(len(x0s)):
    init = x0s[ii]
    XY, Vs, Cs = worm_track(init,1.,mode='dPAW')
    pos_state = np.where(Vs[0,:]>Vs[1,:])[0]
    plt.plot(XY[0,:],XY[1,:],'k')
    plt.plot(XY[0,pos_state],XY[1,pos_state],'r.', markersize=.9)
    plt.scatter(XY[0,0],XY[1,0], color='green', marker='o');plt.scatter(XY[0,-1],XY[1,-1], color='purple', marker='*')

# plt.savefig('worm_circuit_exp2.pdf')

# %%
bin_seq = np.zeros(len(Vs.T))
pos_state = np.where(Vs[0,:]>Vs[1,:])[0]
bin_seq[pos_state] = np.ones(len(pos_state))
plt.figure()
plt.subplot(311)
plt.plot(Cs[1:])
plt.subplot(312)
plt.plot(Vs[:,:].T)
plt.subplot(313)
plt.plot(bin_seq[:])
# plt.savefig('worm_circuit_expt2.pdf')

# %% saving
# import pickle

# pre_text = 'worm_circuit_exp2'
# filename = pre_text + ".pkl"

# # Store variables in a dictionary
# data = {'XY': XY, 'Vs': Vs, 'Cs': Cs}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")

# %% dwell time
reps = 200
stateSt = []
stateTt = []
for rr in range(reps):
    XYi, Vsi,_ = worm_track(np.array([100,100]), .7, mode='staPAW')
    bin_seq = np.zeros(len(XYi.T))
    pos_state = np.where(Vsi[0,:]>Vsi[1,:])[0]
    bin_seq[pos_state] = np.ones(len(pos_state))
    bin_seq = bin_seq[50:]  #remove initial
    stateSt.append(compute_dwell_time(bin_seq, 0))
    stateTt.append(compute_dwell_time(bin_seq, 1))
    print(rr)
    
# %%
wl = 60
bins = np.arange(0,wl,3)
plt.figure()
# plt.hist(np.concatenate(stateSt),bins,alpha=0.5, label='S',density=True)
# plt.hist(np.concatenate(stateTt),bins,alpha=0.5, label='T',density=True)  #
# plt.xlabel('dwell time', fontsize=20); plt.xlim([0,wl]); plt.legend(fontsize=20)

aa,bb = np.histogram(np.concatenate(stateSt),bins, density=True)
plt.plot(bb[:-1],aa, label='S')
aa,bb = np.histogram(np.concatenate(stateTt),bins, density=True)
plt.plot(bb[:-1],aa, label='T')
plt.yscale('log')
plt.xlabel('dwell time', fontsize=20)
# plt.savefig('worm_circuit_dwell.pdf')

# %% CI analysis
reps = 100
Tf = 300
emdpoints = np.zeros((2,reps))
for rr in range(reps):
    XYi, Vsi,_ = worm_track(np.array([100,100]), 1., mode='staPAW', T=Tf)
    if np.sum((XYi[:,-1]-np.array([50,50]))**2)<(eps_target+1)**2:
        emdpoints[0,rr] = 1
    XYi, Vsi,_ = worm_track(np.array([100,100]), 1., mode='dPAW', T=Tf)
    if np.sum((XYi[:,-1]-np.array([50,50]))**2)<(eps_target+1)**2:
        emdpoints[1,rr] = 1
    print(rr)

# %%
plt.figure()
plt.bar([1,2], np.sum(emdpoints,1)/reps)
plt.xlabel('gradient')
plt.ylabel('CI')
plt.xticks([1,2], ['w/ states','w/o'])
# plt.savefig('worm_circuit_CI.pdf')

# %%
###############################################################################
# %% parameter tests
###############################################################################
# %% 
#with behavior
dt = 0.1
T = 300
lt = int(T/dt)
alpha = 4
beta = 1.5
gamma = .2
Ft,St = np.zeros(lt),np.zeros(lt) #fast and slow part of AWC
tau = 1
target = np.array([30,30])
Vs = np.zeros((2,lt))
Cs = np.zeros(lt)
ths = np.zeros(lt)
prt = np.zeros(lt)
XY = np.random.randn(2,lt)
XY[:,0] = np.array([100,0])

proj = np.array([.1,.2])*1
lamb0 = 0.0
gamma0 = 0.5
alpha_p, alpha_s, alpha_g = .4, -.0, .001
dxy = np.random.randn(2)
vv,vs = 2.2, .2 #0.55*2,0.05*2 
K = 20 #10 #np.pi/20 #12
J = np.array([[.3,-2.],[-2,.7]])*0. 
noise = 0.
env_noise = 1
for tt in range(lt-1):
    ### sensor dynamics
    Ft[tt+1] = Ft[tt] + dt*(alpha*Cs[tt] - beta*Ft[tt])  # fast dynamics
    St[tt+1] = St[tt] + dt*gamma*(Ft[tt] - St[tt])  # slow dynamics
    Is = Ft[tt] - St[tt]
    ###neural dynamics
    Vs[:,tt+1] = Vs[:,tt] + dt/tau*( -Vs[:,tt] + proj*Is + J @ sigmoid(Vs[:,tt],1,0) ) \
    + np.random.randn(2)*np.sqrt(dt)*noise
#     ###behavior
#     lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
#     if lambda_p>=np.random.rand():
#         dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
# #        print('k')
#     else:
#         dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
#         dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    
    ### staPAW-like
    if Vs[0,tt+1]>Vs[1,tt+1]:
        lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
        if lambda_p>=np.random.rand():
            dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
    #        print('k')
        else:
            dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
            dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    elif Vs[0,tt+1]<=Vs[1,tt+1]:
        dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
        dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    ###
    
    ths[tt+1] = ths[tt]+dth
    ###environment
    #dxy = np.squeeze(np.array([np.cos(dth), np.sin(dth)]))
    dxy = ang2dis(XY[0,tt],XY[1,tt],ths[tt+1])
    XY[:,tt+1] = XY[:,tt] + dxy*dt
    Cs[tt+1] = environment(XY[0,tt+1],XY[1,tt+1],50) + np.random.randn()*env_noise
    prt[tt+1] = dth
    
# %

pos_state = np.where(Vs[0,:]>Vs[1,:])[0]
plt.figure()
plt.plot(Cs)
plt.figure()
y, x = np.meshgrid(np.linspace(-10, 100, 100), np.linspace(-10, 100, 100))
plt.imshow(environment(x,y),origin='lower',extent = [-10,100,-10,100])
plt.plot(XY[0,:],XY[1,:],'k')
plt.plot(XY[0,pos_state],XY[1,pos_state],'r.', markersize=.9)
plt.scatter(XY[0,0],XY[1,0], color='green', marker='o');plt.scatter(XY[0,-1],XY[1,-1], color='purple', marker='*')
#
# %%
plt.figure()
plt.plot(Vs.T)

# %%
# exp circuit
# exp track
# dwell time
# chemotaxis index vs. without
