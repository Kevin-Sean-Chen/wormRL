# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 18:42:48 2025

@author: kevin
"""

from matplotlib import pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
### this is a new script for wormRL to explore hyperparameters
### then check the marginal distribution of heading changes
### and maybe further analyze dwell time of states...

# %%
### environements
### reward function
### actions: wv and pir-state
### states: up or down ... noisy estimates?

# %% seeed
seed = 37 #37 #37 (RL curve) #13 (bars) #17 (good both but weaker) #42
random.seed(seed) #37 42 17
np.random.seed(seed)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # for numpy random seed
    random.seed(seed)  # for built-in random module
    torch.backends.cudnn.deterministic = True  # to ensure deterministic behavior.
    torch.backends.cudnn.benchmark = False  # if deterministic is set to True, benchmark must be False

# Example usage:
set_seed(seed)

# %%
class dPAW():
    def __init__(self):
        ### initial parameter setting
        self.L = 100
        self.W = 100
        self.xs = np.array([50,50])  # point source
        self.x0 = np.array([0,0])  # initial location
        self.C = 80  #40 source concentration
        self.sigC = 80  # width of the odor bump # 4000 (20,40,60,80,100) -> (x,y,y,yish,yish,x)
        self.theta = np.random.rand()*np.pi  # randomize theta and keep track
        self.state = np.array([0,0])   # keeping track of location
        self.dv = 2.   # speed displacement
        self.xy = np.array([0,0])  # keep track of continous location
        self.state2 = np.random.randn(2,2)  # state-pairs
        self.noise = 0  # noise of sensory environment
        self.dth = 0  # record heading change
        self.bearing = 0  # recording bearing
        
    def odor_environment(self, state):
        """
        odor landscape with a Gaussian bump
        """
        C = np.exp(-np.sum((state - self.xs)**2)/self.sigC**2)*self.C + np.random.randn()*self.noise
        return C 
    
    def odor_profile(self, X,Y):
        exponent = -((X - self.xs[0])**2 / (self.sigC**2) + (Y - self.xs[1])**2 / (self.sigC**2))
        return self.C * np.exp(exponent)
    
    def sample_points(self, radius):
        # Generate random angles
        angle = np.random.uniform(0, 2*np.pi)
        # Calculate x and y coordinates using polar to cartesian conversion
        x_coord = radius * np.cos(angle)
        y_coord = radius * np.sin(angle)
        
        return np.array([x_coord, y_coord])
    
    def reset(self):
        # self.state = self.x0  # reset to the origin for another epoch
        temp = self.sample_points((25**2*2)**0.5)
        self.state =temp
        return
    
    def transition(self, state, action):
        """
        state transition given state-action P(s'|s,a)
        """
        concentration = self.measureC(state)
        if action==0:
            dth = self.WV(concentration)
        elif action==1:
            dth = self.PR(concentration)
        elif action==0.5:  # parallel!!
            dth = self.WV(concentration) + self.PR(concentration)
        
        state_ = self.dth2state(state, dth)  # move to the next state
        self.state = state_*1  # update the state only in the end
        self.state2 = np.array([state_, state])
        self.dth = dth*1  # record heading change       
        return state_
    
    def dth2state(self, state, dth):
        """
        take continuous angle dth and put back to discrete states
        """
        self.theta = self.theta + dth  # update angle
        dx,dy = np.cos(self.theta)*self.dv, np.sin(self.theta)*self.dv
        state_ = state + np.array([dx, dy]).squeeze()
        state_ = np.round(state_).astype(int)
        
        ### measure bearing
        goal_v = self.xs - state_  # goal vector
        heading = np.array([dx, dy])  # continuous heading
        self.bearing = self.cosine_angle(goal_v, heading)  # update bearing to goal
        return state_
    
    def PR(self, concentration):
        """
        given concentration measurements return d_theta with pirouette rule
        """
        dc, dcp = concentration
        Ppr = 1./(1+np.exp(dc*.4))  #.2 the strength has to be altered depending on C!
        if Ppr > np.random.rand():
            dth = np.random.vonmises(np.pi,10,1)
        else:
            dth = np.random.vonmises(0,30,1)  #10
        return dth
    
    def WV(self, concentration):
        """
        given concentration measurements return d_theta with weathervaning rule
        """
        dc, dcp = concentration
        dth = np.random.vonmises(dcp*.025, 15, 1)  #.2 the strength has to be altered depending on C!  #.09, 1.3
        return dth
    
    def reward(self, state):
        """
        reward given the state location, minus distance for now
        """
        # R = 0
        R = -np.sum((self.xs - state)**2)**0.5
        # if np.sum((self.xs - state)**2)**0.5<5:
        #     R = 1
        return R
    
    def measureC(self, state):
        """
        measuring dc and dcp in one step, then update the history state
        """
        dc = self.odor_environment(self.state2[0,:]) - self.odor_environment(self.state2[1,:])
        vec = self.state2[0,:] - self.state2[1,:]
        # print(vec)
        perp_vec = np.array([-vec[1], vec[0]])
        dcp = self.odor_environment(self.state+perp_vec*1) - \
                self.odor_environment((self.state-perp_vec*1))  #np.linalg.norm(perp_vec)
        # self.state = state  # update to keep track after measurements
        concentration = dc, dcp
        
        return concentration
    
    def action(self, prob_pir):
        """
        very simplified action for now...
        """
        action  = 0
        if prob_pir>np.random.rand():
            action = 1
        return action
    
    def cosine_angle(self, vector_a, vector_b):
        """
        used to measure bearing: angle between goal and heading vectors
        """
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        cosine_angle = dot_product / (norm_a * norm_b)   
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        cross_product_z = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    
        # Adjust the angle to be within the range -pi to pi
        if cross_product_z < 0:
            angle_radians = -angle_radians

        return angle_radians
    ###
    # action -> transition -> measurement
    ###

# %%
###############################################################################
# %% controller (using concentration C for action, not state)
# scenario 1: simply learning S-A pair values
# state: up or down gradient
# action: 0 or 1 for WV or PR

# scenario 2: given up ot down gradient, the transition probability for states...need policy gradients

# %%
class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()
        self.theta = nn.Parameter(torch.randn(4, requires_grad=True))

    def forward(self, a_past, dc):
        """
        soft-max policy pi(a|y,m)_theta
        where a is the action, y is the observation (dc) and m is the memory of state (past action)
        """
        w01,w10,b0,b1 = self.theta  # weights and baseline
        if isinstance(a_past, int):
            # if a_past==0:
            #     logit = torch.Tensor([w01*dc, b0])
            #     # P = torch.exp([w01*dc, b0])
            #     # P = P/torch.sum(P)
            # elif a_past==1:
            #     logit = torch.Tensor([b1, w10*dc])
            #     # P = torch.exp([b1, w10*dc])
            #     # P = P/torch.sum(P)
            logit = torch.Tensor([w01*dc+b0, w10*dc+b1])   # remove a dependency
        ######
        ### Check this!!!
        ######
        else:
            # logit = torch.zeros(len(a_past),2)
            # pos0 = torch.where(a_past==0)[0]
            # logit[pos0,:] = torch.cat((w01*dc[pos0], torch.zeros(len(pos0))+b0), dim=0).reshape(2,len(pos0)).T
            # # torch.Tensor([w01*dc[pos0], torch.ones(len(pos0))+b0])
            # pos1 = torch.where(a_past==1)[0]
            # logit[pos1,:] = torch.cat((torch.zeros(len(pos1))+b1, w10*dc[pos1]), dim=0).reshape(2,len(pos1)).T
            # # torch.Tensor([torch.ones(len(pos1))+b1, w10*dc[pos1]])
            logit = torch.zeros(len(dc),2)
            logit[:,0] = dc*w01+b0
            logit[:,1] = dc*w10+b1
        # aa = torch.tensor([0, 1])
        # idx = P.multinomial(num_samples=1, replacement=True)
        # a_ = aa[idx] # np.random.choice([0, 1], p=P)
        return logit #a_

# %% CI function
def pick_sample(a_past, dc):
    with torch.no_grad():
        logits = mylearner(a_past, dc)
        logits = logits.squeeze(dim=0)
        # From logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pick up action's sample
        a = torch.multinomial(probs, num_samples=1)
        # Return
        return a.tolist()[0]
    
def probe_CI(mydpaw, n_tracks, max_steps=300):
    mydpaw.reset()
    # mydpaw.C = 10
    eps_s = 5
    pos = np.zeros(n_tracks)
    ### record heading, action, and bearing
    dths = []
    acts = []
    bs = []
    for n in range(n_tracks):
        mydpaw.reset()
        t = 0
        action = np.random.choice(2)
        dthi = []
        acti = []
        bi = []
        while (np.sum((mydpaw.state-mydpaw.xs)**2)**0.5>eps_s) and (t < max_steps):
            t += 1 # update iteration counter
            ### State S
            old_state = mydpaw.state.copy() # keep track of current state
            ### Action A
            dc = mydpaw.measureC(old_state)[0]
            action = pick_sample(action, dc)
            ### State' S'
            state = mydpaw.transition(old_state, action)
            ### Reward R
            reward = mydpaw.reward(state)
            ### bearing
            bear_t = mydpaw.bearing*1
            
            dthi.append(mydpaw.dth)
            acti.append(action)
            bi.append(bear_t)
            
        pos[n] = -reward
        dths.append(dthi)
        acts.append(acti)
        bs.append(bi)
         
    CI = sum(pos<=5) / n_tracks
    return CI


# %% scann for CI as a function of parameters
K1 = np.arange(-1,1,.2)
K2 = np.arange(-1,1,.2)
b1,b2 = -1, +1 #0.1, 0.1

CIscan = np.zeros((len(K1), len(K2)))

# %%
max_steps = 200 #300
lamb = 1  # test with pirouette regularization
eps_s = 10 #5

mydpaw = dPAW()
mylearner = Learner()
n_tracks = 50 #200

for ii in range(len(K1)):
    print(ii)
    for jj in range(len(K2)):
        print(jj)
        thetai = np.array([K1[ii], K2[jj], b1, b2])
        # mylearner.theta = thetai
        mylearner.theta.data.copy_(torch.from_numpy(thetai).float())
        cii = probe_CI(mydpaw, n_tracks, max_steps)
        CIscan[ii,jj] = cii

# %% plot
plt.figure()
plt.imshow(CIscan, extent=[K1.min(), K1.max(), K2.min(), K2.max()], origin="lower",vmin=0, vmax=1)
plt.colorbar()  # optional
plt.xlabel(r"$K_{S \rightarrow T}$", fontsize=20)
plt.ylabel(r"$K_{T \rightarrow S}$", fontsize=20)
plt.title("CI across parameters", fontsize=20)
plt.show()
