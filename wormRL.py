# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:49:13 2024

@author: kevin
"""

from matplotlib import pyplot as plt
import numpy as np
import random

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
### environements
### reward function
### actions: wv and pir-state
### states: up or down ... noisy estimates?

# %%
class dPAW():
    def __init__(self):
        ### initial parameter setting
        self.L = 100
        self.W = 100
        self.xs = np.array([50,50])  # point source
        self.x0 = np.array([0,0])  # initial location
        self.C = 40  #20 source concentration
        self.sigC = 60  # width of the odor bump # 4000 (20,40,60,80,100) -> (x,y,y,yish,yish,x)
        self.theta = np.random.rand()*np.pi  # randomize theta and keep track
        self.state = np.array([0,0])   # keeping track of location
        self.dv = 2.   # speed displacement
        self.xy = np.array([0,0])  # keep track of continous location
        self.state2 = np.random.randn(2,2)  # state-pairs
        self.noise = 0  # noise of sensory environment
        
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
        return state_
    
    def dth2state(self, state, dth):
        """
        take continuous angle dth and put back to discrete states
        """
        self.theta = self.theta + dth  # update angle
        dx,dy = np.cos(self.theta)*self.dv, np.sin(self.theta)*self.dv
        state_ = state + np.array([dx, dy]).squeeze()
        state_ = np.round(state_).astype(int)
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
            dth = np.random.vonmises(0,10,1)
        return dth
    
    def WV(self, concentration):
        """
        given concentration measurements return d_theta with weathervaning rule
        """
        dc, dcp = concentration
        dth = np.random.vonmises(dcp*.09, 1.3, 1)  #.2 the strength has to be altered depending on C!
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
    ###
    # action -> transition -> measurement
    ###

# %% test runs
mydpaw = dPAW()
x0s = np.array([[0,0], [100,0], [0,100], [100,100]])
max_steps = 300
lamb = 1  # test with pirouette regularization
eps_s = 5

x = np.linspace(-10, 110, 120)
y = np.linspace(-10, 110, 120)
x, y = np.meshgrid(x, y)
c_env = mydpaw.odor_profile(x, y)

plt.figure()
plt.imshow(c_env, extent=[x[0,0], x[-1,-1], y[0,0], y[-1,-1]])
for ww in range(len(x0s)):
    mydpaw.reset()
    mydpaw.state = x0s[ww]
    t = 0
    total_reward = 0
    list_of_trans = []
    transition = {} # # initialize transition dict
    temp = []  # measurements...
    action = 0
    # while np.all(mydpaw.state is not mydpaw.xs) and (t < max_steps):
    while (np.sum((mydpaw.state-mydpaw.xs)**2)**0.5>eps_s) and (t < max_steps):
        t += 1 # update iteration counter
        
        ### State S
        old_state = mydpaw.state.copy() # keep track of current state
        
        ### Action A
        action = np.random.choice(2)  ### random choice before learning
        # action = 1
        ### if learned!
        # dc = mydpaw.measureC(old_state)[0]
        # action = pick_sample(action, dc)
        
        ### State' S'
        state = mydpaw.transition(old_state, action)
        
        ### Reward R
        reward = mydpaw.reward(state)
        # mydpaw.state = state*1
        
        # transition["action2"] = action # keep track of action we took
        # if (t > 1.5) and (not eval): # parameter update
        #   # update values after a_{t+1} so we can do SARSA
        #   learner.update(transition)
        
        # store our transition
        transition = {"state1": old_state, # state we came from
                      "state2": state, # state we ended up in
                      "reward": reward, # how much reward did we get
                      "action": action} # what action did we take
        list_of_trans.append(transition)
        total_reward += reward
        
        conc = mydpaw.measureC(old_state)
        temp.append(conc[1])
    
    
    for ii in range(len(list_of_trans)):
        if list_of_trans[ii]['action']==0:
            plt.plot(list_of_trans[ii]['state1'][0], list_of_trans[ii]['state1'][1], 'k.')
        elif list_of_trans[ii]['action']==1:
            plt.plot(list_of_trans[ii]['state1'][0], list_of_trans[ii]['state1'][1], 'r.')
    plt.plot(list_of_trans[0]['state1'][0], list_of_trans[0]['state1'][1], 'g.')
    plt.plot(list_of_trans[-1]['state1'][0], list_of_trans[-1]['state1'][1], color='purple', marker='*')
    
    plt.xlim(-10,110)
    plt.ylim(-10,110)
plt.colorbar()
    
# plt.savefig('example_plot.pdf')

# %% plotting in time
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(list_of_trans)))  # trajectory color in time
plt.figure()
plt.imshow(c_env)
for ii in range(len(list_of_trans)):
    plt.plot(list_of_trans[ii]['state1'][0], list_of_trans[ii]['state1'][1], 'o', color=colors[ii])
plt.plot(list_of_trans[0]['state1'][0], list_of_trans[0]['state1'][1], 'k.')
plt.plot(list_of_trans[-1]['state1'][0], list_of_trans[-1]['state1'][1], 'purple')

plt.figure()
acts = [list_of_trans[ii]['action'] for ii in range(len(list_of_trans))]
plt.plot(acts[:])

# %% plotting state and action
plt.figure()
plt.imshow(c_env)
for ii in range(len(list_of_trans)):
    if list_of_trans[ii]['action']==0:
        plt.plot(list_of_trans[ii]['state1'][0], list_of_trans[ii]['state1'][1], 'k.')
    elif list_of_trans[ii]['action']==1:
        plt.plot(list_of_trans[ii]['state1'][0], list_of_trans[ii]['state1'][1], 'r.')
plt.plot(list_of_trans[0]['state1'][0], list_of_trans[0]['state1'][1], 'k.')
plt.plot(list_of_trans[-1]['state1'][0], list_of_trans[-1]['state1'][1], 'g*')

act_ = [list_of_trans[ii]['action'] for ii in range(len(list_of_trans))]
plt.figure()
plt.plot(act_)
# %%
###############################################################################
# %% controller (using concentration C for action, not state)
# scenario 1: simply learning S-A pair values
# state: up or down gradient
# action: 0 or 1 for WV or PR

# scenario 2: given up ot down gradient, the transition probability for states...need policy gradients

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %% test running RL!!!
# mydpaw = dPAW()
mylearner = Learner()
reward_records = []
opt = optim.AdamW(mylearner.parameters(), lr=0.01)
n_epochs = 1000
gamma = 0.99  # discount
gamma_act = 0.2  # for action discount (which should be smaller)
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
    
for i in range(n_epochs):
    #
    # Run episode till done
    #
    states = []
    actions = []
    rewards = []
    senses = []
    mydpaw.reset()
    action = np.random.choice(2)
    t = 0
    while (np.sum((mydpaw.state-mydpaw.xs)**2)**0.5>eps_s) and (t < max_steps):
        t += 1 # update iteration counter
        ### State S
        old_state = mydpaw.state.copy() # keep track of current state
        ### Action A
        # action = np.random.choice(2)  ### random choice before learning
        dc = mydpaw.measureC(old_state)[0]
        action = pick_sample(action, dc)
        ### State' S'
        state = mydpaw.transition(old_state, action)
        ### Reward R
        reward = mydpaw.reward(state) #+ -(max_steps-t)
        ### record
        actions.append(action)
        states.append(state)
        rewards.append(reward)
        senses.append(dc)
        
    #
    # Get cumulative rewards
    #
    cum_rewards = np.zeros_like(rewards)
    cum_actions = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)
        cum_actions[j] = actions[j] + (cum_actions[j+1]*gamma_act if j+1 < reward_len else 0)
                            #+ lamb*(cum_[j+1]*gamma if j+1 < reward_len else 0  ### test with regularized action!!!
                            
    cum_rewards = cum_rewards + lamb*cum_actions
    #
    # Train (optimize parameters)
    #
    senses = torch.tensor(senses, dtype=torch.float).to(device)
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    opt.zero_grad()
    logits = mylearner(actions, senses)
    # Calculate negative log probability (-log P) as loss.
    # Cross-entropy loss is -log P in categorical distribution. (see above)
    log_probs = -F.cross_entropy(logits, actions, reduction="none")
    loss = -log_probs * cum_rewards
    loss.sum().backward()
    opt.step()

    # Record total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, sum(rewards)), end="\r")
    reward_records.append(sum(rewards))

print("\nDone")

# %%
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 100:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-99:idx+1]
    average_reward.append(np.average(avg_list))
# Plot
plt.figure()
plt.plot(reward_records)
plt.plot(average_reward)
plt.xlabel('epochs', fontsize=20)
plt.ylabel('expected reward', fontsize=20)
# plt.savefig('example_plot.pdf')

# %% evaluation
def run_chemotaxis(policy, n_tracks, newdpaw=None):
    if newdpaw is None:
        mydpaw = dPAW()
    else:
        mydpaw = newdpaw
    mydpaw.reset()
    # mydpaw.C = 10
    eps_s = 5
    max_steps = 300
    pos = np.zeros(n_tracks)
    for n in range(n_tracks):
        mydpaw.reset()
        t = 0
        action = np.random.choice(2)
        while (np.sum((mydpaw.state-mydpaw.xs)**2)**0.5>eps_s) and (t < max_steps):
            t += 1 # update iteration counter
            ### State S
            old_state = mydpaw.state.copy() # keep track of current state
            ### Action A
            if policy=='RL':
                dc = mydpaw.measureC(old_state)[0]
                action = pick_sample(action, dc)
            elif policy=='random':
                # action = np.random.choice(2)
                action = random.choices([0,1], [4/5,1/5])[0]
                # action = pick_sample(action, 0)
            
            elif policy=='parallel':
                action == 0.5
            elif policy=='WV':
                action = 0
            elif policy=='PR':
                action = 1
            ####
            # make controls here for only WV and only PR!
            ####
            
            ### State' S'
            state = mydpaw.transition(old_state, action)
            ### Reward R
            reward = mydpaw.reward(state)

        pos[n] = -reward
    return pos

n_tracks = 100
rl_pos = run_chemotaxis('RL', n_tracks)
rand_pos = run_chemotaxis('random', n_tracks)

# %%
plt.figure()
plt.plot([1+np.random.randn(n_tracks)*.1,2+np.random.randn(n_tracks)*.1],[rl_pos, rand_pos],'o')
mean1, std1 = np.mean(rl_pos), np.std(rl_pos)
mean2, std2 = np.mean(rand_pos), np.std(rand_pos)

# Plotting
plt.errorbar([1, 2], [mean1, mean2], yerr=[std1, std2], fmt='ko')
plt.xticks([1, 2], ['RL-trained', 'shuffled'])
plt.ylabel('distance to source', fontsize=20)
# plt.savefig('example_plot.pdf')

from scipy.stats import ttest_ind
t_statistic, p_value = ttest_ind(rl_pos, rand_pos)
print(p_value)

###
# next steps:
    # environment slope
    # noise level
    # check transition matrix
# %%
n_tracks = 100
new_dpaw = dPAW() # None
new_dpaw.C = 40
new_dpaw.sigC = 60  #60
new_dpaw.noise = 0.
rl_pos = run_chemotaxis('RL', n_tracks, new_dpaw)
rand_pos = run_chemotaxis('random', n_tracks, new_dpaw)
wv_pos = run_chemotaxis('WV', n_tracks, new_dpaw)
pr_pos = run_chemotaxis('PR', n_tracks, new_dpaw)

# %%
conditions = ['staPAW', 'dPAW','WV','BRW']
plt.figure()
plt.bar(conditions, [sum(rl_pos<=5)/n_tracks, sum(rand_pos<=5)/n_tracks,\
                       sum(wv_pos<=5)/n_tracks,sum(pr_pos<=5)/n_tracks])
# plt.bar(conditions, [np.mean(rl_pos), np.mean(rand_pos),np.mean(wv_pos),np.mean(pr_pos)])
plt.ylabel('chemotaxis index',fontsize=20)    
plt.ylim([0,1])
# plt.savefig('example_plot.pdf')

# %% noise comparison
sd = 37
random.seed(sd) #37 42 17
np.random.seed(sd)

n_tracks = 100
new_dpaw = dPAW() # None
new_dpaw.C = 80   #40
new_dpaw.sigC = 80  #60
new_dpaw.noise = 0.
rl_pos_n = run_chemotaxis('RL', n_tracks, new_dpaw)
rand_pos_n = run_chemotaxis('random', n_tracks, new_dpaw)
wv_pos_n = run_chemotaxis('WV', n_tracks, new_dpaw)
pr_pos_n = run_chemotaxis('PR', n_tracks, new_dpaw)
new_dpaw = dPAW() # None
new_dpaw.C = 80
new_dpaw.sigC = 60  #60
rl_pos = run_chemotaxis('RL', n_tracks, new_dpaw)
rand_pos = run_chemotaxis('random', n_tracks, new_dpaw)
wv_pos = run_chemotaxis('WV', n_tracks, new_dpaw)
pr_pos = run_chemotaxis('PR', n_tracks, new_dpaw)

w_noise = [sum(rl_pos_n<=5)/n_tracks, sum(rand_pos_n<=5)/n_tracks,\
                       sum(wv_pos_n<=5)/n_tracks,sum(pr_pos_n<=5)/n_tracks]
wo_noise = [sum(rl_pos<=5)/n_tracks, sum(rand_pos<=5)/n_tracks,\
                       sum(wv_pos<=5)/n_tracks,sum(pr_pos<=5)/n_tracks]

# %%
# Define the width of the bars
bar_width = 0.35
# Set the positions of the bars on the x-axis
x = np.arange(len(conditions))
plt.figure()
plt.bar(x - bar_width/2, wo_noise, width=bar_width, label='trained')
plt.bar(x + bar_width/2, w_noise, width=bar_width, label='steeper')

# Add labels, title, and legend
plt.ylabel('chemotaxis index', fontsize=20)
plt.xticks(x, conditions)
plt.legend(fontsize=10)
plt.ylim([0,1])
# plt.savefig('example_CI.pdf')