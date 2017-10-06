# IPython log file


# [adapted from an IPython log file]
# This notebook hopes to perform a basic test on the newly-made
# pathfinder environment.

import time; print(time.asctime())
'Fri Oct  6 00:58:37 2017'

import sys
#sys.path.insert(0, '/home/usrnet/Software/installers/gym/')
sys.path.insert(0, '/home/usrnet/Work-Files/Path-Finder')
#sys.path.insert(0, '/home/usrnet/miniconda3/lib/python3.6/site-packages')
sys.path.insert(0, '/home/usrnet/Software/installers/gym/examples/agents')

import gym
from gym import envs, spaces
from gym.spaces import Discrete
from cleaner import *
from random_agent import *
from cem import *
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]),                                 Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()                                                    ['state']
        self.exp_env = exp_env        
        self.metadata = {'render.modes':['human','ansi','PRINT','NOPRINT']}
        self.reward_range = (0,1)
        
    def _reset(self): 
        self.current_state = self.exp_env.get_random_starting_state()['state']
        return self.current_state 
    
    def _render(self, mode=None, close=None): 
        p= print_state(self.current_state,                            'condensed', 'string_ret')
        if mode in ['human','PRINT']: 
            print(p)
            return p
        elif mode=='NOPRINT': 
            return p
        else: raise Exception(mode, 'render mode not defined')

    def sample(self): return random.choice(list(DVECS.keys()))

    def _step(self, actn):
        new_st, succ = self.exp_env.new_statem(            self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.get_agent_loc(new_st) ==                       self.exp_env.get_goal_loc(new_st)
        self.current_state = new_st
        return new_st, int(goalReached), (not succ) or goalReached, {}
    
   
my_env = MyEnv(ex)
''' Adapted from cem.py cross entropy method of rl: '''
def writefile(fname, s):
    with open(path.join(outdir,fname), 'w') as fh: fh.write(s)
    
outdir = '/home/usrnet/Work-Files/Path-Finder/10-6/outdir1'
info={}
from os import path
from cem import cem
from cem import noisy_evaluation
from cem import do_rollout
def noisy_evaluation(theta):
    agent = BinaryActionLinearPolicy(theta)
    rew,T = do_rollout(agent, env, num_steps)
    return rew
env = my_env
params = {n_iter:10, batch_size:25, elite_frac:0.2}
params = dict(n_iter=10, batch_size=25, elite_frac=0.2)
for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros((my_env.exp_env.getGridSize()[0], my_env.exp_env.getGridSize()[1], 4)), **params)):
    print("Iteration %2i. Episode mean reward: %7.3f"%(i, iterdata['y_mean']))
    agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
    if args.display: do_rollout(agent, env, 200, render=True)
    writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))
    
import cPickle as pickle
import pickle
from _policies import BinaryActionLinearPolicy
import numpy as np
for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
    print("Iteration %2i. Episode mean reward: %7.3f"%(i, iterdata['y_mean']))
    agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
    if args.display: do_rollout(agent, env, 200, render=True)
    writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))
    
for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros((25,my_env.exp_env.getGridSize()[0]*my_env.exp_env.getGridSize()[1]*4)), **params)):
    print("Iteration %2i. Episode mean reward: %7.3f"%(i, iterdata['y_mean']))
    agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
    if args.display: do_rollout(agent, env, 200, render=True)
    writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))
    
''' problems? try kerasrl maybe '''
import keras-rl
import keras_rl
import rl
rl
help(rl)
import keras
keras.__version__
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
model = Sequential()
model.add(Flatten(input_shape = (1,11,11,4) ))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(my_env.action_space.n), activation='linear')
model.add(Dense(my_env.action_space.n, activation='linear'))
print(model.summary())
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
help(EpsGreedyQPolicy)
from rl.policy import BoltzmannQPolicy
help(BoltzmannQPolicy)
from rl.memory import SequentialMemory
help(SequentialMemory)
memory = SequentialMemory(limit=1000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions = my_env.action_space.n, memory=memory, nb_steps_warmup=10, enable_dueling_network=True, dueling_type='avg', taret_model_update=1e-3, policy=policy)
dqn = DQNAgent(model=model, nb_actions = my_env.action_space.n, memory=memory, nb_steps_warmup=10, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-3, policy=policy)
dqn.compile(Adam(lr=1e-5), metrics=['mae'])
dqn.fit(my_env, nb_steps=1000, visualize=True, verbose=2)
reload(rl)
# [adapted from an IPython log file]
# This notebook hopes to perform a basic test on the newly-made
# pathfinder environment.

import time; print(time.asctime())
'Fri Oct  6 00:58:37 2017'

import sys
#sys.path.insert(0, '/home/usrnet/Software/installers/gym/')
sys.path.insert(0, '/home/usrnet/Work-Files/Path-Finder')
#sys.path.insert(0, '/home/usrnet/miniconda3/lib/python3.6/site-packages')
sys.path.insert(0, '/home/usrnet/Software/installers/gym/examples/agents')

import gym
from gym import envs, spaces
from gym.spaces import Discrete
from cleaner import *
from random_agent import *
from cem import *
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]),                                 Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()                                                    ['state']
        self.exp_env = exp_env        
        self.metadata = {'render.modes':['human','ansi','PRINT','NOPRINT']}
        self.reward_range = (0,1)
        
    def _reset(self): 
        self.current_state = self.exp_env.get_random_starting_state()['state']
        return self.current_state 
    
    def _render(self, mode=None, close=None): 
        p= print_state(self.current_state,                            'condensed', 'string_ret')
        if mode in ['human','PRINT']: 
            print(p)
            return p
        elif mode=='NOPRINT': 
            return p
        else: raise Exception(mode, 'render mode not defined')

    def sample(self): return random.choice(list(DVECS.keys()))

    def _step(self, actn):
        new_st, succ = self.exp_env.new_statem(            self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.get_agent_loc(new_st) ==                       self.exp_env.get_goal_loc(new_st)
        self.current_state = new_st
        return new_st, int(goalReached), (not succ) or goalReached, {}
    
   
my_env = MyEnv(ex)
model.add(Dense(my_env.action_space.n, activation='linear'))
print(model.summary())
