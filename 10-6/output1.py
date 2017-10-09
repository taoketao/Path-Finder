# IPython log file


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
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]),                                 Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()                                                    ['state']
        self.exp_env = exp_env        
        
    def _reset(self): 
        return self.exp_env.get_random_starting_state()['state']
    
    def _render(self, mode=None, close=None): 
        return print_state(self.current_state,                            'condensed', 'string_ret')

    def sample(self): return random.choice(list(DVECS.keys()))

    def _step(self, actn):
        new_st, succ = self.exp_env.new_statem(self.current_state,                                    actn, valid_move_too=True)
        goalReached = self.exp_env.get_agent_loc(new_st) ==                       self.exp_env.get_goal_loc(new_st)
        self.current_state = new_st
        return new_st, int(goalReached), (not succ) or goalReached, {}
    
    metadata = {'render.modes': ['human', 'ansi']}
    
my_env = MyEnv(ex)
my_env.reset()
my_env.reset().shape
my_env.current_state
my_env.action_space
my_env.observation_space
my_env.metadata
my_env.sample()
my_env.sample()
my_env.sample()
my_env.sample()
my_env.reward_range
type(my_env.reward_range)
help(my_env
)
my_env.render()
my_env.step()
my_env.step(101)
exit()
