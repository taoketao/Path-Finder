# IPython log file


from subprocess import run
run(['python','/home/usrnet/Work-Files/Path-Finder/10-6/output1.py'])
exec('/home/usrnet/Work-Files/Path-Finder/10-6/output1.py')
exec(/home/usrnet/Work-Files/Path-Finder/10-6/output1.py)
exec('/home/usrnet/Work-Files/Path-Finder/10-6/output1.py')
execfile('/home/usrnet/Work-Files/Path-Finder/10-6/output1.py')
exec(open('/home/usrnet/Work-Files/Path-Finder/10-6/output1.py').read())
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
get_ipython().run_line_magic('ls', '')
