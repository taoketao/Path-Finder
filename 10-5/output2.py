# IPython log file


import time; print(time.asctime())
'Fri Oct  6 00:58:37 2017
'
'Fri Oct  6 00:58:37 2017'
''' This log is attempting to essentially embed 
    my environment, with agent, goal, mobile, and
    immobile layers, into gym via an environment 
    that is sourced from code for the (toy) example,
    FrozenLake. '''
import gym
fl_env = gym.make('FrozenLake-v0')
from gym import envs
from gym.envs.registration import register
register(id='TestMeltingLake-v0', entry_point='toy_text.test_melting_lake: MeltingLakeEnv')
tml_env = gym.make('TestMeltingLake-v0')
envs.toy_text
register(id='TestMeltingLake-v0', entry_point='envs.toy_text.test_melting_lake: MeltingLakeEnv')
register(id='TestMeltingLake-v1', entry_point='envs.toy_text.test_melting_lake: MeltingLakeEnv')
tml_env = gym.make('TestMeltingLake-v1')
import sys
sys.path.insert(0, '/home/usrnet/Software/installers/gym/gym/envs')
import test
import test
tml_env = gym.make('TestMeltingLake-v2')
home(/usrnet/Software/installers/gym/gym/envs/test/___init__.py)
exec('/home/usrnet/Software/installers/gym/gym/envs/test/___init__.py')
exec('/home/usrnet/Software/installers/gym/gym/envs/test/__init__.py')
'WHATEVER'
get_ipython().run_line_magic('ls', '')
'See: github.com/openai/gym/issues/626'
help(fl_env.configure)
new_env = gym.Env()
' No, just testing sanity for now...'
[2,2]
from gym import discrete
gym.envs
class MyEnv(gym.Env):
    def __init__(self):
        from cleaner import * 
        try: from gym import spaces
        except: pass; # This should be above
        self.action_space = spaces.Discrete(4)
        #self.observation_space = spaces.Tuple(
from cleaner import *
' imported utilities, constants, and ex, an instance of ExpAPI. '
help(ex)
print(ex.getGridSize())
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        try: from gym import spaces
        except: pass; # This should be above
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple(Discrete(sz[0]), Discrete(sz[1]))
        
        
        
        
        
        
        
        
        
from gym import spaces
my_env = MyEnv(ex)
from gym.spaces import Discrete
my_env = MyEnv(ex)
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple(Discrete(sz[0]), Discrete(sz[1]))
help(MyEnv)
my_env = MyEnv(ex)
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1])))
        
my_env = MyEnv(ex)
help(my_env)
my_env._reset = _my_new_reset
def _my_new_reset(): return ex.get_random_starting_state()['state']
my_env._reset = _my_new_reset
help(my_env._seed)
my_env._seed()
help(my_env)
help(my_env)
' _reset above should not be global :) '
my_env._render = print_state(ex.current_state, 'condensed', 'string_ret')
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
help(gym.spaces)
def my_env._step(actn):
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
     
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _step(self, actn):
        pass
    
get_ipython().run_line_magic('pwd', '')
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _reset(self): return self.exp_env.get_random_starting_state()['state']
    
    def _render(self): return print_state(self.current_state, 'condensed',                         'string_ret')

    def _step(self, actn):
        new, succ = self.exp_env.new_statem(self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.isGoalReached()
        return new, int(goalReached), (not succ) or goalReached, {}
    
my_env = MyEnv(ex)
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', '~/Software/installers/gym/examples/agents')
from cem import *
from random_agent import *
agent = RandomAgent(my_env)
episode_count=100; reward=0; done=False
for i in range(episode_count):
    ob = my_env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = my_env.step(action)
        if done: break
        
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _reset(self): return self.exp_env.get_random_starting_state()['state']
    
    def _render(self): return print_state(self.current_state, 'condensed',                         'string_ret')

    def _step(self, actn):
        new, succ = self.exp_env.new_statem(self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.isGoalReached()
        return new, int(goalReached), (not succ) or goalReached, {}
    
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _reset(self): return self.exp_env.get_random_starting_state()['state']
    
    def _render(self): return print_state(self.current_state, 'condensed',                         'string_ret')


    def sample(self): return random.choice(DVECS.keys())






    def _step(self, actn):
        new, succ = self.exp_env.new_statem(self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.isGoalReached()
        return new, int(goalReached), (not succ) or goalReached, {}
    
agent = RandomAgent(my_env)
for i in range(episode_count):
    
    ob = my_env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = my_env.step(action)
        if done: break
        
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _reset(self): return self.exp_env.get_random_starting_state()['state']
    
    def _render(self): return print_state(self.current_state, 'condensed',                         'string_ret')


    def sample(self): return random.choice(DVECS.keys())






    def _step(self, actn):
        new, succ = self.exp_env.new_statem(self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.isGoalReached()
        return new, int(goalReached), (not succ) or goalReached, {}
    
my_env = MyEnv(ex)
agent = RandomAgent(my_env)
for i in range(episode_count):
    
    ob = my_env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = my_env.step(action)
        if done: break
        
class MyEnv(gym.Env):
    def __init__(self, exp_env):
        self.action_space = Discrete(4)
        sz = exp_env.getGridSize()
        self.observation_space = spaces.Tuple((Discrete(sz[0]), Discrete(sz[1]), Discrete(4)))
        self.current_state = exp_env.get_random_starting_state()['state']
        self.exp_env = exp_env        
        
    def _reset(self): return self.exp_env.get_random_starting_state()['state']
    
    def _render(self): return print_state(self.current_state, 'condensed',                         'string_ret')


    def sample(self): return random.choice(list(DVECS.keys()))






    def _step(self, actn):
        new, succ = self.exp_env.new_statem(self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.isGoalReached()
        return new, int(goalReached), (not succ) or goalReached, {}
    
agent = RandomAgent(my_env)
for i in range(episode_count):
    
    ob = my_env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = my_env.step(action)
        if done: break
        
'  Shelling is about finalized. This should be ready for a notebook. '
exit()
