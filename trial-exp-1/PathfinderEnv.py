#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# A custom environment that can be used instead of an ATARI_GAME
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import time, sys; print (time.asctime())
sys.path.insert(0, '/home/usrnet/Work-Files/Path-Finder')
from Config import Config
sys.path.insert(0, '/home/usrnet/Software/installers/gym/examples/agents')
sys.path.insert(0, '/home/usrnet/Software/installers/gym')
import imp
import gym
import numpy as np
from gym import envs, spaces
from gym.spaces import Discrete

#from random_agent import *
from expenv import *
from environment import *
#from cem import *

def print_state(start_state, mode='condensed', print_or_ret='print'):
    S = ''
    if type(start_state)==np.ndarray:
        st = start_state
    else:
        st = start_state['state']
        S += str(mode+':')
    if mode=='matrices':
        for i in range(st.shape[-1]):
            S += str(st[:,:,i])
    if mode=='condensed':
        for y in range(st.shape[Y]):
            for x in range(st.shape[X]):
                if st[x,y,goalLayer] and st[x,y,agentLayer]: S += str('!')
                elif st[x,y,agentLayer]: S += str('I')
                elif st[x,y,goalLayer] and st[x,y,mobileLayer]: S += str('@')
                elif st[x,y,goalLayer]: S += str('*')
                elif st[x,y,immobileLayer]: S += str('-')
                elif st[x,y,mobileLayer]: S += str('o')
                elif 0==np.sum(st[x,y,:]): S += str(' ')
                else: 
                    S += str('#')
                    print(S)
#                raise Exception("Error", st[x,y,:],S)
            S += str('\n')
    if not type(start_state)==np.ndarray:
        S += str("Flavor signal/goal id: ", start_state['flavor signal'])

    if print_or_ret=='print': print(S)
    else: return S

def pr_st(st):
    return print_state(st, 'condensed', '')



class PathEnv(gym.Env):
    ''' class PathEnv: an openai gym-compliant wrapper for a pathfinder 
            experimental environment. Please provide a well-instantiated 
            ExpAPI (a class that facilitates easy interaction with the 
            core environment objects). Feed this ExpAPI all the experimental
            parameters; this class is strictly just a wrapper.

        Methods defined here: nothing of note. 
    '''
    def __init__(self, exp_env):
        sz = exp_env.getGridSize()
#        self.observation_space = spaces.Tuple((Discrete(sz[0]), \
#                                Discrete(sz[1]), Discrete(NUM_LAYERS)))
#        self.observation_space.shape = tuple(sp.shape for sp in \
#                                        self.observation_space.spaces)
#        self.observation_space = spaces.Box(0,1, (Discrete(sz[0]), Discrete(sz[1]), Discrete(NUM_LAYERS)))
        self.action_space = Discrete(4)
        self.observation_space = spaces.Box(0,1, (sz[0], sz[1], NUM_LAYERS))
                                       
        # ^^ Because gym Tuples don't come with one....!?

        self.current_state = exp_env.get_random_starting_state()['state']
        self.previous_state = self.current_state
#        print ('>> current state:')
#        print_state(self.current_state )
        self.exp_env = exp_env        
        self.metadata = {'render.modes':['human','ansi','PRINT','NOPRINT']}
        self.reward_range = (0,1)
        self.flag=True

        if Config.GAME_NAME=='r-u-ru':
            if Config.CURRICULUM_NAME=='FLAT':
                pass#self.level_1_task = 'r-u-ru'
            elif Config.CURRICULUM_NAME in ['LIN','STEP']:
#                self.level_1_task = 'r-u'
#                self.level_12_task = 'r-u-ru'
#                self.level_2_task = 'ru'
                self.levels = ['r-u', 'ru']
        print("a pathfinder environment wrapper has been imported")

    def reset(self, epoch=-1): 
        if self.flag and not self.exp_env.experiment_name=='r-u-ru':
            print("Flat curriculum used."); 
            self.flag=False;
            curriculum=None
        else:
            curriculum=Config.CURRICULUM_NAME 
        return self._reset(curriculum, epoch) # ugh might be buggy watch out

    def _reset(self, curriculum=None, epoch=-1): 
        ''' Todo: augment so that it can take epoch args (which it 
            passes to the wrapped env for sampling. '''
        self.previous_state = self.current_state = \
                        self.exp_env.get_starting_state(curriculum, epoch)
        return self.current_state
#        if curriculum==None:
#            self.current_state = self.exp_env.get_random_starting_state()['state']
#        else:
#            self.current_state = self.exp_env.get_starting_state_by_epoch()['state']
#        return self.current_state 
    
    def _render(self, mode=None, close=None): 
        #raise Exception("Debug. 442028 Mode: ", mode) This was inconclusive aaa deleted print several lines down.
        p= print_state(self.current_state, \
                           'condensed', 'string_ret')
        if mode in ['human','PRINT']: 
            print(p)
            return p
        elif mode=='NOPRINT': 
            return p
        else: raise Exception(mode, 'render mode not defined')

    def sample(self): 
        return random.choice(list(DVECS.keys()))

    def _step(self, actn):
        new_st, succ = self.exp_env.new_statem(\
            self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.get_agent_loc(new_st) == \
                      self.exp_env.get_goal_loc(new_st)
        self.previous_state = self.current_state
        self.current_state = new_st
        if Config.PLAY_MODE:
            if goalReached: s=' reward: 1'
            elif not succ: s=' reward: 0'
            else: s=''
            print ('action:',actn,'  goalReached:',goalReached,'  acceptable '+\
                    'move:',succ,s,'\n')
        return new_st, int(goalReached), (not succ) or goalReached, {}
    
    def get_num_actions(self): return len(DVECS) # untested


class PathEnvAuto(PathEnv):
    def __init__(self):
        #super(PathEnv.__init__(self, exp_env(Config.GAME_NAME, Config.CENTRISM)))
        try:
            PathEnv.__init__(self, ExpAPI(Config.GAME_NAME, Config.CENTRISM))
        except:
            raise Exception()
