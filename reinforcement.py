'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from environment3 import *
from network import *
from scipy.sparse import coo_matrix

''' system/os constants '''
COMPONENTS_LOC = "./data_files/components/"

''' [Helper] Constants '''
N_ACTIONS = 4
N_LAYERS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
P_ACTION_NAMES = { UDIR:"^U^", DDIR:"vDv", RDIR:"R>>", LDIR:"<<L" }
DEFAULT_EPSILON = 0.9
ALL=-1
A_LAYER=0
G_LAYER=1


class session_results(object):
    ''' Stores results from a session.  Stores state data sparsely.
        Data entries: dicts of
            num_a: num actions taken,
            s0: original state with grid in sparse storage format,
            actions_attempted: fixed size, Action ids with -1 for untaken actions,
            success: bool of whether the goal was reached,
            mode: "train" or "test"         '''
    def __init__(self, n_training_epochs, n_episodes):
        self._data = {} # should be nepochs x nepisodes
        self.nepochs = n_training_epochs
        self.nepisodes = n_episodes

    def put(self, epoch, episode, data): 
        if not 'mode' in data.keys(): raise Exception("Provide mode:train or test.")
        if data['mode']=='train': mode = 0 
        if data['mode']=='test' : mode = 1 
        D = data.copy()
        for d in D:
            if d=='s0':
                D[d] = data[d].copy()
                D[d].grid = [coo_matrix(D[d].grid[:,:,i]) for i in range(N_LAYERS)]
                D[d].sparse=True
        self._data[(epoch, episode, mode)] = D

    def get(self, p1, p2=None, p3=None): 
        if p1 in ['train','test'] and p2==None and p3==None:
            return self._get(ALL,ALL,p1)
        if p1 in ['train','test'] and p2 in ['successes'] and p3==None:
            return np.array( [[ 1.0 if x['success'] else 0.0 for x in X]\
                        for X in self._get(ALL,ALL,p1)])
        if p1 in ['train','test'] and p2 in ['nsteps'] and p3==None:
            return np.array( [[ x['num_a'] for x in X]\
                        for X in self._get(ALL,ALL,p1)])
        if p1=='states' and p2==None and p3==None: # return state tups:
            uniq_st = set()
            for i in range(self.nepochs):
                for j in range(self.nepisodes):
                    st = self._data[(i,j,0)]['s0']
                    st.sparsify()
                    uniq_st.add(( st.grid[A_LAYER].row[0],\
                                  st.grid[A_LAYER].col[0],\
                                  st.grid[G_LAYER].row[0],\
                                  st.grid[G_LAYER].col[0] ))
            return sorted(list(uniq_st))
        return self._get(p1,p2,p3)

    def _get(self, epoch, episode, mode):
        if mode=='train': mode = 0 
        if mode=='test' : mode = 1 
        if epoch == ALL: 
            return [self._get(ep, episode, mode) for ep in range(self.nepochs)]
        if episode == ALL: 
            return [self._get(epoch, ep, mode) for ep in range(self.nepisodes)]
        return self._data[(epoch, episode, mode)].copy()


''' reinforcement: class for managing reinforcement training.  Note that 
    the training programs can take extra parameters; these are not to be 
    confused with the hyperparameters, set at the top of this file. '''
class reinforcement(object):
    ''' Constructor: which_game is the variable that commands entirely what 
        training pattern the network will undergo. 

        v1-single: The network learns to act in the game: [[---],[AG-],[---]].
                Mainly for testing and debugging.
        v1-oriented: The network learns to act in all variants of games which
                are 3x3 and only have component [AG] in any (of 6) locations.
        v1-a_fixedloc: the network learns to do act in a puzzles which have
                the _agent_ at a fixed location ->> 4 states, presumed HARDER
        v1-g_fixedloc: the network learns to do act in a puzzles which have
                the _goal_ at a fixed location ->> 4 states, presumed EASIER
        v1-micro: like a_fixedloc, but *all* other spaces are blocked except
                for A and G.
        v1-corner: This game has the network learn all variants of distance-1
                games in which the 3x3 grid is in the same location.
        v1-all:  not yet implemented.

        v0-a_fixedloc: extra simple - zeros.  A at center.
        ...
    '''
    def __init__(self, which_game, frame, game_shape=(5,5), override=None,
                 load_weights_path=None, data_mode=None, seed=None):
        np.random.seed(seed)
        self.env = environment_handler3(game_shape, frame, world_fill='roll')
        self.sg = state_generator(game_shape)
        self.sg.ingest_component_prefabs(COMPONENTS_LOC)
       
        self.D1_ocorner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env, oriented=True)
        self.D1_corner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env)
        self.D1_micro = self.sg.generate_all_states_micro('v1',self.env)

        pdgm, init_states = {
            'v1-single':    ('V1', [ self.D1_corner[2]]),
            'v1-oriented':  ('V1', self.D1_ocorner),
            'v1-micro':    ('V1', self.D1_micro),
            'v1-corner':    ('V1', self.D1_corner),
            'v1-a_fixedloc':  ('V1', [self.D1_corner[3],  self.D1_corner[8],\
                              self.D1_corner[15], self.D1_corner[20] ]),

            'v1-micro_g_fixedloc': ('V1', self.D1_micro+[self.D1_corner[3], \
                self.D1_corner[8], self.D1_corner[15], self.D1_corner[20] ]),
            'v1-micro_a_fixedloc': ('V1', self.D1_micro+[self.D1_corner[3], \
                self.D1_corner[8], self.D1_corner[15], self.D1_corner[20] ]),
            
            # v *** v
            'v0-a_fixedloc':  ('V0', [self.env.getStateFromFile(\
                            './data_files/states/3x3-basic-G'+f, 'except')\
                            for f in ['U.txt', 'R.txt', 'D.txt', 'L.txt']]),
            # ^ *** ^

            'v0-single':    ('V0', [self.env.getStateFromFile(\
                            './data_files/states/3x3-basic-AU.txt', 'except')]),
            'v1-all':       'stub, not implemented yet!',

            }[which_game]
        if 'printing'in override and override['printing']==True:
            for i,s in enumerate(init_states):
                print("State",i); self.env.displayGameState(s, mode=2); 
                print('---------------------')
        self.which_game = which_game
        self.init_states = init_states
        self.which_paradigm = pdgm
        self.data_mode = data_mode
        self.seed = seed
        net_params=None
        if 'netsize' in override:
            o = override['netsize']
            if not len(o)%2==0: raise Exception("Invalid network structure init.")
            nlayers = len(o)/2
            net_params = { o[i]+str(i+1)+'_size':o[i+nlayers] for i in range(nlayers) }
        if 'optimizer_tup' in override:
            opt = override['optimizer_tup']
        else: opt = ('sgd')
        if 'epsilon' in override:
            self.epsilon_exploration = override['epsilon']
        else:
            self.epsilon_exploration = DEFAULT_EPSILON
            
        # If load_weights_path==None, then initialize weights fresh&random.
        self.Net = network(self.env, 'NEURAL', override=override, \
                load_weights_path=load_weights_path, _game_version=\
                'v0', net_params=net_params, _optimizer_type=opt, seed=seed)

        if not override==None:
            self.max_num_actions = override['max_num_actions'] if \
                    'max_num_actions' in override else  MAX_NUM_ACTIONS
            self.training_epochs = override['nepochs'] if \
                    'nepochs' in override else  MAX_NUM_ACTIONS
    
    def _stateLogic(self, s0, Q0, a0_est, s1_est, rotation=0, printing=True):
        ''' (what should this be named?) This function interfaces the network's
            choices with the environment handler for necessary states.  '''
        motion_est = a0_est % N_ACTIONS
        valid_action_flag = self.env.checkIfValidAction(s0, motion_est)
        if valid_action_flag:
            goal_reached = self.env.isGoalReached(s1_est)
            if printing: print "--> Yields valid", a0_est%4, a0_est/4
            return a0_est, s1_est, REWARD if goal_reached else NO_REWARD, \
                    goal_reached, valid_action_flag

        a0_valid = np.argmax(np.exp(Q0[:4]) * self.env.getActionValidities(s0))
        return a0_valid, self.env.performActionInMode(s0, a0_valid), \
                INVALID_REWARD, False, valid_action_flag

    def dev_checks(self):
        if not self.which_paradigm in ['V0','V1']:
           raise Exception("Reinforcement session is only implemented for V0,V1.")

    def _populate_default_params(self, params):
        if not 'saving' in params:
            params['saving'] = { 'dest':None, 'freq':-1 }
        if not 'mode' in params:
            params['present_mode'] = 'shuffled'
        if not 'buffer_updates' in params:
            params['buffer_updates'] = True
        if not 'dropout-all' in params:
            params['dropout-all'] = True
        if not 'printing' in params:
            params['printing'] = False
        return params

    ''' RUN_SESSION: like Simple_Train but less 'simple': Handles the 
        overall managingof a reinforcement session.  Scope:
         - above me: user interfaces for experiments
         - below me: anything not common to all tests. '''
    def run_session(self, params={}):
        self.dev_checks() # raises errors on stubs
        params = self._populate_default_params(params)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.Net.setSess(sess)
        sess.run(init)
        
        episode_results = session_results(self.training_epochs, \
                len(self.init_states))

        Buff=[]
        with sess.as_default():
          tf.set_random_seed(self.seed)
          for epoch in range(self.training_epochs):
            # Save weights
            save_freq = params['saving']['freq']
            if save_freq>=0 and epoch % save_freq==0:
                self.Net.save_weights(params['saving']['dest'], prefix= \
                        'epoch'+str(epoch)+'--')
            # Take pass over data. Todo: batchify here
            if not self.data_mode == None: 
                params['present_mode']=self.data_mode
            if params['present_mode']=='ordered':
                next_states = self.init_states
            elif params['present_mode']=='shuffled':
                next_states = np.random.permutation(self.init_states)
            else: raise Exception("No provided or default data mode.")

            for episode,s0 in enumerate(next_states):
                ret = self._do_one_episode(s0, epoch, 'train')
                episode_results.put(epoch, episode,  \
                      { 's0': s0.copy(), \
                        'num_a': ret[0], \
                        'success': ret[1], \
                        'attempted_actions':ret[2], \
                        'mode': 'train' })
            for episode,s0 in enumerate(next_states):
                ret = self._do_one_episode(s0, epoch, 'test')
                episode_results.put(epoch, episode,  \
                      { 's0': s0.copy(), \
                        'num_a': ret[0], \
                        'success': ret[1], \
                        'attempted_actions':ret[2], \
                        'mode': 'test' })

            if epoch%250==0: 
                print("Epoch #"+str(epoch)+"/"+str(self.training_epochs))
                if not params['printing']: 
                    continue
                if epoch>0:
                        rng = [-1,-2,-3,-4,-5]
                else:
                    rng = [-1]
                print("Q values for last several actions:")
                for wq in rng:
                    print('\t'+P_ACTION_NAMES[Qvals[wq]['action']]+': ['+\
                            ', '.join(["%1.5f"% p for p in Qvals[wq]['Q']])+\
                            ']; corr?: '+str(int(Qvals[wq]['reward'])))

        return episode_results

    def _get_epsilon(self, epoch):
        if type(self.epsilon_exploration)==float:
            return self.epsilon_exploration
        elif self.epsilon_exploration=='lindecay':
            return 1-float(epoch)/self.training_epochs
        elif self.epsilon_exploration=='1/x':
            return 1.0/(epoch+1)
        elif self.epsilon_exploration[:4]=='1/nx':
            return float(self.epsilon_exploration[4:])/(epoch+1)
        else: raise Exception("Epsilon strategy not implemented")


    def _do_one_episode(self, _s0, epoch, mode, buffer_me=True, printing=False):
        Qs = []; losses=[]; steps=[]
        update_buff = []
        num_a = 0
        wasGoalReached = False
        if printing: print '\n\n'
        last_loss=-1.0
        states_log = [_s0]
        attempted_actions = -1*np.ones((self.max_num_actions,))
        for nth_action in range(self.max_num_actions):
            s0 = states_log[-1]
            if self.env.isGoalReached(s0): 
               wasGoalReached=True
               break
            num_a+=1
            Q0 = self.Net.getQVals([s0])
            eps_chosen=False

            # Choose action
            if mode=='train':
                _epsilon = self._get_epsilon(epoch)
                if np.random.rand() < _epsilon: 
                    eps_chosen=True
                    a0_est = np.random.choice(ACTIONS)
                else:
                    a0_est = np.argmax(Q0)
            elif mode=='test':
                a0_est = np.argmax(Q0)

            attempted_actions[nth_action] = a0_est

            s1_est = self.env.performActionInMode(s0, a0_est)

            tmp = self._stateLogic(s0, Q0, a0_est, s1_est, a0_est / 4, printing)
            a0_valid, s1_valid, R, goal_reached, valid_flag = tmp

            targ = np.copy(Q0)
            targ[a0_est] = R
            if not goal_reached:
                if valid_flag:
                    Q1 = self.Net.getQVals([s1_valid])
                    targ[a0_est] -= GAMMA * np.max(Q1)
                else:
                    targ[a0_est] -= GAMMA * a0_est
            if mode=='train':
                self.Net.update(orig_states=[s0], targ_list=[targ])
           
            states_log.append(s1_valid)

        if self.env.isGoalReached(states_log[-1]): wasGoalReached=True
        return num_a, wasGoalReached, attempted_actions
    


    def save_weights(self, dest, prefix=''): self.Net.save_weights(dest, prefix)
    def displayQvals(self):pass
