'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from environment3 import *
from network import *

''' system/os constants '''
COMPONENTS_LOC = "./data_files/components/"

''' [Helper] Constants '''
N_ACTIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
P_ACTION_NAMES = { UDIR:"^U^", DDIR:"vDv", RDIR:"R>>", LDIR:"<<L" }
DEFAULT_EPSILON = 0.9

''' reinforcemente: class for managing reinforcement training.  Note that 
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
                 load_weights_path=None, data_mode=None):
        self.env = environment_handler3(game_shape, frame, world_fill='roll')
        self.sg = state_generator(game_shape)
        self.sg.ingest_component_prefabs(COMPONENTS_LOC)
       
        self.D1_ocorner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env, oriented=True)
        self.D1_corner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env)
        self.D1_micro = self.sg.generate_all_states_micro('v1',self.env)

        print 'which_game:', which_game
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
        self.rotational = override['rotation']
        net_params=None
        if 'netsize' in override:
            o = override['netsize']
            net_params = { 'cv1_size':o[0], 'cv2_size':o[1], 'fc1_size':o[2] }
        if 'optimizer_tup' in override:
            opt = override['optimizer_tup']
        else: opt = ('sgd')
        if 'epsilon' in override:
            self.epsilon_exploration = override['epsilon']
        else:
            self.epsilon_exploration = DEFAULT_EPSILON
            
        # If load_weights_path==None, then initialize weights fresh&random.
        self.Net = network(self.env, 'NEURAL', rot=self.rotational, override=\
                override, load_weights_path=load_weights_path, _game_version=\
                'v0', net_params=net_params, _optimizer_type=opt)

        if not override==None:
            self.max_num_actions = override['max_num_actions'] if \
                    'max_num_actions' in override else  MAX_NUM_ACTIONS
            self.training_epochs = override['nepochs'] if \
                    'nepochs' in override else  MAX_NUM_ACTIONS
    
    def _stateLogic(self, s0, Q0, a0_est, s1_est, rotation=0, printing=True):
        ''' (what should this be named?) This function interfaces the network's
            choices with the environment handler for necessary states.  '''
        # returns into: a0_valid, s1_valid, R, goal_reached, valid_action_flag

        motion_est = a0_est % N_ACTIONS
        if printing: print "--> State logic: given", a0_est, a0_est%4, rotation, a0_est/4
        valid_action_flag = self.env.checkIfValidAction(s0, motion_est)
        if valid_action_flag:
            goal_reached = self.env.isGoalReached(s1_est)
            if printing: print "--> Yields valid", a0_est%4, a0_est/4
            return a0_est, s1_est, REWARD if goal_reached else NO_REWARD, \
                    goal_reached, valid_action_flag

        # if invalid action, preserve orientation: rotation = 0 degrees.
        a0_valid = np.argmax(np.exp(Q0[:4]) * self.env.getActionValidities(s0))
        # Alternatively:
        #a0_valid =np.argmax(np.exp(Q0[:4]) * self.env.getActionValidities(s0)) 
        #    + 4*np.random.randint(4) for random rotation as well.
                
        if printing: print "--> Yields invalid->", a0_valid, ACTION_NAMES[a0_valid%4],a0_valid/4
        return a0_valid, self.env.performAction(s0, a0_valid, 0), \
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
        train_losses = [];  test_losses = []; # returned
        train_reward = [];  test_reward = []; # returned
        train_nsteps = [];  test_nsteps = []; # returned
        Qvals = [];
        storage = [train_losses, test_losses, train_reward, test_reward, \
                train_nsteps, test_nsteps, Qvals]
        Buff=[]
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
            elif params['present_mode']=='random':
                next_states = np.random.choice(self.init_states, replace=True,\
                        size=len(self.init_states))
            else: raise Exception("No provided or default data mode.")
#            print ''
#            print 'Epoch', epoch, 

            for s_i,s0 in enumerate(next_states):
#                print 'ep',s_i, ', ',
                ret = self._do_one_episode(s0, epoch, 'train', 
                         params['buffer_updates'], params['printing'])

                TrQs, TrR, TrLoss, TrNum_a, TrBuff, Q__data = ret

                if params['buffer_updates']:
                    if epoch % 36 == 35:
                        for b in Buff:
                            self.Net.update(b[0][0], b[0][1]);
                        Buff=[]
                    else:
                        Buff.append(TrBuff)

                TeQs, TeR, TeLoss, TeNum_a, _, Q_data = \
                        self._do_one_episode(s0, epoch, 'test', buffer_me=False, \
                                printing=params['printing'])

                Q__data += Q_data
                train_losses.append(TrLoss)
                train_reward.append(TrR)
                train_nsteps.append(TrNum_a)
                test_losses.append(TeLoss)
                test_reward.append(TeR)
                test_nsteps.append(TeNum_a)
                for a,l,r,q_data,_,tetr,st in Q__data:
                    Qvals.append({'epoch':epoch, 'action':a, 'reward':r, \
                            'Q':q_data, 'mode':tetr, 'loss':l, 'state':st})


            if epoch%1000==0: 
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
        return train_losses, test_losses, train_nsteps, test_nsteps, \
                Qvals, train_reward, test_reward


    def _do_one_episode(self, s0, epoch, mode, buffer_me=True, printing=False):
        Qs = []; losses=[]; steps=[]
        action_q_data = []
        update_buff = []
        num_a = 0.0
        if printing: print '\n\n'
        last_loss=-1.0
        for nth_action in range(self.max_num_actions):
            if mode=='train' and printing: print "\nState:"; self.env.displayGameState(s0)
            if self.env.isGoalReached(s0): break
            num_a+=1
            Q0 = self.Net.getQVals([s0])
            eps_chosen=False
            # Choose action
            if mode=='train':
                randval = random.random()
                if type(self.epsilon_exploration)==float:
                    _epsilon = self.epsilon_exploration
                elif self.epsilon_exploration=='lindecay':
                    _epsilon = 1-float(epoch)/self.training_epochs
                elif self.epsilon_exploration=='1/x':
                    _epsilon = 1.0/(epoch+1)
                elif self.epsilon_exploration[:4]=='1/nx':
                    _epsilon = float(self.epsilon_exploration[4:])/(epoch+1)
                else: raise Exception("Epsilon strategy not implemented")
                if epoch%70==69: print "epsilon at epoch", epoch, ':', _epsilon, self.training_epochs
                if randval < _epsilon: 
                    eps_chosen=True
                    a0_est = np.random.randint(Q0.shape[0])
                    if mode=='train' and printing: print('Action: eps', a0_est)
                else:
                    a0_est = np.argmax(Q0)
                    if mode=='train' and printing: print('Action: select', a0_est)
            elif mode=='test':
                a0_est = np.argmax(Q0)
                if mode=='train' and printing: print('Action: test', a0_est)

            if self.rotational:
                raise Exception("todo: please implement this properly.")
                s1_est = self.env.performAction(s0, a0_est%N_ACTIONS, a0_est/N_ACTIONS)
                # ^ equivalent to: ... a0_est % N_ROTATIONS, a0_est / N_ROTATIONS)
            else:
                s1_est = self.env.performAction(s0, a0_est, 0)

            a0_valid, s1_valid, R, goal_reached, valid_flag = \
                    self._stateLogic(s0, Q0, a0_est, s1_est, a0_est / 4, printing)

            targ = np.copy(Q0)
            targ[a0_est] = R
            if not goal_reached:
                if valid_flag:
                    Q1 = self.Net.getQVals([s1_valid])
                    targ[a0_est] -= GAMMA * np.max(Q1)
                else:
                    targ[a0_est] -= GAMMA * a0_est
           
            if buffer_me==False:# and printing:
                if mode=='train':
                    last_loss = self.Net.update(orig_states=[s0], targ_list=[targ])
                else:
                    last_loss = np.sum(np.square(targ-Q0))
                if epoch%200==0 and printing:
                    if mode=='train': e= ' eps  ' if eps_chosen else ' chose'
                    else:e=' test '
                    if mode=='test': pmode=='test '
                    else: pmode='train'
                    print ' ',epoch,"Last",pmode,"loss, reward: %1.4f"%last_loss,' ',int(R),\
                            'a:',P_ACTION_NAMES[a0_est],e,'  targ-Q:',targ-Q0
            else:
                update_buff.append(([s0], [targ], [Q0]))

            if mode=='train' and printing:
                print "Takes action", ACTION_NAMES[a0_est % N_ACTIONS],
                print "with rotation",(a0_est / 4)*90, '; ', a0_est,
                print "chosen valid action:", a0_valid

            updatedQ0 =  self.Net.getQVals([s0])
            action_q_data.append( ( a0_est, last_loss, R, Q0, updatedQ0, mode, s0 ) )
            #print "degree and reward",R
            #print "Delta Q:", updatedQ0-Q0
            s0 = s1_valid

        if mode=='train':
            pass;#print "Final State:"; self.env.displayGameState(s0)
        return Qs, R, last_loss, num_a, update_buff, action_q_data
    


    def save_weights(self, dest, prefix=''): self.Net.save_weights(dest, prefix)
    def displayQvals(self):pass
