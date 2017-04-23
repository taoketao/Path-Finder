'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
from environment import *
import sys, time, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

''' system/os constants '''
COMPONENTS_LOC = "./data_files/components/"

''' [Helper] Constants '''
N_ACTIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
NULL_ACTION = -1;
XDIM = 0; YDIM = 1
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
P_ACTION_NAMES = { UDIR:"^U^", DDIR:"vDv", RDIR:"R>>", LDIR:"<<L" }
N_LAYERS = 4
LAYER_NAMES = { 0:"Agent", 1:"Goal", 2:"Immobiles", 3:"Mobiles" }

''' Hyper parameters '''
TRAINING_EPISODES = 120;  # ... like an epoch
MAX_NUM_ACTIONS = 15;
EPSILON = 0.1;
REWARD = 1.0;
NO_REWARD = 0.0;
INVALID_REWARD = 0.0;
GAMMA = 0.9;
LEARNING_RATE = 0.01;
VAR_SCALE = 1.0; # scaling for variance of initialized values

''' Architecture parameters '''
C1_NFILTERS = 64;       # number of filters in conv layer #1
C1_SZ = (3,3);          # size of conv #2's  window
C2_NFILTERS = 64;       # number of filters in conv layer #2
C2_SZ = (3,3);          # size of conv #2's  window
FC_1_SIZE = 100;        # number of outputs in dense layer #1
FC_2_SIZE = N_ACTIONS;  # number of outputs in dense layer #2

''' utility functions '''
def softmax(X): return np.exp(X) / np.sum(np.exp(X))
def pShort(x): 
    return ['{:.2f}'.format(xi) for xi in x]
def pShortN(x,n): 
    if n==1: pShort(x)
    else: 
        for xi in x:
            pShort(xi,n-1)
        print ''


'''         network1: Neural Network class
    This is the implementation of an object-oriented deep Q network.
    This object totally houses the Tensorflow operations that might are 
    required of a deep Q agent (and as little else as possible).

    Functions meant for the user are:  (search file for USER-FACING)
     - <constructor>
     - setSess(tensorflow session)
     - getQVals(state)                      -> states: see environment.py
     - update(states, target_Qvals, predicted_Qvals)     
'''
class network1(object):
    ''' USER-FACING network constructor. Parameters:
        _env: Must provide an environment handler object.  Among various 
            reasons, it is needed from the start for setting up the layers.   
            See environment.py for more info.
        _version: one of NEURAL, RANDOM, or STOCH.  NEURAL, RANDOM supported.
        _batch_off: currently untested; keep as True please.
        _train: solidifies learned weights, and is unchangeable after (?)
        _optimizer_type: one of <sgd>, <adam> for now. '''
    def __init__(self, _env, _version="NEURAL", _batch_off=True, _train=True,\
            _optimizer_type='sgd'):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.version = _version;
        self.batch_off = _batch_off; # for now.

        self.nconvs_1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        self.nconvs_2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)

        ''' Network architecture & forward feeding '''
        # setup helper fields for building the network
        #   layer weight shapes
        conv_filter_1_shape = (C1_SZ[0], C1_SZ[1], N_LAYERS, C1_NFILTERS)
        conv_filter_2_shape = (C2_SZ[0], C2_SZ[1], C1_NFILTERS, C2_NFILTERS)
        conv_strides_1 = (1,1,1,1)
        conv_strides_2 = (1,1,1,1)
        fc_weights_1_shape = (C2_NFILTERS*np.prod(self.nconvs_2), FC_1_SIZE)
        fc_weights_2_shape = (FC_1_SIZE, FC_2_SIZE)
        #   helper factors
        inp_var_factor = N_LAYERS * self.gridsz[XDIM] * self.gridsz[YDIM]
        cf1_var_factor = np.prod(conv_filter_1_shape)
        cf2_var_factor = np.prod(conv_filter_2_shape)
        w1_var_factor = np.prod(fc_weights_1_shape)
        w2_var_factor = np.prod(fc_weights_2_shape)

        ''' Initialization Values.  Random init if weights are not provided. 
            Relu activations after conv1, conv2, fc1 -> small positive inital 
            weights.  Last layer gets small mean-zero weights for tanh. '''
        conv_filter_1_init = tf.random_uniform(\
                conv_filter_1_shape, dtype=tf.float32, \
                minval = 0.0, maxval = 1.0/cf1_var_factor**0.5 * VAR_SCALE)
        conv_filter_2_init = tf.random_uniform(\
                conv_filter_2_shape, dtype=tf.float32, \
                minval = 0.0, maxval = 1.0/cf2_var_factor**0.5 * VAR_SCALE)
        fc_weights_1_init = tf.random_uniform(\
                fc_weights_1_shape, dtype=tf.float32,\
                minval=0.0, maxval=1.0/w1_var_factor**0.5 * VAR_SCALE )
        fc_weights_2_init = tf.random_uniform(\
                fc_weights_2_shape, dtype=tf.float32,\
                minval=-0.5/w2_var_factor**0.5 * VAR_SCALE,\
                maxval= 0.5/w2_var_factor**0.5 * VAR_SCALE )

        ''' trainable Tensorflow Variables '''
        self.conv_filter_1 = tf.Variable(conv_filter_1_init, \
                name='filter1', trainable=_train, dtype=tf.float32)
        self.conv_filter_2 = tf.Variable(conv_filter_2_init, \
                name='filter2', trainable=_train, dtype=tf.float32)
        self.fc_weights_1 = tf.Variable(fc_weights_1_init, \
                name='fc1', trainable=_train)
        self.fc_weights_2 = tf.Variable(fc_weights_2_init, \
                name='fc2', trainable=_train)

        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM], self.gridsz[YDIM], N_LAYERS], 'input');    \
        self.l1 = tf.nn.conv2d(\
                input = self.input_layer, \
                filter = self.conv_filter_1,\
                strides = conv_strides_1, \
                padding = 'VALID', \
                name = "conv1")
        self.l1_act = tf.nn.relu( self.l1 )
        self.l2 = tf.nn.conv2d(\
                input = self.l1, \
                filter = self.conv_filter_2,\
                strides = conv_strides_2, \
                padding = 'VALID', \
                name = "conv2")
        self.l2_act = tf.nn.relu( self.l2 )
        self.l2_flat = tf.contrib.layers.flatten(self.l2_act)
        self.l3 = tf.matmul(self.l2_flat, self.fc_weights_1, name='fc1')
        self.l3_act = tf.nn.relu( self.l3 )
        self.l4 = tf.matmul(self.l3_act, self.fc_weights_2, name='fc2')
        self.output_layer = tf.nn.tanh(self.l4, name='output')

        self.trainable_vars = [self.conv_filter_1, self.conv_filter_2, \
                self.fc_weights_1, self.fc_weights_2] # the important variables.

        '''
        for l in self.trainable_vars:
            print "Layer",l.name,"shape:", l.get_shape()
        print "Layer",self.output_layer.name,"shape:",\
                self.output_layer.get_shape()
                '''

        ''' Network operations (besides forward passing) '''
        self.pred_var = self.output_layer
        self.targ_var = tf.placeholder(tf.float32, [None, N_ACTIONS])
        self.loss_op = tf.reduce_sum(tf.square( self.pred_var - self.targ_var ))
        if _optimizer_type=='sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        elif _optimizer_type=='adam':
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.updates = self.optimizer.minimize(self.loss_op)

        self.sess = None

    ''' Use this method to set a default session, so the user does not need 
        to provide a session for any function that might need a sess. 
        USER-FACING'''
    def setSess(self, s): 
        if not self.sess==None:
            print "Warning: sess is already initialized, overwriting. Tag 66"
        self.sess = s;

    def _checkInit(self): 
        if self.sess==None: 
            raise Exception("Please give me the session with setSess(sess).")
 
    ''' The USER-FACING method for getting Q-values (depends on version) '''
    def getQVals(self, States):
        self._checkInit()
        if self.version=='NEURAL':
            return self._forward_pass(States)
        elif self.version=='RANDOM':
            return [self.yieldRandAction() for _ in States]

    ''' This gives Q-values for a random selection, *NOT* a random action! 
        CAUTION this function is tempting but erroneous -- do not use. 
    '''
    def yieldRandAction(self): 
        return np.array([0.25,0.25,0.25,0.25])

    # method for non-neural policy... 
    def addObservedAction(self, action, reward): self.avgActions[action] += reward;

    # internal method for getting Q_est values from states. 
    def _forward_pass(self, States): # handles batch states
        inp = np.array([s.grid for s in States]) 
        Q_sa = self.sess.run(self.output_layer, \
                feed_dict = {self.input_layer : inp});
        if self.batch_off:
            return Q_sa[0]
        return Q_sa
    
    # see self.update(..) below
    def update_with_Debug_Output(self,s0,s1_est,s1_valid,targ,\
            Q0_sa_FP,R_a0_hat,a0_est):
        print 'action',ACTION_NAMES[a0_est], a0_est, ' reward:',R_a0_hat
        print "s0:\t\t", env.printOneLine(s0, 'ret')
        print "s1_est:\t\t", env.printOneLine(s1_est, 'ret')
        print "s1_valid:\t", env.printOneLine(s1_valid, 'ret')
        print "(pred == orig Q(s0).)"
        print 'targ:\t\t ', targ
        print 'pred:\t\t ', Q0_sa_FP
        self.update([s0], [targ], [Q0_sa_FP])
        print 'updated Q(s0):\t ', Net.getQVals([s0])
        print "unchanged Q(s1): ", Q1_sa_FP

    ''' The USER-FACING method for applying gradient descent or other model
        improvements. Please provide lists of corresponding s0, Q_target,
        Q_est as generated elsewhere. '''
    def update(self, orig_states, targ_list, pred_list):
        self._checkInit()
        if self.version=='NEURAL':
            targs = np.array(targ_list, ndmin=2)
            s0s = np.array([s.grid for s in orig_states])
            self.sess.run(self.updates, feed_dict={self.targ_var: targs, \
                    self.input_layer: s0s} )
        elif self.version=='RANDOM': 
            pass
        elif self.version=='STOCH':
            pass # Stub....self.addObservedAction(...

''' reinforcemente: class for managing training.  Note that the training 
    programs can take extra parameters; these are not to be confused with
    the hyperparameters, set at the top of this file.
'''
class reinforcement(object):

    ''' Constructor: which_game is the variable that commands entirely what 
        training pattern the network will undergo. 

        v1-single: The network learns to act in the game: [[---],[AG-],[---]].
                Mainly for testing and debugging.
        v1-corner: This game has the network learn all variants of distance-1
                games.  Can be thought of as a benchmark for learning success.
        v1-all:  not yet implemented.
        ...
    '''
    def __init__(self, which_game, game_shape=(10,10), override=None):
        # The enviroment instance
        self.env = environment_handler(game_shape)
        # The state generator instance
        self.sg = state_generator(game_shape)
        # state generator: utilize components 
        self.sg.ingest_component_prefabs(COMPONENTS_LOC)
       
        self.init_states = []

        self.D1_corner = self.sg.generate_all_states_fixedCenter('v1', self.env)
        if which_game=='v1-single':
            self.init_states.append(self.D1_corner[2])
            # Game 2 is what I was testing on earlier.  Agent is against middle
            # of left wall and must move directly right one block.
        if which_game=='v1-corner':
            for s in self.D1_corner:
                self.init_states.append(self.env.displayGameState(s))
        self.which_game = which_game
        if which_game in ['v1-single', 'v1-corner', 'v1-all']:
            self.which_paradigm = 'V1' 

        self.Net = network1(self.env, 'NEURAL')

        if not override==None and 'max_num_actions' in override:
            self.max_num_actions = override['max_num_actions']
        else:
            self.max_num_actions = MAX_NUM_ACTIONS

    def getStartState(self, order='rand'):
        return random.choice(self.init_states)

    def stateLogic(self, s0, Q0, a0_est, s1_est):
        ''' (what should this be named?) This function interfaces the network's
            choices with the environment handler for necessary states.  '''
        valid_action_flag = self.env.checkIfValidAction(s0, a0_est)
        if valid_action_flag:
            goal_reached = self.env.isGoalReached(s1_est)
            return a0_est, s1_est, s1_est, \
                    REWARD if goal_reached else NO_REWARD, goal_reached
        a0_valid = np.argmax(np.exp(Q0) * self.env.getActionValidities(s0))
        return a0_valid, self.env.performAction(s0, a0_valid), s0, INVALID_REWARD, 0

    def Simple_Train(self, extra_parameters=None):
        ''' The simplest version of training the net to play the game.
            This training takes a single initial game state, plays it to 
            completion, and updates the model. Currently assumes the paradigm 
            is V1.  This uses static learning rates and epsilon exploration.
        '''
        if not self.which_paradigm=='V1': 
          raise Exception("Simple_Train is only implemented for V1 games.")
        if not extra_parameters==None:
          raise Exception("Simple_Train doesn't handle extra parameters currently.")
        init = tf.global_variables_initializer()
        train_losses = [];  test_losses = [];
        train_nsteps = [];  test_nsteps = [];
        tr_outp = [];       te_outp = [];
        sess = tf.Session()
        self.Net.setSess(sess)
        sess.run(init)
        for episode in range(TRAINING_EPISODES):
            s0 = self.getStartState()
            S=''
            for nth_action in range(self.max_num_actions):
                if self.env.isGoalReached(s0): 
                    break

                Q0 = self.Net.getQVals([s0])
                if random.random() < EPSILON: 
                    a0_est = random.choice(ACTIONS)
                else: 
                    a0_est = np.argmax(Q0)
                s1_est = self.env.performAction(s0, a0_est)
                a0_valid, s1_valid, s1_for_update, R, goal_reached = \
                        self.stateLogic(s0, Q0, a0_est, s1_est)

                targ = np.copy(Q0)
                targ[a0_est] = R
                if not goal_reached:
                    Q1 = self.Net.getQVals([s1_for_update])
                    targ[a0_est] -= GAMMA * np.max(Q1)
                self.Net.update(orig_states=[s0], targ_list=[targ], pred_list=[Q0])
                S +=  P_ACTION_NAMES[a0_est]+'\t'
                train_rew = R

                s0 = s1_valid
            if episode<10 and R<1:
                tr_outp.append( str(episode)+'  train  '+S )
            elif R<1:
                tr_outp.append( str(episode)+' train  '+S )
            S=''
            train_losses.append(train_rew)
            train_nsteps.append(nth_action/float(1))

            s0 = self.getStartState()
            for nth_action in range(self.max_num_actions):
                if self.env.isGoalReached(s0): break
                Q0 = self.Net.getQVals([s0])
                a0_est = np.argmax(Q0) # no epsilon
                s1_est = self.env.performAction(s0, a0_est)
                a0_valid, s1_valid, s1_for_update, R, goal_reached = \
                        self.stateLogic(s0, Q0, a0_est, s1_est)
                targ = np.copy(Q0)
                targ[a0_est] = R
                if not goal_reached:
                    Q1 = self.Net.getQVals([s1_for_update])
                    targ[a0_est] -= GAMMA * np.max(Q1)
                S +=  P_ACTION_NAMES[a0_est]+'\t'
                test_rew = R
                s0 = s1_valid
            
            if episode<10 and R<1:
                te_outp.append( str(episode)+'  test   '+S )
            elif R<1:
                te_outp.append( str(episode)+' test   '+S )
            S=''
            test_losses.append(test_rew)
            test_nsteps.append(nth_action/float(1))
        #for t in tr_outp: print t
        #for t in te_outp: print t
        #print 'Final Q values:',  self.Net.getQVals([self.getStartState()])
        return train_losses, test_losses, test_nsteps, train_nsteps


def geomean(X): 
    # return np.prod(X)**(len(X)**-1) # numerically stable geometric mean
    return np.exp(np.sum(np.log(X))*(1.0/len(X)))

def get_time_str(path):
    t = time.localtime()
    return os.path.join(path, str(t[1])+'_'+str(t[2])+'_'+str(t[0])+'_'+\
            str(t[3]).zfill(2)+str(t[4]).zfill(2))

def test_script():
    # testing 50-episode agents with 10 moves max over 50 agents.
    MNA = []
    mnas = [2,3,5,8,10,13,16,20]
    for mna in mnas:
        print "  **********  TRIAL: max number of actions is "+str(mna)+" **********"
        nsamples = 50
        avg_losses5 = np.empty((nsamples, TRAINING_EPISODES/5, 2))
        avg_losses = np.empty((nsamples, TRAINING_EPISODES, 2))
        avg_steps = np.empty((nsamples, TRAINING_EPISODES, 2))
        for ri in range(nsamples):
            ovr = {}; ovr['max_num_actions'] = mna;
            r = reinforcement('v1-single', override = ovr)
            train, test, TrN, TeN = r.Simple_Train()
            Tr5 = np.sum(np.array(train).reshape((TRAINING_EPISODES/5, 5)), axis=1)
            Te5 = np.sum(np.array(test).reshape((TRAINING_EPISODES/5, 5)), axis=1)
            Tr = np.array(train)
            Te = np.array(test)
            avg_losses[ri,:,0] = Tr
            avg_losses[ri,:,1] = Te
            avg_steps[ri,:,:] = np.array([TrN, TeN]).T
            print 'Agent',ri,'\t',"final train, test errors:", Tr[-1], \
                   Te[-1], ';\t avg train, test errs:', np.mean(Tr), np.mean(Te)
        arr = np.array((avg_losses, avg_steps))
        print arr.shape
        # arr shape: (2: losses & steps, nsamples, max_num_actions, 2: train & test)
        MNA.append(arr)
        np.save(get_time_str('./storage/max_num_action_test_on_v1_single/')\
                +'-mna-'+str(mna), arr)
    for i, mna in enumerate(mnas):
        try:
            avg_losses, avg_steps = MNA[i]
            plt.plot(np.mean(avg_losses[:,:,0], axis=0), c='b', linestyle='-')
            plt.plot(np.mean(avg_losses[:,:,1], axis=0), c='y', linestyle='-')
            plt.plot(np.mean(avg_steps[:,:,0], axis=0), c='g', linestyle='-')
            plt.plot(np.mean(avg_steps[:,:,1], axis=0), c='purple', linestyle='-')
            print "blue: avg train err; yellow: avg test err; green: avg train "+\
                    "steps; purple: avg test steps"
            plt.xlabel("Episode/Epoch.  BLUE: train err, YELLOW: test err, \n"+\
                    "GREEN: avg train steps, PURPLE: avg test steps")
            plt.ylabel("Avg reward -or- avg num actions taken. Max num actions:"+\
                    str( MAX_NUM_ACTIONS ))
            plt.title("Training and testing error averaged over "+str(nsamples)+\
                    " samples; eps="+str(EPSILON))
            #plt.show()
            plt.tight_layout(1.2)
            plt.savefig('./storage/max_num_action_test_on_v1_single/trial-4-22-17--mna'+str(mna))
            print '\n-----------------------------------------\n'
        except:
            print "COULD NOT SAVE MNA=", mna
            pass

test_script()

print "Done."
