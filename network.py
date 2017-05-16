'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from environment2 import *
from save_as_plot import *

''' system/os constants '''
COMPONENTS_LOC = "./data_files/components/"

''' [Helper] Constants '''
N_ACTIONS = 4
N_ROTATIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
NULL_ACTION = -1;
XDIM = 0; YDIM = 1
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
P_ACTION_NAMES = { UDIR:"^U^", DDIR:"vDv", RDIR:"R>>", LDIR:"<<L" }
N_LAYERS = 4
LAYER_NAMES = { 0:"Agent", 1:"Goal", 2:"Immobiles", 3:"Mobiles" }
DUMMY_GRIDSZ = (5,5)

''' [Default] Hyper parameters '''
TRAINING_EPISODES = 300;  # ... like an epoch
MAX_NUM_ACTIONS = 15;
EPSILON = 1.0;
REWARD = 1;
NO_REWARD = 0.0;
INVALID_REWARD = 0.0;
GAMMA = 0.9;
LEARNING_RATE = 0.01;
VAR_SCALE = 1.0; # scaling for variance of initialized values

''' Architecture default parameters '''
C1_NFILTERS = 32;       # number of filters in conv layer #1
C1_SZ = (3,3);          # size of conv #2's  window
C2_NFILTERS = 32;       # number of filters in conv layer #2
C2_SZ = (3,3);          # size of conv #2's  window
FC_1_SIZE = 32;        
FC_2_SIZE = 32;       
FC_3_SIZE = 72;        # number of outputs in dense layer #1
N_ROTATIONS = 4; # FC layer 2 out size: # actions or # actions + # rotations.

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
def get_time_str(path, prefix=''):
    t = time.localtime()
    return os.path.join(path, prefix+str(t[1])+'_'+str(t[2])+'_'+str(t[0])\
            +'_'+str(t[3]).zfill(2)+str(t[4]).zfill(2))

'''         network: Neural Network class
    This is the implementation of an object-oriented deep Q network.
    This object totally houses the Tensorflow operations that might are 
    required of a deep Q agent (and as little else as possible).
    This is an augmented version of network1 that now handles rotations.

    Functions meant for the user are:  (search file for USER-FACING)
     - <constructor>
     - setSess(tensorflow session)
     - getQVals(state)                      -> states: see environment.py
     - update(states, target_Qvals)         '''
class network(object):
    ''' USER-FACING network constructor. Parameters:
        _env: Must provide an environment handler object.  Among various 
            reasons, it is needed from the start for setting up the layers.   
            See environment.py for more info.
        _version: one of NEURAL, RANDOM, or STOCH.  NEURAL, RANDOM supported.
        _batch_off: currently untested; keep as True please.
        _train: solidifies learned weights, and is unchangeable after (?)
        _optimizer_type: one of ('sgd'), ('adam', <eps>),.. for now. '''
    def __init__(self, _env, _version="NEURAL", _batch_off=True, _train=True,\
            _optimizer_type=None, override=None, load_weights_path=None, \
            rot=False, _game_version='v1', net_params=None, seed=None):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.version = _version;
        self.batch_off = _batch_off; # for now.
        self.game_version = _game_version
        self.rot=rot;
        self.net_params = net_params
        if not override==None and 'learning_rate' in override:
            self.learning_rate = override['learning_rate']
        else:
            self.learning_rate = LEARNING_RATE
        self.layer_types = net_params.keys()

        if not seed==None:
            tf.set_random_seed(seed)
        self.seed = seed

        if 'cv1_size' in self.layer_types and 'cv2_size' in self.layer_types \
                and 'fc3_size' in self.layer_types:
            self.structure = 'cv-cv-fc-fc'
        elif 'fc1_size' in self.layer_types and 'fc2_size' in self.layer_types \
                and 'fc3_size' in self.layer_types:
            self.structure = 'fc-fc-fc-fc'
        else: raise Exception("Invalid network build requested.")


        self.nconvs_1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        self.nconvs_2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)

        ''' Network architecture & forward feeding '''
        # setup helper fields for building the network
        #   layer weight shapes
        self._init_layer_structure()

        self.conv_filter_1_shape = (C1_SZ[0], C1_SZ[1], N_LAYERS, self.cv1_size)
        self.conv_filter_2_shape = (C2_SZ[0], C2_SZ[1], self.cv1_size, \
                self.cv2_size)
        conv_strides_1 = (1,1,1,1)
        conv_strides_2 = (1,1,1,1)

        self.fc_weights_1_shape =(N_LAYERS, self.fc3_size)
        self.fc_weights_2_shape = (self.fc3_size, self.fc3_size)

        if self.structure=='cv-cv-fc-fc':
            self.fc_weights_3_shape =\
                (self.cv2_size*np.prod(self.nconvs_2), self.fc3_size)
        elif self.structure=='fc-fc-fc-fc':
            self.fc_weights_3_shape = (self.fc2_size, self.fc3_size)
        self.out_weights_4_shape = (self.fc3_size, self.out_size)


        #   helper factors
        try: np_var_factor = N_LAYERS * self.gridsz[XDIM] * self.gridsz[YDIM]
        except: pass
        try: self.c1_var_factor = np.prod(self.conv_filter_1_shape)
        except: pass
        try: self.c2_var_factor = np.prod(self.conv_filter_2_shape)
        except: pass
        try: self.f3_var_factor = np.prod(self.fc_weights_3_shape)
        except: pass
        try: self.o4_var_factor = np.prod(self.out_weights_4_shape)
        except: pass

        ''' Initialization Values.  Random init if weights are not provided. 
            Relu activations after conv1, conv2, fc1 -> small positive inital 
            weights.  Last layer gets small mean-zero weights for tanh. '''
        
        inits = self._initialize_weights(load_weights_path)

        ''' trainable Tensorflow Variables '''
        if self.structure=='cv-cv-fc-fc':
            self.conv_filter_1 = tf.Variable(inits['cv1'], \
                    name='cv1', trainable=_train, dtype=tf.float32)
            self.conv_bias_1 = tf.Variable(inits['cv1_b'], \
                    name='cv1_b', trainable=_train, dtype=tf.float32)
            self.conv_filter_2 = tf.Variable(inits['cv2'], \
                    name='cv2', trainable=_train, dtype=tf.float32)
            self.conv_bias_2 = tf.Variable(inits['cv2_b'], \
                    name='cv2_b', trainable=_train, dtype=tf.float32)
        elif self.structure=='fc-fc-fc-fc':
            self.fc_weights_1 = tf.Variable(inits['fc1'], \
                    name='fc1', trainable=_train)
            self.fc_bias_1 = tf.Variable(inits['fc1_b'], \
                    name='fc1_b', trainable=_train)
            self.fc_weights_2 = tf.Variable(inits['fc2'], \
                    name='fc2', trainable=_train)
            self.fc_bias_2 = tf.Variable(inits['fc2_b'], \
                    name='fc2_b', trainable=_train)

        self.fc_weights_3 = tf.Variable(inits['fc3'], \
                name='fc3', trainable=_train)
        self.fc_bias_3 = tf.Variable(inits['fc3_b'], \
                name='fc3_b', trainable=_train)
        self.fc_weights_4 = tf.Variable(inits['out4'], \
                name='out4', trainable=_train)
        self.fc_bias_4 = tf.Variable(inits['out4_b'], \
                name='out4_b', trainable=_train)

        ''' Construct network: options. '''
        if self.structure=='cv-cv-fc-fc':
            self._construct_2conv_2fc_network(load_weights_path, _train, \
                    _optimizer_type)
        elif self.structure=='fc-fc-fc-fc':
            self._construct_4fc_network(load_weights_path, _train, \
                    _optimizer_type)

        ''' Network operations (besides forward passing) '''
        self.pred_var = self.output_layer
        self.targ_var = tf.placeholder(tf.float32, [None, self.out_size])
        self.loss_op = tf.reduce_sum(tf.square( self.pred_var - self.targ_var ))
        #self.loss_op = tf.reduce_sum(tf.abs( self.pred_var - self.targ_var ))
        if _optimizer_type==None:
            raise Exception("Please provide which optimizer.")
        if _optimizer_type[0]=='sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(\
                    self.learning_rate)
        elif _optimizer_type[0]=='adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, \
                    epsilon=_optimizer_type[1])
        self.updates = self.optimizer.minimize(self.loss_op)
 
        self.sess = None

    def _construct_2conv_2fc_network(self, load_weights_path, _train, \
                    _optimizer_type):
        drp = self.net_params['dropout'] in ['all']
        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM], self.gridsz[YDIM], N_LAYERS], 'input');    \

        self.l1 = tf.nn.conv2d(\
                input = self.input_layer, \
                filter = self.conv_filter_1,\
                strides = conv_strides_1, \
                padding = 'VALID', \
                name = "conv1") + self.conv_bias_1
        if drp=='all': self.l1 = tf.nn.dropout(self.l1, 0.5)
        self.l1_act = tf.nn.relu( self.l1 )

        self.l2 = tf.nn.conv2d(\
                input = self.l1_act, \
                filter = self.conv_filter_2,\
                strides = conv_strides_2, \
                padding = 'VALID', \
                name = "conv2") + self.conv_bias_2 
        if drp=='all': self.l2 = tf.nn.dropout(self.l2, 0.5)
        self.l2_act = tf.nn.relu( self.l2 )

        self.l2_flat = tf.contrib.layers.flatten(self.l2_act)
        self.l3 = tf.matmul(self.l2_flat, self.fc_weights_3, name='fc3')\
                            + self.fc_bias_3
        if drp in ['all', 'last']: self.l3 = tf.nn.dropout(self.l3, 0.5)

        self.l3_act = tf.nn.relu( self.l3  )
        self.l4 = tf.matmul(self.l3_act, self.fc_weights_4, name='out4')
        self.output_layer = tf.nn.tanh(self.l4+self.fc_bias_4, name='output')

        self.trainable_vars = [self.conv_filter_1, self.conv_filter_2, \
                self.fc_weights_3, self.fc_weights_4, self.conv_bias_1, \
                self.conv_bias_2, self.fc_bias_3, self.fc_bias_4]


    def _construct_4fc_network(self, load_weights_path, _train, _optimizer_type):
        drp = self.net_params['dropout'] in ['all']
        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM], self.gridsz[YDIM], N_LAYERS], 'input');    \

        self.l1 = tf.matmul(self.input_layer, self.fc_weights_1, name='fc1')\
                            + self.fc_bias_1
        if drp in ['all']:  self.l1 = tf.nn.dropout(self.l1, 0.5)
        self.l1_act = tf.nn.relu( self.l1 )

        self.l2 = tf.matmul(self.l1_act, self.fc_weights_2, name='fc2')\
                            + self.fc_bias_2
        if drp in ['all']:  self.l2 = tf.nn.dropout(self.l2, 0.5)
        self.l2_act = tf.nn.relu( self.l2 )

        self.l3 = tf.matmul(self.l2_act, self.fc_weights_3, name='fc3')\
                            + self.fc_bias_3
        if drp in ['all', 'last']: self.l3 = tf.nn.dropout(self.l3, 0.5)
        self.l3_act = tf.nn.relu( self.l3 )

        self.l4 = tf.matmul(self.l3_act, self.fc_weights_4, name='out4')
        self.output_layer = tf.nn.tanh(self.l4+self.fc_bias_4, name='output')

        self.trainable_vars = [self.fc_weights_1, self.fc_weights_2, \
                self.fc_weights_3, self.fc_weights_4, self.fc_bias_1, \
                self.fc_bias_2, self.fc_bias_3, self.fc_bias_4]


    def _init_layer_structure(self):
        if self.net_params==None:
            self.cv1_size = C1_NFILTERS
            self.cv2_size = C2_NFILTERS
            self.fc1_size = FC_1_SIZE
            self.fc2_size = FC_2_SIZE
            self.fc3_size = FC_3_SIZE
            self.out_size = N_ACTIONS
            return
        if 'cv1_size' in self.net_params:
            self.cv1_size = self.net_params['cv1_size']
        else: self.cv1_size = C1_NFILTERS
        if 'fc1_size' in self.net_params:
            self.fc1_size = self.net_params['fc2_size']
        else:  self.fc1_size = FC_1_SIZE
        if 'cv2_size' in self.net_params:
            self.cv2_size = self.net_params['cv2_size']
        else:  self.cv2_size = C2_NFILTERS
        if 'fc2_size' in self.net_params:
            self.fc2_size = self.net_params['fc2_size']
        else:  self.fc2_size = FC_2_SIZE
        if 'fc3_size' in self.net_params:
            self.fc3_size = self.net_params['fc3_size']
        else: self.fc3_size = FC_3_SIZE

        if not 'action_mode' in self.net_params:
            self.out_size = N_ACTIONS
        elif self.net_params['action_mode'] in \
                    ['slide_worldcentric','slide_egocentric']:
            self.out_size = N_ACTIONS
        elif self.net_params['action_mode'] in \
                    ['rotSPIN_worldcentric', 'rotSPIN_egocentric',\
                    'rotFwdBack_worldcentric', 'rotFwdBack_egocentric']:
            raise Exception("Warning: rotation actions are currently unsupported.")
        else:
            raise Exception("Warning: unspecified action mode is ambiguous.")
        if not 'dropout' in self.net_params:
            self.net_params['dropout'] = 'all'

    
    def _initialize_weights(self, initialization, mode='1/n'):
        # Mode: 1/n.  Initialization: filename or False 
        weights = {}
        if not initialization == None:
            loaded = np.load(initialization)
            for var in loaded:
                weights[var.split(':')[0]] = tf.constant(loaded[var])
            return weights
        if mode=='1/n':
            if self.structure=='cv-cv-fc-fc':
                ''' First Layers:Conv inits '''
                weights['cv1'] = tf.random_uniform(\
                    self.conv_filter_1_shape, dtype=tf.float32, seed=self.seed,\
                    minval = 0.0, maxval = 2.0/self.c1_var_factor * VAR_SCALE)
                weights['cv1_b'] = tf.random_uniform(\
                    (self.cv1_size,), dtype=tf.float32, seed=self.seed,\
                    minval = 0.0, maxval = 2.0/self.c1_var_factor * VAR_SCALE)
                weights['cv2'] = tf.random_uniform(\
                    self.conv_filter_2_shape, dtype=tf.float32, seed=self.seed, \
                    minval = 0.0, maxval = 2.0/self.c2_var_factor * VAR_SCALE)
                weights['cv2_b'] = tf.random_uniform(\
                    (self.cv2_size,), dtype=tf.float32, seed=self.seed, \
                    minval = 0.0, maxval = 2.0/self.c2_var_factor * VAR_SCALE)

            if self.structure=='fc-fc-fc-fc':
                ''' First Layers:FC inits '''
                weights['fc1'] = tf.random_uniform(\
                    self.fc_weights_1_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f1_var_factor * VAR_SCALE )
                weights['fc1_b'] = tf.random_uniform(\
                    (self.fc3_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f1_var_factor * VAR_SCALE )
                weights['fc2'] = tf.random_uniform(\
                    self.fc_weights_2_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f2_var_factor * VAR_SCALE )
                weights['fc2_b'] = tf.random_uniform(\
                    (self.fc3_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f2_var_factor * VAR_SCALE )

            ''' Last Layers:FC,out inits '''
            weights['fc3'] = tf.random_uniform(\
                self.fc_weights_3_shape, dtype=tf.float32, seed=self.seed,\
                minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )
            weights['fc3_b'] = tf.random_uniform(\
                (self.fc3_size,), dtype=tf.float32, seed=self.seed,\
                minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )

            weights['out4']= tf.random_uniform(\
                self.out_weights_4_shape, dtype=tf.float32, seed=self.seed,\
                minval=-1.0/self.o4_var_factor * VAR_SCALE,\
                maxval= 1.0/self.o4_var_factor * VAR_SCALE )
            weights['out4_b']= tf.random_uniform(\
                (self.out_size,), dtype=tf.float32, seed=self.seed,\
                minval=-1.0/self.o4_var_factor * VAR_SCALE,\
                maxval= 1.0/self.o4_var_factor * VAR_SCALE )

        return weights


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
        print('action',ACTION_NAMES[a0_est], a0_est, ' reward:',R_a0_hat)
        print("s0:\t\t", env.printOneLine(s0, 'ret'))
        print("s1_est:\t\t", env.printOneLine(s1_est, 'ret'))
        print("s1_valid:\t", env.printOneLine(s1_valid, 'ret'))
        print("(pred == orig Q(s0).)")
        print('targ:\t\t ', targ)
        print('pred:\t\t ', Q0_sa_FP)
        self.update([s0], [targ], [Q0_sa_FP])
        print('updated Q(s0):\t ', Net.getQVals([s0]))
        print("unchanged Q(s1): ", Q1_sa_FP)

    ''' The USER-FACING method for applying gradient descent or other model
        improvements. Please provide lists of corresponding s0, Q_target,
        Q_est as generated elsewhere. '''
    def update(self, orig_states, targ_list):
        self._checkInit()
        if self.version=='NEURAL':
            targs = np.array(targ_list, ndmin=2)
            s0s = np.array([s.grid for s in orig_states])
            return self.sess.run([self.loss_op, self.updates], feed_dict=\
                    {self.targ_var: targs, self.input_layer: s0s} )[0]
        else:
            raise Exception("Network version not set to NEURAL; cannot update")
        print("Flag 99")
        return -1

    def save_weights(self, dest, prefix=''):
        self._checkInit()
        all_trainable_weights = self.sess.run({t.name:t for t in self.trainable_vars})
        weightsave_args = {'file':get_time_str(dest,"SavedWeights_"+prefix)}
        for wname,W in all_trainable_weights.items():
            weightsave_args[wname] = W
        np.savez(**weightsave_args)


class NetworkCollector(object):
    ''' Simple wrapping object for managing networks and their initialization.'''
    def __init__(self, argument):
        self.all_nets = []
        """
        elif os.path.isdir(argument):
            ''' Case: interpret the argument as a directory holding 
                network initialization files. '''
            raise Exception("not yet implemented")
            [ self._ingest_netfile(netfile) for netfile in argument  ]"""
        if False:
            pass
        elif os.path.isfile(argument):
            ''' Case: interpret argument as a single initialization file '''
            raise Exception("not yet implemented")
            self._ingest_netfile(argument)
        elif type(argument)==list:
            for a in argument:
                self.all_nets.append(network(net_params=a))
        elif type(argument)==dict:
            self.all_nets.append(network(net_params=argument))
        else:
            raise Exception("NetworkCollector has not implemented that "+\
                    "argument type.  Please provide file(s) for network init.")

        """    ...TODO...    """




# Example usage and check:
if __name__=='__main__':
    OUT_DIR = './storage/5-04/'
    EXP_DIR = './experiments/5-04/'
    env = environment_handler2(DUMMY_GRIDSZ)
    print(os.listdir(OUT_DIR))
    N1 = network(env, load_weights_path=None)

    s = tf.Session()
    s.run(tf.global_variables_initializer())
    N1.setSess(s)

    N1.save_weights(OUT_DIR, prefix='1')
    l = os.listdir(OUT_DIR)
    print(l)
    N2 = network(env,load_weights_path=OUT_DIR+l[-1])
    N1.save_weights(OUT_DIR, prefix='2')
    l = os.listdir(OUT_DIR)
    print(l)

    for li in l:
        if 'SavedWeights' in li:
            os.remove(os.path.join(OUT_DIR,li))
    print os.listdir(OUT_DIR)

    ##################################################

    env = environment_handler2(DUMMY_GRIDSZ, action_mode='slide_worldcentric')
    NC0 = NetworkCollector( {'cv1_size':24} )
    NC1 = NetworkCollector(EXP_DIR)
