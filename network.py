'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time, random
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from environment3 import *
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
SCHEDULED_LR_SIGNAL = -23 # match to 

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
DEFAULT_HUBER_SATURATION = 0.1 # 0.03? 

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
        print('')
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
            rot=False, _game_version='v1', net_params=None, seed=None,
            scheduler=None):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self._train = _train
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.version = _version;
        self.batch_off = _batch_off; # for now.
        self.game_version = _game_version
        self.rot=rot;
        self.net_params = net_params
        self.scheduler=scheduler
        if not scheduler==None:
            self.learning_rate = scheduler.learning_rate_signaller()
        elif not override==None and 'learning_rate' in override:
            self.learning_rate = override['learning_rate']
        self.layer_types = list(net_params.keys())

        if not seed==None:
            tf.set_random_seed(seed)
        self.seed = seed

        if 'cv1_size' in self.layer_types and 'cv2_size' in self.layer_types \
                and 'fc3_size' in self.layer_types:
            self.structure = 'cv-cv-fc-fc'
        elif 'fc1_size' in self.layer_types and 'fc2_size' in self.layer_types \
                and 'fc3_size' in self.layer_types:
            self.structure = 'fc-fc-fc-fc'
        elif 'fc1_size' in self.layer_types and 'fc2_size' in self.layer_types:
            self.structure = 'fc-fc-fc'
        elif 'fc1_size' in self.layer_types:
            self.structure = 'fc-fc'
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
        self.conv_strides_1 = (1,1,1,1)
        self.conv_strides_2 = (1,1,1,1)


        if self.structure=='cv-cv-fc-fc':
            self.fc_weights_1_shape = (self.flat_inp_sz, self.fc1_size)
            self.fc_weights_2_shape = (self.fc1_size, self.fc2_size)
            self.fc_weights_3_shape =\
                (self.cv2_size*np.prod(self.nconvs_2), self.fc3_size)
            self.out_weights_4_shape = (self.fc3_size, self.out_size)
        elif self.structure=='fc-fc-fc-fc':
            self.fc_weights_1_shape = (self.flat_inp_sz, self.fc1_size)
            self.fc_weights_2_shape = (self.fc1_size, self.fc2_size)
            self.fc_weights_3_shape = (self.fc2_size, self.fc3_size)
            self.out_weights_4_shape = (self.fc3_size, self.out_size)
        elif self.structure=='fc-fc-fc':
            self.fc_weights_1_shape = (self.flat_inp_sz, self.fc1_size)
            self.fc_weights_2_shape = (self.fc1_size, self.fc2_size)
            self.out_weights_4_shape = (self.fc2_size, self.out_size)
        elif self.structure=='fc-fc':
            self.fc_weights_1_shape = (self.flat_inp_sz, self.fc1_size)
            self.out_weights_4_shape = (self.fc1_size, self.out_size)


        #   helper factors
        try: np_var_factor = N_LAYERS * self.gridsz[XDIM] * self.gridsz[YDIM]
        except: pass
        try: self.c1_var_factor = np.prod(self.conv_filter_1_shape)
        except: pass
        try: self.c2_var_factor = np.prod(self.conv_filter_2_shape)
        except: pass
        try: self.f1_var_factor = np.prod(self.fc_weights_1_shape)
        except: pass
        try: self.f2_var_factor = np.prod(self.fc_weights_2_shape)
        except: pass
        try: self.f3_var_factor = np.prod(self.fc_weights_3_shape)
        except: pass
        try: self.o4_var_factor = np.prod(self.out_weights_4_shape)
        except: pass

        ''' Initialization Values.  Random init if weights are not provided. 
            Relu activations after conv1, conv2, fc1 -> small positive inital 
            weights.  Last layer gets small mean-zero weights for tanh. '''
        
        inits = self._initialize_weights(load_weights_path)
        self.make_variables(inits)

        ''' Construct network: options. '''
        if self.structure=='cv-cv-fc-fc':
            self._construct_2conv_2fc_network(load_weights_path, _train, \
                    _optimizer_type)
        elif self.structure=='fc-fc-fc-fc':
            self._construct_4fc_network(load_weights_path, _train, _optimizer_type)
        elif self.structure=='fc-fc-fc':
            self._construct_3fc_network(load_weights_path, _train, _optimizer_type)
        elif self.structure=='fc-fc':
            self._construct_2fc_network(load_weights_path, _train, _optimizer_type)
        else:
            raise Exception("Network structure not recognized: "+self.structure)

        ''' Network operations (besides forward passing) '''
        self.pred_var = self.output_layer
        self.targ_var = tf.placeholder(tf.float32, [None, self.out_size])
        self.loss_op_updating = self.getLossOp(self.pred_var, self.targ_var, override)

        self.pred_test_var = tf.placeholder(tf.float32, [None, self.out_size])
        self.targ_test_var = tf.placeholder(tf.float32, [None, self.out_size])
        self.loss_op_test = self.getLossOp(self.pred_test_var, \
                self.targ_test_var, override)

        if _optimizer_type==None:
            raise Exception("Please provide which optimizer.")

        if self.learning_rate == SCHEDULED_LR_SIGNAL:
            self.lr_var = tf.Variable(self.scheduler.get_init_lr())
        else: 
            self.lr_var = tf.constant(self.learning_rate)

        if _optimizer_type[0]=='sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr_var)
        elif _optimizer_type[0]=='adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr_var, \
                    epsilon = _optimizer_type[1])
        self.updates = self.optimizer.minimize(self.loss_op_updating)
 
        self.sess = None

    def adjust_lr(self, new_val):
        if self.learning_rate == SCHEDULED_LR_SIGNAL: raise Exception('lr sched')
        self._cur_lr = new_val

    def _construct_2conv_2fc_network(self, load_weights_path, _train, \
                    _optimizer_type):
        drp = self.net_params['dropout'] in ['all']
        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM], self.gridsz[YDIM], N_LAYERS], 'input');    \

        self.l1 = tf.nn.conv2d(\
                input = self.input_layer, \
                filter = self.conv_filter_1,\
                strides = self.conv_strides_1, \
                padding = 'VALID', \
                name = "conv1") + self.conv_bias_1
        if drp=='all': self.l1 = tf.nn.dropout(self.l1, 0.5)
        self.l1_act = tf.nn.relu( self.l1 )

        self.l2 = tf.nn.conv2d(\
                input = self.l1_act, \
                filter = self.conv_filter_2,\
                strides = self.conv_strides_2, \
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
                self.gridsz[XDIM],self.gridsz[YDIM],N_LAYERS], 'input');    \
        self.inp_l = tf.contrib.layers.flatten(self.input_layer)
        #self.l1 = tf.matmul(self.inp_l, self.fc_weights_1, name='fc1')\
        self.l1 = tf.matmul(self.inp_l, self.fc_weights_1, name='fc1')\
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

    def _construct_3fc_network(self, load_weights_path, _train, _optimizer_type):
        drp = self.net_params['dropout'] in ['all']
        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM],self.gridsz[YDIM],N_LAYERS], 'input');    \
        self.inp_l = tf.contrib.layers.flatten(self.input_layer)
        #self.l1 = tf.matmul(self.inp_l, self.fc_weights_1, name='fc1')\
        self.l1 = tf.matmul(self.inp_l, self.fc_weights_1, name='fc1')\
                            + self.fc_bias_1
        if drp in ['all']:  self.l1 = tf.nn.dropout(self.l1, 0.5)
        self.l1_act = tf.nn.relu( self.l1 )

        self.l2 = tf.matmul(self.l1_act, self.fc_weights_2, name='fc2')\
                            + self.fc_bias_2
        if drp in ['all', 'last']:  self.l2 = tf.nn.dropout(self.l2, 0.5)
        self.l2_act = tf.nn.relu( self.l2 )

        self.l4 = tf.matmul(self.l2_act, self.fc_weights_4, name='out4')
        self.output_layer = tf.nn.tanh(self.l4+self.fc_bias_4, name='output')

        self.trainable_vars = [self.fc_weights_1, self.fc_weights_2, \
                self.fc_weights_4, self.fc_bias_1, \
                self.fc_bias_2, self.fc_bias_4]

    def _construct_2fc_network(self, load_weights_path, _train, _optimizer_type):
        drp = self.net_params['dropout'] in ['all']
        ''' Layer Construction (ie forward pass construction) '''
        self.input_layer = tf.placeholder(tf.float32, [None,                  \
                self.gridsz[XDIM],self.gridsz[YDIM],N_LAYERS], 'input');    \
        self.inp_l = tf.contrib.layers.flatten(self.input_layer)
        self.l1 = tf.matmul(self.inp_l, self.fc_weights_1, name='fc1')\
                            + self.fc_bias_1
        if drp in ['all','last']:  self.l1 = tf.nn.dropout(self.l1, 0.5)
        self.l1_act = tf.nn.relu( self.l1 )

        self.l4 = tf.matmul(self.l1_act, self.fc_weights_4, name='out4')
        self.output_layer = tf.nn.tanh(self.l4+self.fc_bias_4, name='output')

        self.trainable_vars = [self.fc_weights_1,  self.fc_bias_1, \
                self.fc_weights_4,  self.fc_bias_4]



    def _init_layer_structure(self):
        if self.net_params==None:
            self.cv1_size = C1_NFILTERS
            self.cv2_size = C2_NFILTERS
            self.fc1_size = FC_1_SIZE
            self.fc2_size = FC_2_SIZE
            self.fc3_size = FC_3_SIZE
            self.out_size = N_ACTIONS
            return
        self.flat_inp_sz = self.gridsz[XDIM]*self.gridsz[YDIM]*N_LAYERS
        if 'cv1_size' in self.net_params:
            self.cv1_size = self.net_params['cv1_size']
        else: self.cv1_size = C1_NFILTERS
        if 'fc1_size' in self.net_params:
            self.fc1_size = self.net_params['fc1_size']
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
                weights['fc3'] = tf.random_uniform(\
                    self.fc_weights_3_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )
                weights['fc3_b'] = tf.random_uniform(\
                    (self.fc3_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )

            if self.structure[:5]=='fc-fc':
                weights['fc1'] = tf.random_uniform(\
                    self.fc_weights_1_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f1_var_factor * VAR_SCALE )
                weights['fc1_b'] = tf.random_uniform(\
                    (self.fc1_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f1_var_factor * VAR_SCALE )

            if self.structure[:8]=='fc-fc-fc':
                weights['fc2'] = tf.random_uniform(\
                    self.fc_weights_2_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f2_var_factor * VAR_SCALE )
                weights['fc2_b'] = tf.random_uniform(\
                    (self.fc2_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f2_var_factor * VAR_SCALE )
            else: self.fc3_size = self.fc2_size

            if self.structure[:11]=='fc-fc-fc-fc':
                weights['fc3'] = tf.random_uniform(\
                    self.fc_weights_3_shape, dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )
                weights['fc3_b'] = tf.random_uniform(\
                    (self.fc3_size,), dtype=tf.float32, seed=self.seed,\
                    minval=0.0, maxval=2.0/self.f3_var_factor * VAR_SCALE )
            else: self.fc4_size = self.fc3_size

            weights['out4']= tf.random_uniform(\
                self.out_weights_4_shape, dtype=tf.float32, seed=self.seed,\
                minval=-1.0/self.o4_var_factor * VAR_SCALE,\
                maxval= 1.0/self.o4_var_factor * VAR_SCALE )
            weights['out4_b']= tf.random_uniform(\
                (self.out_size,), dtype=tf.float32, seed=self.seed,\
                minval=-1.0/self.o4_var_factor * VAR_SCALE,\
                maxval= 1.0/self.o4_var_factor * VAR_SCALE )

        return weights

    def make_variables(self, inits):
        ''' trainable Tensorflow Variables '''
        if self.structure=='cv-cv-fc-fc':
            self.conv_filter_1 = tf.Variable(inits['cv1'], \
                    name='cv1', trainable=self._train, dtype=tf.float32)
            self.conv_bias_1 = tf.Variable(inits['cv1_b'], \
                    name='cv1_b', trainable=self._train, dtype=tf.float32)
            self.conv_filter_2 = tf.Variable(inits['cv2'], \
                    name='cv2', trainable=self._train, dtype=tf.float32)
            self.conv_bias_2 = tf.Variable(inits['cv2_b'], \
                    name='cv2_b', trainable=self._train, dtype=tf.float32)

            self.fc_weights_3 = tf.Variable(inits['fc3'], \
                    name='fc3', trainable=self._train)
            self.fc_bias_3 = tf.Variable(inits['fc3_b'], \
                    name='fc3_b', trainable=self._train)
        elif 'fc-fc' in self.structure:
            self.fc_weights_1 = tf.Variable(inits['fc1'], \
                    name='fc1', trainable=self._train)
            self.fc_bias_1 = tf.Variable(inits['fc1_b'], \
                    name='fc1_b', trainable=self._train)
            if 'fc-fc-fc' in self.structure:
                self.fc_weights_2 = tf.Variable(inits['fc2'], \
                        name='fc2', trainable=self._train)
                self.fc_bias_2 = tf.Variable(inits['fc2_b'], \
                        name='fc2_b', trainable=self._train)
            if 'fc-fc-fc-fc' in self.structure:
                self.fc_weights_3 = tf.Variable(inits['fc3'], \
                        name='fc3', trainable=self._train)
                self.fc_bias_3 = tf.Variable(inits['fc3_b'], \
                        name='fc3_b', trainable=self._train)
        else: raise Exception("Network structure not recognized: "+self.structure)

        self.fc_weights_4 = tf.Variable(inits['out4'], \
                name='out4', trainable=self._train)
        self.fc_bias_4 = tf.Variable(inits['out4_b'], \
                name='out4_b', trainable=self._train)

    ''' Use this to prescribe new loss functions. Default: square err.
    Reference: http://stackoverflow.com/questions/39106732/how-do-i-combine...
        ...-tf-absolute-and-tf-square-to-create-the-huber-loss-function-in '''
    def getLossOp(self, pred, targ, override):
        if override==None or not 'loss_function' in override:
            return tf.reduce_sum(tf.square( pred - targ ))
        lf = override['loss_function']
        #print('\t'+lf)
        if lf == 'square' or lf==None:
            return tf.reduce_sum(tf.square( pred - targ ))
        if 'huber' in lf:
            if len(lf)>5: max_grad = float(lf[5:])
            else: max_grad = DEFAULT_HUBER_SATURATION

            err = tf.reduce_sum(tf.abs( pred - targ ))
            mg = tf.constant(max_grad, name='max_grad')
            lin = mg*(err-.5*mg)
            quad = .5*err*err
            return tf.where(err < mg, quad, lin)



    ''' Use this method to set a default session, so the user does not need 
        to provide a session for any function that might need a sess. 
        USER-FACING'''
    def setSess(self, s): 
        if not self.sess==None:
            print("Warning: sess is already initialized, overwriting. Tag 66")
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
        print(('action',ACTION_NAMES[a0_est], a0_est, ' reward:',R_a0_hat))
        print(("s0:\t\t", env.printOneLine(s0, 'ret')))
        print(("s1_est:\t\t", env.printOneLine(s1_est, 'ret')))
        print(("s1_valid:\t", env.printOneLine(s1_valid, 'ret')))
        print("(pred == orig Q(s0).)")
        print(('targ:\t\t ', targ))
        print(('pred:\t\t ', Q0_sa_FP))
        self.update([s0], [targ], [Q0_sa_FP])
        print(('updated Q(s0):\t ', Net.getQVals([s0])))
        print(("unchanged Q(s1): ", Q1_sa_FP))

    ''' The USER-FACING method for applying gradient descent or other model
        improvements. Please provide lists of corresponding s0, Q_target,
        Q_est as generated elsewhere.  Returns loss.  '''
    def update(self, orig_states, targ_list):
        self._checkInit()
        if self.version=='NEURAL':
            targs = np.array(targ_list, ndmin=2)
            s0s = np.array([s.grid for s in orig_states])
            if self.learning_rate == SCHEDULED_LR_SIGNAL:
                return self.sess.run([self.loss_op_updating, self.updates], \
                        feed_dict={\
                            self.targ_var: targs, \
                            self.input_layer: s0s       } )[0]
            else:
                return self.sess.run([self.loss_op_updating, self.updates], \
                        feed_dict={\
                            self.targ_var: targs, \
                            self.input_layer: s0s, \
                            self.lr_var: self._cur_lr   } )[0]
        else:
            raise Exception("Network version not set to NEURAL; cannot update")
        print("Flag 99")
        return -1

    def getLoss(self, pred, targ):
        return self.sess.run([self.loss_op_test], feed_dict={\
                    self.pred_test_var: np.array(pred),  \
                    self.targ_test_var: np.array(targ)})[0]

    def save_weights(self, dest, prefix=''):
        self._checkInit()
        all_trainable_weights = self.sess.run({t.name:t for t in self.trainable_vars})
        weightsave_args = {'file':get_time_str(dest,"SavedWeights_"+prefix)}
        for wname,W in list(all_trainable_weights.items()):
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
    env = environment_handler3(DUMMY_GRIDSZ, action_mode='egocentric')
    print((os.listdir(OUT_DIR)))
    N1 = network(env, load_weights_path=None, net_params = {'fc1_size':24}, \
            _optimizer_type='sgd')

    s = tf.Session()
    s.run(tf.global_variables_initializer())
    N1.setSess(s)

    N1.save_weights(OUT_DIR, prefix='1')
    l = os.listdir(OUT_DIR)
    print(l)
    N2 = network(env,load_weights_path=OUT_DIR+l[-1], net_params = \
            {'fc1_size':24}, _optimizer_type='sgd'   )
    N1.save_weights(OUT_DIR, prefix='2')
    l = os.listdir(OUT_DIR)
    print(l)

    for li in l:
        if 'SavedWeights' in li:
            os.remove(os.path.join(OUT_DIR,li))
    print(os.listdir(OUT_DIR))

    ##################################################

    env = environment_handler3(DUMMY_GRIDSZ, action_mode='egocentric')
    NC0 = NetworkCollector( {'cv1_size':24} )
    NC1 = NetworkCollector(EXP_DIR)
