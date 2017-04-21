'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
from environment import *
import sys, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

''' [Helper] Constants '''
N_ACTIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
NULL_ACTION = -1;
XDIM = 0; YDIM = 1
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
N_LAYERS = 4
LAYER_NAMES = { 0:"Agent", 1:"Goal", 2:"Immobiles", 3:"Mobiles" }

''' Hyper parameters '''
training_episodes = 1;  # ... like an epoch
EPSILON_STOCH = 0.2;
REWARD = 1.0;
NO_REWARD = 0.0;
INVALID_REWARD = 0.0;
GAMMA = 0.9;
LEARNING_RATE = 0.1;
VAR_SCALE = 1.0; # scaling for variance of initialized values

''' Architecture parameters '''
C1_NFILTERS = 64;       # number of filters in conv layer #1
C1_SZ = (3,3);          # size of conv #2's  window
C2_NFILTERS = 64;       # number of filters in conv layer #2
C2_SZ = (3,3);          # size of conv #2's  window
FC_1_SIZE = 100;        # number of outputs in dense layer #1
FC_2_SIZE = N_ACTIONS;  # number of outputs in dense layer #2

class network1(object):
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

        for l in self.trainable_vars:
            print "Layer",l.name,"shape:", l.get_shape()
        print "Layer",self.output_layer.name,"shape:",\
                self.output_layer.get_shape()

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

    def setSess(self, s): 
        if not self.sess==None:
            print "Warning: sess is already initialized, overwriting. Tag 66"
        self.sess = s;

    def _softmax(self, X): return np.exp(X) / np.sum(np.exp(X))
    def _checkInit(self): 
        if self.sess==None: 
            raise Exception("Please give me the session with setSess(sess).")
 
    def getQVals(self, States):
        self._checkInit()
        if self.version=='NEURAL':
            return self.forward_pass(States)
        elif self.version=='RANDOM':
            return [self.yieldRandAction() for _ in States]

    def yieldRandAction(self): 
        return np.array([0.25,0.25,0.25,0.25])

    def addObservedAction(self, action, reward):
        self.avgActions[action] += reward;

    def forward_pass(self, States): # handles batch states
        inp = np.array([s.grid for s in States]) 
        Q_sa = self.sess.run(self.output_layer, \
                feed_dict = {self.input_layer : inp});
        if self.batch_off:
            return Q_sa[0]
        return Q_sa
    
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

# STUB: One iteration of a game:
env = environment_handler()
s0 = env.getStateFromFile("./data_files/states/6x6-nextto.txt")
# STUB: consider actions for one state:
Net = network1(env, 'NEURAL')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    Net.setSess(sess)
    sess.run(init);
    LOSSES = []
    for episode in range(training_episodes):
        reward_buffer = []
        losses_buffer = []

        # for # actions permitted....
        #s0 = env.getStartState();

        Q0_sa_FP = Net.getQVals([s0]) # Forward pass: s
        a0_est = np.argmax(Q0_sa_FP)

        s1_est = env.performAction(s0, a0_est)
        valid_action_flag = env.checkIfValidAction(s0, a0_est)
        if valid_action_flag:
            a0_valid = a0_est;
            s1_valid = s1_est;
            R_a0_hat = REWARD if env.isGoalReached(s1_valid) else NO_REWARD
            s1_for_update = s1_est;
        else:
            valid_actions = env.getActionValidities(s0)
            a0_valid = np.argmax(np.exp(Q0_sa_FP) * valid_actions) 
            s1_valid = env.performAction(s0, a0_valid)
            R_a0_hat = INVALID_REWARD;
            s1_for_update = s0; 
        # ^ invalid move -> perform update as if s1=s0 and move to s1_valid.

        reward_buffer.append(R_a0_hat)

        Q1_sa_FP = Net.getQVals([s1_for_update]) # Forward pass: s'

        targ = np.copy(Q0_sa_FP); 
        targ[a0_est] = R_a0_hat + GAMMA * np.max(Q1_sa_FP)

        Net.update([s0], [targ], [Q0_sa_FP]) # Backward pass

        s0 = s1_valid; # Update state for next iteration  STUB

print "Done."
