'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
from environment import *
import sys, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
''' from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import RMSprop '''

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
conv1_nfilters = 64;    # number of filters in conv layer #1
conv1_size = (3,3);     # size of conv #2's  window
conv2_nfilters = 64;    # number of filters in conv layer #2
conv2_size = (3,3);     # size of conv #2's  window
fc_1_size = 100;        # number of outputs in dense layer #1
fc_2_size = 4;          # number of outputs in dense layer #2

class network1(object):
    def __init__(self, _env, _version="NEURAL", _batch_off=True, _train=True):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.version = _version;
        self.batch_off = _batch_off; # for now.

        self.nconvs_1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        self.nconvs_2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)

        ''' Network architecture & forward feeding '''
        # question: at which layer ought we to mix the channels? 
        # Currently, the mixing happens right after the input, into l1.
        # Would switch to depthwise_conv2d.
        '''
        self.input_layer = tf.placeholder(tf.float32, [None,                \
                self.gridsz[XDIM], self.gridsz[YDIM], N_ACTIONS], 'input'); \
        self.l1 = tf.nn.conv2d( name = "layer1",                        \
                input = self.input_layer, kernel_size = conv1_size,           \
                filter = conv1_nfilters)
        self.l1_ = tf.nn.relu( self.l1 )
        self.l2 = tf.nn.conv2d( name = "layer2",                        \
                input = self.l1_, kernel_size=conv2_size,                      \
                filter = conv2_nfilters)
        self.l2_ = tf.nn.relu( self.l2 )
        self.l2_reshaped = tf.contrib.layers.flatten(self.l2)
        self.fc1 = tf.layers.dense( name = "layer3",                        \
                inputs = self.l2_reshaped, units = fc_1_size)
        self.fc1_ = tf.nn.relu( self.fc1 )
        self.fc2 = tf.layers.dense( name = "layer4",                        \
                inputs = self.fc1, units = fc_2_size);
        self.fc2_ = tf.nn.relu( self.fc2 )
        self.output_layer = tf.identity(self.fc2, name='output') 

        self.trainable_layers = [self.l1, self.l2, \
                self.fc1, self.fc2];'''

        # question: output *how many* channels from first conv layer?
        conv_filter_1_shape = (3,3,N_LAYERS,conv1_nfilters)
        cf1_var_factor = np.prod(conv_filter_1_shape)
        conv_strides_1 = (1,1,1,1)
        conv_filter_1_init = tf.random_uniform(\
                conv_filter_1_shape, dtype=tf.float32, \
                minval = -1.0/cf1_var_factor * VAR_SCALE, \
                maxval =  1.0/cf1_var_factor * VAR_SCALE)
        conv_filter_1 = tf.Variable(conv_filter_1_init, \
                name='filter1', trainable=_train, dtype=tf.float32)

        fc_weights_1_shape = (conv1_nfilters*np.prod(self.nconvs_1), N_ACTIONS)
        w1_var_factor = np.prod(fc_weights_1_shape)
        fc_weights_1_init = tf.random_uniform(\
                fc_weights_1_shape, dtype=tf.float32,\
                minval = -1.0/w1_var_factor * VAR_SCALE,\
                maxval =  1.0/w1_var_factor * VAR_SCALE)

        fc_weights_1 = tf.Variable(fc_weights_1_init, \
                name='fc1', trainable=_train)

        self.input_layer = tf.placeholder(tf.float32, [None,                   \
                self.gridsz[XDIM], self.gridsz[YDIM], N_LAYERS], 'input');    \
        self.l1 = tf.nn.conv2d(\
                input = self.input_layer, \
                filter = conv_filter_1,\
                strides = conv_strides_1, \
                padding = 'VALID', \
                name = "conv1")
        self.l1_flat = tf.contrib.layers.flatten(self.l1)
        self.l2 = tf.matmul(self.l1_flat, fc_weights_1, name='fc1') # transpose?
        self.output_layer = tf.identity(self.l2, name='output')
        print self.l1.get_shape(), self.l1_flat.get_shape(), fc_weights_1.get_shape(), self.l2.get_shape()

        self.trainable_layers = [conv_filter_1, fc_weights_1]

        for l in self.trainable_layers:
            print "Layer",l.name,"shape:", l.get_shape()
        print "Layer",self.output_layer.name,"shape:", self.output_layer.get_shape()

        ''' Network operations (besides forward passing) '''
        self.pred_var = self.output_layer
        self.targ_var = tf.placeholder(tf.float32, [None, N_ACTIONS])
        self.loss_op = tf.reduce_sum(tf.square( self.pred_var - self.targ_var ))
        self.adam = tf.train.AdamOptimizer(LEARNING_RATE)
        self.updates = self.adam.minimize(self.loss_op)
        #self.updates = self.adam.compute_gradients(self.loss_op, \
        #        var_list = self.trainable_layers)
        #self.trainable_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #self.trainable_layers = tf.trainable_variables()
        '''
        self.updates = tf.contrib.layers.optimize_loss( 
                loss=self.loss_op, \
                global_step = tf.contrib.framework.get_global_step(), \
                learning_rate = LEARNING_RATE, \
                optimizer='Adam',\
                variables = self.trainable_layers)
        '''
        #self.train_op = tf.contrib.layers.optimize_loss

    def _softmax(self, X): return np.exp(X) / np.sum(np.exp(X))
 
    def chooseRandAction(self): 
        return np.array([0.25,0.25,0.25,0.25])
    def addObservedAction(self, action, reward):
        self.avgActions[action] += reward;

    def forward_pass(self, sess, States): # handles batch states
        inp = np.array([s.grid for s in States]) 
        Q_sa = sess.run(self.output_layer, \
                feed_dict = {self.input_layer : inp});
        if self.batch_off:
            return Q_sa[0]
        return Q_sa
    
    def update(self, sess, targ_list, pred_list):
        targs = np.array(targ_list, ndmin=2)
        preds = np.array(pred_list, ndmin=2)
        print sess.run(self.updates, feed_dict={self.targ_var: targs,\
                self.pred_var: preds})


    def rewardGoalReached(self): pass
    def makeLoss(self, prediction, target, reward): 
        return (reward + GAMMA*target - prediction)**2 # Currently: L2 loss.

def acceptableChoice(choice, valids):
    return bool(valids[choice])

# STUB: One iteration of a game:
env = environment_handler(filename="./state_files/10x10-nextto.txt")
# STUB: consider actions for one state:
RndNet = network1(env, 'NEURAL')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init);
    LOSSES = []
    for episode in range(training_episodes):
        reward_buffer = []
        losses_buffer = []

        # for # actions permitted:
        s0 = env.getStartState();

        Q0_sa_FP = RndNet.forward_pass(sess, [s0])
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
        # s1_est: the state to use for next Q loss. 
        # s1_valid: the state to _update the agent to next.
        ''' ^ question: Which state should be used for s'?
            Should this be _s0_, _argmax(valid next states)_ = s1_valid,
            or the actual estimated state, s1_est?
            Thoughts: s0 with gamma discount < 1 would simply discount the 
            Qval, s1_est could "run off the screen", and s1_valid would 
            potentially reward invalid action choices. 
            
            Currently, we impose this with s1_for_update = s0.
            See [Tag 53] below.
            '''

        reward_buffer.append(R_a0_hat)
        # un-optimal version: 2x redundancy.  Less problematic when s1_est=/=s1_valid.

        ''' Make Loss: All are zero except the chosen action, which 
        computes the loss between the previously chosen action's Qval 
        and the current state's reward + largest Qval of actions from
        this state. '''
        Q1_sa_FP = RndNet.forward_pass(sess, [s1_for_update]) # ?  [Tag 53] 
        '''
        losses = np.zeros((N_ACTIONS,))
        #  This currently uses a0_est, ie, np.argmax(Q0_sa_FP)
        losses[a0_est] = RndNet.makeLoss( \
                target = np.argmax(Q0_sa_FP), \
                prediction = np.argmax(Q1_sa_FP), \
                reward = R_a0_hat) 
        losses_buffer.append(losses)
        '''

        env.displayGameState(s0); 
        print ACTION_NAMES[a0_est], a0_est, R_a0_hat
        env.displayGameState(s1_for_update); 

        targ = np.copy(Q0_sa_FP); 
        targ[a0_est] = R_a0_hat + GAMMA * np.argmax(Q1_sa_FP)
        update_results = RndNet.update(sess, [targ], [Q0_sa_FP])
        print type(update_results)


        #s0 = s1_valid; # Update state for next iteration  STUB


    def test():
      for method in range(2): 
        test_samples = 30
        test_trials = 30
        R = np.empty(shape=(test_samples,))
        B = np.empty(shape=(test_trials,))
        for b in range(test_trials):
            for i in range(test_samples):
                s = env.getStartState();
                for j in range(max_num_actions):
                    if method==0: 
                        Q = RndNet.chooseStochStateIndependentAction()
                    elif method==1:  
                        Q_unnorm = RndNet.chooseRandAction()
                        Q = Q_unnorm / sum(Q_unnorm)
                    a = np.random.choice(ACTIONS, p = Q)
                    if not env.checkIfValidAction(s, a):
                        break;
                        a = np.argmax(np.exp(Q) * env.getActionValidities(s)) 
                    s = env.performAction(s, a)
                    if (env.isGoalReached(s)): 
                        break;
                if (env.isGoalReached(s)): 
                    R[i] = 1.0
                else:
                    R[i] = 0.0
            B[b] = np.mean(R)
        s = "Random" if method==1 else "Stoch"
        print "agent accuracy:", "{0:.2f}".format(np.mean(B)), \
                "correct with std:",  "{0:.2f}".format(np.var(B)**0.5), 'for method:',s
        '''print "agent accuracy:", "{0:.2f}".format(np.mean(B)), \
                "correct with std:",  "{0:.2f}".format(np.var(B)**0.5), "over",\
                test_samples,"samples per",test_trials,'trials for method:',s
                '''

    test()
    print "Sum successful actions:", RndNet.avgActions
    sys.exit()
            
    # Plotting:
    print "Plotting..."
    for a in range(N_ACTIONS):
        plt.plot(plt_arr[a,:], label=ACTION_NAMES[a])
    plt.legend()
    plt.xlabel("Training episode")
    plt.ylabel("Probability")
    plt.title("Learned proportional state-free stochastic actions")
    g = time.localtime()
    s = "./storage/plot-randomcontroller-"+str(g[1])+"_"+str(g[2])+"_"+\
            str(g[0])+"_"+str(g[3]).zfill(2)+str(g[4]).zfill(2)
    plt.savefig(s)
    plt.close()


    print "Done."
