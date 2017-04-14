'''                                                                            |
Morgan Bryant, April 2017
Simplest benchmark controller for pathfinder.
'''
from environment import *
import tensorflow as tf
import numpy as np

''' Hyper parameters '''
''' Setting constants '''
N_ACTIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
XDIM = 0; YDIM = 1

conv1_size = 64; # number of filters in conv layer #1
conv2_size = 64; # number of filters in conv layer #2
fc_1_size = 100; # number of outputs in dense layer #1
fc_2_size = 4;   # number of outputs in dense layer #2

def network1(object):
    def __init__(self, gridsz):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.window1 = (gridsz[XDIM]-2, gridsz[YDIM]-2)
        self.window2 = (gridsz[XDIM]-4, gridsz[YDIM]-4)

        #self.W1 = tf.Variable(tf.random_normal([conv2_size, fc_1_size])
        #self.W2 = tf.Variable(tf.random_normal([fc_1_size, fc_2_size])

        self.input_layer = tf.placeholder(tf.float32,               \
                [None, gridsz[XDIM], gridsz[YDIM], N_ACTIONS]);
        self.l1 = tf.layers.conv2d( name = "layer1",                \
                inputs = input_layer, kernel_size = window1,        \
                filters = conv1_size, activation=tf.nn.relu)
        self.l2 = tf.layers.conv2d( name = "layer2",                \
                inputs = l1, kernel_size=window2,                   \
                filters = 64, activation=tf.nn.relu)
        self.fc1 = tf.layers.dense( name = "layer3",                \
                inputs = l2, units = 100)
        self.fc2 = tf.layers.dense( name = "layer4",                \
                inputs = fc1, units = 4);
        self.out = self.fc2
        # these outputs are q-value estimates.
        self.predict = tf.argmax(self.out)

def chooseRandAction(s):
    return np.random.choice(ACTIONS);



# STUB: One iteration of a game:
env = environment_handler(filename="./state_files/3x3-diag.txt")
# STUB: consider actions for one state:
s0 = env.getStartState();
gridsize = env.getGridSize(); print gridsize
R = network1(gridsize)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    pass



valid_actions = env.getActionValidities(s0);



print "Done."
