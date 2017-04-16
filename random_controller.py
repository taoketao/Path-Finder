'''                                                                            |
Morgan Bryant, April 2017
Simplest benchmark controller for pathfinder.
'''
from environment import *
import tensorflow as tf
import numpy as np

''' [Helper] Constants '''
N_ACTIONS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
NULL_ACTION = -1;
XDIM = 0; YDIM = 1
LAYER_NAMES = { 0:"Agent", 1:"Goal", 2:"Immobiles", 3:"Mobiles" }
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }

''' Hyper parameters '''
max_num_actions = 8;#20; # 50?
training_episodes = 1;  # ... like an epoch

''' Architecture parameters '''
conv1_size = 64; # number of filters in conv layer #1
conv2_size = 64; # number of filters in conv layer #2
fc_1_size = 100; # number of outputs in dense layer #1
fc_2_size = 4;   # number of outputs in dense layer #2

class network1(object):
    def __init__(self, _env):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.window1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        self.window2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)
        '''
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
        '''
    def update(self, cur_state, action_req, reward):
        if reward==0: return; # an obvious optimization
        # Q(s,a) = 

    def chooseRandAction(self, s):
        return np.random.rand(N_ACTIONS);
    def chooseNeuralAction(self, s):
        pass # stub
    def rewardGoalReached(self): pass

def acceptableChoice(choice, valids):
    return bool(valids[choice])

# STUB: One iteration of a game:
env = environment_handler(filename="./state_files/3x3-diag.txt")
# STUB: consider actions for one state:
RndNet = network1(env)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init);
    for episode in range(training_episodes):
        cur_state = env.getStartState();
        print "Initial state:"
        env.displayGameState(cur_state); print ''
        #goal_reached_bit = False
        for nth_action in range(max_num_actions):

            # no advantage function for now...


            Q_sa_scores = RndNet.chooseRandAction(cur_state)
            # Assume Q_sa_scores are >= 0, so that bit masking would work
            action_choice = np.argmax(Q_sa_scores) # epsilon here
            valid_actions = env.getActionValidities(cur_state)
            valid_choice = acceptableChoice(action_choice, valid_actions)
            print str(nth_action)+"th action requested:", ACTION_NAMES[action_choice],'(', action_choice,'). Valid?:', valid_choice
            if not valid_choice:
                action_taken = np.argmax(Q_sa_scores * valid_actions)
                print '\tChose instead:', ACTION_NAMES[action_taken], action_taken
            else:
                action_taken = action_choice
            next_state = env.performAction(cur_state, action_taken)
            print "new state:"
            env.displayGameState(next_state); print ''
            reward = float(env.isGoalReached(next_state))
            RndNet.update(cur_state, action_choice, reward)

            if reward==1.0:
                RndNet.rewardGoalReached()
                #goal_reached_bit = True
                print "Goal Reached!"
                break;

            cur_state = next_state
        print '----------------------------------------'


print "Done."
