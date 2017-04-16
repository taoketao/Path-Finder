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
gamma = 0.95;
REWARD = 1.0;
NO_REWARD = 0.0;

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

        window1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        window2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)

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
        self.output_layer = self.fc2
        # these outputs are q-value estimates.
        self.output_corr = tf.placeholder(tf.float32, [None, N_ACTIONS]);


    def chooseRandAction(self, s): # over actions A, as Q_hat(s,A)
        return np.random.rand(N_ACTIONS);
    def chooseNeuralAction(self):
        Q_sa = Reward
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
        s0 = env.getStartState();
        print "Initial state:"
        env.displayGameState(s0); print ''
        reward_buffer = [] # Apply reward signals after complete path performed
        history = [] # buffer of (state_i, action_i) pairs
        for nth_action in range(max_num_actions):

            # no advantage function for now...


            Q0_sa_FP = RndNet.chooseRandAction(s0)
            a0_est = np.argmax(Q0_sa_FP)
            s1_est = env.performAction(s0, a0_est)
            R_a0_hat = NO_REWARD;
            if env.checkIfValidAction(s0,a0_hat):
                a0_valid = a0_est;
                s1_valid = s1_est;
            else:
                valid_actions = env.getActionValidities(s0)
                a0_valid = np.argmax(np.exp(Q0_sa_FP) * valid_actions) 
                # nonneg + mask
                # ^ TODO: is there a better method than exp?
                # Also, is argmax or random exploration best?
                s1_valid = env.performAction(s0, a1_valid)
            if env.isGoalReached(s1_valid): 
                R_a0_hat = REWARD;

            reward_buffer.append(R_a0_hat)
            if nth_action > 0:
                reward_buffer[-2] += gamma * Q0_sa_FP;
                # ^ max_a' Q(s',a') for prev action: choice, not true, actions
            history.append( (s0, a0_hat) )

            s0 = s1_true;



            # Assume Q_sa_FP are >= 0, so that bit masking would work
            action_choice = np.argmax(Q_sa_FP) # epsilon here
            valid_actions = env.getActionValidities(cur_state)
            valid_choice = acceptableChoice(action_choice, valid_actions)
            print str(nth_action)+"th action requested:", ACTION_NAMES[action_choice],'(', action_choice,'). Valid?:', valid_choice
            if not valid_choice:
                action_taken = np.argmax(Q_sa_FP * valid_actions)
                print '\tChose instead:', ACTION_NAMES[action_taken], action_taken
            else:
                action_taken = action_choice
            next_state = env.performAction(cur_state, action_taken)
            print "new state:"
            env.displayGameState(next_state); print ''
            reward = float(env.isGoalReached(next_state))
            # max_a2_Q_sa2 == max_over_a' (Q(s',a'))
            max_a2_Q_sa2 = np.argmax( RndNet.chooseRandAction(next_state) \
                    * env.getActionValidities(next_state); )

            Q_sa_BP = Q_sa_FP;
            Q_sa_BP[action_choice] = reward + gamma * max_a2_Q_sa2;
            # ^ replace BP slot with the Bellman update:
            #   Q(s,a_hat) = R(s, a_hat) + gamma * max_a(Q(s',a))
RndNet.output_corr = Q_sa_BP;
            
            ;

            if reward==1.0:
                RndNet.rewardGoalReached()
                #goal_reached_bit = True
                print "Goal Reached!"
                break; # TODO: for batches, replace this with a completed
                       # sentinel for future iterations. Terminate early
                       # only if all states have finished early.

            cur_state = next_state
        print '----------------------------------------'


print "Done."
