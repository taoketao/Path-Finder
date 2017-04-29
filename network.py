'''                                                                            |
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from environment import *
from save_as_plot import save_as_plot

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

''' [Default] Hyper parameters '''
TRAINING_EPISODES = 300;  # ... like an epoch
MAX_NUM_ACTIONS = 15;
EPSILON = 1.1;
REWARD = 1;
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
def get_time_str(path, prefix=''):
    t = time.localtime()
    return os.path.join(path, prefix+str(t[1])+'_'+str(t[2])+'_'+str(t[0])\
            +'_'+str(t[3]).zfill(2)+str(t[4]).zfill(2))



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
            _optimizer_type='sgd', override=None, load_weights_path=None):
        ''' inputs: shape [batch, X, Y, 4-action-layers]. '''
        self.gridsz = _env.getGridSize()
        self.env = _env
        self.version = _version;
        self.batch_off = _batch_off; # for now.
        if not override==None and 'learning_rate' in override:
            self.learning_rate = override['learning_rate']
        else:
            self.learning_rate = LEARNING_RATE

        self.nconvs_1 = (self.gridsz[XDIM]-2, self.gridsz[YDIM]-2)
        self.nconvs_2 = (self.gridsz[XDIM]-4, self.gridsz[YDIM]-4)

        ''' Network architecture & forward feeding '''
        # setup helper fields for building the network
        #   layer weight shapes
        self.conv_filter_1_shape = (C1_SZ[0], C1_SZ[1], N_LAYERS, C1_NFILTERS)
        self.conv_filter_2_shape = (C2_SZ[0], C2_SZ[1], C1_NFILTERS, C2_NFILTERS)
        conv_strides_1 = (1,1,1,1)
        conv_strides_2 = (1,1,1,1)
        self.fc_weights_1_shape = (C2_NFILTERS*np.prod(self.nconvs_2), FC_1_SIZE)
        self.fc_weights_2_shape = (FC_1_SIZE, FC_2_SIZE)
        #   helper factors
        inp_var_factor = N_LAYERS * self.gridsz[XDIM] * self.gridsz[YDIM]
        self.cf1_var_factor = np.prod(self.conv_filter_1_shape)
        self.cf2_var_factor = np.prod(self.conv_filter_2_shape)
        self.w1_var_factor = np.prod(self.fc_weights_1_shape)
        self.w2_var_factor = np.prod(self.fc_weights_2_shape)

        ''' Initialization Values.  Random init if weights are not provided. 
            Relu activations after conv1, conv2, fc1 -> small positive inital 
            weights.  Last layer gets small mean-zero weights for tanh. '''
        
        inits = self._initialize_weights(load_weights_path)

        ''' trainable Tensorflow Variables '''
        self.conv_filter_1 = tf.Variable(inits['cv1'], \
                name='filter1', trainable=_train, dtype=tf.float32)
        self.conv_filter_2 = tf.Variable(inits['cv2'], \
                name='filter2', trainable=_train, dtype=tf.float32)
        self.fc_weights_1 = tf.Variable(inits['fc1'], \
                name='fc1', trainable=_train)
        self.fc_weights_2 = tf.Variable(inits['fc2'], \
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

        ''' Network operations (besides forward passing) '''
        self.pred_var = self.output_layer
        self.targ_var = tf.placeholder(tf.float32, [None, N_ACTIONS])
        self.loss_op = tf.reduce_sum(tf.square( self.pred_var - self.targ_var ))
        if _optimizer_type=='sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif _optimizer_type=='adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.updates = self.optimizer.minimize(self.loss_op)
 
        self.sess = None
    
    def _initialize_weights(self, initialization):
        weights = {}
        if initialization:
            loaded = np.load(initialization)
            weights['cv1'] = tf.constant(loaded['cv1'])
            weights['cv2'] = tf.constant(loaded['cv2'])
            weights['fc1'] = tf.constant(loaded['fc1'])
            weights['fc2'] = tf.constant(loaded['fc2'])
            #weights = dict(zip(('cv1','cv2','fc1','fc2'), (tf.constant(
            #    loaded[l]) for l in loaded)))
        else:
            weights['cv1'] = tf.random_uniform(\
                self.conv_filter_1_shape, dtype=tf.float32, \
                minval = 0.0, maxval = 1.0/self.cf1_var_factor**0.5 * VAR_SCALE)
            weights['cv2'] = tf.random_uniform(\
                self.conv_filter_2_shape, dtype=tf.float32, \
                minval = 0.0, maxval = 1.0/self.cf2_var_factor**0.5 * VAR_SCALE)
            weights['fc1'] = tf.random_uniform(\
                self.fc_weights_1_shape, dtype=tf.float32,\
                minval=0.0, maxval=1.0/self.w1_var_factor**0.5 * VAR_SCALE )
            weights['fc2']= tf.random_uniform(\
                self.fc_weights_2_shape, dtype=tf.float32,\
                minval=-0.5/self.w2_var_factor**0.5 * VAR_SCALE,\
                maxval= 0.5/self.w2_var_factor**0.5 * VAR_SCALE )
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

    def save_weights(self, dest, prefix=''):
        self._checkInit()
        l=[cv1, cv2, fc1, fc2]= self.sess.run(self.trainable_vars)
        np.savez(get_time_str(dest,prefix), cv1=cv1, cv2=cv2, fc1=fc1, fc2=fc2)

''' reinforcemente: class for managing training.  Note that the training 
    programs can take extra parameters; these are not to be confused with
    the hyperparameters, set at the top of this file.
'''
class reinforcement(object):

    ''' Constructor: which_game is the variable that commands entirely what 
        training pattern the network will undergo. 

        v1-single: The network learns to act in the game: [[---],[AG-],[---]].
                Mainly for testing and debugging.
        v1-oriented: The network learns to act in all variants of games which
                are 3x3 and only have component [AG] in any (of 6) locations.
        v1-fixedloc: the network learns to do act in a puzzles which have
                the agent at a fixed location ->> 4 states.
        v1-corner: This game has the network learn all variants of distance-1
                games in which the 3x3 grid is in the same location.
        v1-all:  not yet implemented.
        ...
    '''
    def __init__(self, which_game, game_shape=(10,10), override=None,
                 load_weights_path=None):
        # The enviroment instance
        self.env = environment_handler(game_shape)
        # The state generator instance
        self.sg = state_generator(game_shape)
        # state generator: utilize components 
        self.sg.ingest_component_prefabs(COMPONENTS_LOC)
       
        self.init_states = []

        self.D1_ocorner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env, oriented=True)
        self.D1_corner = self.sg.generate_all_states_fixedCenter(\
                'v1', self.env)
        if which_game=='v1-single':
            self.init_states.append(self.D1_corner[2])
            # Game 2 is what I was testing on earlier.  Agent is against middle
            # of left wall and must move directly right one block.
        if which_game=='v1-oriented':
            self.init_states += self.D1_ocorner
        if which_game=='v1-corner':
            self.init_states += self.D1_corner
        if which_game=='v1-fixedloc':
            #self.init_states.append(np.array(self.D1_corner)[[3,8,15,20]])
            self.init_states.append(self.D1_corner[3])
            self.init_states.append(self.D1_corner[8])
            self.init_states.append(self.D1_corner[20])
        self.which_game = which_game
        if which_game in \
                ['v1-single','v1-oriented','v1-corner','v1-all','v1-fixedloc']:
            self.which_paradigm = 'V1' 

        # If load_weights_path==None, then initialize weights fresh&random.
        self.Net = network1(self.env, 'NEURAL', override=override, 
                load_weights_path=load_weights_path)

        if not override==None and 'max_num_actions' in override:
            self.max_num_actions = override['max_num_actions']
        else:
            self.max_num_actions = MAX_NUM_ACTIONS
        if not override==None and 'nepochs' in override:
            self.training_episodes = override['nepochs']
        else:
            self.training_episodes = TRAINING_EPISODES
        '''
        if not override==None and 'save_eps' in override:
            self.save_eps=override['save_eps']
        else:
            self.save_eps=-1
        '''
    
    def getStartState(self, order='rand'):
        return random.choice(self.init_states)

    def stateLogic(self, s0, Q0, a0_est, s1_est):
        ''' (what should this be named?) This function interfaces the network's
            choices with the environment handler for necessary states.  '''
        #print " ^-> state logic.  Req action:", ACTION_NAMES[a0_est]

        valid_action_flag = self.env.checkIfValidAction(s0, a0_est)
        if valid_action_flag:
            goal_reached = self.env.isGoalReached(s1_est)
            return a0_est, s1_est, s1_est, \
                    REWARD if goal_reached else NO_REWARD, goal_reached
        a0_valid = np.argmax(np.exp(Q0) * self.env.getActionValidities(s0))
        return a0_valid, self.env.performAction(s0, a0_valid), s0, INVALID_REWARD, 0

    def Simple_Train(self, extra_parameters=None, save_weights_to=None,
                save_weight_freq=1000):
        ''' The simplest version of training the net to play the game.
            This training takes a single initial game state, plays it to 
            completion, and updates the model. Currently assumes the paradigm 
            is V1.  This uses static learning rates and epsilon exploration.
             - save_weights_to: path of directory to save weights to.
             - save_weight_freq: frequency of saving weights.
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
        self.Qlog=[]
        for episode in range(self.training_episodes):
            if save_weights_to and save_weight_freq and episode % save_weight_freq==0 :
                self.Net.save_weights(save_weights_to, prefix = 'epoch'+str(episode)+'--')
                print("Saving weights at episode #"+str(episode))
            elif episode %1000==0:
                print("Epoch #"+str(episode)+"/"+str(self.training_episodes))
            s0 = self.getStartState()
            #s0 = random.choice(self.getStartState())
            #print "init state: "; self.env.displayGameState(s0)
            S=''
            for nth_action in range(self.max_num_actions):
                #print 'episode',episode,'/',self.training_episodes,', action #',nth_action,':',
                if self.env.isGoalReached(s0): 
                    break

                Q0 = self.Net.getQVals([s0])
                self.Qlog.append((Q0,s0))
                #print "dumping Q0:", Q0
                randval = random.random()
                if randval < EPSILON: 
                    a0_est = random.choice(ACTIONS)
                else: 
                    a0_est = np.argmax(Q0)
                s1_est = self.env.performAction(s0, a0_est)
                a0_valid, s1_valid, s1_for_update, R, goal_reached = \
                        self.stateLogic(s0, Q0, a0_est, s1_est)
                #print ACTION_NAMES[a0_est], 'rand' if randval < EPSILON else 'det'

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
            #print '     - reward:',R,'-     '
            #print '------------------------------------'

            s0 = self.getStartState()
            #s0 = random.choice(self.getStartState())
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
        return train_losses, test_losses, train_nsteps, test_nsteps

    def save_weights(self, dest, prefix=''): self.Net.save_weights(dest, prefix)
    def displayQvals(self):pass




''' Testing / results scripts '''

def geomean(X): 
    # return np.prod(X)**(len(X)**-1) # numerically stable geometric mean
    return np.exp(np.sum(np.log(X))*(1.0/len(X)))

def test_script(version, dest):
    nsamples = 1 # 32
    MNA = []
    #mnas = [4,6,8,10,20]
    mnas = [4]
    #lrs = [0.01, 0.001]
    lrs = [0.001]
    training_eps = 15
    weight_save_eps = None
    saved_time_str = get_time_str(dest)
    gamesize = (5,5)
    for n_mna, mna in enumerate(mnas):
      for n_lr, lr in enumerate(lrs):
        print "\n **********  NEW TRIAL"
        print "\t max number of actions: "+str(mna), '\t'
        print "\t learning rate: "+str(lr)
        print "\t num training episodes: "+str(training_eps), 'and samples:', nsamples
        print "\t Test in series command: ",n_lr,len(lrs),',', n_mna,len(mnas)
        #training_eps_ = training_eps * (1 if lr>0.001 else 2)
        training_eps_ = training_eps
        avg_losses5 = np.empty((nsamples, training_eps_/5, 2))
        avg_losses = np.empty((nsamples, training_eps_, 2))
        avg_steps = np.empty((nsamples, training_eps_, 2))
        for ri in range(nsamples):
            ovr = {'max_num_actions': mna, 'learning_rate':lr, 'nepochs':training_eps_,
                    'save_eps':weight_save_eps};
            r = reinforcement(version, override = ovr, game_shape=gamesize)
            train, test, TrN, TeN = r.Simple_Train(\
                    save_weights_to=dest, save_weight_freq=weight_save_eps)
            Tr5 = np.sum(np.array(train).reshape((training_eps_/5, 5)), axis=1)
            Te5 = np.sum(np.array(test).reshape((training_eps_/5, 5)), axis=1)
            Tr = np.array(train)
            Te = np.array(test)
            avg_losses[ri,:,0] = Tr
            avg_losses[ri,:,1] = Te
            avg_steps [ri,:,:] = np.array([TrN, TeN]).T
            r.displayQvals()
            print '\nAgent',ri,'/'+str(nsamples)+'\t',"final train, test errors:", Tr[-1], \
                           Te[-1], ';\t avg train, test errs:', np.mean(Tr), np.mean(Te),
            if not ri%5: 
                print ''
        print "readout results: "
        print "\t avg tr, te losses:", list(np.mean(avg_losses[:,-1,:], axis=0))
        print "\t avg tr, te nsteps:", list(np.mean(avg_steps[:,-1,:], axis=0))
        arr = np.array((avg_losses, avg_steps))
        #  arr shape: (2: losses & steps, nsamples, max_num_actions, 2: Tr & Te)
        MNA.append(arr)
        np.save(os.path.join(dest,version) + \
                '-mna-' + str(mna) + \
                '-lr' + '%1.e' % lr + \
                '-' + str(gamesize).replace(', ', 'x') + \
                '-nsamples' + str(nsamples) + \
                '', arr)
    for i, mna in enumerate(mnas):
      for lr in lrs:
        avg_losses, avg_steps = MNA[i]
        fn = os.path.join(dest,version) + \
                '-mna-' + str(mna) + \
                '-lr' + '%1.e' % lr + \
                '-' + str(gamesize).replace(', ', 'x') + \
                '-nsamples' + str(nsamples) + \
                '.npy'
        save_as_plot(fn, str(lr), str(mna), str(nsamples) )
        print '-----------------------------------------\n'

#r = reinforcement('v1-fixedloc', override={'max_num_actions': 3, 'learning_rate':0.0001, 'nepochs':3001});
#r.Simple_Train(save_weights_to='./storage/4-28/', save_weight_freq=200)
#r.save_weights('./storage/4-28/')

#test_script('v1-fixedloc', './storage/4-29/')
test_script('v1-fixedloc', './storage/4-29/')
# test_script('v1-oriented', './storage/4-25-p2/')
#test_script('v1-corner', './storage/4-22-17-corner/')
#test_script('v1-oriented', './storage/4-22-17-oriented-gamesize/')

print "Done."
