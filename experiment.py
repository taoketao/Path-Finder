'''                                                                           |
Morgan Bryant, April 2017
Test framework intended to run experiments. 
'''
import sys, time, os
from subprocess import call
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from save_as_plot import *
from reinforcement import reinforcement


''' [Helper] Constants '''
N_ACTIONS = 4
XDIM=0; YDIM=0
agentLayer = 0
goalLayer = 1
immobileLayer = 2
mobileLayer = 3
N_EPS_PER_EPOCH = 4 # upper bound on number of initial start states there are

ALL = -1

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

''' utility functions '''
def get_time_str(path, prefix=''):
    t = time.localtime()
    return os.path.join(path, prefix+str(t[1])+'_'+str(t[2])+'_'+str(t[0])\
            +'_'+str(t[3]).zfill(2)+str(t[4]).zfill(2))


''' Testing / results scripts '''

class experiment(object):
    ''' USER-FACING test harness for experiments. 
        Intention is to read experiment files that spell out the parameters,
        but also a capacity '''
    def __init__(self, mode):
        self.iterator = 0;
        self.seed=42
        if mode=='ego-allo-test':
            self.version='v0-a_fixedloc'
            self.nsamples = 8
            self.curseeds = list(range(self.seed,self.seed+self.nsamples))
            self.no_save = False
            self.dest = './storage/5-17/cv-fc-dev/'
            self.logfile = open(os.path.join(self.dest+'logfile.txt'), 'w')
            self.run_exp('allo-ego')
#            self.run_exp('egocentric')
#            self.nsamples = 1
#            self.run_exp('egocentric')
        self.logfile.close()
        call(["open", self.dest])
    def getseed(self): 
        self.seed += 1
        tf.set_random_seed(self.seed)
        return self.seed



    ''' USER-FACING experiment method.  Please Manually Edit the options! '''
    def run_exp(self, centric, saveweights=False):
        nsamples = self.nsamples
        self.trial_counter=0

        '''------------------'''
        ''' Options to edit: '''
        '''------------------'''
        _training_epochs = [4000]
        mnas = [1]
        lrs = [1e-3]
        epsilons = [0.7]#, 0.3, 'lindecay', '1/nx5', '1/nx15']
        #optimizers = [ ['sgd']]+ [['adam',i] for i in [1e-3,1e-4,1e-5,1e-6]] 
        optimizers = [ ['adam', 1e-6] ] 
        network_sizes = [\
                #('fc','fc','fc',32,32,32),\
                ('cv','cv','fc',36,36,36),\
                ('fc','fc','fc',36,36,36),\
                #('fc','fc','fc',40,40,40),\
                ]
        data_modes = ['shuffled']#, 'ordered']
        gamesizes = [(5,5)]
        smoothing = 25 # <- Adjust for plotting: higher=smoother
        '''--------------------------'''
        ''' end of recommended edits '''
        '''--------------------------'''

        self.tot_num_trials = len(_training_epochs)*len(mnas)*len(lrs)*\
              len(epsilons)*len(optimizers)*len(network_sizes)*len(data_modes)
        saved_time_str = get_time_str(self.dest)
        self.MNA = []
        for mna in mnas:
          [[[[[[[ self.run_trial( epch, mna, lr, nsize, eps_expl, opmzr, gsz, \
                centric, nsamples, dm, smoothing)\
                for epch in _training_epochs ]\
                #for mna in mnas ]\
                for lr in lrs ]\
                for nsize in network_sizes ]\
                for eps_expl in epsilons ]\
                for opmzr in optimizers ]\
                for gsz in gamesizes]\
                for dm in data_modes]
          if self.no_save: continue#return
          [[[[[[[ save_as_plot1(self.get_filesave_str(mna, lr, gsz, eps_expl,\
                    opmzr, epch, nsize, data_mode, centric, 0) +\
                    '-loss-graph.npy', \
                    str(lr), str(mna), str(nsamples), which='l', \
                    div=N_EPS_PER_EPOCH, smoothing=smoothing)
                for epch in _training_epochs ]\
                #for mna in mnas ]\
                for lr in lrs ]\
                for nsize in network_sizes ]\
                for eps_expl in epsilons ]\
                for opmzr in optimizers ]\
                for gsz in gamesizes]\
                for data_mode in data_modes]

        print '\n-----------------------------------------'
        print '-----------------------------------------\n'





    def run_trial(self, training_epochs, mna, lr, nsize, eps_expl, opmzr, gsz,\
            centric, nsamples, data_mode, smooth_factor=50):
        print("\n **********  NEW TRIAL, number "+str(1+\
            self.trial_counter)+'/'+str(self.tot_num_trials))
        print("\t max number of actions: "+str(mna))
        print("\t learning rate: "+str(lr))
        print("\t num training epochs: "+str(training_epochs))
        print("\t samplesl "+str(self.nsamples))
        print("\t frame: "+centric)
        print("\t data mode: "+str(data_mode))
        print("\t exploration epsilon: "+str(eps_expl))
        print("\t network shape: "+str(nsize))
        print("\t game input shape: "+str(gsz))
        print("\t optimizer: "+str(opmzr))
        self.trial_counter+=1
        s=self.get_filesave_str(mna, lr, gsz, eps_expl, opmzr, \
                training_epochs, nsize, data_mode, centric, 0)

        if centric in ['allocentric', 'egocentric']:
            tr_successes, te_successes, states = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, centric,  s)
            save_as_successes(s+'-successes', tr_successes, te_successes, \
                states, smooth_factor, centric)
            return;

        elif centric=='allo-ego':
            tr_successes_e, te_successes_e, st_e = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'egocentric',  s)
            tr_successes_a, te_successes_a, st_a = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'allocentric',  s)
            assert(st_e==st_a)
            save_as_successes(s+'-successes', tr_successes_e, te_successes_e, \
                st_e, smooth_factor, ['ego','allo'],
                tr_successes_a, te_successes_a)
            return
        

    def run_single_train_sess(self, nsamples, mna, lr, training_epochs, \
            nsize, eps_expl, opmzr, gsz, data_mode, centric, s=''):
        Tr_Successes = [];      Tr_losses = []; 
        Te_Successes = [];      Te_losses = [];  
        states = None
        for ri in range(nsamples):
            ovr = {'max_num_actions': mna, 'learning_rate':lr, \
                    'nepochs': training_epochs, 'netsize':nsize, \
                    'epsilon':eps_expl, 'loss_function':'square', \
                    'optimizer_tup':opmzr, 'rotation':False };
            r = reinforcement(self.version, centric, override=ovr, \
                    game_shape=gsz, data_mode=data_mode, \
                    seed=self.curseeds[ri])
            print "Running sample # "+str(ri+1)+'/'+str(nsamples)
            results = r.run_session(params={ 'disp_avg_losses':20,\
                'buffer_updates':False, 'rotational':False, 'printing':False}) 
            Tr_losses.append(results.get('train', 'losses'))
            Te_losses.append(results.get('test' , 'losses'))

            Tr_Successes.append(results.get('train', 'successes'))
            test_results = results.get('test', 'successes')
            Te_Successes.append(test_results)
            if training_epochs > 30 and len(s)>0:
                self.logfile.write(s+' sample #'+str(ri)+\
                    ' last 30 test accs: '+'\n'+str(test_results[-30:])+'\n')
            if states==None: 
                states = results.get('states')
        
        np.save(s+'-loss-graph.npy', np.array([Tr_losses, Te_losses]))

        return  np.mean(np.array(Tr_Successes), axis=0), \
                np.mean(np.array(Te_Successes), axis=0), states



    def __deprecated__(self):
        '''print "readout results: "
    print "\t avg tr, te losses:", list(np.mean(avg_losses[:,-1,:], axis=0))
    print "\t avg tr, te nsteps:", list(np.mean(avg_steps[:,-1,:], axis=0))
    print "\t avg tr, te reward:", list(np.mean(avg_reward[:,-1,:], axis=0))'''
        arr = np.array((avg_losses, avg_steps, avg_reward))
        # arr shape: (2: losses & steps, nsamples, max_num_actions, 2: Tr & Te)
        self.MNA.append(arr)

        # Plot Q's actions & rewards, per epoch:
        ActMat3=np.zeros((3,training_epochs,N_ACTIONS,2)) 
        # ^  Plot reward,loss,Q per epoch per action in train, test.
        Q_sorted = []
        for ep in range(training_epochs): 
            Q_sorted.append([])
        for q in Q:
            Q_sorted[q['epoch']].append(q)
        for q_ in Q_sorted:
            for q in q_:
                if q['mode']=='train': tr_te = 0
                else: tr_te = 1
                ActMat3[0,q['epoch'],q['action'],tr_te] += q['reward']
                ActMat3[1,q['epoch'],q['action'],tr_te] += q['loss']
                ActMat3[2,q['epoch'],q['action'],tr_te] += q['Q'][q['action']]
        for e in range(1,training_epochs): # Smoothing
          for tt in range(2):
            for a in range(4):
              for lqr in [1,2]:
                continue
                if ActMat3[lqr,e,a,tt]==0.0:
                  ActMat3[lqr,e,a,tt] = ActMat3[lqr,e-1,a,tt]; # ...
        self.iterator += 1
        
        

        # Plot R, Q, ...? by state:
        if self.version=='v0-a_fixedloc' and gsz==(5,5):
            if mna==1: 
                num_states = (16 if centric=='allocentric' else 12)
            if mna==2:
                num_states = (44 if centric=='allocentric' else 21)
        elif self.version=='v0-single' and gsz==(5,5):
            if mna==1:  num_states = 4
            if mna==2:  num_states = 10
        else: raise Exception("version not implemented: num states unk")
        StateMat = np.zeros((4,training_epochs,num_states,2)) 
        # ^ Plot [action,] per epoch per state in train, test.

        # info: q_ is dict of epoch, action, reward, Q_data, 
        #   mode[train/test], loss, state.
        # Q_data has format: (a0_est, last_loss, R, Q0, updated_Q0, 
        # mode['train'/'test'])
        states_found = {} # To be sorted later;  format: (agent_x, 
        # agent_y, goal_x, goal_y) -> index in state_mat
        for q_ in Q_sorted:
            for q in q_:
                st = q['state']
                state_id = tuple([pos[0] for pos in np.nonzero(\
                        q['state'].grid[:,:,agentLayer])] + [pos[0] for \
                        pos in np.nonzero(q['state'].grid[:,:,goalLayer])])
                if not state_id in states_found: 
                    states_found[state_id] = len(states_found)
                state_index = states_found[state_id]

                StateMat[q['action'], q['epoch'], state_index, \
                        (0 if q['mode']=='train' else 1) ] = 1;
                # Overwriting: happens only if in the same *trial* the agent 
                # backtracks in mna >= 3 and takes a different action the 
                # second time: ie, very infrequently since Q udpate would 
                # have to happen
        if self.no_save: return

        #'-nsamples' + str(nsamples) + \
        s=self.get_filesave_str(mna, lr, gsz, eps_expl, opmzr, \
                training_epochs, nsize, data_mode, centric, 0)
            
        
        save_as_state_actions1(states_found, StateMat, gsz, \
                        s+'-per_state_actions')

        #self.iterator+=1
        np.save(s+'-loss-graph.npy', arr)





    def get_filesave_str(self, mna, lr, gsz, eps_expl, opmzr, \
            nepochs, nsize, data_mode, centric, seed=None): 
        if type(eps_expl)==float:
            eps = '%2.e' % eps_expl 
        else: eps = eps_expl.replace('/','_')
        if type(opmzr)==list and len(opmzr)==2:
            opmzr = '_'.join([opmzr[0], '%1.e' % opmzr[1]])
        s= os.path.join(self.dest, self.version) + \
                '-mna' + str(mna) + \
                '-lr' + '%1.e' % lr + \
                '-nepochs'+str(nepochs) +\
                '-' + str(gsz).replace(', ', 'x') + \
                '-nsamps_' + str(self.nsamples) + \
                '-eps_' + str(eps) + \
                '-opt_'+str(opmzr) + \
                '-net_'+'_'.join([str(i) for i in nsize]) + \
                '-data_'+data_mode+'-frame_'+centric +\
                ('-seed_'+str(seed) if not seed==None else '')
        return s


#r = reinforcement('v1-fixedloc', override={'max_num_actions': 3, \
#'learning_rate':0.0001, 'nepochs':3001});
#r.Simple_Train(save_weights_to='./storage/4-28/', save_weight_freq=200)
#r.save_weights('./storage/4-28/')

#test_script('v1-single', './storage/4-29/')
#test_script('v1-micro', './storage/5-02/', no_save=False)
#test_script('v1-single', './storage/5-02/', no_save=False)

#test_script('v0-single', './storage/5-03/', no_save=False)
#test_script('v1-micro_fixedloc', './storage/5-02/', no_save=False)
# test_script('v1-oriented', './storage/4-25-p2/')
#test_script('v1-corner', './storage/4-22-17-corner/')
#test_script('v1-oriented', './storage/4-22-17-oriented-gamesize/')

experiment(mode='ego-allo-test')

print "Done."
