'''                                                                           |
Morgan Bryant, April 2017
Test framework intended to run experiments. 
'''
import sys, time, os
from subprocess import call
from socket import gethostname
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import open

from save_as_plot import *
from reinforcement_batch import reinforcement_b


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
            self.dest = './storage/5-21/14/'
            if not os.path.exists(self.dest): os.makedirs(self.dest)
            self.nsamples = 1
            self.curseeds = list(range(self.seed,self.seed+self.nsamples))
            self.no_save = False
            self.fin_logfile = open(get_time_str(self.dest,'fin_logfile.txt'), \
                    'w+',encoding="utf-8")
            self.tot_logfile = open(get_time_str(self.dest,'tot_logfile.txt'), \
                    'w+',encoding="utf-8")
            self.run_exp('allo-ego')

        self.fin_logfile.close()
        self.tot_logfile.close()
        if not gethostname()=='PDP':
            call(["open", self.dest])
    def getseed(self): 
        self.seed += 1
        return self.seed



    ''' USER-FACING experiment method.  Please Manually Edit the options! '''
    def run_exp(self, centric, saveweights=False):
        nsamples = self.nsamples
        self.trial_counter=0

        '''------------------'''
        ''' Options to edit: '''
        '''------------------'''
        _training_epochs = [10000]
        mnas = [2]
        gameversions = [ 'v0-a_fixedloc','v2-a_fixedloc_leq','v2-a_fixedloc_eq' ]
        #gameversions = [ 'v0-a_fixedloc' ]
        loss_fns = [ 'square', 'huber0.1', 'huber1', 'huber10' ]
        curricula = [ None ]
        lrs = [3e-4]
        epsilons = ['decay_99']
        optimizers = [ ['adam',1e-7] ] 
        network_sizes = [\
                ('fc','fc',64,32),\
                ]
        data_modes = ['shuffled']
        smoothing = 25 # <- Adjust for plotting: higher=smoother
        '''--------------------------'''
        ''' end of recommended edits '''
        '''--------------------------'''

        self.tot_num_trials = len(_training_epochs)*len(mnas)*len(lrs)*\
              len(epsilons)*len(optimizers)*len(network_sizes)*len(data_modes)*\
              len(gameversions)*len(curricula)*len(loss_fns)
        saved_time_str = get_time_str(self.dest)
        self.MNA = []
        for mna in mnas:
          [[[[[[[[[ self.run_trial( epch, mna, lr, nsize, eps_expl, opmzr, None, \
                centric, nsamples, dm, gv, lossfn, curr, smoothing)\
                for epch in _training_epochs ]\
                #for mna in mnas ]\
                for lr in lrs ]\
                for nsize in network_sizes ]\
                for eps_expl in epsilons ]\
                for opmzr in optimizers ]\
                for dm in data_modes]\
                for lossfn in loss_fns]\
                for curr in curricula]\
                for gv in gameversions]
          if self.no_save: continue#return
          [[[[[[[[[ save_as_plot1(self.get_filesave_str(mna, lr, None, eps_expl,\
                    opmzr, epch, nsize, data_mode, centric, gv, lossfn, curr,\
                    0) + '-loss-graph.npy', \
                    str(lr), str(mna), str(nsamples), which='l', \
                    div=N_EPS_PER_EPOCH, smoothing=smoothing)
                for epch in _training_epochs ]\
                #for mna in mnas ]\
                for lr in lrs ]\
                for nsize in network_sizes ]\
                for eps_expl in epsilons ]\
                for opmzr in optimizers ]\
                for data_mode in data_modes]\
                for lossfn in loss_fns]\
                for curr in curricula]\
                for gv in gameversions]

        print('\n-----------------------------------------')
        print('-----------------------------------------\n')





    def run_trial(self, training_epochs, mna, lr, nsize, eps_expl, opmzr, gsz,\
            centric, nsamples, data_mode, gameversion, loss_fn, curr, smooth_factor=50):
        s=self.get_filesave_str(mna, lr, gsz, eps_expl, opmzr, training_epochs,\
                nsize, data_mode, centric, gameversion, loss_fn, curr, 0)

        if centric in ['allocentric', 'egocentric']:
            tr_successes, te_successes, states = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, centric, gameversion, loss_fn, curr, s)
            save_as_successes(s+'-successes', tr_successes, te_successes, \
                states, smooth_factor, centric)
            self.trial_counter+=1
            return;

        elif centric=='allo-ego':
            tr_successes_e, te_successes_e, st_e = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'egocentric', gameversion, \
                    loss_fn, curr, s)
            tr_successes_a, te_successes_a, st_a = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'allocentric',  gameversion, \
                    loss_fn, curr, s)
            self.trial_counter+=1
            assert(st_e==st_a)
            save_as_successes(s+'-successes', tr_successes_e, te_successes_e, \
                st_e, smooth_factor, ['ego','allo'],
                tr_successes_a, te_successes_a)
            return
        

    def run_single_train_sess(self, nsamples, mna, lr, training_epochs, \
            nsize, eps_expl, opmzr, gsz, data_mode, centric, gameversion, \
            loss_fn, curr, s=''):
        Tr_Successes = [];      Tr_losses = []; 
        Te_Successes = [];      Te_losses = [];  
        states = None
        for ri in range(nsamples):
            ovr = {'max_num_actions': mna, 'learning_rate':lr, \
                    'nepochs': training_epochs, 'netsize':nsize, \
                    'epsilon':eps_expl, 'loss_function':loss_fn, \
                    'gamesize':gsz, 'curriculum':curr, \
                    'optimizer_tup':opmzr, 'rotation':False };
            r = reinforcement_b(gameversion, centric, override=ovr, \
                    data_mode=data_mode, seed=self.curseeds[ri])

            print(("\n **********  NEW Network, trial number "+str(1+\
                self.trial_counter)+'/'+str(self.tot_num_trials)))
            print(("\t max number of actions: "+str(mna)))
            print(("\t learning rate: "+str(lr)))
            print(("\t num training epochs: "+str(training_epochs)))
            print(("\t samples: "+str(self.nsamples)))
            print(("\t frame: "+centric))
            if not curr==None: print(("\t data mode: "+str(data_mode)))
            print(("\t exploration epsilon: "+str(eps_expl)))
            print(("\t network shape: "+str(nsize)))
            if not gsz==None: print(("\t game input shape: "+str(gsz)))
            print(("\t loss: "+loss_fn))
            if not curr==None: print(("\t curriculum: "+curr))
            print(("\t game version: "+str(gameversion)))
            print(("\t optimizer: "+str(opmzr)))
            print("Running sample # "+str(ri+1)+'/'+str(nsamples)+': '+centric)
            results = r.run_session(params={ 'disp_avg_losses':20,\
                'buffer_updates':False, 'rotational':False, 'printing':False}) 
            Tr_losses.append(results.get('train', 'losses'))
            Te_losses.append(results.get('test' , 'losses'))

            Tr_Successes.append(results.get('train', 'successes'))
            test_results = results.get('test', 'successes')
            Te_Successes.append(test_results)
            if training_epochs > 30 and len(s)>0:
                s_ = s+' sample #'+str(ri)+' last 30 test accs: '+'\n'+\
                        str(test_results[-30:])+'\n'
                try:    self.fin_logfile.write(unicode(s_))
                except: self.fin_logfile.write(s_)
                s_ = s+' sample #'+str(ri)+' last 30 test accs: '+'\n'+\
                        str(test_results)+'\n'
                try:    self.tot_logfile.write(unicode(s_))
                except: self.tot_logfile.write(s_)
            if states==None: 
                states = results.get('states')
        
        np.save(s+'-loss-graph.npy', np.array([Tr_losses, Te_losses]))

        return  np.mean(np.array(Tr_Successes), axis=0), \
                np.mean(np.array(Te_Successes), axis=0), states



    def get_filesave_str(self, mna, lr, gsz, eps_expl, opmzr, \
            nepochs, nsize, data_mode, centric, gameversion, loss_fn, curr, \
            seed=None): 
        if type(eps_expl)==float:
            eps = '%2.e' % eps_expl 
        else: eps = eps_expl.replace('/','_')
        if type(opmzr)==list and len(opmzr)==2:
            opmzr = '_'.join([opmzr[0], '%1.e' % opmzr[1]])
        s= os.path.join(self.dest, gameversion) + \
                '-mna' + str(mna) + \
                '-lr' + '%1.e' % lr + \
                '-nepochs'+str(nepochs) +\
                ('-' + str(gsz).replace(', ', 'x') if not gsz==None else '') + \
                '-nsamps_' + str(self.nsamples) + \
                '-eps_' + str(eps) + \
                '-opt_'+str(opmzr) + \
                '-net_'+'_'.join([str(i) for i in nsize]) + \
                '-data_'+data_mode+'-frame_'+centric +\
                '-loss_'+loss_fn +\
                ('-curr'+curr if not curr==None else '')+\
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

if __name__=='__main__':
    if len(sys.argv)==1:mode='ego-allo-test'
    else: mode = sys.argv[1]
    experiment(mode)

print("Done.")
