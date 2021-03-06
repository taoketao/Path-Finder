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
from reinforcement_batch import reinforcement_b, CurriculumGenerator 


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
    def __init__(self, mode, Seed=None):
        self.iterator = 0;
        if not Seed==None:
            self.seed = Seed
        else:
            self.seed = 0
        if mode=='ego-allo-test':
            self.dest = './storage/6-02/ego-allo-2/'
            if not os.path.exists(self.dest): os.makedirs(self.dest)
            self.nsamples = 20
            self.curseeds = list(range(self.seed,self.seed+2*self.nsamples))
            self.no_save = False
            self.fin_logfile = open(get_time_str(self.dest,'fin_logfile')\
                    +'.txt', 'w+',encoding="utf-8")
            self.run_exp('allo-ego')

        self.fin_logfile.close()
        if not 'pdp' in gethostname().lower():
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
#        _training_epochs = [5000]
#        mnas = [ 2, '2_anneal_linear_b300_e800', '2_anneal_linear_b100_e1000' ] 
        #mnas = [ '2_anneal_linear_b0_e750', '2_anneal_linear_b1000_e1050' ] 
#        mnas = [ '2_anneal_linear_b1000_e2000', '2_anneal_linear_b500_e1000' ] 
        #mnas = [ '2_anneal_linear_b500_e1500' ] 
        _training_epochs = [2000]
#        mnas = [ 2, '2_anneal_linear_b300_e800', '2_anneal_linear_b100_e1000' ] 
        #mnas = [ '2_anneal_linear_b0_e750', '2_anneal_linear_b1000_e1050' ] 
#        mnas = [ '2_anneal_linear_b1000_e2000', '2_anneal_linear_b500_e1000' ] 
        mnas = [ 2 ] 
        gameversions = [ 'v2-a_fixedloc_leq' ]
        loss_fns = [ 'huber3e-5' ]

        ''' Curriculum options: <all>, <any1step, u r diag>, <r or u only>, 
            <r, u, ru-diag only>, <uu ur>, <poles>, <all diag>, <1step>, <1step split> '''
        curricula = []
#        for task_2groups in ['any1step, u r diag', 'r, u, ru-diag only', 'poles']:
#        if len(_training_epochs)>1: raise Exception()
#        timings = []
#        for i in range(_training_epochs[0]//1000-1):
#            for j in range(i+1,_training_epochs[0]//1000):
#                timings.append({'b1':i*1000, 'e1':j*1000})
#        curricula += CurriculumGenerator( scheme='cross parameters', inp={\
#                    #'schedule kind': 'linear anneal', 'which ids':'r, u, ru-diag only', \
#                    #'schedule kind': 'linear anneal', 'which ids':'any1step, u r diag', \
#                    'schedule kind': 'no anneal', \
#                    'which ids':['all diag', 'any1step, u r diag', 'r, u, ru-diag only'], \
#                    'schedule strengths': ['20-80 flat group 1'], \
#                    'schedule timings': [{'t1':750}, {'t1':1000}, {'t1':1250}] })
#        curricula += CurriculumGenerator( scheme='cross parameters', inp={\
#                    'schedule kind':        'linear anneal', \
#                    'which ids':            [ 'any1step, u r diag', 'r, u, ru-diag only'], \
#                    'schedule strengths':   [ '20-80 flat group 1' ], \
#                    'schedule timings':     [ {'b1':1000, 'e1':2000 }, {'b1':0, 'e1':300}, {'b1':0, 'e1': 6000}]})
#        curricula += CurriculumGenerator( scheme='cross parameters', inp={\
#                    'schedule kind':        'no anneal', \
#                    'which ids':            [ 'any1step, u r diag', 'r, u, ru-diag only'], \
#                    'schedule strengths':   [ '20-80 flat group 1' ], \
#                    'schedule timings':     [ {'t1':50}, {'t1':100}, {'t1':400}  ]})
#                    #] } )#{'b1':0,'e1':5000},{'b1':1000, 'e1':5000},{'b1':4000, 'e1':8000},\
##        curricula += CurriculumGenerator( scheme='cross parameters', inp={\
#                    'schedule kind': 'no anneal', 'which ids':'r, u, ru-diag only', \
#                    'schedule strengths': ['egalitarian', '20-80 flat group 1'], \
#                    'schedule timings': [ {'t1':500},  {'t1':1000}, {'t1':1500}, \
#                                          {'t1':2000}, {'t1':3000}  ] } )
        curricula += CurriculumGenerator( scheme='cross parameters', inp={\
                    'schedule kind': 'no anneal', 'which ids':'r, u, ru-diag only', \
#                    'schedule kind': 'no anneal', 'which ids':'1step', \
                    'schedule strengths': ['20-80 flat group 1'],\
                    'schedule timings': [{'t1': 250},{'t1':1}]} )
#        curricula += CurriculumGenerator( inp={\
#                    'schedule kind': 'uniform', 'which ids':'1step split'} )
#
#        curricula = CurriculumGenerator( { 'schedule kind':'uniform', 'which ids': 'all diag' } )
#        curricula += CurriculumGenerator( { 'schedule kind':'uniform', 'which ids': 'r, u, ru-diag only' } )
#            CurriculumGenerator( { 'schedule kind':'uniform', 'which ids': '1step split' } )]
#            CurriculumGenerator( { 'schedule kind':'uniform', 'which ids': \
#                    'any1step, u r diag', 'schedule strengths':'egalitarian',
#                    'schedule timings':{'b1':0, 'e1':800}} )]
#                    'schedule timings': [{'b1':0, 'e1':250}, {'b1':0, 'e1':500},\
#                        {'b1':0, 'e1':1000}, {'b1':250, 'e1':500}, \
#                        {'b1':250, 'e1':1000}, {'b1':250, 'e1':1500}, \
#                        {'b1':500, 'e1':1000}, {'b1':500, 'e1':1500}] } )
#        'schedule timings': [{'b1':0, 'e1':250}, {'b1':0, 'e1':500},\
#            {'b1':0, 'e1':1000}, {'b1':250, 'e1':500}, \
#            {'b1':250, 'e1':1000}, {'b1':250, 'e1':1500}, \
#            {'b1':500, 'e1':1000}, {'b1':500, 'e1':1500}] } )

        #lrs = [ 4e-4 ]
        lrs = [ 1e-4 ]
        epsilons = [ 8e-1 ]#, 4e-1, 'decay_995' ]
        optimizers = [ ['adam',1e-6] ] 
        network_sizes = [\
                ('fc',64),\
#                ('fc',72),\
#                ('fc','fc',128,128),\
                ]
        data_modes = ['shuffled']
#        smoothing = 100 # <- Adjust for plotting: higher=smoother
#        self.test_frequency = 10
        smoothing = 25 # <- Adjust for plotting: higher=smoother
        smoothing = 10 # <- Adjust for plotting: higher=smoother
        self.test_frequency = 10

        '''--------------------------'''
        ''' end of recommended edits '''
        '''--------------------------'''

        self.curr_counter = 0; # update for curricula accesses
        self.curr_map = {}
        fn = get_time_str(self.dest,'curriculum_key_'+str(centric)+'_')+'.txt'
        with open(fn,'w+',encoding="utf-8") as curr_out:
            print("Curriculum information stored at "+fn)
            for ci, cr in enumerate(curricula): 
                s = str(ci)+'\t:\t'+cr.toString().replace('\n',\
                        '\n\t\t') + '\n'
                try:        curr_out.write(s)
                except:     curr_out.write(unicode(s))
                self.curr_map[cr] = ci

        self.tot_num_trials = len(_training_epochs)*len(mnas)*len(lrs)*\
              len(epsilons)*len(optimizers)*len(network_sizes)*len(data_modes)*\
              len(gameversions)*len(curricula)*len(loss_fns)
        saved_time_str = get_time_str(self.dest)
        self.MNA = []
        [[[[[[[[[[ self.run_trial( epch, mna, lr, nsize, eps_expl, opmzr, None, \
                centric, nsamples, dm, gv, lossfn, curr, smoothing)\
                for epch in _training_epochs ]\
                for mna in mnas ]\
                for lr in lrs ]\
                for nsize in network_sizes ]\
                for eps_expl in epsilons ]\
                for opmzr in optimizers ]\
                for dm in data_modes]\
                for lossfn in loss_fns]\
                for curr in curricula]\
                for gv in gameversions]
        if self.no_save: return
        [[[[[[[[[[ save_as_plot1(self.get_filesave_str(mna, lr, None, eps_expl,\
                    opmzr, epch, nsize, data_mode, centric, gv, lossfn, curr,\
                    self.seed).replace('_curr','\ncurr') + '-loss-graph.npy', \
                    str(lr), str(mna), str(nsamples), which='l',\
                    div=N_EPS_PER_EPOCH, smoothing=smoothing)
                for epch in _training_epochs ]\
                for mna in mnas ]\
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
                nsize, data_mode, centric, gameversion, loss_fn, curr, self.seed)

#        states = [ (3,3,1,3), (3,3,2,2), (3,3,3,1), (3,3,4,2), (3,3,5,3),\
#                       (3,3,4,4), (3,3,3,5), (3,3,2,4), (3,3,4,3), (3,3,2,3), 
#                       (3,3,3,2), (3,3,3,4) ]
#       Lex: [ dd dl dr ll lu rr ru uu _d _l _r _u ]
        states = [ (3,3,5,3), (3,3,4,2), (3,3,4,4), (3,3,3,1), (3,3,2,2),\
                   (3,3,3,5), (3,3,2,4), (3,3,1,3), (3,3,4,3), (3,3,3,2),\
                   (3,3,3,4), (3,3,2,3)]

        if centric in ['allocentric', 'egocentric']:
            tr_successes, te_successes, sm, states2 = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, centric, gameversion, loss_fn, curr, s)
            save_as_successes(s+'-successes', tr_successes, te_successes, \
                    smooth_factor, centric, curr=curr, statemap=sm, \
                    tf=self.test_frequency)
            self.trial_counter+=1
            return;

        elif centric=='allo-ego':
            tr_successes_e, te_successes_e, sm, states2 = self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'egocentric', gameversion, \
                    loss_fn, curr, s)
            tr_successes_a, te_successes_a, _,_= self.run_single_train_sess(\
                    self.nsamples, mna, lr, training_epochs, nsize, eps_expl, \
                    opmzr, gsz, data_mode, 'allocentric',  gameversion, \
                    loss_fn, curr, s)
            self.trial_counter+=1

            save_as_successes(s+'-successes', tr_successes_e, te_successes_e, \
                states, smooth_factor, ['ego','allo'], tr_successes_a, \
                te_successes_a, curr=curr, statemap=sm, tf=self.test_frequency,\
                dest=self.dest)
            return
        

    def run_single_train_sess(self, nsamples, mna, lr, training_epochs, \
            nsize, eps_expl, opmzr, gsz, data_mode, centric, gameversion, \
            loss_fn, curr, s=''):
        Tr_Successes = [];      Tr_losses = []; 
        Te_Successes = [];      Te_losses = [];  
        statemap = None;        StateMaps = [];
        for ri in range(nsamples):
            ovr = {'max_num_actions': mna, 'learning_rate':lr, \
                    'nepochs': training_epochs, 'netsize':nsize, \
                    'epsilon':eps_expl, 'loss_function':loss_fn, \
                    'gamesize':gsz, 'curriculum':curr, \
                    'optimizer_tup':opmzr, 'rotation':False };
            r = reinforcement_b(gameversion, centric, override=ovr, \
                    data_mode=data_mode, seed=self.curseeds[ri], dest=self.dest)

            print(("\n **********  NEW Network, trial number "+str(1+\
                self.trial_counter)+'/'+str(self.tot_num_trials)))
            if not curr==None: print(("\t max number of actions: "+str(mna)))
            print(("\t learning rate: "+str(lr)))
            print(("\t num training epochs: "+str(training_epochs)))
            print(("\t samples: "+str(self.nsamples)))
            print(("\t frame: "+centric))
            if not curr==None: print(("\t data mode: "+str(data_mode)))
            print(("\t exploration epsilon: "+str(eps_expl)))
            print(("\t network shape: "+str(nsize)))
            if not gsz==None: print(("\t game input shape: "+str(gsz)))
            print(("\t loss: "+loss_fn))
            print(("\t game version: "+str(gameversion)))
            print(("\t optimizer: "+str(opmzr)))
            print(("\t seed: "+str(self.curseeds[ri])))
            if not curr==None: print(("\t curriculum: "+str(self.curr_map[curr])))
            print("\t"+s)
            print("Running sample # "+str(ri+1)+'/'+str(nsamples)+': '+centric)
            results, states2 = r.run_session(params={ 'disp_avg_losses':10, \
                    'test_frequency':self.test_frequency,\
                    'buffer_updates':False, 'rotational':False, 'printing':False}) 
#            with open('temp_results_keys.txt', 'w') as f:
#                for k in sorted(results._data.keys()):
#                    f.write(str(k)+'\n')
#            sys.exit()
            Tr_losses.append(results.get('train', 'losses'))
            Te_losses.append(results.get('test' , 'losses'))

            Tr_Successes.append(results.get('train', 'successes'))
            test_results = results.get('test', 'successes')
            Te_Successes.append(test_results)
            #if training_epochs > 30 and len(s)>0:
            if True:
                s_ = s+' sample #'+str(ri)+' seed '+str(self.curseeds[ri])+\
                        ' last 30 test accs: '+'\n'+\
                        str(test_results[-30:])+'\n'
                try:    self.fin_logfile.write(unicode(s_))
                except: self.fin_logfile.write(s_)
                s_ = s+' sample #'+str(ri)+' last 30 test accs: '+'\n'+\
                        str(test_results)+'\n'
#                try:    self.tot_logfile.write(unicode(s_))
#                except: self.tot_logfile.write(s_)
            if statemap==None: 
                statemap = r.scheduler.statemap.copy()
            # TODO: load pickled states and reorder them locally.
            StateMaps.append(results.lex_to_ints.copy()) 

        np.save(s+'-loss-graph.npy', np.array([Tr_losses, Te_losses]))
        for S in StateMaps:
            print(S)
        print (statemap)
        return  np.mean(np.array(Tr_Successes), axis=0), \
                np.mean(np.array(Te_Successes), axis=0), statemap, states2



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
                ('-seed_'+str(seed) if not seed==None else '')+\
                ('-curr_'+str(self.curr_map[curr]).zfill(2) if not curr==None else '')
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
    mode='ego-allo-test'
    if len(sys.argv)<2:
        seed=None
    else:
        seed = int(sys.argv[1])
    print (sys.argv)
    experiment(mode, seed)

print("Done.")
