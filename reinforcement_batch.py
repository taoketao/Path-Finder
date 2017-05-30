'''
Morgan Bryant, April 2017
test framework for making sure the NN works - absent any reinforcement context
'''
import sys, time
import tensorflow as tf
from socket import gethostname
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from environment3 import *
from network import *
from scipy.sparse import coo_matrix
from subprocess import call

''' system/os constants '''
COMPONENTS_LOC = "./data_files/components/"

''' [Helper] Constants '''
N_ACTIONS = 4
N_LAYERS = 4
ACTIONS = [UDIR, RDIR, DDIR, LDIR]
ACTION_NAMES = { UDIR:"UDIR", DDIR:"DDIR", RDIR:"RDIR", LDIR:"LDIR" }
P_ACTION_NAMES = { UDIR:"^U^", DDIR:"vDv", RDIR:"R>>", LDIR:"<<L" }
DEFAULT_EPSILON = 0.9
DEFAULT_BATCH_SIZE = 12
ALL=-1
A_LAYER=0
G_LAYER=1
#SCHEDULED_LR_SIGNAL = -23 # some value
VALID_FLAG_PLACEHOLDER=True
DEFAULT_TEST_FREQ = 10          # Trials should have >100 epochs
DEFAULT_DISP_NUM_LOSSES = 10    # Trials should have >100 epochs


class session_results(object):
    ''' Stores results from a session.  Stores state data sparsely.
        Data entries: dicts of
            num_a: num actions taken,
            s0: original state with grid in sparse storage format,
            actions_attempted: fixed size, Action ids with -1 for untaken actions,
            success: bool of whether the goal was reached,
            mode: "train" or "test"         '''
    def __init__(self, n_training_epochs, n_episodes, batch):
        self._data = {} # should be nepochs x nepisodes
        self.nepochs = n_training_epochs
        self.nepisodes = n_episodes
        self.batch=batch

    def put(self, epoch, episode, data): 
        if not 'mode' in list(data.keys()): 
            raise Exception("Provide mode: train or test.")
        if data['mode']=='train': mode = 0 
        if data['mode']=='test' : mode = 1 
        D = data.copy()
        for d in D:
            if d=='s0':
                D[d] = data[d].copy()
                D[d].grid = [coo_matrix(D[d].grid[:,:,i]) for i in range(N_LAYERS)]
                D[d].sparse=True
        if episode=='all':
            self._data[(epoch, mode)] = D
        else:
            self._data[(epoch, episode, mode)] = D

    def get(self, p1, p2=None, p3=None, batch=True): 
        if p1 in ['train','test'] and p2==None and p3==None:
            return self._get(ALL,ALL,p1)
        if p1 in ['train','test'] and p2 in ['successes'] and p3==None:
            x__ = []
            for X in self._get(ALL,ALL,p1):
                x__.append(np.array([ 1.0 if x['success'] else 0.0 for x in X]))
            return np.array( x__ )
        if p1 in ['train','test'] and p2 in ['nsteps'] and p3==None:
            return np.array( [[ x['num_a'] for x in X]\
                        for X in self._get(ALL,ALL,p1)])
        if p1 in ['train','test'] and p2 in ['losses'] and p3==None:
            x__ = []
            for X in self._get(ALL,ALL,p1):
                x__.append(np.array([ x['loss'] for x in X]))
            return np.array( x__ )

            return np.array( [[ x['loss'] for x in X]\
                        for X in self._get(ALL,ALL,p1)])

        if p1=='states' and p2==None and p3==None: # return state tups:
            uniq_st = set()
            for i in range(self.nepochs):
                for j in range(self.nepisodes):
                    st = self._data[(i,j,0)]['s0']
                    st.sparsify()
                    uniq_st.add(( st.grid[A_LAYER].row[0],\
                                  st.grid[A_LAYER].col[0],\
                                  st.grid[G_LAYER].row[0],\
                                  st.grid[G_LAYER].col[0] ))
            return sorted(list(uniq_st))
        return self._get(p1,p2,p3)

    def _get(self, epoch, episode, mode):
        if mode=='train': mode = 0 
        if mode=='test' : mode = 1 
        if epoch == ALL: 
            return [self._get(ep, episode, mode) for ep in range(self.nepochs)]
        if episode == ALL: 
            return [self._get(epoch, ep, mode) for ep in range(self.nepisodes)]
        return self._data[(epoch, episode, mode)].copy()

def CurriculumGenerator(inp, scheme=None):
    ''' Generates numerous curricula according to a scheme. '''
    if scheme==None and type(inp)==dict and 'schedule kind' in inp.keys() and \
            'which ids' in inp.keys(): scheme = 'default'

    def repeat_specs(List, ncopies):
        newL = [None]*(ncopies*len(List))
        for i in range(len(List)):
            for j in range(ncopies):
                newL[i*ncopies+j] = List[i].copy()
        return newL

    if scheme=='default':
        ''' Case: take the input <inp> as a literal dictionary. '''
        return CurriculumSpecifier(inp)
    elif scheme == 'cross parameters':
        curr_specs = [{}]
        for spec, specvals in inp.items():
            n_orig = len(curr_specs)
            if type(specvals)==str:
                specvals = [specvals]
            n_vals = len(specvals)

            if n_vals==1:
                for i in range(n_orig):
                    curr_specs[i][spec] = specvals[0] 
                continue
            curr_specs = repeat_specs(curr_specs, n_vals)
            for i in range(n_orig):
                for j in range(n_vals):
                    curr_specs[i*n_vals + j][spec] = specvals[j]
        return [CurriculumSpecifier(specs) for specs in curr_specs]
    else: assert(False)

class CurriculumSpecifier(object):
    ''' Usage: For each state of state group id (SID), specify (a) the
        (partial?) ordering of the SIDs and (b) the English schedule for
        occurrence. Class Scheduler implements which states to actually return.
        
    Doc: 
        dynamic:  is a bool indicating if the curr should respond to training.
        which ids:  an object that encapsulates which states go to which IDs.
            Case Xdir: which is a list of states to keep (or a shorthand)
        schedule kind:  how different groups should be presented: curve shapes.
        schedule strengths:  what percentages parameterize the groups over epochs.
            Give schedule-kind-specific float values for each group (or all but 1st)
            or a keyword such as 'egalitarian', which ends all at the same strength.
            These can be thought of as FACTORS, ie, unnormalized probabilities.
        schedule timings:  which epochs to enact the changes
            Give schedule-kind-specific integer values for each group (or all but 1st)
        groups: a dict.  Keys are lexically-ordered tuples where X<Y means
            X will be presented to the network before Y will.

    Accessible fields (do your best not to overwrite them):
        inp: the original input.  
        dynamic, which_ids, sched_kind: 'symlinks' to cleaned inp fields
        groups: a dict of group lexicoIDs to the group's tasks (as strs)
        nstates: total number of states that are (potentially) ever active 
        begVal, endVal, begTime, endTime: dicts of group lexicoIDs to their 
                parameterizations; may be empty depending on curriculum.
        '''
    def __init__(self, inp, controller_format='default'):
        if not controller_format=='default': 
            raise Exception('input format error')
        self.inp = inp
        self.groups = {}
        assert( type(inp)==dict)
        for required_key in ['which ids','schedule kind','schedule strengths',\
                'schedule timings']:
            if (not required_key in inp.keys() and inp['which ids']=='uniform'):
                print (inp.keys())
                raise Exception("Please provide parameters: %s" % required_key)
        self.dynamic = inp['dynamic'] if 'dynamic' in inp.keys() else False  
        self.which_ids = which = inp['which ids']
        if which=='any': which='all'
        self.groups = {\
            'all': { (1,): {'_u','_r','_d','_l'}, (2,): {'dl','dr','lu','ru'}, \
                    (3,): {'uu','dd','ll','rr'} }, \
            'any1step, u r diag': { (1,): {'_u','_r','_d','_l'}, (2,): {'ru'} }, \
            'r, u, ru-diag only': { (1,): {'_u','_r'}, (2,): {'ru'} }, \
            'r or u only': { (1,): {'_u','_r'}, (2,): {'ru'}, (3,): {'rr','uu'} }, \
            'uu ur': { (1,): {'_l','_d','_u','_r'}, (2,): {'ru'}, (3,): {'uu'} }, \
            'poles': { (1,): {'_l','_d','_u','_r'}, (2,): {'rr','ll'} }, \
            'all diag': { (1,): {'_u','_r','_l','_r'}, (2,): {'dl','dr','lu','ru'} },\
            '1step': { (1,): {'_u','_d','_l','_r'} }, \
            '1step split': { (1,): {'_u','_d'}, (2,):{'_l','_r'} }, \
          }[which]
        self.nstates = { 'any1step, u r diag': 5, 'r, u, ru-diag only': 3, \
                'r or u only': 5, 'uu ur': 6, 'poles': 6, 'all diag': 8, \
                'all': 12, '1step': 4, '1step split': 4 }[which]
            
        self.sched_kind = sched_kind = inp['schedule kind']
        if not sched_kind == 'uniform': 
            if not 'schedule strengths' in inp.keys():
                raise Exception("Please provide parameters to the schedule.")
            if not 'schedule timings' in inp.keys():
                raise Exception("Please provide parameters to the schedule.")


        ''' Strengths: What (relative) percentages should each group take?'''
        if sched_kind == 'uniform':
            self.begVal = {};  self.endVal = {} 
            self.begTime = {}; self.endTime = {} 
            for i in range(len(self.groups)):
                self.begVal[(i+1,)] = 1.0;    self.endVal[(i+1,)] = 1.0;
                self.begTime[(i+1,)] = 0;    self.endTime[(i+1,)] = -1;
            return


        begVal = {}; endVal = {} 
        begTime = {}; endTime = {} 
        schv = inp['schedule strengths']
        if type(schv)==str and schv=='egalitarian' and len(self.groups)==2:
            begVal[(1,)] = 1.0; endVal[(1,)] = 1.0
            begVal[(2,)] = 0.0; endVal[(2,)] = 1.0
        elif sched_kind in ['linear anneal', 'no anneal']:
            if schv=='20-80 flat group 1':
                begVal[(1,)] = 20.0;   endVal[(1,)] = 20.0;
                begVal[(2,)] = 0.0;    endVal[(2,)] = 80.0;
            elif schv=='20-70-10 flat group 1':
                begVal[(1,)] = 20.0;   endVal[(1,)] = 20.0;
                begVal[(2,)] = 0.0;    endVal[(2,)] = 70.0;
                begVal[(3,)] = 0.0;    endVal[(3,)] = 10.0;
            else:
                begVal[(1,)] = schv['b0'] if 'b0' in schv else 1.0
                endVal[(1,)] = schv['e0'] if 'e0' in schv else 1.0
                begVal[(2,)] = schv['b1'] if 'b1' in schv else 0.0
                endVal[(2,)] = schv['e1'] if 'e1' in schv else 1.0
                begVal[(3,)] = schv['b2'] if 'b2' in schv else 0.0
                endVal[(3,)] = schv['e2'] if 'e2' in schv else 1.0
        elif sched_kind == 'sigmoid':
            # Values correspond to logistic sigmoid's temperature
            begVal[(1,)] = schv['s0'] if 's0' in schv else 1.0
            begVal[(2,)] = schv['s1'] if 's1' in schv else 1.0
            begVal[(3,)] = schv['s2'] if 's2' in schv else 1.0
        else: raise Exception("Schedule value format not recognized.")

        ''' Timings: At what epochs should the schedule kind operate at? '''
        scht = inp['schedule timings']
        if sched_kind in ['linear anneal', 'egalitarian']:
            begTime[(1,)] = scht['b0'] if 'b0' in scht else 0
            if 'e0' in scht: endTime[(1,)] = scht['e0']
            if 'b1' in scht: begTime[(2,)] = scht['b1'] 
            endTime[(2,)] = scht['e1'] if 'e1' in scht else -1 # last epoch
            if 'b2' in scht: begTime[(3,)] = scht['b2'] 
            if 'e2' in scht: endTime[(3,)] = scht['e2']
        elif sched_kind == 'no anneal':
            begTime[(1,)] = scht['t0'] if 't0' in scht else 0
            begTime[(2,)] = scht['t1'] if 't1' in scht else 0
            begTime[(3,)] = scht['t2'] if 't2' in scht else 0
        elif sched_kind == 'sigmoid':
            begTime[(1,)] = scht['s0'] if 's0' in scht else 0
            begTime[(2,)] = scht['s1'] if 's1' in scht else 0
            begTime[(3,)] = scht['s2'] if 's2' in scht else 0
        else: raise Exception("Schedule times format not recognized.")

        self.begVal = begVal
        self.endVal = endVal
        self.begTime = begTime
        self.endTime = endTime

    def toString(self):
        return '\n  '.join(['inp:'+str(self.inp), 'groups:'+str(self.groups), 'begTime:'+str(self.begTime), 'endTime:'+str(self.endTime), 'begVal:'+str(self.begVal), 'endVal:'+str(self.endVal)])
        s=''
        s += 'Curriculum: [  schedule kind: '+ self.sched_kind + ',  '
        s += 'which states (' + self.inp['which ids'] + '): '
        for g,(vs,bt,et,bv,ev) in enumerate(zip(self.groups, self.begTime,
                    self.endTime, self.begVal, self.endVal)):
            s += 'Group '+str(g)+' <'+ ' '.join(str(vs[g])) + '>: '+'schedule ('\
                +str(bt[g])+','+str(et[g])+') & factors ('+str(bv[g])+','+str(ev[g])+');  '
        return s
    
    
class Scheduler(object):
    ''' Scheduler: various modes:
        - uniform curriculum: each unique state is chosen for batches with the 
            same probability, for all epochs.
        - linear_anneal_eX_bY curriculum: the states are sorted into 'easy' and
            'hard' (currently only implemented for v2-a_fixedloc_leq).  Batches 
            are chosen uniquely from 'easy' set up until epoch number Y, are 
            chosen uniformly randomly from either 'easy' or 'hard' after epoch
            number X, and are linearly annealed in between.
        - upguided_eX_bY: same as linear_anneal_eX_bY, except 'hard' set is 
            reduced from all states with a goal 2 away from A to the singleton
            set with the state where G is 2 up from A.

        Supported curricula: (provide as 'curr' dict value in override dict)
            all:                 1: u,r,d,l, 2: dl,dr,lu,ru
            any1step, u r diag:  1: u,r,d,l, 2: ru
            r, u, ru-diag only:  1: u,r,     2: ru
            r or u only:         1: u,r,     2: ru,           3: rr,uu
            uu ur:               1: l,d,u,r, 2: ru,           3: uu
            poles:               1: l,d,u,r, 2: rr,ll
            all diag:            1: u,r,l,r, 2: dl,dr,lu,ru
            all:                 1: u,r,d,l, 2: dl,dr,lu,ru,  3: dd,ll,rr,uu
            1step:               1: u,r,d,l
            1step split:         1: u,r      2: d,l

    '''


    def __init__(self, data_mode, which_game, init_states, t_epchs,\
            override ):
        self.data_mode = data_mode
        self.mna_type = override['max_num_actions'] if \
                    'max_num_actions' in override else MAX_NUM_ACTIONS

        ''' >>>>    Curriculum schedule setup. '''
        self.curriculum = override['curriculum'] 
        if not which_game=='v2-a_fixedloc_leq':
            self.batchsize = max(len(init_states), 4)
        else: 
            self.batchsize = DEFAULT_BATCH_SIZE
        if type(self.curriculum)==str and self.curriculum=='uniform':
            pass # for now...
        elif type(self.curriculum)==str and ('linear_anneal' in self.curriculum\
                or 'upguided' in self.curriculum):
            print("Note: curriculum [%s] is deprecated." % self.curriculum)
            self.tasks = {}
            if self.override['curr'] == 'linear_anneal':
                self.tasks['easy'] = self.tasks[0]
                self.tasks['hard'] = self.tasks[1]
                self.easy_ids = list(self.tasks['easy'])
            self.easy_ids = list(self.tasks['easy'])
            self.n_easy = len(self.tasks['easy'])
            self.n_hard = len(self.tasks['hard'])
        elif type(self.curriculum) == CurriculumSpecifier:
            if not self.curriculum.sched_kind in ['uniform', 'linear anneal', \
                    'no anneal']: raise Exception('bad curriculum shaping given.')
            self.statemap = self.get_lexico(init_states, self.curriculum.groups)
#            print('statemap:', '; \n\t  '.join([str(stid)+': gr'+str(st['group'])+', state:'+\
#                    st['state'].name+', lexid: '+st['lex_id'] for stid, st in self.\
#                    statemap.items()]))
            # Then, refurbish get_next_states(), the rest of this init, and
            # the reinforcement process for sampling states.
        else:
            raise Exception("dev err 3")
        self.training_epochs = t_epchs
        self.init_states = init_states

        ''' >>>>    Epsilon scheduling setup. '''
        if 'epsilon' in override:  
            self.epsilon_exploration = override['epsilon']
        else:  
            self.epsilon_exploration = DEFAULT_EPSILON
        eps_e = self.epsilon_exploration
        if type(eps_e)==str and eps_e[:5] == 'decay':
            self.eps_schedule = [1.0]
            exp = float(eps_e[6:])
            for i in range(t_epchs):
                self.eps_schedule.append( self.eps_schedule[-1]*exp )

        ''' >>>>    Learning rate schedule setup. '''
        param_lr = override['learning_rate']
        if type(param_lr) in (float, int): self.lr_schedule = param_lr
        else:
            if 'one_log_decrease' in param_lr: 
                self.lr_schedule = {'dependencies':'epoch-only-dependent', \
                        'kind': 'one_log_decrease', 'init': float(param_lr[10:]),\
                        }
        self.adaptive_lr = False if type(self.lr_schedule) in [int, float] else True 
            
    
    ''' State management. '''
    def get_lexico(self, states, groups):
        ''' This function returns a list-accessible-formatted dictionary of 
            states that ease curriculum operations. 

            Returns dict of  {init_state indexed state : -> {'state': state object, 
                'lex_id': str lex id, 'group': group tup}}
            '''
        statemap = {}
        if not len(states)==12:
            raise Exception("Lexicograph not yet implemented for these states.")
        invg = {}
        for k,V in groups.items():
            for v in V:
                invg[v]=k
        for i, st in enumerate(states):
            if not st.get_agent_loc()==(3,3) and st.gridsz==(7,7):
                raise Exception("Lexicograph not yet implemented for these states.")
            lex_id = self._get_nameid(st, 2)
            grp = invg[lex_id] if lex_id in invg.keys() else ()
            statemap[i] = {'state':st, 'lex_id':lex_id, 'group':grp }
        return statemap
    def _get_nameid(self, st, minchars): 
        '''  Lowercase:   _ < d < l < r < u   string inequalities hold. '''
        ax, ay = st.get_agent_loc(); gx, gy = st.get_goal_loc(); 
        s = ''.join( ['d' for _ in range(gy-ay)] + ['l' for _ in range(ax-gx)] \
                   + ['r' for _ in range(gx-ax)] + ['u' for _ in range(ay-gy)] )
        while len(s)<minchars: s = '_'+s
        return s
        

    ''' Learning rate scheduling. '''
    def learning_rate_signaller(self): 
        return SCHEDULED_LR_SIGNAL
    def get_init_lr(self): 
        if type(self.lr_schedule) in [int, float]: 
            return self.lr_schedule
        else: 
            return self.lr_schedule['init']

    def get_lr(self, epoch=None ):
        if not self.adaptive_lr: raise Exception
        if 'epoch-only-dependent' in self.lr_schedule['dependencies']:
            if self.lr_schedule['kind'] == 'one_log_decrease':
                return self.lr_schedule['init'] * (10**-epoch/self.training_epoch)

    def getBatchSize(self): return self.batchsize

    ''' Curriculum scheduling. '''
    def get_next_states(self, epoch):
        curr = self.curriculum
        #print ('\n\n',curr.toString())
        ps = [0.0]*len(self.init_states)
        for i in range(len(ps)):
            my_group = self.statemap[i]['group']
            if len(my_group)==0:    continue
            sched_kind = curr.sched_kind
#            print(i, my_group, self.init_states[i].name, sched_kind, curr.which_ids)
            if sched_kind == 'sigmoid': raise Exception("Dev: sigm not implemented")
            try:
                begT = curr.begTime[my_group]
                if sched_kind in ["no anneal", 'sigmoid'] or not my_group in \
                        curr.endTime.keys():
                    endT = self.training_epochs
                else:
                    endT = curr.endTime[my_group] #######
                begS = curr.begVal[my_group]
                endS = curr.endVal[my_group]
            except:
                raise Exception("Group not logged in this curriculum object: "+\
                        str(my_group)+'  '+str(sched_kind)+' <'+\
                        curr.which_ids +'> '+ self.statemap[i]['lex_id']+', -- '+\
                       ', '.join([str(s) for s in [curr.begVal, curr.endVal, curr.begTime, curr.endTime]]) )
            #print(i, my_group, begT, endT, begS, endS)
            if endT<0: endT = self.training_epochs
            if sched_kind=='uniform':
                assert(begS==endS)
                ps[i] = begS;   continue # endT==begT.
            if epoch <= begT:   ps[i] = begS;   continue
            elif sched_kind=='no anneal':   ps[i] = endS ; continue
            if epoch >= endT:   ps[i] = endS;   continue
            if sched_kind=='no anneal':
                raise Exception("dev err: epoch should have been caught already.")
            if sched_kind=='linear anneal':
                pct = float(epoch-begT) / (endT-begT)
                ps[i] = pct*begS + (1-pct)*endS

        #print(['%1.3f' % p for p in ps])
        if np.sum(ps)==0: raise Exception()
        ps /= np.sum(ps)
        selections = np.random.choice(range(len(self.init_states)), \
                self.batchsize, p=ps)
        return [(i,self.init_states[i]) for i in selections]

    def action_drop(self, epoch):
        if type(self.mna_type)==int: return False
        mna_parts = self.mna_type.split('_')
        if 'anneal' in mna_parts and 'linear' in mna_parts:
            for m in mna_parts:
                if m[0]=='e': end_epch = int(m[1:])
                if m[0]=='b': burn_in  = int(m[1:])
            # Burn-in: set to 100 by default.
            thresh = max(min(float(epoch-burn_in)/\
                    (end_epch-burn_in),1),0)
            #print(thresh, end=' ')
            if np.random.rand() < thresh: 
                return False
            return True
        raise Exception("MNA method not recognized: "+self.mna_type)

    def get_epsilon(self, epoch):
        if type(self.epsilon_exploration)==float:
            return self.epsilon_exploration
        elif self.epsilon_exploration=='lindecay':
            return 1-float(epoch)/self.training_epochs
        elif self.epsilon_exploration=='1/x':
            return 1.0/(epoch+1)
        elif self.epsilon_exploration[:4]=='1/nx':
            return float(self.epsilon_exploration[4:])/(epoch+1)
        elif self.epsilon_exploration[:5]=='decay':
            return self.eps_schedule[epoch]
        else: raise Exception("Epsilon strategy not implemented")







''' reinforcement: class for managing reinforcement training.  Note that 
    the training programs can take extra parameters; these are not to be 
    confused with the hyperparameters, set at the top of this file. '''
class reinforcement_b(object):
    ''' Constructor: which_game is the variable that commands entirely what 
        training pattern the network will undergo. 

        v1-single: The network learns to act in the game: [[---],[AG-],[---]].
                Mainly for testing and debugging.
        v1-oriented: The network learns to act in all variants of games which
                are 3x3 and only have component [AG] in any (of 6) locations.
        v1-a_fixedloc: the network learns to do act in a puzzles which have
                the _agent_ at a fixed location ->> 4 states, presumed HARDER
        v1-g_fixedloc: the network learns to do act in a puzzles which have
                the _goal_ at a fixed location ->> 4 states, presumed EASIER
        v1-micro: like a_fixedloc, but *all* other spaces are blocked except
                for A and G.
        v1-corner: This game has the network learn all variants of distance-1
                games in which the 3x3 grid is in the same location.
        v1-all:  not yet implemented.

        v0-a_fixedloc: extra simple - zeros.  A at center.

        v2-a_fixedloc:  A at center, all variations where G is within two
                steps of A in a (7x7) world. 7x7: bc agent doesn't have 
                'which # action is this' identifiability.
        ...
    '''
    def __init__(self, which_game, frame, game_shape=None, override=None,
                 load_weights_path=None, data_mode=None, seed=None):
        np.random.seed(seed)
        if game_shape == None: 
            if 'v3' in which_game: game_shape = (9,9)
            if 'v2' in which_game: game_shape = (7,7)
            if 'v1' in which_game: game_shape = (5,5)
            if 'v0' in which_game: game_shape = (5,5)
        self.env = environment_handler3(game_shape, frame, world_fill='roll')
        self.sg = state_generator(game_shape)
        self.sg.ingest_component_prefabs(COMPONENTS_LOC)
       
        if 'v1' in which_game or 'v0' in which_game:
            self.D1_ocorner = self.sg.generate_all_states_fixedCenter(\
                    'v1', self.env, oriented=True)
            self.D1_corner = self.sg.generate_all_states_fixedCenter(\
                    'v1', self.env)
            self.D1_micro = self.sg.generate_all_states_micro('v1',self.env)
            pdgm, init_states = {
                'v1-single':    ('v1', [ self.D1_corner[2]]),
                'v1-oriented':  ('v1', self.D1_ocorner),
                'v1-micro':    ('v1', self.D1_micro),
                'v1-corner':    ('v1', self.D1_corner),
                'v1-a_fixedloc':  ('v1', [self.D1_corner[3],  self.D1_corner[8],\
                                  self.D1_corner[15], self.D1_corner[20] ]),

                'v1-micro_g_fixedloc': ('v1', self.D1_micro+[self.D1_corner[3], \
                    self.D1_corner[8], self.D1_corner[15], self.D1_corner[20] ]),
                'v1-micro_a_fixedloc': ('v1', self.D1_micro+[self.D1_corner[3], \
                    self.D1_corner[8], self.D1_corner[15], self.D1_corner[20] ]),
                
                # v *** v
                'v0-a_fixedloc':  ('v0', [self.env.getStateFromFile(\
                                './data_files/states/3x3-basic-G'+f, 'except')\
                                for f in ['U.txt', 'R.txt', 'D.txt', 'L.txt']]),
                # ^ *** ^

                'v0-single':    ('v0', [self.env.getStateFromFile(\
                                './data_files/states/3x3-basic-AU.txt', 'except')]),
                'v1-all':       'stub, not implemented yet!',
            }[which_game]

        if 'v2' in which_game:
            if which_game == 'v2-a_fixedloc_eq':
                init_states = self.sg.generate_all_states_only_2away('v2',self.env)
            if which_game == 'v2-a_fixedloc_leq':
                init_states = self.sg.generate_all_states_upto_2away('v2',self.env)
            pdgm = 'v2'

        if 'printing'in override and override['printing']==True:
            for i,s in enumerate(init_states):
                print(("State",i)); self.env.displayGameState(s, mode=2); 
                print('---------------------')
        self.which_game = which_game
        self.init_states = init_states
        self.which_paradigm = pdgm
        self.data_mode = data_mode
        self.seed = seed
        net_params=None

        if not override==None:
            self.mna_type = override['max_num_actions'] if \
                    'max_num_actions' in override else MAX_NUM_ACTIONS

            self.max_num_actions = self.mna_type if type(self.mna_type)==int \
                    else int(self.mna_type[:self.mna_type.find('_')])

            self.training_epochs = override['nepochs'] if \
                    'nepochs' in override else  MAX_NUM_ACTIONS
        if not 'rotation' in override: override['rotation']=False
        if 'netsize' in override:
            o = override['netsize']
            if not len(o)%2==0: raise Exception("Invalid network structure init.")
            nlayers = len(o)//2
            net_params = \
                    { o[i]+str(i+1)+'_size':o[i+nlayers] for i in range(nlayers) }
        if 'optimizer_tup' in override:
            opt = override['optimizer_tup']
        else: opt = ('sgd')

        if type(override['epsilon'])==str and override['epsilon'][:5]=='decay':
            f = float('0.'+override['epsilon'][6:])
            self.eps_schedule = [1.0]
            for i in range(self.training_epochs):
                self.eps_schedule.append(self.eps_schedule[-1]*f)
        self.curriculum = override['curriculum']
        

#        for i,ii in enumerate(init_states):
#            print(i, end='  '); print(ii.get_goal_loc(), ii.get_agent_loc()) 
#            self.env.displayGameState(ii)


        self.scheduler = Scheduler(self.data_mode, self.which_game, \
                self.init_states, self.training_epochs, override )
        self.minibatchsize = self.scheduler.getBatchSize()

        # If load_weights_path==None, then initialize weights fresh&random.
        self.Net = network(self.env, 'NEURAL', override=override, \
                load_weights_path=load_weights_path, _game_version=\
                pdgm, net_params=net_params, _optimizer_type=opt, seed=seed,\
                _batch_off=False) # batch mode!

    
    def _stateLogic(self, s0, Q0, a0_est, s1_est, rotation=0, printing=True):
        ''' (what should this be named?) This function interfaces the network's
            choices with the environment handler for necessary states.  '''
        motion_est = a0_est % N_ACTIONS
        valid_action_flag = self.env.checkIfValidAction(s0, motion_est)
        if valid_action_flag:
            goal_reached = self.env.isGoalReached(s1_est)
            if printing: print("--> Yields valid", a0_est%4, a0_est/4)
            return a0_est, s1_est, REWARD if goal_reached else NO_REWARD, \
                    goal_reached, valid_action_flag

        a0_valid = np.argmax(np.exp(Q0[:4]) * self.env.getActionValidities(s0))
        return a0_valid, self.env.performActionInMode(s0, a0_valid), \
                INVALID_REWARD, False, valid_action_flag

    def dev_checks(self):
        if not self.which_paradigm in ['v0','v1','v2']:
           raise Exception("Reinforcement session is only implemented for v0,v1,v2.")

    def _populate_default_params(self, params):
        if not 'saving' in params:
            params['saving'] = { 'dest':None, 'freq':-1 }
        if not 'mode' in params:
            params['present_mode'] = 'ordered' # batch mode
        if not 'buffer_updates' in params:
            params['buffer_updates'] = False
        if not 'dropout-all' in params:
            params['dropout-all'] = True
        if not 'printing' in params:
            params['printing'] = False
        if not 'disp_avg_losses' in params:
            params['disp_avg_losses'] = DEFAULT_DISP_NUM_LOSSES
        if not 'test_frequency' in params:
            params['test_frequency'] = DEFAULT_TEST_FREQ
        return params

    ''' RUN_SESSION: like Simple_Train but less 'simple': Handles the 
        overall managingof a reinforcement session.  Scope:
         - above me: user interfaces for experiments
         - below me: anything not common to all tests. '''
    def run_session(self, params={}):
        self.dev_checks() # raises errors on stubs
        params = self._populate_default_params(params)
        init = tf.global_variables_initializer()
#        sess_config = tf.ConfigProto(inter_op_parallelism_threads=1,\
#                                     intra_op_parallelism_threads=1)

        sess_config = tf.ConfigProto()       
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        self.Net.setSess(sess)
        sess.run(init)
        
        episode_results = session_results(self.training_epochs, \
                self.minibatchsize, batch=True)

        Buff=[]
        last_n_test_losses = []
        last_n_successes = []

        tf.set_random_seed(self.seed)

        self.Net.adjust_lr(self.scheduler.get_init_lr())
        init_time = time.time()
        last_test_eps = None 

        for epoch in range(self.training_epochs):
            # Save weights
            save_freq = params['saving']['freq']
            if save_freq>=0 and epoch % save_freq==0:
                self.Net.save_weights(params['saving']['dest'], prefix= \
                        'epoch'+str(epoch)+'--')
            # Setup scheduling
            if self.scheduler.adaptive_lr:
                if 'epoch-only-dependent' in self.scheduler.lr_schedule:
                    self.Net.adjust_lr(self.scheduler.get_lr(epoch=epoch))

            # Take pass over data.
            if not (self.curriculum == None or self.curriculum=='uniform'):
                next_states = self.scheduler.get_next_states(epoch)
            else:
                if not self.data_mode == None: 
                    params['present_mode']=self.data_mode
                if params['present_mode']=='ordered':
                    next_states = [(i,s) for i,s in enumerate(self.init_states)]
                elif params['present_mode']=='shuffled':
                    order = np.random.permutation(range(len(self.init_states)))
                    next_states = [(i,self.init_states[i]) for i in order]
                else: raise Exception("Provided or default data mode not "+\
                            "implemented or not provided..")
            if not last_test_eps: last_test_eps=[None]*len(next_states)

            ep_losses = None
            for __mode in ['train', 'test']:
                if __mode=='test' and epoch>0 and \
                            not epoch % params['test_frequency']==0:
                    # Use previously stored test values
                    for si in range(len(next_states)):
                        #episode_results.put(epoch, si, episode_results._get(epoch-1,si,'test'))
                        episode_results.put(epoch, si, last_test_eps[si])
                else:
                    num_as, goals, actns, ep_losses = \
                            self._do_batch(next_states, epoch, __mode)
                    for si, (_,s0) in enumerate(next_states):
                        result_info = { 's0': s0, \
                                        'num_a': num_as[si], \
                                        'success': goals[si], \
                                        'attempted_actions': actns[:,si], \
                                        'loss': ep_losses[-1], \
                                        'mode': __mode }
                        if __mode=='test':
                            last_test_eps[si] = result_info
                        episode_results.put(epoch, si, result_info)

            ret_te = ep_losses
            ret_sc = goals

            if params['disp_avg_losses'] > 0:
                last_n_test_losses.append(ret_te)
                if len(last_n_test_losses) > params['disp_avg_losses']\
                        * params['test_frequency']:
                    last_n_test_losses.pop(0)
                last_n_successes.append(ret_sc)
                if len(last_n_successes) > params['test_frequency']\
                        * params['test_frequency']:
                    last_n_successes.pop(0)

#            if gethostname()=='PDP' and epoch==1000: 
#                call(['nvidia-smi'])

            if (epoch%1000==0 and epoch>0) or \
                    (epoch<=params['disp_avg_losses'] and epoch%5==0): 
                s = "Epoch #"+str(epoch)+"/"+str(self.training_epochs)
                s += '\tlast '+str(params['disp_avg_losses'])+' losses'+\
                        ' averaged over all states:'
                s += '  '+str(np.mean(np.array(last_n_test_losses)))
                s += '  and successes:' 
                s += '  '+str(np.mean(np.array(last_n_successes)))
                print(s) # Batch mode!

                if not params['printing']: 
                    continue
                if epoch>0:
                        rng = [-1,-2,-3,-4,-5]
                else:
                    rng = [-1]
                print("Q values for last several actions:")
                for wq in rng:
                    print(('\t'+P_ACTION_NAMES[Qvals[wq]['action']]+': ['+\
                            ', '.join(["%1.5f"% p for p in Qvals[wq]['Q']])+\
                            ']; corr?: '+str(int(Qvals[wq]['reward']))))
        t = time.time()-init_time
        print("Elapsed time in seconds: %i mins, %.3f seconds" % \
                (t//60), (t % 60)
        return episode_results

    def _do_batch(self, states, epoch, mode, buffer_me=False, printing=False):
        #print("EPISODE"); sys.exit()
        update_buff = []
        nstates = len(states)
        num_a = [0]*nstates
        mna_cutoffs = [-1]*nstates
        wasGoalReached = [False]*nstates
        if printing: print('\n\n')
        #print([(i,s.name) for i,s in states])
        cur_state_tups = [ [(i,s.copy()) for i,s in states] ]
        cur_states = [ [c[1] for c in cur_state_tups[0]] ]
        attempted_actions = -1*np.ones((self.max_num_actions, nstates))
        losses = []
        for nth_action in range(self.max_num_actions):
            for si,s0 in enumerate(cur_states[-1]):
                if mna_cutoffs[si]>=0: continue
                if self.env.isGoalReached(s0): 
                    wasGoalReached[si]=True
                elif nth_action>0 and mode=='train' and \
                        self.scheduler.action_drop(epoch):
                    mna_cutoffs[si] = nth_action # wp, drop actions
                else:
                    num_a[si] += 1
            Q0_s = self.Net.getQVals(cur_states[-1])
            eps_chosen= np.zeros(shape=(nstates,), dtype=bool)
            
            a0_est = np.empty( (nstates,), dtype=int )
            # Choose action
            if mode=='train':
                _epsilon = self.scheduler.get_epsilon(epoch)
                eps_chs = np.random.rand(nstates)
                for eps in np.where(eps_chs < _epsilon)[0]:
                    if mna_cutoffs[eps]>= 0:  a0_est[eps]=-2; continue
                    if wasGoalReached[eps]:  a0_est[eps]=-1; continue
                    eps_chosen[eps]=True
                    a0_est[eps] = np.random.choice(ACTIONS).astype(int)
                for eps in np.where(eps_chs >= _epsilon)[0]:
                    if mna_cutoffs[eps]>= 0:  a0_est[eps]=-2; continue
                    if wasGoalReached[eps]:  a0_est[eps]=-1; continue
                    a0_est[eps] = np.argmax(Q0_s[eps,:]).astype(int)
            elif mode=='test':
                a0_est = np.argmax(Q0_s, axis=1).astype(int)

            attempted_actions[nth_action,:] = a0_est
            
            targ = np.copy(Q0_s)
            s1_ests = []
            logic_rets = []
            for si,a in np.ndenumerate(a0_est):
                si = si[0]
                a = int(a)
                s0 = cur_states[-1][si]
                if a<0:
                    s1_est = s0  
                else:
                    s1_est = self.env.performActionInMode(s0, a)
                s1_ests.append(s1_est)
                if a>=0: 
                    tmp = self._stateLogic(s0, Q0_s[si], a, s1_est, a, printing)
                    _,s1_valid,R,_,valid_flag = tmp
                    logic_rets.append( (s1_valid, R, valid_flag) )
                else:
                    logic_rets.append( (s0, NO_REWARD, VALID_FLAG_PLACEHOLDER) )

            s1_valids = [s[0] for s in logic_rets]
            Q1_s = self.Net.getQVals(s1_valids)

            valid_flags = np.zeros( (nstates,), dtype=bool)
            for i in range(nstates):
                if a0_est[i]>=0:
                    targ[i,a0_est[i]] = logic_rets[i][1]
                    valid_flag = logic_rets[i][2]
                    if valid_flag:
                        targ[i,a0_est[i]] -= GAMMA * np.max(Q1_s[i,:])
                    else:
                        targ[i,a0_est[i]] -= GAMMA * a0_est[i]

            losses.append(self.Net.getLoss(Q0_s, targ))
            if mode=='train':
                self.Net.update(orig_states=cur_states[-1], targ_list=targ)
            elif not mode=='test': raise Exception(mode)
           
            cur_states.append(s1_valids)
        return num_a, wasGoalReached, attempted_actions, losses
        return num_a, wasGoalReached, attempted_actions, np.mean(losses)
    


    def save_weights(self, dest, prefix=''): self.Net.save_weights(dest, prefix)
    def displayQvals(self):pass


if __name__=='__main__':
    ovr = {'max_num_actions':2, 'learning_rate':1e-4, 'nepochs':1, \
            'netsize':('fc',24), 'epsilon':0.5, 'loss_function':'huber10',\
            'gamesize':(7,7), 'optimizer_tup':('adam',1e-6)}
    r = reinforcement_b('v2-a_fixedloc', 'egocentric', override=ovr, \
            game_shape=(7,7), data_mode='ordered')
    _ = r.run_session()
    print("DONE")
