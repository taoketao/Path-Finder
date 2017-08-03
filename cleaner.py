# The point of this script is to help me develop on 
# issues that I still need to identify since it's been
# almost two months since I felt the code.
#
# Task: Run a simple curriculum learning experiment on 
# the flavor-place location task by Tse (2007): that is,
# with four starting locations, one of six signals, and
# six corresponding possible targets. Mimic 'digging'
# by placing a block over each starting location.

import sys

from environment import *
from experiment import *


# Global sentinels:
X=0; Y=1;
MOVE_NORTH, MOVE_SOUTH, MOVE_EAST,  MOVE_WEST = 100,101,102,103
MOVE_FWD,   MOVE_BACK                         = 110,111
ROT_R90_C,  ROT_R180_C,  ROT_R270_C           = 120,121,122
DVECS = {MOVE_NORTH: (0,-1), MOVE_SOUTH: (0,1), MOVE_EAST: (1,0), \
        MOVE_WEST: (-1,0)}
DIRVECS = {(0,-1):'N', (0,1):'S', (1,0):'E', (-1,0):'W'}
OLAYERS = [agentLayer, goalLayer, immobileLayer, mobileLayer]
# OLAYERS: ordered layers, where pos corresponds to value

AL,GL,IL,ML = [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]
AL[agentLayer]=1;       AL=np.array(AL)
GL[goalLayer]=1;        GL=np.array(GL)
ML[mobileLayer]=1;      ML=np.array(ML)
IL[immobileLayer]=1;    IL=np.array(IL)
not_a_layer = np.array([0,0,0,0])

'''     Template guide: 
            x   =   immobile block
            .   =   empty space
            a   =   an agent starting place
            !   =   goal
            r   =   end line
            e   =   end map
            o   =   start options list, after e
            D   =   'dirt': place mobile blocks over all '!' locs
            *   =   'cross': cross each agent location by each goal location
'''
TEMPLATE_TSE = ''' x x x x x x x x x x x  r
                   x x x x x a x x x x x  r
                   x x . . . . . . . x x  r
                   x x . ! . . . . ! x x  r
                   x x . . . . ! . . x x  r
                   x a . . . . . . . a x  r
                   x x . . ! . . . . x x  r
                   x x ! . . . . ! . x x  r
                   x x . . . . . . . x x  r
                   x x x x x a x x x x x  r
                   x x x x x x x x x x x  e
        o: D, *  
                   '''

'''   Printing legend for condensed mode:
    !  agent and goal       I  agent        -  immobile
    @  goal and mobile      *  goal         o  mobile
    <space>  empty          #  ERROR                        '''
def print_state(start_state, mode):
    if type(start_state)==np.ndarray:
        st = start_state
    else:
        st = start_state['state']
        print mode+':'
    if mode=='matrices':
        for i in range(st.shape[-1]):
            print st[:,:,i]
    if mode=='condensed':
        for y in range(st.shape[Y]):
            for x in range(st.shape[X]):
                if st[x,y,goalLayer] and st[x,y,agentLayer]: print '!',
                elif st[x,y,agentLayer]: print 'I',
                elif st[x,y,goalLayer] and st[x,y,mobileLayer]: print '@',
                elif st[x,y,goalLayer]: print '*',
                elif st[x,y,immobileLayer]: print '-',
                elif st[x,y,mobileLayer]: print 'o',
                elif 0==np.sum(st[x,y,:]): print ' ',
                else: 
                    print '#'
                    raise Exception("Error")
            print ''
    if not type(start_state)==np.ndarray:
        print "Flavor signal: ", start_state['flavor signal']




# General utilities:
def map_nparr_to_tup(Iterable):
    return tuple([value.tolist()[0] for value in Iterable])

def addvec(Iterable, m, optn=None):
    try: 
        return tuple([i+m for i,m in zip(Iterable,m)])
    except:
        return tuple([i+m for i in Iterable])

def multvec(Iterable, m, optn=None):
    if optn=='//':  return tuple([i//m for i in Iterable])
    if optn=='/':   return tuple([i/m for i in Iterable])
    if optn==int:   return tuple([int(i*m) for i in Iterable])
    return tuple([i*m for i in Iterable])

def at(mat, pos, lyr): return mat[pos[X], pos[Y], lyr]
def empty(mat, pos): return np.any(mat[pos[X], pos[Y], :])
def what(mat, pos): return np.array([at(mat, pos, lyr) for lyr in OLAYERS])
def put(mat, pos, lyr, v): mat[pos[X], pos[Y], lyr] = v
def put_all(mat, pos_list, lyr, v):
    for p in pos_list: put(mat, p, lyr, v)




# Experiment class:
class ExpAPI(environment_handler3):
    def __init__(self, experiment_name, centr, debug=False):
        environment_handler3.__init__(self, \
                gridsize    = { 'tse2007': (11,11) }[experiment_name], \
                action_mode = centr )
        self.state_gen = state_generator(self.gridsz)
        self.start_states = []
        self.set_starting_states(\
                {'tse2007':TEMPLATE_TSE}[experiment_name], debug)

    #  scan from a template string (eg, TEMPLATE_TSE)
    def find_all(self, a_str, char):
        s = a_str.replace(' ','')
        startX, startY = 0,0
        for c in s:
            if c==char: 
                yield((startX, startY))
            elif c=='r': 
                startY += 1
                startX = 0
            if c in 'a!x.': 
                startX += 1

    # Set this experiment's possible starting states
    def set_starting_states(self, state_template, debug=False):
        oind = state_template.index('o')
        if state_template.index('e') > oind: raise Exception()
        num_start_locs = state_template.count('a')
        num_goal_locs = state_template.count('!')
        if not state_template.find('*') > oind: raise Exception()

        start_locs = list(self.find_all(state_template, 'a'))
        goal_locs = list(self.find_all(state_template, '!'));
        block_locs = list(self.find_all(state_template, 'x'));
        if 'D' in state_template:
            mobile_locs = list(self.find_all(state_template, '!'));
            self.valid_states = np.array( [AL, GL, AL|GL, IL, ML, ML|GL] ).T
        else:
            self.valid_states = np.array( [AL, GL, AL|GL, IL, ML] ).T
#        self.valid_states = np.append(self.valid_states, np.expand_dims(\
#                np.array([0,0,0,0], dtype=bool)), axis=0)

        rx = [0,1,self.gridsz[X]-2, self.gridsz[X]-1]
        ry = [0,1,self.gridsz[Y]-2, self.gridsz[Y]-1]

        for start_box in start_locs:
            for flav_id, flavor_loc in enumerate(goal_locs):
                st = np.zeros( (self.gridsz[X], self.gridsz[Y], NUM_LAYERS))
                put(st, start_box, agentLayer, True)
                put(st, flavor_loc, goalLayer, True)
                put_all(st, mobile_locs, mobileLayer, True)
                put_all(st, block_locs,  immobileLayer, True)

                self.start_states.append( { 'flavor signal': flav_id, 'state': st, \
                        '_whichgoal':flav_id, '_startpos':start_box })
        rnd_state = self.start_states[np.random.choice(range(24))]
        if debug: print_state(rnd_state, 'condensed')

    def get_random_starting_state(self): 
        st = self.start_states[np.random.choice(range(24))]
        sret = {}
        for key in ('_startpos','flavor signal','_whichgoal'):
            sret[key] = st[key]
        sret['state'] = np.copy(st['state'])
        return sret

    def get_agent_loc(self,s):    return self._get_loc(s,targ='agent')
    def get_goal_loc(self,s):     return self._get_loc(s,targ='goal')
    def get_allo_loc(self,s):     return self._get_loc(s,targ='map') # center
    
    def _get_loc(self, state_matrix, targ):
        if targ=='agent': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,agentLayer]==1))
        if targ=='goal': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,goalLayer]==1))
        if targ=='map':
            return multvec(self.gridsz, 2, '//') # center

    def out_of_bounds(self, pos):
        return (pos[X] < 0 or pos[X] >= self.gridsz[X] or \
                pos[Y] < 0 or pos[Y] >= self.gridsz[Y])

    def is_valid_move(self, st, move): 
        aloc = self.get_agent_loc(st)
        newaloc = addvec(aloc, move)
        if self.out_of_bounds(newaloc): return False
        if at(st, newaloc, immobileLayer): return False
        if at(st, newaloc, mobileLayer):
            st2 = np.copy(st)
            put(st2, newaloc, agentLayer, True)
            put(st2, aloc, agentLayer, False)
            return self.is_valid_move(st2, move)
        return True

    def move_ent_from_to(self, mat, loc, nextloc, lyr):
        m2 = np.copy(mat)
        if not at(m2,loc,lyr): raise Exception()
        put(m2,loc,lyr, False)
        put(m2,nextloc,lyr, True)
        return m2

    def adjust_blocks(self, mat, loc, dir_vec, debug=True):
        nloc = addvec(loc, dir_vec)
        if self.out_of_bounds(nloc): return mat, False
        arr = [what(mat, loc), what(mat, nloc)]
        ploc=nloc
        while True:
            nloc = addvec(ploc, dir_vec)
            if self.out_of_bounds(nloc): return mat, False
            if not arr[-1][mobileLayer]: return mat, not arr[-1][immobileLayer]
            nmat = self.move_ent_from_to(mat, ploc, nloc, mobileLayer)
            if len(arr)>2: put(nmat, ploc, mobileLayer, True)
            arr.append(what(mat, nloc))
            ploc=nloc
            mat=nmat
        raise Exception()

    def move_agent(self, state_mat, dir_vec):

        print DIRVECS[dir_vec], self.is_valid_move(state_mat, dir_vec)
        aloc = self.get_agent_loc(state_mat)
        newL = addvec(aloc, dir_vec)
        
        state_mat2, success = self.adjust_blocks(state_mat, aloc, dir_vec)

        state_mat2 = self.move_ent_from_to(state_mat2, aloc, newL, agentLayer)
        '''
        if st[newaloc[X],newaloc[Y],mobileLayer]: 
            st2 = np.copy(st)
            st2[newaloc[X],newaloc[Y],agentLayer]=1; 
            st2[aloc[X], aloc[Y], agentLayer]=0;
            return self.is_valid_move(st2, move)
            '''

        if self.is_valid_move(state_mat, dir_vec): return state_mat2
        return state_mat

    def new_statem(self, orig_state, action):
        next_state = self.move_agent(orig_state, DVECS[action])
                
        return next_state

def _____dont_do_this__stub():
    for centr in ['egocentric', 'allocentric']:
        ExpAPI('tse2007', centr).set_starting_states(TEMPLATE_TSE)

ex = ExpAPI('tse2007', 'egocentric')
cur_state = ex.get_random_starting_state()['state']
while True:
    print 'current state:' 
    print_state(cur_state, 'condensed')
    print 'current location:', ex.get_agent_loc(cur_state)
    inp = raw_input(' interface input >> ')
    if not len(inp)==1: break
    try:
        inp_to_mov = {\
            'N': MOVE_NORTH,
            'S': MOVE_SOUTH,
            'E': MOVE_EAST,
            'W': MOVE_WEST, }[inp.upper()]
    except: 
        break
    next_state = ex.new_statem(cur_state, inp_to_mov)

    cur_state = next_state
