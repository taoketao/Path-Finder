'''                                                                           |
Morgan Bryant, April 2017                                                     |
Environment class for creating and handling world states.                     |
'''
import numpy as np
import sys, os

agentLayer = 0;             UDIR=0;
goalLayer = 1;              RDIR=1;
immobileLayer = 2;          DDIR=2;
mobileLayer = 3;            LDIR=3;
XDIM = 0;  YDIM = 1;
NUM_LAYERS = 4; # agent, goal, immobile, mobile

LAYER_MAP = {'A':0, 'G':1, 'I':2, 'M':3}
all_actions = [UDIR, RDIR, DDIR, LDIR]

layer_names = {\
        0:"Agent Layer", \
        1:"Goal Layer", \
        2:"Immobile Block Layer",\
        3: "Mobile Block Layer"}

class state(object):
    '''          ~~ state class ~~
        An object centered about a game state.  Holds data,
        provides methods, and facilitates any [valid] edits desired.
        Roughly a 'RESTful' class: access is based around get, post, del, put.
    '''

    def __init__(self, gridsize):
        self.gridsz = gridsize
        self.grid = np.zeros((self.gridsz[XDIM], self.gridsz[YDIM], \
                NUM_LAYERS), dtype='float32')
 
    def _post(self, loc, whichLayer): # set grid to on
        assert(len(loc)==2)
        #print "Adding", layer_names[whichLayer], "to loc", loc 
        self.grid[loc[XDIM], loc[YDIM], whichLayer] = 1 
    def _del(self, loc, whichLayer):
        assert(len(loc)==2)
        #print "Deleting", layer_names[whichLayer], "from loc", loc
        self.grid[loc[XDIM], loc[YDIM], whichLayer] = 0 

    ''' Public methods for the AGENT '''
    def get_agent_loc(self): return self.a_loc;
    def post_agent(self, agent_loc):
        self._post(agent_loc, agentLayer)
        self.a_loc = agent_loc
    def put_agent(self, new_loc):
        self._del(self.a_loc, agentLayer)
        self.post_agent(new_loc)

    ''' Public methods for the GOAL '''
    def get_goal_loc(self): return self.g_loc;
    def post_goal(self, goal_loc):
        self._post(goal_loc, goalLayer)
        self.g_loc = goal_loc

    ''' Public methods for the IMMOBILE BLOCKS '''
    def isImBlocked(self, loc): 
        return self.grid[loc[XDIM], loc[YDIM], immobileLayer]
    def post_immobile_blocks(self, opts):
        if opts=='border':
            [self._post((x,y), immobileLayer) for x in range(self.gridsz[XDIM])\
                    for y in [0, self.gridsz[YDIM]-1] ]
            [self._post((x,y), immobileLayer) for x in [0, self.gridsz[XDIM]-1]\
                    for y in range(1, self.gridsz[YDIM]-1) ]
        elif type(opts)==list:
            for l in opts: self._post(l, immobileLayer);
        elif len(opts)==2:
            self._post(opts, immobileLayer);

    ''' Public methods for the MOBILE BLOCKS '''
    def isMoBlocked(self, loc): 
        return self.grid[loc[XDIM], loc[YDIM], mobileLayer]
    def post_mobile_blocks(self, locs_list):
        if type(locs_list)==list:
            for loc in locs_list: 
                self._post(loc, mobileLayer)
        elif len(locs_list)==2:
            self._post(locs_list, mobileLayer);
    def put_mobile_block(self, ploc, nloc):
        self._del(ploc, mobileLayer)
        self._post(nloc, mobileLayer)

    ''' Method: find out what kind of object is located at a queried location '''
    def getQueryLoc(self, loc):
        if self.grid[loc[XDIM], loc[YDIM],immobileLayer]==1:return immobileLayer
        if self.grid[loc[XDIM], loc[YDIM], mobileLayer]==1:   return mobileLayer
        if self.grid[loc[XDIM], loc[YDIM], agentLayer]==1:    return agentLayer
        if self.grid[loc[XDIM], loc[YDIM], goalLayer]==1:     return goalLayer
        return -1 # <- flag for 'empty square'

    def getAllLocs(self, m_or_i):
        if m_or_i=='m':
          return [list(x) for x in np.argwhere(self.grid[:,:,mobileLayer]>0)]
        if m_or_i=='i':
          return [list(x) for x in np.argwhere(self.grid[:,:,immobileLayer]>0)]
            
    ''' Dump the entire grid representation, for debugging '''
    def dump_state(self):
        for i in range(NUM_LAYERS):
            print "Layer #"+str(i)+", the "+layer_names[i]+":"
            print self.grid[:,:,i], '\n'

    ''' Return an identical but distinct version of this state object '''
    def copy(self):
        s_p = state(self.gridsz)
        s_p.grid = np.copy(self.grid)
        s_p.a_loc = self.a_loc
        s_p.g_loc = self.g_loc
        return s_p

    ''' Use this function for testing equality between states. '''
    def equals(self, s_p):
        return np.array_equal(self.grid, s_p.grid) and self.gridsz==s_p.gridsz \
                and s_p.a_loc == self.a_loc and s_p.g_loc == self.g_loc

    ''' Get a boolean of whether or not I am a valid, consistent state. '''
    def checkValidity(self, long_version):
        if self.isImBlocked(self.a_loc)==1.0: return False
        if long_version: # <- check extra properties for new init. Bloated!!
            if self.isImBlocked(self.a_loc)==1.0: 
                print "FLAG 42"
                return False
            if self.isMoBlocked(self.a_loc)==1.0: 
                print "FLAG 25"
                return False
            for x in range(self.gridsz[XDIM]):
              for y in range(self.gridsz[YDIM]):
                if (self.grid[x, y, goalLayer] and \
                        self.grid[x, y, immobileLayer]): 
                    print "FLAG 95"
                    return False
                if (self.grid[x, y, goalLayer] and \
                        self.grid[x, y, mobileLayer]): 
                    print "FLAG 85"
                    return False
                if (self.grid[x, y, immobileLayer] and \
                        self.grid[x, y, mobileLayer]):
                    print "FLAG 50"
                    return False
                if ((x==0 or y==0 or x==self.gridsz[XDIM]-1 or y==self.gridsz[YDIM]-1)\
                        and self.grid[x, y, immobileLayer]==0.0): 
                    print "FLAG 05"
                    return False
        return True


class environment_handler(object):
    '''
    class that handles objects.
    '''
    def __init__(self, gridsize=None):
        if gridsize:
            self.gridsz = gridsize;
        self.allstates = []
    
    ''' Initialize a new state with post_state. '''
    def post_state(self, parameters):
        '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobiles_locs' in:
            {'borders' which fills only the borders, list of points}, 
            'mobiles_locs' = list of points.
        '''
        S = state(self.gridsz)
        S.post_agent(parameters['agent_loc'])
        S.post_goal(parameters['goal_loc'])
        S.post_immobile_blocks(parameters['immobiles_locs'])
        if 'mobiles_locs' in parameters.keys():
            S.post_mobile_blocks(parameters['mobiles_locs'])
        # S.dump_state();
        if not self.checkValidState(S, long_version=True):
            raise Exception("Invalid state attempted initialization: Flag 84")
        self.allstates.append(S)
        return S;
    
    ''' Helper: return a new location following an action '''
    def newLoc(self, loc, action):
        if (action in ['u','^',UDIR]): return (loc[XDIM],loc[YDIM]-1) # u and d are reversed
        if (action in ['d','v',DDIR]): return (loc[XDIM],loc[YDIM]+1) # because of file 
        if (action in ['r','>',RDIR]): return (loc[XDIM]+1,loc[YDIM])
        if (action in ['l','<',LDIR]): return (loc[XDIM]-1,loc[YDIM])

    ''' Assertion: Verifies that a state is consistent '''
    def checkValidState(self, State, long_version=False):
        if State==None or not type(State)==state: return False
        return State.checkValidity(long_version)

    ''' Determines whether an action is valid given a state '''
    def checkIfValidAction(self, State, action) :
        a_curloc = State.get_agent_loc();
        queryloc = self.newLoc(a_curloc, action)
        ''' First, check if the agent tries to push a block, that it is valid: '''
        if State.isMoBlocked(queryloc)==1.0:
            blockQueryloc = self.newLoc(queryloc, action)
            if State.isImBlocked(blockQueryloc)==1.0: return False
        ''' Second, check if the agent direction is not blocked by immobile:'''
        if State.isImBlocked(queryloc)==1.0: return False
        return True

    ''' Returns the State that should result from the action.  if the action 
    was invalid, it returns the original state. newState: a bool flag for if 
    you want to overwrite the input State or return a new one, with the 
    original untouched.  '''
    def performAction(self, State, action, newState=True):
        valid = self.checkValidState(State) and \
                self.checkIfValidAction(State, action)
        if not valid: return State # same state as before: current state.

        newAgentLoc = self.newLoc(State.get_agent_loc(), action)
        if newState:
            State_prime = State.copy()
        else:
            State_prime = State
        if State_prime.isMoBlocked(newAgentLoc)==1.0: 
            newBlockLoc = self.newLoc(newAgentLoc, action)
            State_prime.put_mobile_block(newAgentLoc, newBlockLoc)
        State_prime.put_agent(newAgentLoc)
        return State_prime
    ''' utilities '''
    def isGoalReached(self, State):
        if State==None:
            return False;
        return State.get_agent_loc() == State.get_goal_loc()
    def getActionValidities(self, s): return np.array(\
        [ self.checkIfValidAction(s, a) for a in all_actions ], dtype='float32')
    def getGridSize(self): return self.gridsz;

    '''  Debugging methods  '''
    def displayGameState(self, State=None):
        ''' Print the state of the game in a visually appealing way. '''
        for y in range(self.gridsz[YDIM]):
            l = []
            for x in range(self.gridsz[XDIM]):
                flag = State.getQueryLoc((x,y))
                if flag==agentLayer: l += '@'
                if flag==goalLayer: l += 'X'
                if flag==immobileLayer: l += '#'
                if flag==mobileLayer: l += 'O'
                if flag==-1: l += ' '
            print ' '.join(l)

    def printOneLine(self, State, mode='print', ret_Is=False):
        ''' Prints the state as succintly as possible '''
        s = ''; 
        s+= 'agent:'+str(State.get_agent_loc())+'\t'
        s+= ' goal:'+str(State.get_goal_loc())+'\t'
        Ms = State.getAllLocs('m')
        for m in Ms:
            s+= ' M:'+str(m)
        if len(Ms)==0:
            s+= ' [no M found]'
        s += '\t'
        if ret_Is:
          Is = State.getAllLocs('I')
          for i in Is:
            if (i[XDIM]>0 and i[XDIM]<State.gridsz[XDIM]-1 and \
                    i[YDIM]>0 and i[YDIM]<State.gridsz[YDIM]-1):
                s+= ' I:'+str(m)
          if len(Is)==0:
            s+= ' [no inner I found]'
        if mode=='print': print s; 
        elif mode=='ret': return s;
        else: print "ERR tag 82"

    def getStateFromFile(self, filename): return self._read_state_file(filename)
    def _read_state_file(self, fn):
        '''
        _read_init_state_file: this function takes a file (sourced from main's 
        directory path) and processes it as an initialized state file.
        Format: first line is the X-size of the map, second line is the Y-size,
        and the following lines are the y'th grid squares, X-long, with key:
            A = agent    G = goal    I = immobile block    M = mobile block
        '''
        with open(fn, 'r') as f:
            self.gridsz = (int(f.readline()), int(f.readline()))
            #revgs = (int(f.readline()), int(f.readline()))
            #self.gridsz = (revgs[1], revgs[0])
            parameters = {}
            parameters['immobiles_locs']=[]
            parameters['mobiles_locs']=[]
            for y in range(self.gridsz[YDIM]):
                read_data = f.readline().strip()
                for x,c in enumerate(read_data):
                    if c=='I': parameters['immobiles_locs'].append((x,y))
                    if c=='M': parameters['mobiles_locs'].append((x,y))
                    if c=='A': parameters['agent_loc'] = (x,y)
                    if c=='G': parameters['goal_loc'] = (x,y)
        self.states = []
        return self.post_state(parameters)

# flags for state components, which I think are mildly redundant, tag 90:
ARR=0; FLIPLR=1; FLIPUD=2; ROT90=3; ROT180=4; XSZ=5; YSZ=6; ALOC=7; GLOC=8;
class state_generator(object):
    ''' State generator class. Difficulty classes:
        v1: Empty map, 1 step from goal.    Implemented? N
        v2: Empty map, N steps from goal.   Implemented? N
        v3: ...
    '''
    def __init__(self, gridsz): 
        # For now, FIX gridsz so that a network architecture need not change
        self.gridsz = gridsz;
        self.state_diffs = {}
        self.components = {}
        self.prefabs = [] # named list of prefabs

    ''' Component prefab reader. (use the flags at line tagged 90!) :
        Each component prefab is an 'uninitialized' component of a map,
        a particular arrangement of objects that are not spatially fixed but
        are reproducible for various purposes.  KEY:
        - XSZ and YSZ are the size of the rectangular hull of the component
        - ARR is the array of the component as an indiactor array: the 
            first two dims are XSZ and YSZ and the third is NUM_LAYERS=4.
        - Boolean symmetry indicators: 
            LR if the component is invariant over flipping over Y axis.
            UD if the component is invariant over flipping over X axis.
            90 if the component is invariant over 90 degree rotations.
            180 if the component is invariant over 180 degree rotations, ie
            has LR and UD symmetry.  ''' 
    def ingest_component_prefabs(self, pathname):
        component_files = [ f for f in os.listdir(pathname) \
                if os.path.isfile(os.path.join(pathname, f))]
        for c in component_files:
            slot = {}
            A = self._read_component(pathname, c)
            slot[ARR] = A
            slot[XSZ] = A.shape[0]
            slot[YSZ] = A.shape[1]
            R1 = np.rot90(A, 1, axes=(0,1))
            R2 = np.rot90(A, 2, axes=(0,1))
            name = os.path.splitext(c)[0]
            slot[FLIPUD] = np.array_equal(A, np.fliplr(A)) # * note:
            slot[FLIPLR] = np.array_equal(A, np.flipud(A)) # reversed!
            slot[ROT90]  = np.array_equal(A, R1) and R1.shape==A.shape
            slot[ROT180] = np.array_equal(A, R2) and R2.shape==A.shape
            alocs = np.argwhere(A[:,:,agentLayer])
            if alocs.shape[0]>=1:
                slot[ALOC] = alocs[0]
                if alocs.shape[0]>1: raise Exception("Invalid state, tag73")
            glocs = np.argwhere(A[:,:,goalLayer])
            if glocs.shape[0]>=1:
                slot[GLOC] = glocs[0]
                if glocs.shape[0]>1: raise Exception("Invalid state, tag72")
            self.components[name] = slot
            self.prefabs.append(name)

    def _read_component(self, pathname, component):
        with open(os.path.join(pathname,component), 'r') as c:
            c_sz = (int(c.readline()), int(c.readline()))
            prefab = np.zeros((c_sz[XDIM], c_sz[YDIM], NUM_LAYERS))
            for y in range(c_sz[YDIM]):
                read_data = c.readline()
                for x,obj in enumerate(read_data):
                    if obj in LAYER_MAP:
                        prefab[x,y,LAYER_MAP[obj]]=1.0
        return prefab

    def _readable_component(self, c, print_or_ret):
        cmpnt = self.components[c]
        s = c+': { sz('+str(cmpnt[XSZ])+','+str(cmpnt[YSZ])+'); ' 
        if np.any(cmpnt[ARR][:,:,agentLayer]): s+= 'A: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,agentLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,goalLayer]): s+= 'G: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,goalLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,immobileLayer]): s+= 'I: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,immobileLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,mobileLayer]): s+= 'M: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,mobileLayer])).replace('\n','/')+'; '
        s+= 'Symms: '
        if cmpnt[FLIPLR]: s+= 'LR '
        if cmpnt[FLIPUD]: s+= 'UD '
        if cmpnt[ROT90]: s+= '90 '
        if cmpnt[ROT180]: s+= '180 '
        s+='}'
        if print_or_ret=='print': print s
        elif print_or_ret=='return': return s
            
    def generate_all_states_fixedCenter(self, version, env):
        if version=='v1': 
            return self._generate_v1('default_center', env)

    def generate_all_states_floatCenter(self, version):pass
    def generate_N_states_fixedCenter(self, version, replacement=False):pass
    def generate_N_states_floatCenter(self, version, replacement=False):pass

    ''' -- verify where: assert that the requested [where] location fits in 
        the specified map size and in which the non-blocked-out space is of 
        dims (field_x, field_y). '''
    def _verifyWhere(self, where, field_sz):
        field_x, field_y = field_sz
        if not (len(where)==2 and where[XDIM]>=1 and where[YDIM]>=1 and \
                where[XDIM]+field_x < self.gridsz[XDIM] and \
                where[YDIM]+field_y < self.gridsz[YDIM] ):
            print "Invalid center for v1 init:", where
            return None
        return where
    
    def _adjust_for_orientation(self, whichCmp, cmpOrien, field_sz, cmpLoc):
        ''' Rearrange the component's matrix based on requested orientation.'''
        cmpnt = self.components[whichCmp]
        c_arr = cmpnt[ARR]
        if cmpOrien[0]: c_arr = np.fliplr(c_arr)
        if cmpOrien[1]: c_arr = np.flipud(c_arr)
        c_arr = np.rot90(c_arr, cmpOrien[2], axes=(0,1))
        arr = np.zeros((field_sz[XDIM], field_sz[YDIM], NUM_LAYERS))
        if cmpOrien[2]%2==0:
            arr[ cmpLoc[XDIM] : cmpLoc[XDIM] + cmpnt[XSZ], \
                cmpLoc[YDIM] : cmpLoc[YDIM] + cmpnt[YSZ], :] = c_arr
        else:
            arr[ cmpLoc[XDIM] : cmpLoc[XDIM] + cmpnt[YSZ], \
                cmpLoc[YDIM] : cmpLoc[YDIM] + cmpnt[XSZ], :] = c_arr
        return arr

    def _initialize_component(self, template, rootloc, field_sz, \
                  whichCmp, cmpLoc, cmpOrien=(0,0,0), mode='arr'):
        ''' Enrichens the template via masking. No verifications! 
            cmpLoc: *relative* to the root location.
            cmpOrien: orientation of (flipUD?, flipLR?, 90*x rotation?)'''

        # Create new parameters dict that interfaces the state system
        if template==None and mode=='param':
            parameters = {}
            parameters['mobiles_locs'] = []
            parameters['immobiles_locs'] = []
            self._fill_Immobiles_dict(parameters, field_sz, rootloc)

        arr = self._adjust_for_orientation(whichCmp, cmpOrien, field_sz, cmpLoc)
        

        if mode=='param':
            # Set the (single) agent parameter
            alocs = np.argwhere(arr[:,:,agentLayer])
            if alocs.shape[0]==1: aloc = alocs[0]
            aloc[XDIM] += rootloc[XDIM]# + cmpLoc[XDIM] <- mystery fix...
            aloc[YDIM] += rootloc[YDIM]# + cmpLoc[YDIM]
            parameters['agent_loc'] = aloc

            # Set the (single) goal parameter
            glocs = np.argwhere(arr[:,:,goalLayer])
            if glocs.shape[0]==1: gloc = glocs[0]
            gloc[XDIM] += rootloc[XDIM]# + cmpLoc[XDIM]
            gloc[YDIM] += rootloc[YDIM]# + cmpLoc[YDIM]
            parameters['goal_loc'] = gloc

            # Set the immobile block parameters
            if not 'immobiles_locs' in parameters:
                parameters['immobiles_locs']=[]
            ilocs = np.argwhere(arr[:,:,immobileLayer])
            for iloc in ilocs:
                iloc[XDIM] += rootloc[XDIM] + cmpLoc[XDIM]
                iloc[YDIM] += rootloc[YDIM] + cmpLoc[YDIM]
                parameters['immobiles_loc'].append(iloc)

            # Set the mobile block parameters
            if not 'mobiles_locs' in parameters:
                parameters['mobiles_locs']=[]
            mlocs = np.argwhere(arr[:,:,mobileLayer])
            for mloc in mlocs:
                mloc[XDIM] += rootloc[XDIM] + cmpLoc[XDIM]
                mloc[YDIM] += rootloc[YDIM] + cmpLoc[YDIM]
                parameters['mobiles_loc'].append(mloc)

            return parameters

        elif mode=='arr':
            template[rootloc[XDIM] : rootloc[XDIM] + field_sz[XDIM], \
                     rootloc[YDIM] : rootloc[YDIM] + field_sz[YDIM], :] = arr
            return template

    '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobiles_locs' in:
            {'borders' which fills only the borders, list of points}, 
            'mobiles_locs' = list of points.
    '''
    def _fill_Immobiles_dict(self, parameters, field_shape, centerloc):
        ''' _fill_immobiles but with parameter dict retval '''
        parameters['immobiles_locs']=[]
        for a in range(self.gridsz[XDIM]):
            for b in range(self.gridsz[YDIM]):
                if a==0 or b==0 or a==self.gridsz[XDIM] or b==self.gridsz[YDIM] \
                    or a>=field_shape[XDIM]+centerloc[XDIM] \
                    or b>=field_shape[YDIM]+centerloc[YDIM]:
                        parameters['immobiles_locs'].append((a,b))
        return parameters

    def _generate_v1(self, rootloc, env):
        ''' V1, version one easiest state. These states are 3x3 and place the 
        agent directly next to the goal. '''
        field_shape = (3,3) # for all V1 states.
        if rootloc=='default_center': rootloc=(1,1)
        if not self._verifyWhere(rootloc, field_shape): sys.exit()
        states = []
        for x in range(field_shape[XDIM]-1):
            for y in range(field_shape[YDIM]):
                print '  ',x,y,'aloc, gloc:'
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,0,0), 'param')
                print sp['agent_loc'], sp['goal_loc']
                states.append(env.post_state(sp))
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,1,0), 'param')
                print sp['agent_loc'], sp['goal_loc']
                states.append(env.post_state(sp))
        for y in range(field_shape[YDIM]-1):
            for x in range(field_shape[XDIM]):
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,0,1), 'param')
                states.append(env.post_state(sp))
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,1,1), 'param')
                states.append(env.post_state(sp))
        print len(states)
        return states

    def _calculate_minsteps(self, s):pass # TODO



def test_script1():
    # this script tests the ability to generate all game states for vers1.
    foo = state_generator((10,10))
    env = environment_handler((10,10))
    foo.ingest_component_prefabs("./data_files/components/")
    X = foo.generate_all_states_fixedCenter('v1', env)

    for i,x in enumerate(X):
        env.displayGameState(x); print ''
    print "Number of states generated:", len(X)
    print "Above are all the possible valid game states that have a 3x3",
    print " grid in a fixed location in which the agent is directly next",
    print " to the goal (in any direction); that is, the first possible task."
    print '--------------------------------------------------------'
    print '--------------------------------------------------------'


def test_script2():
    # This example script demonstrates the ability for a 
    print "The following is a test example.  For reference, @ is the agent, X is"+\
      " the goal, O is a movable block, and # is an immovable block."

    for fn in ["./data_files/states/3x3-diag+M.txt"]:#, "./data_files/states/3x4-diag+M.txt"]:
        env = environment_handler()
        si = env.getStateFromFile(fn)
        print "\nvalid initial state:", not si==None;
        env.displayGameState(si); 
        s0 = env.performAction(si, 'd');  print "\naction: d, action success:", \
            env.checkIfValidAction(si, 'd');  env.displayGameState(s0);
        s1 = env.performAction(s0, 'd');  print "\naction: d, action success:", \
            env.checkIfValidAction(s0, 'd');  env.displayGameState(s1);
        s2 = env.performAction(s1, 'r');  print "\naction: r, action success:", \
            env.checkIfValidAction(s1, 'r');  env.displayGameState(s2);
        s3 = env.performAction(s2, 'r');  print "\naction: r, action success:", \
            env.checkIfValidAction(s2, 'r');  env.displayGameState(s3);
        s4 = env.performAction(s3, 'u');  print "\naction: u, action success:", \
            env.checkIfValidAction(s3, 'u');  env.displayGameState(s4);
        s5 = env.performAction(s4, 'd');  print "\naction: d, action success:", \
            env.checkIfValidAction(s4, 'd');  env.displayGameState(s5);
        s6 = env.performAction(s5, 'l');  print "\naction: l, action success:", \
            env.checkIfValidAction(s5, 'l');  env.displayGameState(s6);
        s7 = env.performAction(s6, 'u');  print "\naction: u, action success:", \
            env.checkIfValidAction(s6, 'u');  env.displayGameState(s7);
    print '--------------------------------------------------------'
    print "Above is a test script that demonstrates that the state-environment-actor ",
    print "situation is coherent and functional."

if __name__=='__main__':
    test_script1()
    test_script2()
    print "DONE"
