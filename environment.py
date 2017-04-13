'''                                                                           |
Morgan Bryant, April 2017                                                     |
Environment class for establishing world states.                              |
'''
import numpy as np
import sys

agentLayer = 0;             UDIR=0;
goalLayer = 1;              RDIR=1;
immobileLayer = 2;          DDIR=2;
mobileLayer = 3;            LDIR=3;

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

    def __init__(self, gridsize, num_lyrs):
        self.gridsz = gridsize
        self.num_layers = num_lyrs
        self.grid = np.zeros((self.gridsz[0], self.gridsz[1], self.num_layers),
                dtype='float32')

    def _post(self, loc, whichLayer): # set grid to on
        assert(len(loc)==2)
        self.grid[loc[0], loc[1], whichLayer] = 1 
    def _del(self, loc, whichLayer):
        assert(len(loc)==2)
        self.grid[loc[0], loc[1], whichLayer] = 0 

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
        return self.grid[loc[0], loc[1], immobileLayer]
    def post_immobile_blocks(self, opts):
        if opts=='border':
            [ self._post((x,y), immobileLayer) for x in range(self.gridsz[0])\
                    for y in [0, self.gridsz[1]-1] ]
            [ self._post((x,y), immobileLayer) for x in [0, self.gridsz[0]-1] \
                    for y in range(1, self.gridsz[1]-1) ]
        elif type(opts)==list:
            for l in opts: self._post(l, immobileLayer);
        elif len(opts)==2:
            self._post(opts, immobileLayer);

    ''' Public methods for the MOBILE BLOCKS '''
    def isMoBlocked(self, loc): 
        return self.grid[loc[0], loc[1], mobileLayer]
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
        if self.grid[loc[0], loc[1], immobileLayer]==1: return immobileLayer
        if self.grid[loc[0], loc[1], mobileLayer]==1:   return mobileLayer
        if self.grid[loc[0], loc[1], agentLayer]==1:    return agentLayer
        if self.grid[loc[0], loc[1], goalLayer]==1:     return goalLayer
        return -1 # <- flag for 'empty square'
            
    ''' Dump the entire grid representation, for debugging '''
    def dump_state(self):
        for i in range(self.num_layers):
            print "Layer #"+str(i)+", the "+layer_names[i]+":"
            print self.grid[:,:,i], '\n'

    ''' Return an identical but distinct version of this state object '''
    def copy(self):
        s_p = state(self.gridsz, self.num_layers)
        s_p.grid = np.copy(self.grid)
        s_p.a_loc = self.a_loc
        s_p.g_loc = self.g_loc
        return s_p

    ''' Use this function for testing equality between states. '''
    def equals(self, s_p):
        return np.array_equal(self.grid, s_p.grid) and self.gridsz==s_p.grid_sz \
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
            for x in range(self.gridsz[0]):
              for y in range(self.gridsz[1]):
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
                if ((x==0 or y==0 or x==self.gridsz[0]-1 or y==self.gridsz[0]-1)\
                        and self.grid[x, y, immobileLayer]==0.0): 
                    print "FLAG 05"
                    return False
        return True


class environment_handler(object):
    '''
    class that handles objects.
    '''
    num_layers = 4; # agent, goal, immobile, mobile
    
    def __init__(self, gridsize=None, filename=None):
        if gridsize:
            self.gridsz = gridsize;
            self.states = []
        elif filename:
            self._read_init_state_file(filename)

    ''' Initialize a new state with post_state. '''
    def post_state(self, parameters):
        '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobiles_locs' in:
            {'borders' which fills only the borders, list of points}, 
            'mobiles_locs' = list of points.
        '''
        S = state(self.gridsz, self.num_layers)
        S.post_agent(parameters['agent_loc'])
        S.post_goal(parameters['goal_loc'])
        S.post_immobile_blocks(parameters['immobiles_locs'])
        if 'mobiles_locs' in parameters.keys():
            S.post_mobile_blocks(parameters['mobiles_locs'])
        # S.dump_state();
        if not self.checkValidState(S, long_version=True):
            print "Invalid state attempted initialization: Flag 84"
            raise Exception()
        self.states.append(S)
        return S;
    
    ''' Helper: return a new location following an action '''
    def newLoc(self, loc, action):
        if (action in ['u','^',UDIR]): return (loc[0]+1,loc[1])
        if (action in ['r','>',RDIR]): return (loc[0],loc[1]+1)
        if (action in ['d','v',DDIR]): return (loc[0]-1,loc[1])
        if (action in ['l','<',LDIR]): return (loc[0],loc[1]-1)

    ''' Assertion: Verifies that a state is consistent '''
    def checkValidState(self, State, long_version=False):
        if State==None or not type(State)==state: return False
        return State.checkValidity(long_version)

    ''' Determines whether an action is valid given a state '''
    def checkIfValidAction(self, state, action) :
        a_curloc = state.get_agent_loc();
        queryloc = self.newLoc(a_curloc, action)
        ''' First, check if the agent tries to push a block, that it is valid: '''
        if state.isMoBlocked(queryloc)==1.0:
            blockQueryloc = self.newLoc(queryloc, action)
            if state.isImBlocked(blockQueryloc)==1.0: return False
        ''' Second, check if the agent direction is not blocked by immobile:'''
        if state.isImBlocked(queryloc)==1.0: return False
        return True

    def performAction(self, State, action, newState=True):
        ''' Returns the State that should result from the action;
            if the action was invalid, it returns the original state.
            newState: a bool flag for if you want to overwrite the input State
            or return a new one, with the original untouched.
        '''
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

    def displayGameState(self, State=None):
        ''' Print the state of the game in a visually appealing way. '''
        for x in range(self.gridsz[0]):
            l = []
            for y in range(self.gridsz[1]):
                flag = State.getQueryLoc((x,y))
                if flag==agentLayer: l += '@'
                if flag==goalLayer: l += 'X'
                if flag==immobileLayer: l += '#'
                if flag==mobileLayer: l += 'O'
                if flag==-1: l += ' '
            print ' '.join(l)

    def isGoalReached(self, State):
        ''' query: if the agent is at the goal '''
        if State==None:
            return False;
        return State.get_agent_loc() == State.get_goal_loc()

    def _read_init_state_file(self, fn):
        '''
        _read_init_state_file: this function takes a file (sourced from main's 
        directory path) and processes it as an initialized state file.
        Format: first line is the X-size of the map, second line is the Y-size,
        and the following lines are the y'th grid squares, X-long, with key:
            A = agent    G = goal    I = immobile block    M = mobile block
        '''
        with open(fn, 'r') as f:
            self.gridsz = (int(f.readline()), int(f.readline()))
            parameters = {}
            parameters['immobiles_locs']=[]
            parameters['mobiles_locs']=[]
            for y in range(self.gridsz[1]):
                read_data = f.readline().strip()
                for x,c in enumerate(read_data):
                    if c=='I': parameters['immobiles_locs'].append((y,x))
                    if c=='M': parameters['mobiles_locs'].append((y,x))
                    if c=='A': parameters['agent_loc'] = (y,x)
                    if c=='G': parameters['goal_loc'] = (y,x)
        self.states = []
        self.default_state = self.post_state(parameters)
    



# implementation test script:



fn = "./3x3-diag+M.txt"
env = environment_handler(filename=fn)
si = env.default_state; print "\nvalid initial state:", not si==None;env.displayGameState(si); 
s0 = env.performAction(si, 'u');  print "\naction: u, action success:", not si.equals(s0), \
      env.checkIfValidAction(si, 'u'); env.displayGameState(s0);
s1 = env.performAction(s0, 'u');  print "\naction: u, action success:", not s0.equals(s1), \
      env.checkIfValidAction(s0, 'u');  env.displayGameState(s1);

s2 = env.performAction(s1, 'r');  print "\naction: r, action success:", not s1.equals(s2), \
      env.checkIfValidAction(s1, 'r');  env.displayGameState(s2);
s3 = env.performAction(s2, 'r');  print "\naction: r, action success:", not s2.equals(s3), \
      env.checkIfValidAction(s2, 'r');  env.displayGameState(s3);
s4 = env.performAction(s3, 'd');  print "\naction: d, action success:", not s3.equals(s4), \
      env.checkIfValidAction(s3, 'd');  env.displayGameState(s4);

s5 = env.performAction(s4, 'u');  print "\naction: u, action success:", not s4.equals(s5), \
      env.checkIfValidAction(s4, 'u');  env.displayGameState(s5);
s6 = env.performAction(s5, 'l');  print "\naction: l, action success:", not s5.equals(s6), \
      env.checkIfValidAction(s5, 'l');  env.displayGameState(s6);
s7 = env.performAction(s6, 'd');  print "\naction: d, action success:", not s6.equals(s7), \
      env.checkIfValidAction(s6, 'd');  env.displayGameState(s7);

print "DONE"
