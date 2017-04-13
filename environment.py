'''                                                                           |
Morgan Bryant, April 2017                                                     |
Environment class for establishing world states.                              |
'''
import numpy as np

agentLayer = 0;             UDIR=0
goalLayer = 1;              RDIR=1
immobileLayer = 2;          DDIR=2
mobileLayer = 3;            LDIR=3

class state(object):
    def __init__(self, gridsize, num_lyrs):
        self.gridsz = gridsize
        self.num_layers = num_lyrs
        self.grid = np.zeros((self.gridsz[0], self.gridsz[1], self.num_layers),
                dtype='float32')

    def A(self, (x, y)): return (agentLayer,x,y)
    def G(self, (x, y)): return (goalLayer,x,y)
    def S(self, (x, y)): return (immobileLayer,x,y)
    def A(self, (x, y)): return (mobileLayer,x,y)

    def _post(self, loc, whichLayer): # set grid to on
        assert(len(loc)==2)
        self.grid[loc[0], loc[1], whichLayer] = 1 
    def _del(self, loc, whichLayer):
        assert(len(loc)==2)
        self.grid[loc[0], loc[1], whichLayer] = 0 

    def get_agent_loc(self): return self.a_loc;
    def post_agent(self, agent_loc):
        self._post(agent_loc, agentLayer)
        self.a_loc = agent_loc
    def put_agent(self, new_loc):
        self._del(self.a_loc, agentLayer)
        self.post_agent(new_loc)

    def get_goal_loc(self): return self.g_loc;
    def post_goal(self, goal_loc):
        self._post(goal_loc, goalLayer)
        self.g_loc = goal_loc

    def isBlocked(self, loc): return self.grid[loc[0], loc[1], immobileLayer]
    def post_immobile_blocks(self, opts):
        if opts=='border':
            [ self._post((x,y), immobileLayer) for x in range(self.gridsz[0])\
                    for y in [0, self.gridsz[1]-1] ]
            [ self._post((x,y), immobileLayer) for x in [0, self.gridsz[0]-1] \
                    for y in range(1, self.gridsz[1]-1) ]
            
    def dump_state(self):
        for i in range(self.num_layers-1):
            print "Layer "+str(i)+":"
            print self.grid[:,:,i], '\n'




class environment_handler(object):
    '''
    class that handles objects.
    '''
    num_layers = 4; # agent, goal, immobile, mobile
    
    def __init__(self, gridsize):
        self.gridsz = gridsize;
        self.states = []

    def post_state(self, parameters):
        '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobile_loc' in:
            {'borders' which fills only the borders, }
        '''
        S = state(self.gridsz, self.num_layers)
        S.post_agent(parameters['agent_loc'])
        S.post_goal(parameters['goal_loc'])
        S.post_immobile_blocks(parameters['immobiles_loc'])
        if 'mobiles_loc' in parameters.keys():
            S.post_mobile_blocks(parameters['mobiles_loc'])
        # S.dump_state();
        self.states.append(S)
        return S;
    
    def adjLoc(self, loc, action):
        if (action in ['u','^',UDIR]): return (loc[0]+1,loc[1])
        if (action in ['r','>',RDIR]): return (loc[0],loc[1]+1)
        if (action in ['d','v',DDIR]): return (loc[0]-1,loc[1])
        if (action in ['l','<',LDIR]): return (loc[0],loc[1]-1)

    def checkValidState(self, state):
        if state==None: return False
        if state.isBlocked(state.get_agent_loc())==1.0: return False
        return True

    def checkIfValidAction(self, state, action) :
        # First, check if the agent direction is not blocked by immobile:
        a_curloc = state.get_agent_loc();
        queryloc = self.adjLoc(a_curloc, action)
        print a_curloc, '->', queryloc, 'with', action
        ##print 'is blocked?: ', state.isBlocked(queryloc);
        if state.isBlocked(queryloc)==0.0: return True
        # Second, check if the agent tries to push a block, that it is valid:
        pass # TODO stub

        return False
        #return state.isBlocked(queryloc)==0.0

    def perform_Action(self, state, action):
        ''' 
            Returns a new state if valid, else null if invalid state.
        '''
        valid = self.checkValidState(state) and \
                self.checkIfValidAction(state, action)
                
        print "isValidAction?\t\t", valid
        if not valid: return None;
        newAgentLoc = self.adjLoc(state.get_agent_loc(), action)
        state.put_agent(newAgentLoc)
        return state
        # state.dump_state()

    def isGoalReached(self, state):
        if state==None:
            return False;
        return state.get_agent_loc() == state.get_goal_loc()


# implementation test script:
env = environment_handler((5,5))

state0 = env.post_state( 
        {'agent_loc':(1,1), 'goal_loc':(2,2), 'immobiles_loc':'border'})
print "isGoalReached after init? ", env.isGoalReached(state0)
state1 = env.perform_Action(state0 ,'d');
print "isGoalReached after d? ", env.isGoalReached(state1)
print '----------------------------------------------------'

init_state = env.post_state( 
        {'agent_loc':(1,1), 'goal_loc':(2,2), 'immobiles_loc':'border'})
newState1 = env.perform_Action(init_state ,'u');
print "isGoalReached after u? ", env.isGoalReached(newState1)
newState2= env.perform_Action(newState1 ,'r');
print "isGoalReached after ur? ", env.isGoalReached(newState2)

print '----------------------------------------------------'

state0 = env.post_state( 
        {'agent_loc':(1,1), 'goal_loc':(2,2), 'immobiles_loc':'border'})
state1 = env.perform_Action(state0, 'r');
print "isGoalReached after r? ", env.isGoalReached(state1)
state2 = env.perform_Action(state1, 'r');
print "isGoalReached after rr? ", env.isGoalReached(state2)
state3 = env.perform_Action(state2, 'r');
print "isGoalReached after rrr? ", env.isGoalReached(state3)
state4 = env.perform_Action(state3, 'r');
print "isGoalReached after rrrr? ", env.isGoalReached(state4)
print '----------------------------------------------------'

print "DONE"
