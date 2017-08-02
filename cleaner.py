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

#class ExpAPI(object):
#    def __init__(self, experiment_name):
#        if experiment_name=='tse2007':
#            self.get_tse_environment()
#    def get_tse_environment(self):
#        env = env_h(gridsize=(9,9), 

X=0; Y=1;
#START_LOCS = [ (1,5), (5,1), (9,5), (5,9) ] # U L D R
#FLAV_LOCS = [ (3,3), (9,3), (6,4), (4,6), (2,7), (8,7) ]

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

''' Printing legend for condensed mode:
    !  agent and goal       I  agent        -  immobile
    @  goal and mobile      *  goal         o  mobile
    <space>  empty          #  ERROR                        '''
def print_state(start_state, mode):
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
                elif st[x,y,immobileLayer]: print '-', # '◊', 
                elif st[x,y,mobileLayer]: print 'o',
                elif 0==np.sum(st[x,y,:]): print ' ',
                else: 
                    print '#'
                    raise Exception("Error")
            print ''
    print "Signal: ", start_state['signal']







class ExpAPI(environment_handler3):
    def __init__(self, experiment_name, centr):
        gridsize = (11,11)
        environment_handler3.__init__(self, gridsize=\
                    { 'tse2007': (11,11) }[experiment_name], \
                    action_mode=centr )
        try:pass
        except:
            raise Exception('impl')
        self.state_gen = state_generator(gridsize)
        self.start_states = []
        self.gridsize=gridsize

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

    def set_starting_states(self, state_template):
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

        rx = [0,1,self.gridsize[X]-2, self.gridsize[X]-1]
        ry = [0,1,self.gridsize[Y]-2, self.gridsize[Y]-1]

        for start_box in start_locs:
            for flav_id, flavor_loc in enumerate(goal_locs):
                st = np.zeros( (self.gridsize[X], self.gridsize[Y], NUM_LAYERS))
                st[ start_box[X],  start_box[Y],   agentLayer ] = 1.0
                st[ flavor_loc[X], flavor_loc[Y],  goalLayer  ] = 1.0
                for mx,my in mobile_locs:
                    st[ mx, my, mobileLayer  ] = 1.0
                for bx,by in block_locs:
                    st[ bx, by, immobileLayer  ] = 1.0

#                for gx,gy in goal_locs:                 st[gx, gy,  mobileLayer] = 1.0
#                for x in range(self.gridsize[XDIM]):    st[x, ry, immobileLayer] = 1.0
#                for y in range(self.gridsize[YDIM]):    st[rx, y, immobileLayer] = 1.0
#                for _x,_y in start_locs:                st[_x, _y, immobileLayer] = 0.0
                self.start_states.append( { 'signal': flav_id,
                    'state': st})
#            print self.start_states[-1]['state'][:,:,agentLayer]
        rnd_state = self.start_states[np.random.choice(range(24))]
        print_state(rnd_state, 'condensed')


for centr in ['egocentric', 'allocentric']:
    ExpAPI('tse2007', centr).set_starting_states(TEMPLATE_TSE)
