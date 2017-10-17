
'''     Template guide: 
            x   =   immobile block
            m   =   mobile block
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
TEMPLATE_R_U_RU_orig = '''   x x x x x x x x x  r
                        x x x x x x x x x  r
                        x x x x x x x x x  r
                        x x x . ! ! x x x  r
                        x x x . a ! x x x  r
                        x x x . . . x x x  r
                        x x x x x x x x x  r
                        x x x x x x x x x  r
                        x x x x x x x x x  e
                o: *    '''
TEMPLATE_R_U_RU = '''   x x x x x  r
                        x . ! ! x  r
                        x . a ! x  r
                        x . . . x  r
                        x x x x x  e
                o: D, *    '''
#TEMPLATE_R_U =    '''   x x x x x  r
#                        x . ! . x  r
#                        x . a ! x  r
#                        x . . . x  r
#                        x x x x x  e
#                o: D, *    '''
TEMPLATE_R_U =    '''   x x x x x x x  r
                        x m m m m m x  r
                        x m . ! . m x  r
                        x m . a ! m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''

TEMPLATE_RU =    '''    x x x x x x x  r
                        x m m m m m x  r
                        x m . . ! m x  r
                        x m . a . m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''
TEMPLATE_R_U_RU =    '''    x x x x x x x  r
                        x m m m m m x  r
                        x m . ! ! m x  r
                        x m . a ! m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''



'''   Printing legend for condensed mode:
    !  agent and goal       I  agent        -  immobile
    @  goal and mobile      *  goal         o  mobile
    <space>  empty          #  ERROR                        '''

# Warning: this function got all screwed up when copy-pasted.
def print_state(start_state, mode, print_or_ret='print'):
    S = ''
    if type(start_state)==np.ndarray:
        st = start_state
    else:
        st = start_state['state']
        S += str(mode+':')
    if mode=='matrices':
        for i in range(st.shape[-1]):
            S += str(st[:,:,i])
    if mode=='condensed':
        for y in range(st.shape[Y]):
            for x in range(st.shape[X]):
                if st[x,y,goalLayer] and st[x,y,agentLayer]: 
                    S += str('!')
                elif st[x,y,agentLayer]: 
                    S += str('I')
                elif st[x,y,goalLayer] and st[x,y,mobileLayer]:
                    S += str('@')
                elif st[x,y,goalLayer]: 
                    S += str('*')
                elif st[x,y,immobileLayer]: 
                    S += str('-')
                elif st[x,y,mobileLayer]: 
                    S += str('o')
                elif 0==np.sum(st[x,y,:]): 
                    S += str(' ')
                else: 
                    S += str('#')
                    print(S)
                # raise Exception("Error")
            S += str('\n')
    if not type(start_state)==np.ndarray:
        S += str("Flavor signal/goal id: ", start_state['flavor signal'])

    if print_or_ret=='print': print(S)
    else: return S


