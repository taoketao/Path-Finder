import sys, os, time 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, font_manager, patches
from math import ceil, floor
from functools import reduce
from reinforcement_batch import *

ACTION_NAMES = { 0:"U", 1:'R', 2:"D", 3:'L' } 
UDIR = 0; RDIR = 1; DDIR = 2; LDIR = 3
MAX_DISPLAY_EPOCHS = 100



def get_time_str(path, prefix=''):
    t=time.localtime()
    return os.path.join(path, prefix+str(t[1])+'_'+str(t[2])+'_'+str(t[0])+\
            '_'+str(t[3]).zfill(2)+str(t[4]).zfill(2))

def smooth(arr, smoothing):
    if smoothing==1: return arr
    try: int(smoothing); assert(smoothing>0);
    except: raise Exception("bad smoothing value: "+str(smoothing))

    if not arr.shape[0]%smoothing==0: 
        print(("Warning: smoothing factor does not evenly divide array: "+\
                str(arr.shape[0])+' & '+str(smoothing)))

    Arr = np.empty(shape=arr.shape)
    for i in range(arr.shape[0] // smoothing):
        arr_val = np.mean(arr[i*smoothing:(i+1)*smoothing,:], axis=0)
        for smth in range(smoothing):
            Arr[i*smoothing+smth,:] = arr_val
    return Arr


def display_curriculum_curves(curr, disp_or_save='disp'):
    ''' Displays a curriculum as object curves for different groups. '''
    pass








    
def save_as_plot1(fns, lr=None, mna=None, nsamples=None, which='l', \
        div=1.0, delete_npy=False, smoothing=20):
    # all parameters should be strings

    # shape convention: (2,x,y,2) where the first 2 is losses & nsteps, x is
    # nsamples, y is max_num_actions, and the last 2 is train / test.
    print(fns)
    if type(fns)==str:
        tmp = fns
        fns = [tmp]
    files = []
    for fn in fns:
        if fn in files: continue
        files.append(fn)
        x = np.load(fn)
        try:
            lr = fn.split('lr')[1].split('-')[0]
            print(("lr:"+str(lr)))
        except: lr=0
        try:
            mna = fn.split('mna-')[1].split('-')[0]
            print(("mna:"+str(mna)))
        except: mna=0
        try:
            nsamples = max(fn.split('nsamples')[1].split('.npy')[0],\
                            fn.split('nsamps')[1].split('-')[0])
            print(("nsamples:"+str(nsamples)))
        except:  nsamples=0

        #if not nsamples: nsamples = str(x.shape[1])
        try:
            losses, nsteps, rewards = x;
        except:
            try: 
                losses, nsteps = x;
            except:
                losses = x;
        #print "SHAPES:", losses.shape, nsteps.shape
        factor=20; lmax = (losses.shape[1]//factor)*factor
        if not which=='l':
            meanmean = np.mean(rewards[:,:lmax,1], axis=0).reshape(\
                    rewards.shape[1]//factor, factor)
            meanmean = np.mean(meanmean, axis=0)
            X = np.repeat(meanmean, rewards.shape[1]//factor)
        if 'O' in which:
            plt.plot(X, c='orange')
            o_string = ", orange=interpolated test n steps"
        else:
            o_string = ""
        if 'R' in which:
            plt.plot(np.mean(rewards[:,:,0], axis=0), c='red', linestyle='-') 
            # avg train R
            plt.plot(np.mean(rewards[:,:,1], axis=0), c='black', linestyle='-') 
            # avg test R
        if 'l'==which:
            plt.plot(np.mean(smooth(losses[0,:,:], smoothing), axis=1), \
                    c='red', linestyle='-') 

        if 'L' in which:
            plt.plot(np.mean(losses[:,:,0], axis=0), c='blue', linestyle='-') 
            # train loss
            plt.plot(np.mean(losses[:,:,1], axis=0), c='yellow', linestyle='-') 
            # test loss
        if 'S' in which:
            print(nsteps.shape)
            plt.plot(np.mean(nsteps[:,:,0], axis=0), c='green', linestyle='-') 
            # avg train steps
            plt.plot(np.mean(nsteps[:,:,1], axis=0), c='purple', linestyle='-') 
            # avg test steps
        if 'R' in which:
            plt.xlabel("Epoch.  blue=train err, yellow=test err,\n green=avg "+\
                    "train steps, purple=avg test steps"+o_string+\
                    "\nred: avg train reward, black: avg test reward")
        elif 'l' in which:
            plt.xlabel("test err per epoch, smoothed in chunks of "+str(smoothing))
        else:
            plt.xlabel("Episode.  blue=train err, yellow=test err,\n green=avg "+\
                       "train steps, purple=avg test steps"+o_string)

        if not 'l' in which:
          plt.title("Training and testing error averaged over "+str(nsamples)+\
                " samples with lr "+lr)
          plt.ylabel("Avg reward, loss, num actions taken.  Max num actions: "+str(mna))
        else:
          plt.title("Testing error averaged over "+str(nsamples)+\
                " samples with lr "+lr, fontsize=12)
          plt.ylabel("Avg loss.  Max num actions: "+str(mna))

        plt.tight_layout(1.2)
        #plt.ylim(-0.1, 1.1)
        #plt.ylim(bottom=0)
        plt.yscale('log')
        plt.ylim(top=1.5)
        plt.savefig(fn[:-4]+'.png')
        #plt.savefig(fn[:-4]+'.pdf', format='pdf')
        #plt.show()
#        plt.savefig(fn[:-4])
        plt.close()
        if delete_npy:
            os.remove(fn)
    print("Success: last file stored at", fn[:-4])


def save_as_Qplot2(mat, save_to_loc):
    f,ax = plt.subplots( 2,4, sharex=True, sharey=True)
    f.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
    ax[0,0].plot(mat[0,:,0,1], c='red')  # test,  U, Loss
    ax[0,0].plot(mat[0,:,0,0], c='blue') # train, U, Loss
    ax[1,0].plot(mat[1,:,0,1], c='red')  # test,  U, Q
    ax[1,0].plot(mat[1,:,0,0], c='blue') # train, U, Q

    ax[0,1].plot(mat[0,:,1,1], c='red')  # test,  R, Loss
    ax[0,1].plot(mat[0,:,1,0], c='blue') # train, R, Loss
    ax[1,1].plot(mat[1,:,1,1], c='red')  # test,  R, Q
    ax[1,1].plot(mat[1,:,1,0], c='blue') # train, R, Q

    ax[0,2].plot(mat[0,:,2,1], c='red')  # test,  d, Loss
    ax[0,2].plot(mat[0,:,2,0], c='blue') # train, d, Loss
    ax[1,2].plot(mat[1,:,2,1], c='red')  # test,  d, Q
    ax[1,2].plot(mat[1,:,2,0], c='blue') # train, d, Q

    ax[0,3].plot(mat[0,:,3,1], c='red')  # test,  l, Loss
    ax[0,3].plot(mat[0,:,3,0], c='blue') # train, l, Loss
    ax[1,3].plot(mat[1,:,3,1], c='red')  # test,  l, Q
    ax[1,3].plot(mat[1,:,3,0], c='blue') # train, l, Q
    f.suptitle("Qval of actions, loss of actions over epochs.\n Blue: train, Red: test")

    ax[1,0].set_xlabel("Up")
    ax[1,1].set_xlabel("Right")
    ax[1,2].set_xlabel("Down")
    ax[1,3].set_xlabel("Left")
    ax[0,0].set_ylabel("Losses")
    ax[1,0].set_ylabel("Q values")
    f.set_size_inches(20,6)
    plt.plot()
    #plt.show()
#    plt.savefig(save_to_loc+'.pdf', dpi=100, format='pdf')
    plt.savefig(save_to_loc+'.png', dpi=100)
    plt.close()

def save_as_Qplot3(mat, save_to_loc, trialinfo='[none provided]'):
    f,ax = plt.subplots(3,4,sharex=True)
    #f.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
    for a in range(4):
        for LQR in range(3):
            ax[LQR,a].plot(mat[LQR,:,a,0], c='red')
            ax[LQR,a].plot(mat[LQR,:,a,1], c='blue')
#            ax[LQR,a].get_xticklabels()
#            ax[LQR,a].get_yticklabels()
            plt.setp(ax[LQR,a].get_xticklabels(), visible=True)
            plt.setp(ax[LQR,a].get_yticklabels(), visible=True)
            ax[LQR,a].minorticks_on()
    f.suptitle("Reward, Qval, Loss of actions over epochs.  Blue: train, Red: test"+\
            "\n Trial info: "+trialinfo)
    ax[2,0].set_xlabel("Up", fontsize=16)
    ax[2,1].set_xlabel("Right", fontsize=16)
    ax[2,2].set_xlabel("Down", fontsize=16)
    ax[2,3].set_xlabel("Left", fontsize=16)
    ax[0,0].set_ylabel("Reward", fontsize=16)
    ax[1,0].set_ylabel("Losses", fontsize=16)
    ax[2,0].set_ylabel("Q values", fontsize=16)
    f.set_size_inches(20,9)
    plt.plot()
    #plt.show()
#    plt.savefig(save_to_loc+'.pdf', dpi=100, format='pdf')
    print(save_to_loc)
    plt.savefig(save_to_loc+'.png', dpi=100)
    plt.close()









def _dist(x,y): return abs(x[0]-y[0])+abs(x[1]-y[1])

def _make_str2(states, gridsz, trte, c1, c2):
    if not trte==-1: raise Exception('not implemented yet')
    s='\nStates:\n'
    for which in range(int(np.ceil(len(c1)/3.0))):
        s += '\n'
        for i in range(gridsz[0]):
            for j in range(gridsz[1]):
                S = states[which*2]
                if (i==S[1] and j==S[0]):   s += 'A'
                elif (i==S[3] and j==S[2]): s += 'G'
                else: s += '-'
            s+='      '
            for j in range(gridsz[1]):
                S = states[which*2+1]
                if (i==S[1] and j==S[0]):   s += 'A'
                elif (i==S[3] and j==S[2]): s += 'G'
                else: s += '-'
            s+='      '
            for j in range(gridsz[1]):
                S = states[which*2+2]
                if (i==S[1] and j==S[0]):   s += 'A'
                elif (i==S[3] and j==S[2]): s += 'G'
                else: s += '-'
            if not i==gridsz[0]-1:
                s += '\n'
        s += '\n'
    return s


def _make_str3(gs, make_states,gridsz, grp_colors, used_clrs):
    ndivs = 4
    state_legend = []
    grp_ittrs = [0]*40
    gsg = gs.get_geometry()
    for i in range(len(make_states)):
        ax_tmp = plt.subplot(gs[2*(i//ndivs), 8+i%ndivs])
        ax_tmp.get_xaxis().set_visible(False)
        ax_tmp.get_yaxis().set_visible(False)
        s_ = ''
        grp, S = make_states[i]
        for y in range(gridsz[0]):
            for x in range(gridsz[1]):
                if (x==S[0] and y==S[1]):   s_ += 'A'
                elif (x==S[2] and y==S[3]): s_ += 'G'
                else: s_ += '-'
            s_ += '\n'

        #state_legend[i].text(0, -1.0, s_, \
        ax_tmp.text(0, -1.0, s_, \
                horizontalalignment='left',\
                verticalalignment='center', family='monospace')
        #state_legend[i].add_patch( patches.Rectangle\
        ax_tmp.add_patch( patches.Rectangle\
                ( (0,0), 1, 1, \
                color=grp_colors[grp[0]-1][grp_ittrs[grp[0]]]))

        grp_ittrs[grp[0]] += 1
    ax_tmp = plt.subplot(gs[2*(len(make_states)//ndivs), 8+len(make_states)%ndivs])
    ax_tmp.set_axis_off()
#    ax_tmp.get_xaxis().set_visible(False)
#    ax_tmp.get_yaxis().set_visible(False)
    return ''



def get_lex_id(state):
    ax,ay,gx,gy = state
    s=''.join( ['d' for _ in range(gy-ay)] + ['l' for _ in range(ax-gx)] \
              +['r' for _ in range(gx-ax)] + ['u' for _ in range(ay-gy)])
    while len(s)<2: s = '_'+s
    return s

def _make_str(s_id, gridsz, trte, debug=False):
    if trte==0: s ='train\n'
    if trte==1: s ='test \n'
    if trte==-1: s ='    '
    for i in range(gridsz[0]):
        for j in range(gridsz[1]):
            if (i==s_id[1] and j==s_id[0]):
                s += 'A'
            elif (i==s_id[3] and j==s_id[2]):
                s += 'G'
            else:
                s += '-'
        if not i==gridsz[0]-1:
            s += '\n'
            if trte==-1: s += '     '
    if debug:
        if _dist((s_id[0], s_id[1]), (s_id[2], s_id[3])):
            s += '\n('+str(s_id[2]-s_id[0])+', '+str(s_id[3]-s_id[1])+')\n'
        s += str(s_id)
    return s

def get_group(st, statemap, l):
    lex = get_lex_id(st)
    grp=None
    for j in range(l):
        if statemap[j]['lex_id']==lex:
            return statemap[j]['group']
    return grp

def save_as_successes(s, tr, te, states=None, smoothing=10, centric=None,\
        tr2=None, te2=None, curr=None, statemap=None, tf=None, dest=None):
    if states==None:
        raise Exception()
    #f, ax = plt.subplots(lX*2, lY*2, sharex=True)
    if centric in ['allocentric','egocentric']:
        twoplots = False
    elif type(centric)==list and len(centric)==2: 
        twoplots = True
    else:
        raise Exception("Please tell me if this is egocentric, allocentric, etc")

    Tr = smooth(tr, smoothing)
    if tf: Te = smooth(te, max(4, smoothing//tf))
    else:  Te = smooth(te, smoothing)
    if twoplots: 
        Tr2 = smooth(tr2, smoothing)
        if tf: 
            try: 
                Te2 = smooth(te, smoothing//(tf//2))
            except: 
                Te2 = smooth(te, smoothing)
        else:  Te2 = smooth(te, smoothing)

    #f,ax = plt.subplots(2,2, sharex=True, sharey=True)
    gs = gridspec.GridSpec(8, 12)
    plt.subplots_adjust(\
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
    ax={}
    ax[(0,0)] = plt.subplot(gs[ :4, :4])
    ax[(0,1)] = plt.subplot(gs[ :4,4:8], sharey = ax[(0,0)])
    ax[(1,0)] = plt.subplot(gs[4: , :4], sharey = ax[(0,0)])
    ax[(1,1)] = plt.subplot(gs[4: ,4:8], sharey = ax[(0,0)])
    plt.gca().set_ylim([-0.1,1.1])


    if Tr.shape[1]==4:
        colors = ['blue','red','green','khaki']
        darkcolors = ['dark'+c for c in colors]
    elif Tr.shape[1]>4:
        colors = ['#ccccff','#9999ff','#6666ff','#3333ff','#0000cc','#000080',\
            '#ccffff','#80ffff','#00ffff','#00cccc','#009999','#008080'] # blues
        darkcolors = ['#ffcccc','#ff8080','#ff1a1a','#e60000','#cc0000','#990000',\
            '#ffe6f0','#ffcce0','#ff99c2','#ff66a3','#ff1a75','#e6005c'] # reds
    else: raise Exception(str(Tr.shape))

    if not curr==None: 
        accessible_states = reduce(set.union, [v for v in curr.groups.values()])
#        grp1_colors_tr = ['#ff99c2','#ffe6f0','#ff1a75','#ffcce0','#ff66a3','#e6005c'] # pinks
#        grp2_colors_tr = ['#00cccc','#008080','#ccffff','#00ffff','#009999','#80ffff'] # cyans
#        grp3_colors_tr = ['#cfffcc','#afffa9','#9ae095','#7fb77b','#5e895b','#496b47'] # lightgreens
#        grp1_colors_te = ['#ff1a1a','#ffcccc','#990000','#ff8080','#e60000','#cc0000'] # reds
#        grp2_colors_te = ['#000080','#6666ff','#9999ff','#3333ff','#ccccff','#0000cc'] # blues
#        grp3_colors_te = ['#ffde7a','#ffd145','#ffc000','#dba602','#c19204','#987301'] # oranges
        grp_colors =[['#3333ff','#000080','#9999ff','#ccccff','#0000cc','#6666ff'], # blues
                     ['#e60000','#990000','#ff8080','#ffcccc','#cc0000','#ff1a1a'], # reds
                     ['#dba602','#987301','#ffd145','#ffde7a','#c19204','#ffc000'], # oranges
                     ['#7fb77b','#496b47','#afffa9','#cfffcc','#5e895b','#9ae095'], # lightgreens
                     ['#00cccc','#008080','#80ffff','#ccffff','#009999','#00ffff'], # cyans
                     ['#ff66a3','#e6005c','#ffcce0','#ffe6f0','#ff1a75','#ff99c2']] # pinks
        g1,g2,g3 = 0,0,0
    else: accessible_states='all'

    used_clrs = []
#    incl = np.zeros(Tr.shape, dtype=bool)
    incl = []
    for ind, S in statemap.items():
        if len(S['group'])>0:
            print(ind, S['group'], S['lex_id'])
            incl.append((ind, S))
    if not accessible_states=='all':
        for i,S in incl:
            grp = S['group']
    #STATES = [pickle.load(open(os.path.join(dest, 'statesdir','state_'+str(i)),'rb')) for i in range(12)]
            if len(grp)==0: # ie, no group
                continue
            if not grp: raise Exception(str(states[i]))
            if grp[0]==1: 
                clr = grp_colors[0][g1]
                used_clrs.append( (1, g1) )
                g1+=1
            if grp[0]==2: 
                clr = grp_colors[1][g2]
                used_clrs.append( (2, g2) )
                g2+=1
            if grp[0]==3: 
                clr = grp_colors[2][g3]
                used_clrs.append( (3, g3) )
                g3+=1
            ax[(0,0)].plot(Tr[:,i]+i*5e-4*(-1)**i, c=clr)
            ax[(0,1)].plot(Te[:,i]+i*5e-4*(-1)**i, c=clr)
            if twoplots:
                ax[(0,0)].plot(Tr2[:,i]+i*5e-4*(-1)**i, c=clr)
                ax[(0,1)].plot(Te2[:,i]+i*5e-4*(-1)**i, c=clr)
    else:
        for i in range(len(states)):
            ax[(0,0)].plot(Tr[:,i], c=colors[i])
            ax[(0,1)].plot(Te[:,i], c=colors[i])
            if twoplots:
                ax[(0,0)].plot(Tr2[:,i], c=darkcolors[i])
                ax[(0,1)].plot(Te2[:,i], c=darkcolors[i])
#    print(incl,states, [states[i[0]] for i in incl]) 
    inds = [i[0] for i in incl]
    if not accessible_states=='all':
        ax[(1,0)].plot(np.mean(Tr[:,inds], axis=1)+np.random.random(\
                (Tr.shape[0],))*0.03, c='orange')
        ax[(1,1)].plot(np.mean(Te[:,inds], axis=1)+np.random.random(\
                (Te.shape[0],))*0.03, c='orange')
        if twoplots:
            ax[(1,0)].plot(np.mean(Tr2[:,inds], axis=1)+np.random.random(\
                    (Tr2.shape[0],))*0.03, c='black')
            ax[(1,1)].plot(np.mean(Te2[:,inds], axis=1)+np.random.random(\
                    (Te2.shape[0],))*0.03, c='black')
    else:
        ax[(1,0)].plot(np.mean(Tr, axis=1), c='orange')
        ax[(1,1)].plot(np.mean(Te, axis=1), c='orange')
        if twoplots:
            ax[(1,0)].plot(np.mean(Tr2, axis=1), c='black')
            ax[(1,1)].plot(np.mean(Te2, axis=1), c='black')

    if curr and curr.which_ids=='r, u, ru-diag only':
        ax[(1,0)].plot([2*(3**-1)]*Tr.shape[0], c='green')
        ax[(1,1)].plot([2*(3**-1)]*Tr.shape[0], c='green')
    elif curr and curr.which_ids=='all diag':
        ax[(1,0)].plot([0.5]*Tr.shape[0], c='green')
        ax[(1,1)].plot([0.5]*Tr.shape[0], c='green')
    elif curr and curr.which_ids=='r or u only':
        ax[(1,0)].plot([0.4]*Tr.shape[0], c='green')
        ax[(1,1)].plot([0.4]*Tr.shape[0], c='green')
        ax[(1,0)].plot([0.8]*Tr.shape[0], c='green')
        ax[(1,1)].plot([0.8]*Tr.shape[0], c='green')
    elif curr and curr.which_ids=='any1step, u r diag':
        ax[(1,0)].plot([0.8]*Tr.shape[0], c='green')
        ax[(1,1)].plot([0.8]*Tr.shape[0], c='green')
    else:
        ax[(1,0)].plot([3**-1]*Tr.shape[0], c='green')
        ax[(1,1)].plot([3**-1]*Tr.shape[0], c='green')
    ax[(0,0)].set_ylabel("accuracy per epoch by start state")
    ax[(1,0)].set_ylabel("accuracy per epoch")
    ax[(1,0)].set_xlabel("Training")
    ax[(1,1)].set_xlabel("Testing")
    plt.gcf().set_size_inches(16,10)
    s2 = "Successes taken per state, per epoch."
    if not states==None and Tr.shape[1]==4 and acccessible_states=='all':
        for sc in range(len(states)):
            s2 += '\n    '+colors[sc]+':\n'+ _make_str(states[sc],(5,5),-1)
    if not accessible_states=='all':
        make_states = []
        for st in states:
            grp = get_group(st, statemap, Tr.shape[1])
            if len(grp)==0: # ie, no group
                continue
            c = { 1: 'pinks/reds', 2: 'cyans/blues', 3: 'lightgreens/oranges' }[grp[0]]
            make_states.append((grp,st))
        #cnames =['pink/red   ','cyan/blue  ','lightgreens/oranges'] 
        s2 += '\n'+ _make_str3(gs, make_states,(7,7),grp_colors,used_clrs)
    elif not states==None and Tr.shape[1]>4:
        s2 += _make_str2(states,(7,7), -1, colors, darkcolors)


    if twoplots:
        if Tr.shape[1]==4:
            s2 += '\nLight / orange: '+centric[0]
            s2 += '\nDark / black: '+centric[1]
            fs = 14
        elif not accessible_states=='all':
            s2 += '\norange: '+centric[0]
            s2 += '\nblack: '+centric[1]
            fs = 10
        elif Tr.shape[1]>4:
            s2 += '\nblues: '+centric[0]
            s2 += '\nreds: '+centric[1]
            fs = 10
    else: fs=10
    ax[(1,1)].text(1.2*Tr.shape[0],0, s2, fontsize=fs, family='monospace')
    s.replace('-0','TMP').replace('_',':').replace('-','  ').replace('TMP','-0')
    plt.gcf().suptitle( s[s.find('v'):s.find('net')] +'\n'+\
            s[s.find('net'):s.find(' successes')])

    plt.plot()
    plt.savefig(s+'.png')
    plt.close()






def save_as_state_actions1(state_indices, state_mat, gridsz, dest):
    sorted_states_indices = sorted(list(state_indices.keys())) # dict sorted
    print(state_mat.shape, len(sorted_states_indices))
    
    lY = int(round(len(sorted_states_indices)**0.5))
    lX = int(ceil(len(sorted_states_indices)**0.5))
    tiles = np.arange(lX*lY).reshape(lY,lX)
    # ^ tile states roughly squarely; 

    gs = gridspec.GridSpec(lX*2, lY*2)
    #f, ax = plt.subplots(lX*2, lY*2, sharex=True)
    ax = {}
    for i in range(lX):
        for j in range(lY):
            for trte in range(2):
                ax[(i,j,trte)] = plt.subplot(gs[2*j+trte, 2*i:2*i+2])
                ax[(i,j,trte)].get_yaxis().set_visible(False)

    y_font = font_manager.FontProperties()
    y_font.set_size('small')
    y_font.set_family("monospace")

    nepochs = state_mat.shape[1]
    if nepochs > MAX_DISPLAY_EPOCHS:
        #ep_skip = floor(nepochs/MAX_DISPLAY_EPOCHS)
        effective_eps =np.linspace(0,nepochs-1,MAX_DISPLAY_EPOCHS)
    else:
        effective_eps = list(range(nepochs))
    if effective_eps[-1]==nepochs: effective_eps[-1] -= 1
    effective_eps = [int(floor(e)) for e in effective_eps]

    plt.subplots_adjust(\
            left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=2, hspace=0.5)


    for st_i, St_key in enumerate(sorted_states_indices):
        for trte in range(2):
            data = state_mat[:,effective_eps, state_indices[St_key],trte]

            clrs = [(0,'blue'), (1,'red'), (2,'green'), (3,'yellow')]
            for ci,clr in clrs:
                ax[st_i // lX, st_i % lY, trte].plot(effective_eps, \
                            data[ci,:], color=clr)
                ax[st_i // lX, st_i % lY, trte].fill_between(effective_eps, \
                        data[ci,:], color=clr)
                #np.random.shuffle(clrs)

            ax[st_i//lX,st_i%lY,trte].text(-0.35*nepochs,0,\
                _make_str(St_key, gridsz, trte), fontsize=9, family='monospace')

    plt.gcf().set_size_inches(4*lX,4*lY)
    plt.gcf().suptitle("Actions taken per state, per epoch. BLUE:"+\
            ACTION_NAMES[0]+"; RED:"+ACTION_NAMES[1]+"; GREEN:"+\
            ACTION_NAMES[2]+"; YELLOW:"+ACTION_NAMES[3])
    
    s = " / "+str(nepochs // MAX_DISPLAY_EPOCHS) \
                if nepochs>MAX_DISPLAY_EPOCHS else ''
    ax[0,lY-1,1].set_xlabel("Epoch")

    plt.plot()
    #plt.show()
    plt.savefig(dest+'.png')
    plt.close()
        



def save_as_plot(fns, lr, mna, nsamples=None):
    # all parameters should be strings

    # shape convention: (2,x,y,2) where the first 2 is losses & nsteps, x is
    # nsamples, y is max_num_actions, and the last 2 is train / test.
    if type(fns)==str:
        tmp = fns
        fns = [tmp]
    files = []
    for fn in fns:
        if fn in files: continue
        files.append(fn)
        x = np.load(fn)
        if not nsamples: nsamples = str(x.shape[1])
        losses, nsteps = x;
        plt.plot(np.mean(losses[:,:,0], axis=0), c='blue', linestyle='-') # train err
        plt.plot(np.mean(losses[:,:,1], axis=0), c='yellow', linestyle='-') # test err
        plt.plot(np.mean(nsteps[:,:,0], axis=0), c='green', linestyle='-') # avg train steps
        plt.plot(np.mean(nsteps[:,:,1], axis=0), c='purple', linestyle='-') # avg test steps
        plt.title("Training and testing error averaged over "+nsamples+" samples with lr "+lr)
        plt.xlabel("Episode.  blue=train err, yellow=test err,\n green=avg train steps, purple=avg test steps")
        plt.ylabel("Avg reward // avg num actions taken.  Max num actions: "+mna)
        plt.tight_layout(1.2)
#        plt.ylim(0, 4)
#        plt.savefig(fn[:-4]+'.pdf', format='pdf')
        plt.savefig(fn[:-4]+'.png')
        plt.close()

    print("Success: last file stored at", fn[:-4])


def freeze_dict(d):
    if type(d)==set: return '<'+str(', '.join([str(di) for di in sorted(d)]))+'>'
    if type(d)==tuple and len(d)==1: return 'g'+str(d[0])
    if not type(d)==dict: return str(d)
    return '{'+', '.join([freeze_dict(k)+':'+freeze_dict(v) for k,v in sorted(d.items())])+'}'


def get_attributes(s_,i, curr):
    s=s_.strip()
    s=s[s.rfind('/')+1:]
    sl = s.split('-')
    attrs = {}
    itr = 0
    while itr<len(sl):
        if 'mna' in sl[itr]:
            attrs['mna'] = 'mna'+sl[itr][3:]
            while len(attrs['mna'])<30: attrs['mna'] += ' '
            itr += 1;continue
        if 'lr' == sl[itr][:2]:
            attrs['lr'] = 'lr'+sl[itr][2:]+'-'+sl[itr+1]
            while len(attrs['lr'])<7: attrs['lr'] += ' '
            itr += 2;continue
        if 'nepochs' == sl[itr][:7]:
            attrs['nepochs'] = 'epoch'+sl[itr][7:]
            while len(attrs['nepochs'])<13: attrs['nepochs'] += ' '
            itr += 1;continue
        if '(' == sl[itr][0] and ')'==sl[itr][-1]:
            attrs['gamesize'] = sl[itr]
            itr += 1;continue
        if 'eps' == sl[itr][:3]:
            if sl[itr][-1]=='e':
                attrs['eps'] = 'eps'+sl[itr][4:]+sl[itr+1]
                itr += 2;continue
            else:
                attrs['eps'] = 'eps'+sl[itr][4:]
                itr += 1;continue
        if 'nsamps_' in sl[itr]:
            attrs['nsamples'] = int(sl[itr][7:])
            itr += 1;continue
        if 'net' in sl[itr]:
            attrs['net'] = 'net'+sl[itr][3:]
            while len(attrs['net'])<16: attrs['net'] += ' '
            itr += 1;continue
        if 'frame' == sl[itr][:5] and sl[itr+1]=='ego':
            attrs['frame'] = 'allo-ego'
            while len(attrs['frame'])<5: attrs['frame'] += ' '
            itr += 2;continue
#        if 'seed' == sl[itr][:4]:
#            attrs['sample'] = 'whichsamp'+sl[itr][sl[itr].find('#')+1:sl[itr].find(\
#                    'last')].strip()
#            itr += 1;continue
        if 'opt' == sl[itr][:3]:
            if 'adam' in sl[itr]:
                attrs['opt'] = 'opt'+sl[itr][4:]+'-'+sl[itr+1]
                while len(attrs['opt'])<10: attrs['opt'] += ' '
                itr += 2;continue
            else:
                attrs['opt'] = 'opt'+sl[itr][4:]
                while len(attrs['opt'])<10: attrs['opt'] += ' '
                itr += 1;continue
        if 'loss' == sl[itr][:4]:
            pass#if 'huber' in sl['TODO']: pass
        if 'curr' == sl[itr][:4]:
            if curr==None:
                attrs['curr'] = sl[itr]
                while len(attrs['curr'])<30: attrs['curr'] += ' '
                itr += 1; continue    
            else:
                currid = int(sl[itr][5:sl[itr].find('sample')-1])
        itr+=1
    if not curr==None:
        for k,v in curr[currid].items():
            if k=='currID': continue
            if k=='inp':
                for inpk, inpv in v.items():
                    attrs['curr--'+inpk] = inpv if type(inpv)==int else freeze_dict(inpv)
            else:
                attrs['curr--'+k] = v if type(v)==int else freeze_dict(v)
    if not 'curr--schedule timings' in attrs.keys():
        attrs['curr--schedule timings'] = -1
    if not 'curr--schedule strengths' in attrs.keys():
        attrs['curr--schedule strengths'] = -1
    for k,v in attrs.items():
        K = str(k)
        while len(K)<20: K = K+' '
        print('\t'+K+'\t'+str(v))
    print('\n\n')

    return attrs

def ingest_curr_file(fn):
    cs = []
    with open(fn, 'r') as f:
      while True:
        line = f.readline()
        if not line: break
        c_obj = {'currID':int(line[:line.find(':')-1])}
        c_obj['inp'] = eval(line[line.find('inp:')+4:])
        for key in ['groups', 'begTime', 'endTime', 'begVal', 'endVal']:
            line = f.readline()
            c_obj[key] = eval(line[line.find(':')+1:])
        cs.append(c_obj)
    return cs

def samp_tup(ndim, pct):
    t = []
    for i in range(ndim):
        if random.random()<pct: t.append(i)
    return tuple(t)

def save_final_losses_process(dest): 
    #for now, only supports one file processing.
    curr=None
    datas = []
    tmp_data = []
    trials = []
    if type(dest)==list:
        if len(dest)==1 and 'curriculum' in str(os.listdir(dest[0])):
            dest.append( dest[0][:dest[0].rfind('/')] )
        if len(dest)>1 and 'logfile' in dest[0] and 'curriculum' in dest[1]:
            dests = [dest[0]]
            curr = ingest_curr_file(dest[1])
        else:
            dests = dest
    else:
        dests = [dest]
    for d in dests:
        DF = open(d, 'r')
        data = []
        a = ' '; counter = 0.0;
        while not len(a)==0:
            a = DF.readline().strip(' ')
            if not len(a)==0:
                if 'test accs:' in a: 
                    trials.append(a)
                    continue
                if a[:2] == '[[':
                    tmp_data = []
                    a = a[1:]
                al = list(a)
#                if len(al)<40:
#                    tmp_data += [int(al[2])]
#                    tmp_data += [int(al[6])]
#                    tmp_data += [int(al[10])]
#                    tmp_data += [int(al[14])]
#                else:
                for i in [2+4*j for j in range(12)]:
#                    print(len(al), i)
                    try:
                        tmp_data += [int(al[i])]
                    except:pass
                counter += 1.0
                if ']]' in a:
                    for i in range(4): 
                        tmp_data[i] /= counter
                    counter = 0.0
                    data.append(tmp_data)
        DF.close(); 
        datas.append(data)
    if not len(trials)==len(datas):
        datas = datas[0]
    D = np.array(datas)
    print(D.shape, len(trials), len(datas), len(datas[0]))
        #raise Exception("inconsistency: "+str(len(trials))+';'+str(len(datas)))
    n_entities = len(datas)
    attributes = []
    for i in range(n_entities):
        attributes.append( get_attributes(trials[i], i, curr) )
    nsamps = attributes[-1]['nsamples']
    # insider knowledge....sloppy.....:
    if attributes[-1]['frame']=='allo-ego':
        for i,a in enumerate(attributes):
            a['ego_or_allo'] = 'ego ' if (i/nsamps)%2==0 else 'allo'

    hyperparams = set() 
    for a in attributes:
        for k in list(a.keys()):
            hyperparams.add(k)
    hyperparams = list(hyperparams)
    hp_map = { h:k for k,h in enumerate(hyperparams) }
    nversions = []
    for i,h in enumerate(hyperparams):
        nversions.append(set())
        for a in attributes:
            nversions[-1].add(a[h])
    nversc = [ len(s) for i,s in enumerate(nversions) ]
    nwhere_ge1 = [(1 if nversc[i] > 1 else 0) for i in range(len(nversc))]
    nz = np.nonzero(nwhere_ge1)[0]
    ndim = sum(nwhere_ge1) 
    WhichVary = [hyperparams[i] for i in nz]
    WhichVaryVals = [sorted(list(nversions[i])) for i in nz]
    WhichVaryValsD ={}

    arr_shape = [nversc[z] for z in nz]
    AvgAccsIsolated = np.zeros( shape=arr_shape )
    MinAccsIsolated = np.zeros( shape=arr_shape )
    MaxAccsIsolated = np.zeros( shape=arr_shape )
    VarAccsIsolated = np.zeros( shape=arr_shape )

    for hp in sorted(WhichVaryVals):
        for i,a in enumerate(hp):
            WhichVaryValsD[a]=i
    #print '\n>>> Differences per trial over each starting state:'
    for i,a in enumerate(attributes):
        index = tuple( WhichVaryValsD[a[hp]] for hp in WhichVary )
        try: d__ = D[i,:]
        except: d__ = D[i]
        AvgAccsIsolated [ index ] = np.mean(d__)
        MinAccsIsolated [ index ] = np.min(d__)
        MaxAccsIsolated [ index ] = np.max(d__)
        VarAccsIsolated [ index ] = np.var(d__)**0.5

#    print hyperparams; print hp_map; print nversions; print nversc; 
#    print nwhere_ge1; print nz; print arr_shape; print ndim; print WhichVary; 
#    print WhichVaryVals; print WhichVaryValsD;


#        continue
#        for hp in WhichVary:
#            print hp, a[hp], '\t',
#        print ':',
#        print '  avg', '{:1.3f}'.format(AvgAccsIsolated [ index ]), 
#        print '  max', '{:1.3f}'.format(MaxAccsIsolated [ index ]), 
#        print '  min', '{:1.3f}'.format(MinAccsIsolated [ index ]), 
#        print '  rt var', '{:1.3f}'.format(VarAccsIsolated [ index ]), 
#        print ''

    print('\n===============================\n')
    print("VARIABLES: "+str(WhichVary)+' hyperparameters were varied, taking on:')
    for h in hyperparams:
        if h in WhichVary:
            for X in sorted(list(nversions[hp_map[h]])):
                #print('\t'+'{0[0]:<15}{0[1]:<15}'.format((h+':'+str(X))))
                hstr = h+':'
                while len(hstr)<30: hstr = hstr+' '
                print('\t'+hstr+'  '+str(X))
    print("CONSTANTS: these hyperparameters were held constant:")
    for h in hyperparams:
        if h not in WhichVary:
            #print '\t',h,':\t', 
#            print('\t'+'{0[0]:<15}{0[1]:<15}'.format((h+':'+\
#                    str(tuple(nversions[hp_map[h]])[0]))))
            print('\t'+h+':\t'+str(tuple(nversions[hp_map[h]])[0]))
#    nversions = []
#    for _ in WhichVaryVals: print '\n',_
    print('\nThis analysis studies '+str(len(attributes)*attributes[-1]\
            ['nsamples'])+' total trained networks.')
    if ndim==0: 
        print('All recorded values:')
        for m in (AvgAccsIsolated, MaxAccsIsolated, MinAccsIsolated):
            print(m)
        #Margs = [tuple([])]
        return
    print("\n\nThe following are marginals over certain variables.\n")
    print(ndim)
    if ndim==1: Margs = [ tuple([])]
    elif ndim==2: Margs = [ (0,), (1,), tuple([])]
    elif ndim==3: Margs = [ (0,), (1,), (2,), (0,2), (1,2), (0,1), tuple([]) ]
    elif ndim==4: Margs = [ (0,), (1,), (2,), (3,), (0,1), (0,2), (1,2),\
         (0,3), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3), tuple([])] # ALL
    else: Margs = [tuple([])]+[ tuple([i for i in range(ndim) if not i==j]) for \
            j in range(ndim)] + [ samp_tup(ndim, 0.3) for _ in range(20)]
        
        
#        [ (0,), (1,), (2,), (3,), (0,1), (0,2), (1,2),\
#         (0,3), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3), tuple([])] # ALL
#    if ndim==4: Margs = [ (0,), (1,), (2,), (3,), (0,1), (0,2), (1,2),\
#         (0,3), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3), tuple([])] # ALL
#    if ndim==4: Margs = [ (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3), \
#            tuple([])] # all except ego/allo distinction
#    if ndim==4: Margs = [ (0,1,2), (0,1,3), (0,2,3), (1,2,3), tuple([])] # ALL
#    elif ndim==4: Margs = [ (2,), (3,), (2,3), tuple([])] # don't marg over
        # erroneous LR
#    else: raise Exception(ndim)

    for margs in Margs:
        s = []
        for i in range(len(WhichVary)):
            if not i in margs: s.append(i)
        s = tuple(s); 

        #print 'present over', [WhichVary[m] for m in s] ,'&',
        if len(margs)==0: print("all responses:")
        else: print('marginalize over'+str([WhichVary[m] for m in margs])+':')
        MEAN = np.mean(AvgAccsIsolated, axis=margs)
        MINS = np.min(MinAccsIsolated, axis=margs)
        MAXS = np.max(MaxAccsIsolated, axis=margs)
        if len(s)==1:
            for i in range(arr_shape[s[0]]):
                wv = str(WhichVaryVals[s[0]][i])
                while len(wv)<30: wv += ' '
                print(wv+\
                        '\tavg  '+'{:1.3f}'.format(MEAN[i])+\
                        '\tmin  '+'{:1.3f}'.format(MINS[i])+\
                        '\tmax  '+'{:1.3f}'.format(MAXS[i]))
        if len(s)==2:
          for i in range(arr_shape[s[0]]):
#            print(str(WhichVaryVals[s[0]][i])+\
#                '\tavg  '+'{:1.3f}'.format(np.mean(MEAN[i]))+\
#                '\tmin  '+'{:1.3f}'.format(np.min(MINS[i]))+\
#                '\tmax  '+'{:1.3f}'.format(np.max(MAXS[i])))
            for j in range(arr_shape[s[1]]):
                print(str(WhichVaryVals[s[0]][i])+'  '+str(WhichVaryVals[s[1]][j])+\
                    '\tavg  '+'{:1.3f}'.format(float(MEAN[i,j]))+\
                    '\tmin  '+'{:1.3f}'.format(float(MINS[i,j]))+\
                    '\tmax  '+'{:1.3f}'.format(float(MAXS[i,j])))

        if len(s)==3:
          for i in range(arr_shape[s[0]]):
           for j in range(arr_shape[s[1]]):
            for k in range(arr_shape[s[2]]):
                print(str(WhichVaryVals[s[0]][i])+'  '+str(WhichVaryVals[s[1]][j])+'  '+\
                        str(WhichVaryVals[s[2]][k])+\
                        '\tavg '+ '{:1.3f}'.format(MEAN[i,j,k])+\
                        '\tmin '+ '{:1.3f}'.format(MINS[i,j,k])+\
                        '\tmax '+ '{:1.3f}'.format(MAXS[i,j,k]))

        print('') 

if __name__=='__main__':
    print(len(sys.argv), sys.argv)
    if len(sys.argv)==1:
        print("\nsave_as_plot information: use first argument <logfile> and log"+\
            "\n\tfile(s) to generate an analysis of final values of an experiment."+\
            "\n\tUse first argument <state_actions> to generate a Successes plot"+\
            "\n\tof testing and training success rates per state.  Give just files "+\
            "\n\tof .npy format to develop loss plots for each." )
    elif 'logfile'==sys.argv[1]:
#        if len(sys.argv)>3:
#            Z = np.load(sys.argv[2])
#            Y = np.load(sys.argv[3])
#            X = np.mean( [X, Y], axis=1)
#            print X.shape, Y.shape, Z.shape
        save_final_losses_process(sys.argv[2:])
    elif 'state_actions'==sys.argv[1]:
        X = np.mean(np.load(sys.argv[2]), axis=1)
        print(X.shape)

        _2away = [ (0,1), (1,0), (0,-1), (-1,0), (2,0), (1,1), (0,2), (1,-1), \
                    (0,-2), (-1,-1), (-2,0), (-1,1) ]
        print(len(_2away))
        states = [ (3,3,3+i,3+j) for i,j in _2away]
        save_as_successes(get_time_str(sys.argv[2][:sys.argv[2].find('v2')])+'output_state_actions', 
                X[0,:,:], X[1,:,:], states=states, \
                smoothing=100, centric='egocentric',    tr2=None, te2=None)
    else:
        save_as_plot1(sys.argv[1:])
