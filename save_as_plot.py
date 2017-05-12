import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, font_manager
from math import ceil, floor

ACTION_NAMES = { 0:"U", 1:'R', 2:"D", 3:'L' } 
UDIR = 0; RDIR = 1; DDIR = 2; LDIR = 3
MAX_DISPLAY_EPOCHS = 100

def save_as_plot1(fns, lr=None, mna=None, nsamples=None, which='L S', \
        div=1.0, delete_npy=False):
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
        try:
            lr = fn.split('lr')[1].split('-(')[0]
            print("lr:"+str(lr))
        except: lr=0
        try:
            mna = fn.split('mna-')[1].split('-lr')[0]
            print("mna:"+str(mna))
        except: mna=0
        try:
            nsamples = fn.split('nsamples')[1].split('.npy')[0]
            print("nsamples:"+str(nsamples))
        except: nsamples=0

        #if not nsamples: nsamples = str(x.shape[1])
        try:
            losses, nsteps, rewards = x;
        except:
            losses, nsteps = x;
        print "SHAPES:", losses.shape, nsteps.shape
        factor=20; lmax = int(rewards.shape[1]/factor)*factor
        meanmean = np.mean(rewards[:,:lmax,1], axis=0).reshape(rewards.shape[1]/factor, factor)
        meanmean = np.mean(meanmean, axis=0)
        X = np.repeat(meanmean, rewards.shape[1]/factor)
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
        if 'L' in which:
            plt.plot(np.mean(losses[:,:,0], axis=0), c='blue', linestyle='-') 
            # train loss
            plt.plot(np.mean(losses[:,:,1], axis=0), c='yellow', linestyle='-') 
            # test loss
        if 'S' in which:
            print nsteps.shape
            plt.plot(np.mean(nsteps[:,:,0], axis=0), c='green', linestyle='-') 
            # avg train steps
            plt.plot(np.mean(nsteps[:,:,1], axis=0), c='purple', linestyle='-') 
            # avg test steps
        if 'R' in which:
            plt.xlabel("Epoch.  blue=train err, yellow=test err,\n green=avg "+\
                    "train steps, purple=avg test steps"+o_string+\
                    "\nred: avg train reward, black: avg test reward")
        else:
            plt.xlabel("Episode.  blue=train err, yellow=test err,\n green=avg "+\
                       "train steps, purple=avg test steps"+o_string)

        plt.title("Training and testing error averaged over "+str(nsamples)+" samples with lr "+lr)
        plt.ylabel("Avg reward, loss, num actions taken.  Max num actions: "+str(mna))

    
        plt.tight_layout(1.2)
#        plt.ylim(0, 4)
        plt.savefig(fn[:-4])
        #plt.savefig(fn[:-4]+'.pdf', format='pdf')
        #plt.show()
#        plt.savefig(fn[:-4])
        plt.close()
        if delete_npy:
            os.remove(fn)
    print "Success: last file stored at", fn[:-4]


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
    plt.savefig(save_to_loc, dpi=100)
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
    print save_to_loc
    plt.savefig(save_to_loc, dpi=100)
    plt.close()


def _dist(x,y): return abs(x[0]-y[0])+abs(x[1]-y[1])

def _make_str(s_id, gridsz, trte, debug=False):
    if trte==0: s ='train\n'
    if trte==1: s ='test \n'
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
    if debug:
        if _dist((s_id[0], s_id[1]), (s_id[2], s_id[3])):
            s += '\n('+str(s_id[2]-s_id[0])+', '+str(s_id[3]-s_id[1])+')\n'
        s += str(s_id)
    return s



def save_as_state_actions1(state_indices, state_mat, gridsz, dest):
    sorted_states_indices = sorted(list(state_indices.keys())) # dict sorted
    print state_mat.shape, len(sorted_states_indices)
    
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
                ax[st_i / lX, st_i % lY, trte].plot(effective_eps, \
                            data[ci,:], color=clr)
                ax[st_i / lX, st_i % lY, trte].fill_between(effective_eps, \
                        data[ci,:], color=clr)
                #np.random.shuffle(clrs)

            ax[st_i/lX,st_i%lY,trte].text(-0.35*nepochs,0,\
                _make_str(St_key, gridsz, trte), fontsize=9, family='monospace')

    plt.gcf().set_size_inches(4*lX,4*lY)
    plt.gcf().suptitle("Actions taken per state, per epoch. BLUE:"+\
            ACTION_NAMES[0]+"; RED:"+ACTION_NAMES[1]+"; GREEN:"+\
            ACTION_NAMES[2]+"; YELLOW:"+ACTION_NAMES[3])
    
    s = " / "+str(nepochs / MAX_DISPLAY_EPOCHS) \
                if nepochs>MAX_DISPLAY_EPOCHS else ''
    ax[0,lY-1,1].set_xlabel("Epoch")

    plt.plot()
    #plt.show()
    plt.savefig(dest)
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
        plt.savefig(fn[:-4])
        plt.close()

    print "Success: last file stored at", fn[:-4]

if __name__=='__main__':
    save_as_plot1(sys.argv[1:])
    '''
    try:
        save_as_plot(sys.argv[1:-3], sys.argv[-3], sys.argv[-2], sys.argv[-1])
    except:
        #save_as_plot(sys.argv[1:], sys.argv[-2], sys.argv[-1])
        save_as_plot1(sys.argv[1:])
        '''
