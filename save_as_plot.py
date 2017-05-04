import sys
import numpy as np
import matplotlib.pyplot as plt

def save_as_plot1(fns, lr=None, mna=None, nsamples=None, which='L S'):
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
            plt.plot(np.mean(rewards[:,:,0], axis=0), c='red', linestyle='-') # avg train R
            plt.plot(np.mean(rewards[:,:,1], axis=0), c='black', linestyle='-') # avg test R
        if 'L' in which:
            plt.plot(np.mean(losses[:,:,0], axis=0), c='blue', linestyle='-') # train loss
            plt.plot(np.mean(losses[:,:,1], axis=0), c='yellow', linestyle='-') # test loss
        if 'S' in which:
            plt.plot(np.mean(nsteps[:,:,0], axis=0), c='green', linestyle='-') # avg train steps
            plt.plot(np.mean(nsteps[:,:,1], axis=0), c='purple', linestyle='-') # avg test steps
        if 'R' in which:
            plt.xlabel("Episode.  blue=train err, yellow=test err,\n green=avg "+\
                    "train steps, purple=avg test steps"+o_string+\
                    "\nred: avg train reward, black: avg test reward")
        else:
            plt.xlabel("Episode.  blue=train err, yellow=test err,\n green=avg "+\
                       "train steps, purple=avg test steps"+o_string)
        plt.title("Training and testing error averaged over "+nsamples+" samples with lr "+lr)
        plt.ylabel("Avg reward, loss, num actions taken.  Max num actions: "+mna)

    
        plt.tight_layout(1.2)
#        plt.ylim(0, 4)
        plt.savefig(fn[:-4]+'.pdf', format='pdf')
        plt.close()
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
    plt.savefig(save_to_loc+'.pdf', dpi=100, format='pdf')
    plt.close()

def save_as_Qplot3(mat, save_to_loc, trialinfo='[none provided]'):
    f,ax = plt.subplots(3,4,sharex=True, sharey=True)
    #f.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
    for a in range(4):
        for LQR in range(3):
            ax[LQR,a].plot(mat[LQR,:,a,0], c='red')
            ax[LQR,a].plot(mat[LQR,:,a,0], c='blue')
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
    plt.savefig(save_to_loc+'.pdf', dpi=100, format='pdf')
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
        plt.savefig(fn[:-4]+'.pdf', format='pdf')
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
