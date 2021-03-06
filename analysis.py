import sys, os, pwd, grp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, rc
from subprocess import call
from time import asctime
from scipy.stats import binom_test

def init_process(x, num_epochs): 
    if num_epochs==2000: R = 13
    if num_epochs==7500: R = 78
    if "output.txt" in x:
        y = x[:-10]+'log.txt'
        INP = open(x, 'r')
        OUT = open(y, 'w')
        try: os.chown(x,501,20)
        except: pass#os.chown(x,501,20)
        def onetrial(skip=True):
            if skip:
                for _ in range(18): 
                    x=INP.readline();
            for i in range(R):
                l = INP.readline()
                if l=='Done.': break
                OUT.write( l.split(':  ')[-1] )
            OUT.write('\n')
        INP.readline()
        OUT.write("Ego\n")
        for J in range(20): onetrial()
        OUT.write("Allo\n")
        INP.readline()
        if not len(INP.readline())==1:
            for _ in range(19): INP.readline()
            for J in range(20): onetrial()
        else:
            for _ in range(16): INP.readline()
            onetrial(False)
            for J in range(19): onetrial()
        INP.close(); 
        OUT.close()
        targ= y
    
    elif not "log.txt" in x:
        raise Exception(x)
    else: targ=x
    with open(targ, 'r') as f:
        while not 'Ego' in f.readline(): pass
        EGOS = np.zeros((20,R))
        ALLOS = np.zeros((20,R))
        for e in range(20):
            for j in range(R):
                EGOS[e,j] = float(f.readline())
            f.readline()
        while not 'Allo' in f.readline(): pass
        for e in range(20):
            for j in range(R):
                l = f.readline()
                try:
                    ALLOS[e,j] = float(l)
                except:
                    raise Exception(e,j,l,targ)
            f.readline()
    return EGOS, ALLOS

def run_stat_compare(targs, show=False):
#    rc('text', usetex=True)
    ''' This analysis tests 1) [amount of] success and 2) time to success
        for successful trials, across trials.  Facilitates multiple 
        experiments.'''
    R = [0,5,10,200,400,600,800,1000,1200,1400,1600,1800,1999]
    R2 = [0,5,10]+[100*(i+1) for i in range(75)]
    ego_mats, allo_mats = [],[]

#    targs = ['./storage/6-01/ego-allo-4/output.txt', 
#                './storage/6-01/ego-allo-5/output.txt', 
#                './storage/6-01/ego-allo-6/output.txt', 
#                './storage/6-03/ego-allo-3/output.txt']

    for i, targ in enumerate(targs):
        print i, targ
        if i<=2:
            tup = init_process(targ, 2000)
        else:
            tup = init_process(targ, 7500)
        ego_mats.append(tup[0]); 
        allo_mats.append(tup[1])
    Shape = ego_mats[0].shape # Shape: (trials) x (time-series datapoints)
    n_compare = len(ego_mats)
    if show: print(n_compare, Shape)
    if not (n_compare==3 or n_compare==4): raise Exception("Not implemented yet")
    ''' Assumption: this data comes in curr order GRADUAL, NO, STEPWISE'''

    etimes = np.zeros   ( (n_compare, Shape[0] ), dtype=int) # bucketed
    atimes = np.zeros   ( (n_compare, Shape[0] ), dtype=int ) # bucketed 
    esccsamt = np.zeros ( (n_compare, Shape[0] ) ) # final acc
    asccsamt = np.zeros ( (n_compare, Shape[0] ) ) # final acc
    esuccess = np.zeros ( (n_compare, Shape[0] ), dtype=bool ) # Bool: successful?
    asuccess = np.zeros ( (n_compare, Shape[0] ), dtype=bool ) # Bool: successful?
    e_categ = np.ones( (n_compare, len(R)+1) )*-0.5
    a_categ = np.ones( (n_compare, len(R)+1) )*-0.5
    for m in range(n_compare):
        esuccess[m,:] = np.floor(ego_mats[m][:,-1])
        asuccess[m,:] = np.floor(allo_mats[m][:,-1])
        esccsamt[m,:] = ego_mats[m][:,-1]
        asccsamt[m,:] = allo_mats[m][:,-1]

        for smp in range(Shape[0]):
            etimes[m,smp] = -1
            atimes[m,smp] = -1
            Ve = 1; Va = 1
            for N in reversed(range(len(R))):
                Ve = min(ego_mats[m][smp,N],Ve)
                if not Ve<1: 
                    etimes[m,smp] = N;#R[N]
                Va = min(allo_mats[m][smp,N],Va)
                if not Va<1: 
                    atimes[m,smp] = N;#R[N]
            if e_categ[m,etimes[m,smp]] == -1:
                e_categ[m,etimes[m,smp]] += 1
            if a_categ[m,etimes[m,smp]] == -1:
                a_categ[m,etimes[m,smp]] += 1
            e_categ[m,etimes[m,smp]] += 1
            a_categ[m,atimes[m,smp]] += 1
        if show:
            print(etimes[m,:])
            print(e_categ[m,:])
            print(atimes[m,:])
            print(a_categ[m,:])
            print('')
    if show: 
        print(esuccess, asuccess)
        print(np.mean(esuccess), np.mean(asuccess), np.mean(esuccess, \
                axis=1), np.mean(asuccess, axis=1))

    raw_success = [[ ['all ego',    np.sum(esuccess)], \
                     ['all allo',   np.sum(asuccess)], ],\
                   [ ['gradu ego',  np.sum(esuccess, axis=1)[0]],\
                     ['gradu allo', np.sum(asuccess, axis=1)[0]], ],\
                   [ ['nocur ego',  np.sum(esuccess, axis=1)[1]],\
                     ['nocur allo', np.sum(asuccess, axis=1)[1]], ],\
                   [ ['stepw ego',  np.sum(esuccess, axis=1)[2]],\
                     ['stepw allo', np.sum(asuccess, axis=1)[2]] ]]
    pct_success = [[ ['all ego',    np.mean(esuccess)], \
                     ['all allo',   np.mean(asuccess)], ],\
                   [ ['gradu ego',  np.mean(esuccess, axis=1)[0]],\
                     ['gradu allo', np.mean(asuccess, axis=1)[0]], ],\
                   [ ['nocur ego',  np.mean(esuccess, axis=1)[1]],\
                     ['nocur allo', np.mean(asuccess, axis=1)[1]], ],\
                   [ ['stepw ego',  np.mean(esuccess, axis=1)[2]],\
                     ['stepw allo', np.mean(asuccess, axis=1)[2]] ]]
    RR = np.array(range(len(R)+1))

    width = 0.4

    # Good for 2x2: plt.subplots_adjust(top=0.85, left=0.15, bottom=0.15, wspace=0.2, hspace=0.45)
    fig, ax = plt.subplots(4,4)
    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1, wspace=0.5, hspace=0.6)
 #   fig, ax = plt.subplots(4,2)
  #  plt.subplots_adjust(top=0.9, left=0.1, right=1.1, bottom=0.1, wspace=0.5, hspace=0.5)

    fig.set_size_inches(10,10)

    inds = [13,0,1,2,3,4,5,6,7,8,9,10,11,12]

    lblue = '#9999ff'
    lgreen = '#9ae095'
    lred = '#ff8080'
    blue = '#3333ff'
    green = '#7fb77b'
    red = '#e60000'
    lblue = 'c'
    lgreen = 'y'
    lred = 'magenta'
    blue = 'b'
    green = 'g'
    red = 'r'

    ax[0,0].bar(RR-5*width/6, e_categ[0,inds], width/3, color=lblue)
    ax[0,0].bar(RR-3*width/6, e_categ[1,inds], width/3, color=lgreen)
    ax[0,0].bar(RR-1*width/6, e_categ[2,inds], width/3, color=lred)
    ax[0,0].bar(RR+1*width/6, a_categ[0,inds], width/3, color=blue)
    ax[0,0].bar(RR+3*width/6, a_categ[1,inds], width/3, color=green)
    ax[0,0].bar(RR+5*width/6, a_categ[2,inds], width/3, color=red)

    ax[1,0].bar(RR+width/2, e_categ[0,inds], width, color=lblue)
    ax[1,0].bar(RR-width/2, a_categ[0,inds], width, color=blue)
    ax[2,0].bar(RR+width/2, e_categ[1,inds], width, color=lgreen)
    ax[2,0].bar(RR-width/2, a_categ[1,inds], width, color=green)
    ax[3,0].bar(RR+width/2, e_categ[2,inds], width, color=lred)
    ax[3,0].bar(RR-width/2, a_categ[2,inds], width, color=red)

    ax[0,0].set_title('All', fontsize=9)
    ax[1,0].set_title('Gradual curriculum', fontsize=9)
    ax[2,0].set_title('No curriculum', fontsize=9)
    ax[3,0].set_title('Stepwise curriculum', fontsize=9)

    fig.suptitle('Comparison of ego/allocentrism: histogram each'\
            +" curriculum's time to success.\nDark: allocentric, "\
            +"light: egocentric.  Did Not Learn indicated as DNL.", fontsize=10)
    fig.suptitle('Comparison of ego/allocentrism: histogram each'\
            +" curriculum's time to success.\n RBG: allocentric, "\
            +"CYM: egocentric.  Did Not Learn indicated as DNL.", fontsize=10)

    for r in  raw_success: print(r)
    for i in range(4):
        xticklabels = ['','DNL']+[str(s) for s in R]
        xtickNames = plt.setp(ax[i,0], xticklabels=xticklabels )
        ax[i,0].locator_params(nbins=15, axis='x')
        ax[i,0].set_xlim(-1, 14)
        plt.setp(xtickNames, rotation=85, fontsize=6)
        ax[i,0].set_ylim(-1, 20)
        ax[i,0].plot([-width,13+width],[0,0], c='black', linewidth=0.5)

        p= binom_test( (raw_success[i][0][1], raw_success[i][1][1]), alternative='greater')
        s = {0: 'Overall:\n\n', 1: 'Gradual curriculum:\n\n', 2:'No curriculum:\n\n',\
                3:'Stepwise curriculum:\n\n'}[i]
        if n_compare==4:
            ax[i,1].text(-0.35,0, s+'Under a binomial test, the\nsignificance of '\
                    +'the hypothesis that a\nsuccessful trial came from an\n'\
                    +'Egocentric run a than an\nAllocentric run: $\\bf{p=%1.3e}$' % p,\
                    fontsize=8)
        else:
            ax[i,1].text(-0.35,0, s+'Under a binomial test, the significance\nof '\
                    +'the hypothesis that a successful trial\ncame from an '\
                    +'Egocentric run than\nan Allocentric run: $\\bf{p=%1.3e}$' % p,\
                    fontsize=12)
        

        for j in [1]:#,2,3]:
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_xlim(0,1)
            ax[i,j].set_axis_off()
            plt.setp(ax[i,j].get_xticklabels(), visible=False)
            plt.setp(ax[i,j].get_yticklabels(), visible=False)
    ax[3,0].set_xlabel('First epoch with 100% success', fontsize=8)
    ax[0,0].set_ylabel('Count out of 20 trials', fontsize=8)
    ax[1,0].set_ylabel('Count out of 20 trials', fontsize=8)
    ax[2,0].set_ylabel('Count out of 20 trials', fontsize=8)
    ax[3,0].set_ylabel('Count out of 20 trials', fontsize=8)
    t = './Experiments/statplot-'+asctime().replace(' ','_').replace(':','')

    if n_compare==4:
        gs = gridspec.GridSpec(4,4)
        ax2 = plt.subplot(gs[0,2:])
        etimes = np.zeros   ( (Shape[0] ), dtype=int) # bucketed
        atimes = np.zeros   ( (Shape[0] ), dtype=int ) # bucketed 

        e_categ = np.ones( (len(R2)+1) )*-0.5
        a_categ = np.ones( (len(R2)+1) )*-0.5
        esuccess = np.floor(ego_mats[-1][:,-1])
        asuccess = np.floor(allo_mats[-1][:,-1])
        esccsamt = ego_mats[-1][:,-1]
        asccsamt = allo_mats[-1][:,-1]

        for smp in range(Shape[0]):
            etimes[smp] = -1
            atimes[smp] = -1
            Ve = 1; Va = 1
            for N in reversed(range(len(R2))):
                Ve = min(ego_mats[-1][smp,N],Ve)
                if not Ve<1: 
                    etimes[smp] = N;#R[N]
                Va = min(allo_mats[-1][smp,N],Va)
                if not Va<1: 
                    atimes[smp] = N;#R[N]
            if e_categ[etimes[smp]] == -1:
                e_categ[etimes[smp]] += 1
            if a_categ[etimes[smp]] == -1:
                a_categ[etimes[smp]] += 1
            e_categ[etimes[smp]] += 1
            a_categ[atimes[smp]] += 1
        inds = [78]+list(range(78))
        RR = np.array(range(len(R2)+1))
        ax2.bar(RR+width/2, e_categ[inds], width, color='lime')#'0.35')
        ax2.bar(RR-width/2, a_categ[inds], width, color='darkmagenta')#'0.65')
        ax2.set_title('Linear curriculum. Lime: egocentric, Purple: allocentric', fontsize=9)

        xticklabels = ['DNL','DNL']+[str(s) for s in R2[::2]]
        xtickNames = plt.setp(ax2, xticklabels=xticklabels )
        ax2.locator_params(nbins=80, axis='x')
        ax2.set_xlim(-1, 80)
        plt.setp(xtickNames, rotation=85, fontsize=6)
        ax2.set_ylim(bottom=-1)
        ax2.plot([-width,78+width],[0,0], c='black', linewidth=0.5)

    # Get successful trials; among these, do t-test for value of success.





    if show:
        plt.show()
    else:
        plt.savefig(t+'.png', dpi=100)
        call(['open', t+'.png'])

    print('done')









def run_6_02_analysis(targ):
    ''' This analysis plots a number of trials of ego and allo on the same
        graph.  This is useful for comparing both ego vs allo utility as 
        well as providing graphs that can be contrasted across parameter
        settings, such as curriculum style. '''
    R = [0,5,10,200,400,600,800,1000,1200,1400,1600,1800,1999]
    EGOS, ALLOS = init_process(targ)
    plt.plot([0,2000], [1.5**-1, 1.5**-1], c='lightgreen')
    plt.plot([0,2000], [1, 1], c='darkgreen')
    for i in range(20):
        plt.plot(R, EGOS[i,:], c='cyan')
        plt.plot(R, ALLOS[i,:], c='pink')
    plt.plot(R, np.mean(EGOS,axis=0), c='blue')
    plt.plot(R, np.mean(ALLOS,axis=0), c='red')
    plt.plot(R, np.std(EGOS,axis=0), c='navy')
    plt.plot(R, np.std(ALLOS,axis=0), c='darkred')
    plt.xlabel("Training Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Results from task collection < R, U, RU > over 20 samples "\
            +'using curriculum that mixes task RU\n to 4/5 of training data '\
            +'from the initial training epoch. Blue: egocentric, Red: '\
            +'allocentric.  \n Dark Blue, Red: data 1 standard deviation.  '\
            +'Light Green: 2 of <R,U,RU> solved; dark green: all solved. ', fontsize=8)
#    plt.title("Results from task collection < R, U, RU > over 20 samples "\
#            +'using curriculum that mixes task RU\n to 4/5 of training data '\
#            +' linearly in the first 500 epochs. Blue: egocentric, Red:'\
#            +' allocentric. \n Dark Blue, Red: data 1 standard deviation.  '\
#            +'Light Green: 2 of <R,U,RU> solved; dark green: all solved. ', fontsize=8)
    plt.show()
    print('done')


if __name__=='__main__':
    if not len(sys.argv)>=3:
        raise Exception(str(sys.argv)+":   please use one of the following arguments"+\
                " a target file: <plot-ego-allo-diffs>, <stat-compare>")
    else:
        if sys.argv[1]=='plot-ego-allo-diffs':
            run_6_02_analysis(sys.argv[2]);
        elif sys.argv[1]=='stat-compare':
            run_stat_compare(sys.argv[2:])
        elif sys.argv[1]=='stat-compare-show':
            run_stat_compare(sys.argv[2:], show=True)
        else: print("argument not recognized.")
