import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def process_outputfile(x): 
    y = x[:-10]+'log.txt'
    print(x, y)
    INP = open(x, 'r')
    OUT = open(y, 'w')
    def onetrial():
        for _ in range(18): INP.readline()
        for i in range(13):
            OUT.write( INP.readline().split(':  ')[-1] )
        OUT.write('\n')
    INP.readline()
    OUT.write("Ego\n")
    for J in range(20): onetrial()
    for _ in range(21): INP.readline()
    OUT.write("Allo\n")
    for J in range(20): onetrial()
    INP.close(); 
    OUT.close()
    return y

def run_6_02_analysis(targ):
    R = [0,5,10,200,400,600,800,1000,1200,1400,1600,1800,1999]
    if "output.txt" in targ:
        targ = process_outputfile(targ)
    elif not "log.txt" in targ:
        raise Exception(targ)

    with open(targ, 'r') as f:
        while not 'Ego' in f.readline(): pass
        EGOS = np.zeros((20,13))
        ALLOS = np.zeros((20,13))
        for e in range(20):
            for j in range(13):
#                EGOS[e,j] = float(f.readline())+np.random.rand()*0.0001#*(-1)**j
                EGOS[e,j] = float(f.readline())
            f.readline()
        while not 'Allo' in f.readline(): pass
        for e in range(20):
            for j in range(13):
#                ALLOS[e,j] = float(f.readline())+np.random.rand()*0.0001#*(-1)**j
                ALLOS[e,j] = float(f.readline())
            f.readline()
        print("DONE")
#    f, ax = plt.subplots(1,1, sharex = True, sharey = True)
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


if __name__=='__main__':
    if not len(sys.argv)==3:
        raise Exception(sys.argv)
    else:
        if sys.argv[1]=='6-02':
            run_6_02_analysis(sys.argv[2]);
