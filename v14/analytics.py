import pickle
import matplotlib.pyplot as plt

EXP = 3
NAME = 'U-E-T-200'

while EXP > 0:
    with open('results/'+NAME+'-'+str(EXP)+'.pkl', 'rb') as f:
        e = pickle.load(f)

    ax1 = plt.subplot(2,2,1)
    plt.plot(list(e[0]))
    plt.ylabel('predicted utility')
    plt.ylim(0,2)

    ax2 = plt.subplot(2,2,2, sharex=ax1)
    plt.plot(list(e[1]))
    plt.ylabel('perceived utility')
    plt.ylim(0,2)

    ax3 = plt.subplot(2,2,3, sharex=ax1)
    plt.plot(list(e[2]))
    plt.ylabel('knowledge gain')
    plt.ylim(-1,1)

    ax4 = plt.subplot(2,2,4, sharex=ax1)
    plt.plot(list(e[3]))
    plt.ylabel('remaining utility')
    plt.ylim(0,2)

    plt.show()
    EXP -= 1
