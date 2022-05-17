import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensor_generation import *


def plot_events():
    bag = rosbag.Bag('datasets/back6.bag')

    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(projection='3d')

    pxs = []
    pys = []
    pzs = []
    nxs = []
    nys = []
    nzs = []

    for i, (topic, msg, t) in enumerate(bag.read_messages()):
        if i == 105:
            break

        if i % 3 == 0:
            events = msg.events
            for j, ev in enumerate(events):
                if j % 1 == 0:
                    if ev.polarity:
                        pxs.append(ev.ts.secs * 1e9 + ev.ts.nsecs)
                        pys.append(ev.x)
                        pzs.append(ev.y)

                    else:
                        nxs.append(ev.ts.secs * 1e9 + ev.ts.nsecs)
                        nys.append(ev.x)
                        nzs.append(ev.y)
    
            print(i, end=" ", flush=True)
    
    print()

    ax.scatter(pxs, pys, pzs, s=.3)
    ax.scatter(nxs, nys, nzs, s=.3, color='darkgrey')

    ax.set_box_aspect(aspect=(8, 2, 2))


    plt.savefig('events.png')


def plot_mp4():
    vt = mp4_to_tensor('datasets/back6.mp4', 3)
    plt.imsave('1.png', vt[0], cmap='Greys_r')
    plt.imsave('2.png', vt[1], cmap='Greys_r')
    plt.imsave('3.png', vt[2], cmap='Greys_r')


def main():
    plot_mp4()

if __name__ == '__main__':
    main()