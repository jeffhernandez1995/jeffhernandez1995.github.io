import numpy as np
from datasets import MovingMNIST
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import rc
mpl.rcParams['savefig.pad_inches'] = 0
# mpl.rcParams['animation.html'] = 'html5'


num_digits = 5
digits = np.array([0, 2, 4, 6, 8])
seq_len = 100

dset = MovingMNIST(True, seq_len=seq_len, num_digits=num_digits, digits=digits)

seq, bbox = dset[0]

def plot_gt(seq):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    def update(i):
        im.set_data(seq[i, :, :, 0])
        for j in range(bbox.shape[1]):
            x, y = bbox[i, j, 0], bbox[i, j, 1]
            w, h = 28, 28
            rects[j].set_width(w)
            rects[j].set_height(h)
            rects[j].set_xy([x, y])
    im = ax.imshow(seq[0, :, :, 0], cmap='gray', aspect='auto')
    rects = [0]*bbox.shape[1]
    for j in range(bbox.shape[1]):
        x, y = bbox[0, j, 0], bbox[0, j, 1]
        w, h = 28, 28
        rects[j] = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rects[j])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ani = animation.FuncAnimation(fig,
                              update,
                              seq.shape[0],
                              interval=1000/5,
                              blit=False)
    return ani

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani = plot_gt(seq)
ani.save('anim.mp4', writer=writer)