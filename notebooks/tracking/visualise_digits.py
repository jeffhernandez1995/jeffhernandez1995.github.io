import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import rc
mpl.rcParams['savefig.pad_inches'] = 0
# mpl.rcParams['animation.html'] = 'html5'


seq = np.load('data/mnist_training_fast_videos.npy')[0]
bbox = trajectory_tensor = np.load('data/mnist_training_fast_trajectories.npy')[0]
bbox = bbox.swapaxes(0, 1)
bbox[:, :, 1] = seq.shape[1] - bbox[:, :, 1]
bbox[:, :, 3] = seq.shape[1] - bbox[:, :, 3]


def plot_gt(seq):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    def update(i):
        im.set_data(seq[i, :, :])
        for j in range(bbox.shape[1]):
            x, y = bbox[i, j, 0], bbox[i, j, 1]
            w, h = bbox[i, j, 2] - bbox[i, j, 0], bbox[i, j, 3] - bbox[i, j, 1]
            rects[j].set_width(w)
            rects[j].set_height(h)
            rects[j].set_xy([x, y])
    im = ax.imshow(seq[0, :, :], cmap='gray', aspect='auto')
    rects = [0]*bbox.shape[1]
    for j in range(bbox.shape[1]):
        x, y = bbox[0, j, 0], bbox[0, j, 1]
        w, h = bbox[0, j, 2] - bbox[0, j, 0], bbox[0, j, 3] - bbox[0, j, 1]
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
