import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import rc
from sort import Sort
mpl.rcParams['savefig.pad_inches'] = 0
# mpl.rcParams['animation.html'] = 'html5'


# seq = np.load('datasets/X_train.npy')[0].reshape(10, 28, 28)

# for i in range(10):
#     fig = plt.figure(frameon=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     im = ax.imshow(seq[i, :, :], cmap='gray', aspect='auto')
#     ax.set_xticks([])
#     ax.set_yticks([])

#     ax.get_xaxis().set_ticklabels([])
#     ax.get_yaxis().set_ticklabels([])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     fig.savefig(f'resources/fig_{i}.pdf')
#     plt.close()

# seq = np.load('data/mnist_training_fast_videos.npy')[0]
# bbox = trajectory_tensor = np.load('data/mnist_training_fast_trajectories.npy')[0]
# bbox = bbox.swapaxes(0, 1)
# bbox[:, :, 1] = seq.shape[1] - bbox[:, :, 1]
# bbox[:, :, 3] = seq.shape[1] - bbox[:, :, 3]


# def plot_gt(seq):
#     fig = plt.figure(frameon=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)

#     def update(i):
#         im.set_data(seq[i, :, :])
#         for j in range(bbox.shape[1]):
#             x, y = bbox[i, j, 0], bbox[i, j, 1]
#             w, h = bbox[i, j, 2] - bbox[i, j, 0], bbox[i, j, 3] - bbox[i, j, 1]
#             rects[j].set_width(w)
#             rects[j].set_height(h)
#             rects[j].set_xy([x, y])
#     im = ax.imshow(seq[0, :, :], cmap='gray', aspect='auto')
#     rects = [0]*bbox.shape[1]
#     for j in range(bbox.shape[1]):
#         x, y = bbox[0, j, 0], bbox[0, j, 1]
#         w, h = bbox[0, j, 2] - bbox[0, j, 0], bbox[0, j, 3] - bbox[0, j, 1]
#         rects[j] = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rects[j])
#     ax.set_xticks([])
#     ax.set_yticks([])

#     ax.get_xaxis().set_ticklabels([])
#     ax.get_yaxis().set_ticklabels([])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ani = animation.FuncAnimation(fig,
#                                   update,
#                                   seq.shape[0],
#                                   interval=1000/5,
#                                   blit=False)
#     return ani


# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# ani = plot_gt(seq)
# ani.save('anim.mp4', writer=writer)


seq = np.load('data/icons8_testing_fast_videos.npy')[0]
bbox = np.load('data/icons8_testing_fast_trajectories.npy')[0]
bbox = bbox.swapaxes(0, 1)
bbox[:, :, 1] = seq.shape[1] - bbox[:, :, 1]
bbox[:, :, 3] = seq.shape[1] - bbox[:, :, 3]
ids = np.repeat(np.arange(5), 100).reshape(5, 100, 1)
ids = ids.swapaxes(0, 1)
bbox = np.concatenate((bbox, ids), axis=2)

tracker = Sort()
trks = np.zeros(bbox.shape)
for t in range(bbox.shape[0]):
    dets = bbox[t].copy()
    dets[:, -1] = 1
    trks_ = tracker.update(dets)
    trks_[:, 2] = trks_[:, 2] - trks_[:, 0]
    trks_[:, 3] = trks_[:, 3] - trks_[:, 1]
    trks[t, :trks_.shape[0], :] = trks_
print(np.unique(trks[:, :, 4].flatten()))
assert 2 == 1
colors = ['r', 'g', 'b', 'c', 'm']


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