import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from datasets import SiameseMNIST
from losses import ContrastiveLoss
from models import EmbeddingNet, SiameseNet


def plot_embedding(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    print(x_min, x_max)
    X = (X - x_min) / (x_max - x_min)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(imgs[i, 0, :, :], cmap=plt.cm.gray_r),
            X[i])
        ax.add_artist(imagebox)
    ax.set_xticks([]), ax.set_yticks([])
    return fig

embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
if device:
    model.to(device)
model.load_state_dict(torch.load('models/SiameseNet_SiameseNet_11_loss=0.001986655.pth'))
model.eval()
print(model)

# Known digits
digits = np.array([0, 2, 4, 6, 8])
imgs = np.load('data/X_test.npy')
labels = np.load('data/Y_test.npy')
inds = [i for i in range(labels.shape[0]) if labels[i] in digits]
imgs = imgs[inds].reshape((-1, 1, 28, 28))
y = labels[inds]

inp = torch.from_numpy(imgs[:1000]).float()
if device:
    inp = inp.to(device)
with torch.no_grad():
    preds = model.get_final(inp)
    preds = preds.to('cpu').numpy()

# tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
# X_tsne = tsne.fit_transform(preds)
fig = plot_embedding(preds)
# plt.title('Embedding created by Siamese Network \n On testing set, known classes')
fig.savefig('embedding_known.png', bbox_inches='tight')

# Known digits
digits = np.array([1, 3, 5, 7, 9])
imgs = np.load('data/X_test.npy')
labels = np.load('data/Y_test.npy')
inds = [i for i in range(labels.shape[0]) if labels[i] in digits]
imgs = imgs[inds].reshape((-1, 1, 28, 28))
y = labels[inds]

inp = torch.from_numpy(imgs[:1000]).float()
if device:
    inp = inp.to(device)
with torch.no_grad():
    preds = model.get_final(inp)
    preds = preds.to('cpu').numpy()

# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(preds)
fig = plot_embedding(preds)
# plt.title('Embedding created by Siamese Network \n On testing set, known classes')
fig.savefig('embedding_unknown.png', bbox_inches='tight')