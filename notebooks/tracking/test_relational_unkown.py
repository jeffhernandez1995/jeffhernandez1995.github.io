import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lapsolver import solve_dense
from losses import AssociationLoss
from models import InteractionNet


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

digits = np.array([0, 2, 4, 6, 8])

X_train = np.load('datasets/X_train.npy')
y_train = np.load('datasets/y_train.npy')
X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Self connected graph
off_diag = np.triu(np.ones((digits.shape[0], digits.shape[0])))
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
n_iter = rel_rec.shape[0]
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)
if device:
    rel_rec = rel_rec.to(device)
    rel_send = rel_send.to(device)

model = InteractionNet(28*28, 256*2, digits.shape[0], n_iter, 0.25)
if device:
    model.to(device)
model.load_state_dict(torch.load('models/InteractionNet__11_loss=0.2297398.pth'))
weights = [1/3]*3
model.eval()
criterion = AssociationLoss(weights)
softmax = nn.Softmax(dim=1)
n_digits = 3

with torch.no_grad():
    for i, (inp, target) in enumerate(test_loader):
        if device:
            inp = inp.to(device)
            target = target.to(device)
        # inp[0, 4, 0, :] = torch.zeros(256, device=device) # (5, 256)

        y_pred = model(inp, rel_rec, rel_send)
        loss = criterion(y_pred, target)
        y_pred_ = softmax(y_pred[0])
        matrix = y_pred_.to('cpu').detach().numpy()
        y_pred_ = (y_pred_ * 10**n_digits).round() / (10**n_digits)
        rids, cids = solve_dense(-matrix)
        matched_indices = np.array([rids, cids]).T
        fig, ax = plt.subplots(nrows=5, ncols=2)
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                r = matched_indices[i, j]
                img = inp[0, r, j, :, :].to('cpu').detach().numpy()
                col.imshow(img, cmap='gray', aspect='auto')
                col.set_axis_off()
                col.set_xticks([])
                col.set_yticks([])
                col.get_xaxis().set_ticklabels([])
                col.get_yaxis().set_ticklabels([])
                col.get_xaxis().set_visible(False)
                col.get_yaxis().set_visible(False)
        plt.show()
        plt.close()
        rids, cids = solve_dense(-target[0].to('cpu').detach().numpy())
        matched_indices = np.array([rids, cids]).T
        fig, ax = plt.subplots(nrows=5, ncols=2)
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                r = matched_indices[i, j]
                img = inp[0, r, j, :, :].to('cpu').detach().numpy()
                col.imshow(img, cmap='gray', aspect='auto')
                col.set_axis_off()
                col.set_xticks([])
                col.set_yticks([])
                col.get_xaxis().set_ticklabels([])
                col.get_yaxis().set_ticklabels([])
                col.get_xaxis().set_visible(False)
                col.get_yaxis().set_visible(False)
        plt.show()
        plt.close()

        assert 2 == 1
