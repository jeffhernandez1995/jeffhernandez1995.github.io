import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lapsolver import solve_dense
from datasets import AssociationMNIST
from losses import AssociationLoss
from models import InteractionNet


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

digits = np.array([1, 3, 5, 7, 9])
digit_size = 28
train_dataset = AssociationMNIST(True, digits=digits)
test_dataset = AssociationMNIST(False, digits=digits)

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

model = InteractionNet(256, 256*2, digits.shape[0], n_iter, 0.5)
if device:
    model.to(device)
model.load_state_dict(torch.load('models/InteractionNet__29_loss=0.01199802.pth'))
weights = [1/3]*3
model.eval()
criterion = AssociationLoss(weights)
softmax = nn.Softmax(dim=1)
n_digits = 3

with torch.no_grad():
    for i, (inp, target) in enumerate(train_loader):
        if device:
            inp = inp.to(device)
            target = target.to(device)
        # inp[0, 4, 0, :] = torch.zeros(256, device=device) # (5, 256)
        y_pred = model(inp, rel_rec, rel_send)
        loss = criterion(y_pred, target)
        y_pred_ = softmax(y_pred[0])
        y_pred_ = (y_pred_ * 10**n_digits).round() / (10**n_digits)
        matrix = y_pred[0].to('cpu').detach().numpy()
        rids, cids = solve_dense(-matrix)
        matched_indices = np.array([rids, cids]).T
        print(matched_indices)
        print(target[0])
        print(loss.item())
        assert 2 == 1
    
