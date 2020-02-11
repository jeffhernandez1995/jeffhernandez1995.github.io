import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociationLoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(AssociationLoss, self).__init__()
        self.w = weights
        self.bce = nn.BCELoss(reduction=reduction)
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        y1 = self.softmax(y_pred)
        y2 = self.softmax(y_pred.transpose(1, 2)).transpose(2, 1)
        loss1 = self.bce(y1, y_true)
        loss2 = self.bce(y2, y_true)
        loss3 = self.mse(y1, y2)
        loss = self.w[0] * loss1 + self.w[1] * loss2 + self.w[2] * loss3
        return loss
