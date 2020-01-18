import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import EmbeddingNet, SiameseNet
# mostly based on https://github.com/edenton/svg/blob/master/data/moving_mnist.py


class MovingMNIST(Dataset):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self,
                 train,
                 seq_len=100,
                 num_digits=3,
                 digits=np.arange(10)):

        self.seq_len = seq_len
        self.num_digits = num_digits
        self.digits = digits
        self.digit_size = 28
        self.image_size = self.digit_size * self.num_digits
        self.channels = 1

        if train:
            self.X = np.load('data/X_train.npy')
            self.Y = np.load('data/Y_train.npy')
        else:
            self.X = np.load('data/X_test.npy')
            self.Y = np.load('data/Y_test.npy')
        self.N = self.X.shape[0]
        inds = [i for i in range(self.Y.shape[0]) if self.Y[i] in self.digits]
        self.X = self.X[inds]
        self.Y = self.Y[inds]

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = np.zeros((self.seq_len,
                      self.image_size,
                      self.image_size,
                      self.channels), dtype=np.float32)
        bbox = np.zeros((self.seq_len, self.num_digits, 5))
        for n in range(self.num_digits):
            idx = np.random.randint(self.X.shape[0])
            digit = self.X[idx]
            sx = np.random.randint(self.image_size - self.digit_size)
            sy = np.random.randint(self.image_size - self.digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    dy = np.random.randint(1, 5)
                    dx = np.random.randint(-4, 5)
                elif sy >= self.image_size - self.digit_size:
                    sy = self.image_size - self.digit_size - 1
                    dy = np.random.randint(-4, 0)
                    dx = np.random.randint(-4, 5)
                if sx < 0:
                    sx = 0
                    dx = np.random.randint(1, 5)
                    dy = np.random.randint(-4, 5)
                elif sx >= self.image_size - self.digit_size:
                    sx = self.image_size - self.digit_size - 1
                    dx = np.random.randint(-4, 0)
                    dy = np.random.randint(-4, 5)
                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit
                bbox[t, n, :] = [sx, sy, sx+self.digit_size, sy+self.digit_size, n]
                sy += dy
                sx += dx
        x[x > 1] = 1.
        for t in range(self.seq_len):
            np.random.shuffle(bbox[t, :, :])
        return torch.from_numpy(x), torch.from_numpy(bbox)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train, digits=np.arange(10)):
        self.train = train
        self.digits = digits
        if self.train:
            self.Y = np.load('data/Y_train.npy').astype(float)
            self.X = np.load('data/X_train.npy').astype(float)
            inds = [i for i in range(self.Y.shape[0]) if self.Y[i] in self.digits]
            self.X = self.X[inds]
            self.Y = self.Y[inds]
            self.labels_set = set(self.Y)
            self.label_to_indices = {label: np.where(self.Y == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.X = np.load('data/X_test.npy').astype(float)
            self.Y = np.load('data/Y_test.npy').astype(float)
            inds = [i for i in range(self.Y.shape[0]) if self.Y[i] in self.digits]
            self.X = self.X[inds]
            self.Y = self.Y[inds]
            self.labels_set = set(self.Y)
            self.label_to_indices = {label: np.where(self.Y == label)[0]
                                     for label in self.labels_set}

            positive_pairs = [[i,
                               np.random.choice(self.label_to_indices[self.Y[i]]),
                               1]
                              for i in range(0, len(self.X), 2)]

            negative_pairs = [[i,
                               np.random.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.Y[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.X), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = torch.as_tensor(np.random.randint(0, 2))
            img1, label1 = torch.as_tensor(self.X[index]), torch.as_tensor(self.Y[index])
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1.item()])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1.item()])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = torch.as_tensor(self.X[siamese_index])
        else:
            img1 = torch.as_tensor(self.X[self.test_pairs[index][0]])
            img2 = torch.as_tensor(self.X[self.test_pairs[index][1]])
            target = torch.as_tensor(self.test_pairs[index][2])
        return img1.unsqueeze(0).float(), img2.unsqueeze(0).float(), target

    def __len__(self):
        return self.X.shape[0]


class AssociationMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train, digits=np.arange(10)):
        self.train = train
        self.digits = digits
        self.size = digits.shape[0]

        embedding_net = EmbeddingNet()
        self.model = SiameseNet(embedding_net)
        self.device = "cpu"
        self.model.load_state_dict(torch.load('models/SiameseNet_SiameseNet_11_loss=0.001986655.pth'))
        if self.device:
            self.model.to(self.device)
        self.model.eval()
        if train:
            self.X = np.load('data/X_train.npy')
            self.Y = np.load('data/Y_train.npy')
        else:
            self.X = np.load('data/X_test.npy')
            self.Y = np.load('data/Y_test.npy')
        self.N = self.X.shape[0]
        inds = [i for i in range(self.Y.shape[0]) if self.Y[i] in self.digits]
        self.X = self.X[inds]
        self.Y = self.Y[inds]

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = np.zeros((self.size, 2, 1, self.X.shape[1], self.X.shape[1]))
        y = np.zeros((self.size, self.size))
        for i in range(self.size):
            idx = np.random.randint(self.X.shape[0])
            x[i, 0, 0, :, :] = self.X[idx]
        permidx = np.random.permutation(self.size)
        for i, index in enumerate(permidx):
            x[index, 1, 0, :, :] = x[i, 0, 0, :, :]
            y[i, index] = 1
        inp =  torch.as_tensor(x).float()
        inp = inp.view(self.size * 2, 1, self.X.shape[1], self.X.shape[1])
        with torch.no_grad():
            if self.device:
                inp = inp.to(self.device)
            inp = self.model.get_embedding(inp)
        inp = inp.view(self.size, 2, -1).to('cpu')
        target =  torch.as_tensor(y).float()
        return inp, target

