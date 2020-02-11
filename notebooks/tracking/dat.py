import numpy as np
import torch
import torch.nn as nn
from models import InteractionNet
from lapsolver import solve_dense


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class Tracker(object):
    count = 0

    def __init__(self, state, bbox):
        self.time_since_update = 0
        self.id = Tracker.count
        Tracker.count += 1
        self.state = state
        self.bbox = bbox
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, state, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.state = state
        self.bbox = bbox
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        return self.state

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.bbox


class DAT(object):
    def __init__(self, max_age=10000, min_hits=0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        Tracker.count = 0
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        off_diag = np.triu(np.ones((5, 5)))
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        n_iter = rel_rec.shape[0]
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)
        if self.device:
            self.rel_rec = self.rel_rec.to(self.device)
            self.rel_send = self.rel_send.to(self.device)
        self.internet = InteractionNet(28 * 28, 256*2, 5, n_iter, 0.5)
        self.internet.to(self.device)
        self.softmax = nn.Softmax(dim=1)
        self.internet.load_state_dict(torch.load('models/InteractionNet__29_loss=0.01199802.pth'))

    def update(self, images, bboxs):
        self.frame_count += 1
        if self.device:
            images = images.to(self.device)
            bboxs = bboxs.to(self.device)
        with torch.no_grad():
            features = self.siamesenet.get_embedding(images)
        if len(self.trackers) == 0:
            for (feat, bbox) in zip(features, bboxs):
                trk = Tracker(feat, bbox)
                self.trackers.append(trk)
            assert len(self.trackers) == 5
        else:
            inp = torch.zeros((5, 2, 256), device=self.device)
            inp[:features.size(0), 0, :] = features
            temp = []
            for trk in self.trackers:
                state = trk.predict()
                temp.append(state.view(1, -1))
            temp = torch.cat(temp, dim=0)
            assert temp.size(0) == 5
            inp[:temp.size(0), 1, :] = temp
            with torch.no_grad():
                matrix = self.internet(inp.unsqueeze(0), self.rel_rec, self.rel_send)
                matrix = self.softmax(matrix)
            matrix = matrix[0].to('cpu').detach().numpy()
            rids, cids = solve_dense(-matrix)
            matched_indices = np.array([rids, cids]).T
            matched_indices = matched_indices[:features.size(0)]
            for (d, t) in matched_indices:
                self.trackers[t].update(features[d], bboxs[d])
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state().to('cpu').detach().numpy()[:4]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
