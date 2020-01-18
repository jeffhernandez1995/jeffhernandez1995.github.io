from collections import OrderedDict

import numpy as np
import pandas as pd
from datasets import MovingMNIST
from sort import Sort

import motmetrics as mm


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts.keys():
            accs.append(mm.utils.compare_to_groundtruth(list(gts.items())[k][1], tsacc, 'iou', distth=0.5))
            names.append(k)
    return accs, names

num_digits = 5
test_digits = np.array([1, 3, 5, 7, 9])
seq_len = 100
test_dset = MovingMNIST(True, seq_len=seq_len, num_digits=num_digits, digits=test_digits)

gt = []
dt = []
names = ['FrameId', 'X', 'Y', 'Width', 'Height', 'Id', 'Confidence', 'ClassId', 'Visibility']
for i in range(100): # len(test_dset)
    seq, bbox = test_dset[i]
    bbox = bbox.numpy()
    tracker = Sort()
    trks_ = np.zeros((0, 9))
    for t in range(bbox.shape[0]):
        dets = bbox[t].copy()
        if np.random.uniform() < 0.1:
            idel = np.random.randint(dets.shape[0])
            dets = np.delete(dets, (idel), axis=0)
        dets[:, -1] = 1 # remove ground truth id replace with condifence
        trks = tracker.update(dets)
        trks[:, 2] = trks[:, 2] - trks[:, 0]
        trks[:, 3] = trks[:, 3] - trks[:, 1]
        trks = np.concatenate(((t + 1) * np.ones((trks.shape[0],1)), trks), axis=1)
        trks = np.concatenate((trks, np.ones((trks.shape[0],3))), axis=1)
        trks_ = np.concatenate((trks_, trks), axis=0)
    dt_df = pd.DataFrame(trks_,
                         columns=names)
    dt_df.index = pd.MultiIndex.from_arrays(dt_df[['FrameId', 'Id']].values.T, names=['FrameId', 'Id'])
    del dt_df['FrameId']
    del dt_df['Id']
    dt.append(dt_df)
    # ground truth
    framesid = np.repeat(np.arange(1, seq_len+1), num_digits)
    bbox_ = bbox.reshape((-1, 5))
    bbox_ = np.concatenate((framesid.reshape(framesid.shape[0],1), bbox_), axis=1)
    bbox_ = np.concatenate((bbox_, np.ones((framesid.shape[0],3))), axis=1)
    bbox_[:, 3] = 28
    bbox_[:, 4] = 28
    gt_df = pd.DataFrame(bbox_,
                         columns=names)
    gt_df.index = pd.MultiIndex.from_arrays(gt_df[['FrameId', 'Id']].values.T, names=['FrameId', 'Id'])
    del gt_df['FrameId']
    del gt_df['Id']
    gt.append(gt_df)

gt = OrderedDict([(i, df) for i, df in enumerate(gt)])
dt = OrderedDict([(i, df) for i, df in enumerate(dt)])
mh = mm.metrics.create()
accs, names = compare_dataframes(gt, dt)
summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

summary.to_csv('kalman_results.csv', index=False)