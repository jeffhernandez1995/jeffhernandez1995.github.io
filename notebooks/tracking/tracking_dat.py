from collections import OrderedDict

import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from dat import DAT
import matplotlib.pyplot as plt
import motmetrics as mm
from copy import deepcopy


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts.keys():
            accs.append(mm.utils.compare_to_groundtruth(list(gts.items())[k][1], tsacc, 'iou', distth=0.5))
            names.append(k)
    return accs, names


seq_len = 100
num_digits = 5
vids = np.load('data/icons8_testing_fast_videos.npy')[:seq_len]
bboxs = np.load('data/icons8_testing_fast_trajectories.npy')[:seq_len]
bboxs[:, :, :, 3] = vids.shape[2] - bboxs[:, :, :, 3]
bboxs[:, :, :, 1] = vids.shape[2] - bboxs[:, :, :, 1]
bboxs = bboxs.swapaxes(1, 2)
ids = np.repeat(np.arange(num_digits), seq_len * seq_len).reshape(num_digits, seq_len, seq_len, 1)
ids = ids.swapaxes(0, 2)
bboxs = np.concatenate((bboxs, ids), axis=3)

gt = []
dt = []
names = ['FrameId', 'X', 'Y', 'Width', 'Height', 'Id', 'Confidence', 'ClassId', 'Visibility']
for i in range(100):  # len(test_dset)
    seq = vids[i]
    bbox = bboxs[i]
    temp_bbox = deepcopy(bbox)
    tracker = DAT()
    trks_ = np.zeros((0, 9))
    for t in range(bbox.shape[0]):
        dets = bbox[t].copy()
        # if np.random.uniform() < 0.1 and t > 0:
        #     idx = torch.ones(dets.shape[0]).long()
        #     idel = np.random.randint(dets.shape[0])
        #     idx[idel] = 0
        #     dets = dets[idx]
        dets[:, -1] = 1  # remove ground truth id replace with condifence
        images = []
        for det in dets:
            pts = det.astype(int)
            y1 = min(127, max(0, pts[1]))
            y2 = min(127, max(0, pts[3]))
            x1 = min(127, max(0, pts[0]))
            x2 = min(127, max(0, pts[2]))
            img = seq[t, y1:y2, x1:x2]
            img = cv2.resize(img, (28, 28))
            img = torch.from_numpy(img / 255.).float()
            images.append(img.unsqueeze(0))
        images = torch.cat(images, dim=0)
        dets = torch.from_numpy(dets).float()
        trks = tracker.update(images, dets)
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
    # print(dt_df.head(10))
    dt.append(dt_df)
    # ground truth
    framesid = np.repeat(np.arange(1, seq_len+1), num_digits)
    bbox_ = temp_bbox.reshape((-1, 5))
    bbox_ = np.concatenate((framesid.reshape(framesid.shape[0], 1), bbox_), axis=1)
    bbox_ = np.concatenate((bbox_, np.ones((framesid.shape[0], 3))), axis=1)
    bbox_[:, 3] = bbox_[:, 3] - bbox_[:, 1]
    bbox_[:, 4] = bbox_[:, 4] - bbox_[:, 2]
    gt_df = pd.DataFrame(bbox_,
                         columns=names)
    gt_df.index = pd.MultiIndex.from_arrays(gt_df[['FrameId', 'Id']].values.T, names=['FrameId', 'Id'])
    del gt_df['FrameId']
    del gt_df['Id']
    # print(gt_df.head(10))
    gt.append(gt_df)

gt = OrderedDict([(i, df) for i, df in enumerate(gt)])
dt = OrderedDict([(i, df) for i, df in enumerate(dt)])
mh = mm.metrics.create()
accs, names = compare_dataframes(gt, dt)
summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

summary.to_csv('dat_results.csv', index=False)