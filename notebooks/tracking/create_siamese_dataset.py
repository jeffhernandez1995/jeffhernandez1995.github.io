import numpy as np
import cv2

train_vids = np.load('data/icons8_training_fast_videos.npy')
train_bbox = np.load('data/icons8_training_fast_trajectories.npy')
print(train_vids.shape)
print(train_bbox.shape)
