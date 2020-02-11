import numpy as np
import cv2
import matplotlib.pyplot as plt

# vids = np.load('data/mnist_training_fast_videos.npy')
# bbox = np.load('data/mnist_training_fast_trajectories.npy')

# bbox[:, :, :, 3] = vids.shape[2] - bbox[:, :, :, 3]
# bbox[:, :, :, 1] = vids.shape[2] - bbox[:, :, :, 1]
# bbox = bbox.swapaxes(1, 2)

# length = 20000

# X_train = np.zeros((length, 5, 2, 28, 28))
# y_train = np.zeros((length, 5, 5))


# i = 0
# count = 0
# while count < length:
#     print(f'{i/length}: There are {count} examples so far')
#     num_t = vids.shape[1]
#     indexes = np.triu(np.ones((num_t, num_t)) - np.eye(num_t))
#     indexes = np.array([np.where(indexes)[0], np.where(indexes)[1]]).T
#     inds = np.random.choice(indexes.shape[0], 250)
#     indexes = indexes[inds]
#     for idx in indexes:
#         seq1 = vids[i, idx[0], :, :]
#         seq2 = vids[i, idx[1], :, :]
#         bbox1 = bbox[i, idx[0], :, :].astype(int)
#         bbox2 = bbox[i, idx[1], :, :].astype(int)
#         for j in range(5):
#             # print(bbox1[j])
#             y1 = min(127, max(0, bbox1[j, 1]))
#             y2 = min(127, max(0, bbox1[j, 3]))
#             x1 = min(127, max(0, bbox1[j, 0]))
#             x2 = min(127, max(0, bbox1[j, 2]))
#             img = seq1[y1:y2, x1:x2]
#             # print(x1, x2, y1, y2)
#             img = cv2.resize(img, (28, 28))
#             X_train[count, j, 0, :, :] = img / 255.
#         permidx = np.random.permutation(5)
#         for j, index in enumerate(permidx):
#             # print(bbox2[j])
#             y1 = min(127, max(0, bbox2[j, 1]))
#             y2 = min(127, max(0, bbox2[j, 3]))
#             x1 = min(127, max(0, bbox2[j, 0]))
#             x2 = min(127, max(0, bbox2[j, 2]))
#             # print(x1, x2, y1, y2)
#             img = seq2[y1:y2, x1:x2]
#             img = cv2.resize(img, (28, 28))
#             X_train[count, index, 1, :, :] = img / 255.
#             y_train[count, j, index] = 1
#         # assert 2 == 1
#         count += 1
#     i += 1

# permidx = np.random.permutation(X_train.shape[0])
# X_train = X_train[permidx]
# y_train = y_train[permidx]
# np.save('datasets/X_train.npy', X_train)
# np.save('datasets/y_train.npy', y_train)

vids = np.load('data/icons8_testing_fast_videos.npy')
bbox = np.load('data/icons8_testing_fast_trajectories.npy')

bbox[:, :, :, 3] = vids.shape[2] - bbox[:, :, :, 3]
bbox[:, :, :, 1] = vids.shape[2] - bbox[:, :, :, 1]
bbox = bbox.swapaxes(1, 2)

length = 20000

X_test = np.zeros((length, 5, 2, 28, 28))
y_test = np.zeros((length, 5, 5))


i = 0
count = 0
while count < length:
    print(f'{i/length}: There are {count} examples so far')
    num_t = vids.shape[1]
    indexes = np.triu(np.ones((num_t, num_t)) - np.eye(num_t))
    indexes = np.array([np.where(indexes)[0], np.where(indexes)[1]]).T
    inds = np.random.choice(indexes.shape[0], 250)
    indexes = indexes[inds]
    for idx in indexes:
        seq1 = vids[i, idx[0], :, :]
        seq2 = vids[i, idx[1], :, :]
        bbox1 = bbox[i, idx[0], :, :].astype(int)
        bbox2 = bbox[i, idx[1], :, :].astype(int)
        for j in range(5):
            # print(bbox1[j])
            y1 = min(127, max(0, bbox1[j, 1]))
            y2 = min(127, max(0, bbox1[j, 3]))
            x1 = min(127, max(0, bbox1[j, 0]))
            x2 = min(127, max(0, bbox1[j, 2]))
            img = seq1[y1:y2, x1:x2]
            # print(x1, x2, y1, y2)
            img = cv2.resize(img, (28, 28))
            X_test[count, j, 0, :, :] = img / 255.
        permidx = np.random.permutation(5)
        for j, index in enumerate(permidx):
            # print(bbox2[j])
            y1 = min(127, max(0, bbox2[j, 1]))
            y2 = min(127, max(0, bbox2[j, 3]))
            x1 = min(127, max(0, bbox2[j, 0]))
            x2 = min(127, max(0, bbox2[j, 2]))
            # print(x1, x2, y1, y2)
            img = seq2[y1:y2, x1:x2]
            img = cv2.resize(img, (28, 28))
            X_test[count, index, 1, :, :] = img / 255.
            y_test[count, j, index] = 1
        count += 1
    i += 1

permidx = np.random.permutation(X_test.shape[0])
X_test = X_test[permidx]
y_test = y_test[permidx]
np.save('datasets/X_test.npy', X_test)
np.save('datasets/y_test.npy', y_test)
