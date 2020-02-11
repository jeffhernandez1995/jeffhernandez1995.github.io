import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.metrics import Loss, RunningAverage
from torch.utils.data import TensorDataset, DataLoader

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

# plt.imshow(X_test[0, 1, 0, :, :], cmap='gray')
# plt.show()
# plt.imshow(X_test[0, 0, 1, :, :], cmap='gray')
# plt.show()
# print(y_test[0])
# assert 2 == 1

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
weights = [1/3]*3
criterion = AssociationLoss(weights)
lr = 1e-4
optimizer = optim.RMSprop(model.parameters(), lr=lr)


def evaluate_function(engine, batch):
    if device:
        model.to(device)
    model.eval()
    with torch.no_grad():
        inp, target = batch
        if device:
            inp = inp.to(device)
            target = target.to(device)
        y_pred = model(inp, rel_rec, rel_send)
    return y_pred, target


def process_function(engine, batch):
    if device:
        model.to(device)
    model.train()
    optimizer.zero_grad()
    inp, target = batch
    if device:
        inp = inp.to(device)
        target = target.to(device)
    y_pred = model(inp, rel_rec, rel_send)
    loss = criterion(y_pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
training_history = {'loss':[]}
validation_history = {'loss':[]}

Loss(criterion).attach(evaluator, 'AssociationLoss')
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

pbar = ProgressBar(persist=True,
                   bar_format="")
pbar.attach(trainer, ['loss'])


def score_function(engine):
    val_loss = engine.state.metrics['AssociationLoss']
    return -val_loss


handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    loss = engine.state.metrics['loss']
    training_history['loss'].append(loss)
    print("Training Results - Epoch: {}  Avg loss: {:.2f}".format(trainer.state.epoch, loss))


def log_validation_results(engine):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    loss = metrics['AssociationLoss']
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {} Avg loss: {:.2f}".format(trainer.state.epoch, loss))


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)


def val_score(engine):
    evaluator.run(test_loader)
    avg_loss = evaluator.state.metrics['AssociationLoss']
    return -avg_loss


checkpointer = ModelCheckpoint('./models',
                               'InteractionNet',
                               score_function=val_score,
                               score_name='loss',
                               n_saved=2,
                               create_dir=True,
                               save_as_state_dict=True,
                               require_empty=False)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'': model})
trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

trainer.run(train_loader, max_epochs=30)

training_history['loss'] = training_history['loss'][1:]
validation_history['loss'] = validation_history['loss'][1:]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(len(training_history['loss'])), training_history['loss'], 'dodgerblue', label='training')
ax.plot(range(len(validation_history['loss'])), validation_history['loss'], 'orange', label='validation')
ax.set_xlim(0, len(validation_history['loss']))
ax.set_xlabel('Epoch')
ax.set_ylabel('Association Loss')
plt.title('Association Loss on Training/Validation Set')
plt.legend()
fig.savefig('association_loss.png', bbox_inches='tight')
