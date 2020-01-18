import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.metrics import Loss, RunningAverage
from torch.utils.data import DataLoader

from datasets import SiameseMNIST
from losses import ContrastiveLoss
from models import EmbeddingNet, SiameseNet

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

train_dataset = SiameseMNIST(True, digits=np.array([0, 2, 4, 6, 8]))
# print(len(train_dataset))
test_dataset = SiameseMNIST(False, digits=np.array([0, 2, 4, 6, 8]))
# print(len(test_dataset))
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
# if cuda:
#     model.cuda()
criterion = ContrastiveLoss(margin, reduction='mean')
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

def evaluate_function(engine, batch):
    if device:
        model.to(device)
    model.eval()
    with torch.no_grad():
        x1, x2, target = batch
        if device:
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)
        y1, y2 = model(x1, x2)
        kwargs = {'target': target} # 'output1': y1, 'output2': y2, 
    return y1, y2, kwargs

def process_function(engine, batch):
    if device:
        model.to(device)
    model.train()
    optimizer.zero_grad()
    x1, x2, target = batch
    if device:
        x1 = x1.to(device)
        x2 = x2.to(device)
        target = target.to(device)
    y1, y2 = model(x1, x2)
    loss = criterion(y1, y2, target)
    loss.backward()
    optimizer.step()
    return loss.item()
        
trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
training_history = {'loss':[]}
validation_history = {'loss':[]}

Loss(criterion).attach(evaluator, 'nll')
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

def score_function(engine):
    val_loss = engine.state.metrics['nll']
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
    loss = metrics['nll']
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {} Avg loss: {:.2f}".format(trainer.state.epoch, loss))
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

def val_score(engine):
    evaluator.run(test_loader)
    avg_loss = evaluator.state.metrics['nll']
    return -avg_loss


checkpointer = ModelCheckpoint('./models',
                               'SiameseNet',
                               score_function=val_score,
                               score_name='loss',
                               n_saved=2,
                               create_dir=True,
                               save_as_state_dict=True,
                               require_empty=False)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'SiameseNet': model})
trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

trainer.run(train_loader, max_epochs=30)

training_history['loss'] = training_history['loss'][1:]
validation_history['loss'] = validation_history['loss'][1:]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(len(training_history['loss'])), training_history['loss'], 'dodgerblue', label='training')
ax.plot(range(len(validation_history['loss'])), validation_history['loss'], 'orange', label='validation')
ax.set_xlim(0, len(validation_history['loss']));
ax.set_xlabel('Epoch')
ax.set_ylabel('Contrastive Loss')
plt.title('Contrastive Loss on Training/Validation Set')
plt.legend()
fig.savefig('contrastive_loss.png', bbox_inches='tight')