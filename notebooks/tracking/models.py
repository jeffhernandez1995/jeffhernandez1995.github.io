import torch
import torch.nn as nn
import torch.nn.functional as F

# taken from https://github.com/ethanfetaya/NRI/blob/master/modules.py


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(10, 32, 5), nn.BatchNorm2d(32), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 10, 1), nn.BatchNorm2d(10), nn.ReLU(),
                                     )

        self.fc = nn.Sequential(nn.Linear(8 * 8, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU())
        self.drop_out = nn.Dropout()

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), output.size(1), -1)
        output = self.drop_out(output)
        output = self.fc(output)
        return output


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class InteractionNet(nn.Module):
    def __init__(self, n_in, n_hid, n_dig, n_iter, do_prob=0.):
        super(InteractionNet, self).__init__()
        self.n_dig = n_dig
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_iter * n_hid, self.n_dig ** 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def node2edge_temp(self, x1, x2, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x2)
        senders = torch.matmul(rel_send, x1)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [B, num_objs, 2, 28, 28]
        shape = list(inputs.size())

        x = inputs.view(shape[0], shape[1] * shape[2], -1)
        # New shape: [B, num_objs * 2, 28 * 28]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = x.view(shape[0], shape[1], shape[2], -1)
        x1 = x[:, :, 0, :].view(shape[0], shape[1], -1)
        x2 = x[:, :, 1, :].view(shape[0], shape[1], -1)
        x = self.node2edge_temp(x1, x2, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        x = self.mlp3(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x).view(shape[0], -1)
        return self.fc_out(x).view(shape[0], self.n_dig, self.n_dig)
