import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class owareNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.action_size = 6
        self.args = args

        super(owareNNet, self).__init__()

        # self.fc1 = nn.Linear(12+2+1, 64)
        self.fc1 = nn.Linear(12, 64)
        self.fc_bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, args.num_channels)
        self.fc_bn2 = nn.BatchNorm1d(args.num_channels)
        
        self.fc3 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn3 = nn.BatchNorm1d(args.num_channels)
        
        self.fc4 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn4 = nn.BatchNorm1d(args.num_channels)
        
        self.fc5 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn5 = nn.BatchNorm1d(args.num_channels)

        self.fc_pi = nn.Linear(args.num_channels, self.action_size)
        self.fc_v = nn.Linear(args.num_channels, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 12)                                    # batch_size x 1 x board(12)+point(2)+turn(1)
        # s = s.view(-1, 12+2+1)                                    # batch_size x 1 x board(12)+point(2)+turn(1)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 64
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        s = F.dropout(F.relu(self.fc_bn3(self.fc3(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        s = F.dropout(F.relu(self.fc_bn4(self.fc4(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        s = F.dropout(F.relu(self.fc_bn5(self.fc5(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc_pi(s)                                                                         # batch_size x action_size
        v = self.fc_v(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
