import random
import numpy as np
import os
import torch
from gym import make
from torch import nn

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DQN_nn(nn.Module):
    def __init__(self, state_dim, action_dim, seed, hidden=256):
        super(DQN_nn, self).__init__()
        self.seed = random.seed(seed)
        self.hidden = hidden
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self):
        self.Q = DQN_nn(state_dim=8, action_dim=4, seed=1, hidden=256)
        #self.Q.load_state_dict(torch.load('agent.pth'))
        self.Q.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl", map_location=device))
        #self.Q.load_state_dict(torch.load("agent.pkl", map_location=device))
        self.Q.eval()
        
    def act(self, state):
        Q_value = self.Q(torch.from_numpy(state).type(torch.FloatTensor))
        return torch.argmax(Q_value).item()

    def reset(self):
        pass
    