import gym
import torch
from torch import nn
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F
import copy


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
C = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_state(state):
    return np.array(state)

class ReplayBuffer:
    """we store a fixed amount of the last transitions <f_t, a_t, r_t, f_t+1>"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, transition):
        e = self.experience(*transition)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

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

class DQN:
    def __init__(self, state_dim, action_dim, seed):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.Q = DQN_nn(state_dim, action_dim, seed)
        self.Q.apply(init_weights)
        self.TQ = copy.deepcopy(self.Q)
        self.Q.to(device)
        self.TQ.to(device)
        self.buffer = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=LR)
        self.loss = F.smooth_l1_loss
        self.t_step = 0
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad(): 
            Q_value = self.Q(state)
        self.Q.train()
        return torch.argmax(Q_value).item()        
    
    def step(self, transition):
        self.t_step = (self.t_step + 1) % C
        self.buffer.add(transition)
        if len(self.buffer) > BATCH_SIZE:
            train_batch = self.buffer.sample()
            self.train(train_batch)
        if self.t_step == 0:
            self.TQ = copy.deepcopy(self.Q)        
        
    def train(self, train_batch):
        states, actions, rewards, next_states, dones = train_batch
        
        Q = self.Q(states).gather(1, actions)
        Q1 = self.TQ(next_states)
        maxQ1 = torch.max(Q1, -1)[0].unsqueeze(1)
        TQ = rewards + (self.gamma * maxQ1.detach() * (1 - dones))
        
        loss = self.loss(Q, TQ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="agent.pkl"):
        torch.save(self.Q.state_dict(), path)
        #torch.save(self.Q, path)

import time
env = gym.make("LunarLander-v2")
env.seed(1); #torch.manual_seed(1); np.random.seed(1)
dqn = DQN(state_dim=8, action_dim=4, seed=1)
#actions: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
#state: coordinates are the first two numbers in state vector.
eps = 0.1 
episodes = 600
max_steps = 1000
max_reward = float('-inf')
rewards_window = deque(maxlen=100)
start = time.time()

for i in range(1, episodes+1):
    state = transform_state(env.reset())
    total_reward = 0
    steps = 0
    done = False
    for _ in range(max_steps):
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = transform_state(next_state)
        total_reward += reward
        steps += 1
        dqn.step((state, action, reward, next_state, done))
        state = next_state
        if done: 
            break
            
    rewards_window.append(total_reward)
    eps = max(eps*0.95, 0.001) 
    
    if total_reward > max_reward:
        max_reward = total_reward
        dqn.save()

    if (i % 20 == 0) and (i >= 100):
        if np.mean(rewards_window) >= 250.0:
            print(f'Environment solved in {i} episodes!\tAverage Score: {np.mean(rewards_window):.1f}')            
        else:
            result = time.time() - start
            print(f'{i} episode: {np.mean(rewards_window):.1f} average reward, {result:.1f} seconds')
