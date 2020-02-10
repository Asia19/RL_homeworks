from gym import make
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
import random

N_STEP = 1
GAMMA = 0.96
max_epsilon = 0.5
min_epsilon = 0.1
episodes = 2000

def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result).reshape(2, 1)

def test():
    local_env = make("MountainCar-v0")
    rewards = []
    for i in range(100):
        state = local_env.reset()
        state = transform_state(state)
        total_reward = 0
        done = False
        while not done:
            action = aql.act(state)
            next_state, reward, done, _ = local_env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards

class AQL:
    def __init__(self, state_dim, action_dim, gamma, lr):
        #self.gamma = GAMMA ** N_STEP
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.W = np.random.normal(loc=0, scale=np.sqrt(1 / ((action_dim + state_dim))), size=(action_dim, state_dim))
        self.b = np.zeros((action_dim,1))
        
    def forward(self, state):
        return self.W @ state + self.b
    
    def update(self, transition):
        state, action, next_state, reward, done = transition        
        Q = self.forward(state)[action]
        Q_target = reward + self.gamma * np.max(self.forward(next_state))
        self.W[action] = self.W[action] + self.lr * (Q_target - Q) * state.T
        self.b[action] = self.b[action] + self.lr * (Q_target - Q)

    def act(self, state, target=False):
        return np.argmax(self.W @ state + self.b)

    def save(self, path="agent.pth"):
        np.savez("agent.npz", self.W, self.b)

def get_epsilon(step):
    return max_epsilon - (max_epsilon - min_epsilon) * step / episodes
    
if __name__ == "__main__":
    
    env = make("MountainCar-v0")
    env.seed(1)
    lr = 0.05
    gamma = 0.95
    aql = AQL(state_dim=2, action_dim=3, gamma=gamma, lr=lr)
    #eps = 1
    rewards = []
    best_result = -200

    for i in range(episodes):
        state = transform_state(env.reset())
        eps = get_epsilon(i+1)
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward += 15 * (abs(next_state[1])) 
            #reward += 300 * (gamma * abs(next_state[1]) - abs(state[1]))
            next_state = transform_state(next_state)
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        rewards.append(total_reward)
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))
        if i % 15 == 0:
            test_result = np.mean(test())
            if test_result > best_result:
                best_result = test_result
                print('TEST result', best_result)
                aql.save()
        eps *= 0.95