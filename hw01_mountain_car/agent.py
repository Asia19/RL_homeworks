import random
import numpy as np
import os
#from train import transform_state
#from gym import make

def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result).reshape(2, 1)

class Agent:
    def __init__(self):
        #self.weight, self.bias = np.load(__file__[:-8] + "/agent.npz")
        file = np.load(__file__[:-8] + "/agent.npz")
        #file = np.load("agent.npz")
        self.weight = file[file.files[0]]
        self.bias = file[file.files[1]]
        
    def act(self, state):
        return np.argmax(self.weight.dot(transform_state(state)) + self.bias)

    def reset(self):
        pass
    