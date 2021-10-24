import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def act(self, state):
        res = self.model(torch.FloatTensor(state).to(self.device))
        return res.argmax().item()

