import random

import numpy as np
import torch

random.seed(99)
torch.manual_seed(99)
np.random.seed(99)


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(self.device)
            action, _, _ = self.model.act(state)
        return action.cpu().numpy()

    def reset(self):
        pass
