import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, state_dim, action_dim) -> None:
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_dim, 512)
        self.sigm1 = nn.Sigmoid()
        self.linear2 = nn.Linear(512, 256)
        self.sigm2 = nn.Sigmoid()
        self.linear3 = nn.Linear(256, 128)
        self.sigm3 = nn.Sigmoid()
        self.linear4 = nn.Linear(128, 64)
        self.sigm4 = nn.Sigmoid()
        self.linear5 = nn.Linear(64, action_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_batch = batch
        input_batch = self.linear1(input_batch)
        input_batch = self.sigm1(input_batch)
        input_batch = self.linear2(input_batch)
        input_batch = self.sigm2(input_batch)
        input_batch = self.linear3(input_batch)
        input_batch = self.sigm3(input_batch)
        input_batch = self.linear4(input_batch)
        input_batch = self.sigm4(input_batch)
        result = self.linear5(input_batch)
        return result
