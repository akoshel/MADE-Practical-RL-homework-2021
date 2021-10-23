import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 16)
        self.sigm1 = nn.Sigmoid()
        self.linear2 = nn.Linear(16, 16)
        self.sigm2 = nn.Sigmoid()
        self.linear3 = nn.Linear(16, 16)
        self.sigm3 = nn.Sigmoid()
        self.linear4 = nn.Linear(16, 4)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_batch = batch
        input_batch = self.linear1(input_batch)
        input_batch = self.sigm1(input_batch)
        input_batch = self.linear2(input_batch)
        input_batch = self.sigm2(input_batch)
        input_batch = self.linear3(input_batch)
        input_batch = self.sigm3(input_batch)
        result = self.linear4(input_batch)
        return result
