import random
from collections import deque, namedtuple
from copy import deepcopy
from random import sample

import numpy as np
import torch
import torch.nn as nn
from gym import make
from torch.nn import functional as F
from torch.optim import Adam


class Model(nn.Module):

    def __init__(self, state_dim, action_dim) -> None:
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_dim, 512)
        self.sigm1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.sigm2 = nn.ReLU()
        self.linear3 = nn.Linear(256, action_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_batch = batch
        input_batch = self.linear1(input_batch)
        input_batch = self.sigm1(input_batch)
        input_batch = self.linear2(input_batch)
        input_batch = self.sigm2(input_batch)
        result = self.linear3(input_batch)
        return result


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

GAMMA = 0.99
INITIAL_STEPS = 2048
TRANSITIONS = 700000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4


class DQN:
    def __init__(self, state_dim, action_dim) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.steps = 0  # Do not change
        self.model = Model(state_dim, action_dim).to(self.device)  # Torch model
        self.target = Model(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(deepcopy(self.model.state_dict()))  # Torch model
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=500000)
        self.batch_size = BATCH_SIZE

    def consume_transition(self, transition) -> None:
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.memory.append(Transition(*transition))

    def sample_batch(self) -> np.ndarray:
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = sample(self.memory, self.batch_size)
        return batch

    def train_step(self, transitions):
        batch = Transition(*zip(*transitions))
        done_mask = torch.Tensor(batch.done)
        # Use batch to update DQN's network.
        state_batch = torch.Tensor(batch.state).to(self.device)
        action_batch = torch.Tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.Tensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_values = torch.Tensor(batch.next_state).to(self.device)
        state_action_values = self.model(state_batch).gather(1, action_batch.long())

        expected_state_action_values = (self.target(next_state_values).max(1)[0].detach().unsqueeze(
            1) * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target.load_state_dict(deepcopy(self.model.state_dict()))

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        res = self.model(torch.FloatTensor(state).to(self.device))
        return res.argmax().item()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
