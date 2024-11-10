import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from env import Environment as Env

seed = 695

def reseed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

reseed(seed)

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(PolicyNet, self).__init__()

        self.input = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        X_0 = torch.relu(self.input(state))
        X_1 = self.output(X_0)

        return torch.softmax(X_1, dim=-1)
    

class PolicyGradient:
    def __init__(self, env: Env, policy_net, seed, reward_to_go: bool = True):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
        self.reward_to_go = reward_to_go
        self.seed = seed

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def select_action(self, state):
        state = torch.tensor(state).to(self.device)

        return torch.distributions.Categorical(self.policy_net.forward(state)).sample().item()

    def compute_loss(self, episode, gamma):
        states = torch.tensor([s for (s, a, r) in episode]).to(self.device)
        action = torch.tensor([a for (s, a, r) in episode]).to(self.device)
        reward = torch.tensor([r for (s, a, r) in episode]).to(self.device)

        R = []

        if not self.reward_to_go:
            cum_reward = 0.0
            for i in range(len(reward)):
              cum_reward += (gamma ** i) * reward[i]

            for r in reward: R.append(cum_reward)

        else:
            cum_reward = 0.0
            T = range(len(states))
            for t in T:
              for i in range(t, len(reward)):
                cum_reward += (gamma ** (i - t)) * reward[i]
              R.append(cum_reward)

        sum = 0
        for (s, a) in zip(states, action):
          sum += torch.log(self.policy_net(s)[a]) * R[0]
          R.pop(0)

        loss = -1  * sum
        return loss

    def update_policy(self, episodes, optimizer, gamma):
        optimizer.zero_grad()

        loss = 0
        for episode in episodes: loss += self.compute_loss(episode, gamma)
        loss /= len(episodes)
        loss.backward()
        optimizer.step()


    ## TODO ##
    def run_episode(self):
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        state = self.env.reset()
        episode = []
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, num_iterations, batch_size, gamma, lr):
        self.policy_net.train()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        lst = []

        for i in range(num_iterations):
          episodes = []

          for _ in range(batch_size): episodes.append(self.run_episode())
          self.update_policy(episodes, optimizer, gamma)

          if i % 10 == 0:
            lst.append(self.evaluate(10))

        return lst

    def evaluate(self, num_episodes = 100):
        self.policy_net.eval()
    
        reward = 0
        for _ in range(num_episodes):
          episode = self.run_episode()
          reward += sum([reward for (_, _, reward) in episode])

        average_reward = reward / num_episodes
        return average_reward
    