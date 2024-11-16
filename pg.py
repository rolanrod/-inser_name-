import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from env import Environment
import pygame
from colors import Colors

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
        state = state.float()
        X_0 = torch.relu(self.input(state))
        X_1 = self.output(X_0)

        return torch.softmax(X_1, dim=-1)
    

class PolicyGradient:
    def __init__(self, env: Environment, policy_net: PolicyNet, seed: int, reward_to_go: bool = True):
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
        states = torch.tensor(np.array([s for (s, _, _) in episode])).to(self.device)
        action = torch.tensor(np.array([a for (_, a, _) in episode])).to(self.device)
        reward = torch.tensor(np.array([r for (_, _, r) in episode])).to(self.device)

        R = []

        if not self.reward_to_go:
            cum_reward = 0.0
            for i in range(len(reward)):
              cum_reward += (gamma ** i) * reward[i]

            for _ in reward: R.append(cum_reward)

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

    def run_episode(self):
        render = False
        if render:
           pygame.init()

        states, actions, rewards, episode = [], [], [], []
        self.env.reset()
        features = self.env.features_to_vector(self.env.extract_features())
        state = np.array(features, dtype=np.float32)

        done = False
        timestep = 0
        while not done:
            if render:
               self.render_game()

            action = self.select_action(state)
            next_state, reward, done, timestep = self.env.step(action, timestep)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, num_iterations, batch_size, gamma, lr):
        self.policy_net.train()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        lst = []

        for i in range(num_iterations):
          print(f"EPISODE {i}")
          episodes = []

          for j in range(batch_size): 
            print(f"EPISODE {j + (i * 10)}")
            episodes.append(self.run_episode())
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
    

    def render_game(self):
        ## Ugly render_game code ##
        game = self.env
        title_font = pygame.font.Font(None, 40)
        score_surface = title_font.render("Score", True, Colors.white)
        next_surface = title_font.render("Next", True, Colors.white)
        game_over_surface = title_font.render("GAME OVER", True, Colors.white)

        score_rect = pygame.Rect(320, 55, 170, 60)
        next_rect = pygame.Rect(320, 215, 170, 180)

        screen = pygame.display.set_mode((500, 620))
        pygame.display.set_caption("Python Tetris")

        clock = pygame.time.Clock()

        GAME_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(GAME_UPDATE, 200)
        score_value_surface = title_font.render(str(game.score), True, Colors.white)

        screen.fill(Colors.dark_blue)
        screen.blit(score_surface, (365, 20, 50, 50))
        screen.blit(next_surface, (375, 180, 50, 50))

        if game.game_over == True:
            screen.blit(game_over_surface, (320, 450, 50, 50))

        pygame.draw.rect(screen, Colors.light_blue, score_rect, 0, 10)
        screen.blit(score_value_surface, score_value_surface.get_rect(centerx = score_rect.centerx, 
            centery = score_rect.centery))
        pygame.draw.rect(screen, Colors.light_blue, next_rect, 0, 10)
        game.draw(screen)

        pygame.display.update()
        clock.tick(60)
    
    
def main():
    env = Environment()
    reseed(seed)
    env.reset()

    input_dim = len(env.features_to_vector(env.extract_features()))
    action_dim = 5 # left, right, down, rotate, do nothing

    nn = PolicyNet(input_dim, action_dim, hidden_dim=128)
    reinforce = PolicyGradient(env, nn, seed, reward_to_go=True)
    reinforce.train(num_iterations=10, batch_size=10, gamma=0.99, lr=0.001)
    print(reinforce.evaluate(100))

if __name__ == '__main__':
   main()