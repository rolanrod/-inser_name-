import numpy as np
from game import Game

class Environment(Game):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.step_count = 0
    
    def step(self, action, timestep):
        if self.game_over: return self.grid, 0, True

        reward_0 = self.score

        # Apply transitions and update state
        if action == 'l':
            self.move_left()
        if action == 'r':
            self.move_right()
        if action == 'down':
            self.move_down()
            self.update_score(0, 1)
        if action == 'rot':
            self.rotate()
        
        self.step_count += 1

        if self.step_count >= self.interval and not self.game_over:
            self.move_down()
            self.step_count = 0  # Reset the step counter after each automatic move
            
        if self.game_over:
            self.update_score(0, -100)

        timestep = timestep + 1

        # Calculate reward added by taking `action`
        delta_reward = self.score - reward_0
        return self.grid, delta_reward, self.game_over, timestep