import numpy as np
from game import Game
import rewards

class Environment(Game):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.step_count = 0
    
    def step(self, action, timestep):
        # if self.game_over: return np.zeros(10 + 9 + 1 + 1), 0, True

        initial_score = self.score

        old_state = self.extract_compact_state()

        # Apply transitions and update state
        if action == 0:
            self.move_left()
        if action == 1:
            self.move_right()
        if action == 2:
            self.move_down()
            # Reward for choosing 'down' action
            self.update_score(0, rewards.move_down)
        if action == 3:
            self.rotate()
        
        self.step_count += 1

        if self.step_count >= self.interval and not self.game_over:
            self.move_down()
            self.step_count = 0  # Reset the step counter after each automatic move
            
        if self.game_over:
            # Negative reward for losing
            self.update_score(0, rewards.game_over)

        timestep = timestep + 1

        new_state = self.extract_compact_state()

        # Punish increasing max heights:
        self.score -= (new_state[-2] - old_state[-2])

        # Punish the creation of holes:
        self.score -= (new_state[-1] - old_state[-1]) * 10
    
        # Calculate reward added by taking `action`
        reward = self.score - initial_score
        # print("ACTION: " + str(action))
        # print("REWARD: " + str(reward))

        return new_state, reward, self.game_over
    
    def extract_compact_state(self):
        # First 10 are heights
        # Next 9 are height diffs
        # Last is number of hoels
        # Second to last is max height
        # Third to last is current piece
        new_state = np.zeros(10 + 9 + 1 + 1 + 1)
        for i in range(self.grid.num_rows):
            for j in range(self.grid.num_cols):
                if self.grid.grid[i][j] == 1:
                    # 0 is the top-most row
                    new_state[j] = max(20 - i, new_state[j]) # Height of each column
                if self.grid.grid[i][j] == 1 and (i == self.grid.num_rows-1 or self.grid.grid[i+1][j] == 0):
                    # Number of holes
                    new_state[-1] += 1

        new_state[-2] = np.max(new_state[0:10])

        new_state[-3] = self.current_block.id - 1
        for i in range(self.grid.num_cols-1):
            new_state[10+i] = new_state[i] - new_state[i+1]

        return new_state