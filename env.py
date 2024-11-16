import numpy as np
from game import Game

class Environment(Game):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.step_count = 0

    def extract_features(self):
        """
        Extracts compact features from the board.

        Args:
            board (2D list): The Tetris board.

        Returns:
            dict: A dictionary of features.
        """
        num_rows = 20
        num_cols = 10

        grid_arr = np.array(self.grid.grid)

        heights = [np.max(np.where(grid_arr[:, col] > 0, num_rows - np.argmax(grid_arr[:, col] > 0), 0)) for col in range(num_cols)]
        max_height = max(heights)
        aggregate_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        holes = sum((num_rows - np.argmax(grid_arr[:, col] > 0, axis=0)) - np.sum(grid_arr[:, col]) for col in range(num_cols))
    
        tetromino = self.current_block.id - 1

        return {
            "heights": heights,
            "max_height": max_height,
            "aggregate_height": aggregate_height,
            "bumpiness": bumpiness,
            "holes": holes,
            "tetromino": tetromino,
            "tetromino_cells": tetromino.cells
        }
    
    def features_to_vector(self, features):
        """
        Converts the feature dictionary into a flattened vector.

        Args:
            features (dict): The extracted features.

        Returns:
            list: Flattened feature vector.
        """
        return [
            *features["heights"],            # Heights of all columns (length = BOARD_COLS)
            features["max_height"],          # Maximum height
            features["aggregate_height"],    # Aggregate height
            features["bumpiness"],           # Bumpiness
            features["holes"],               # Number of holes
        ]
    
    def step(self, action, timestep):
        if self.game_over: 
            return np.array(self.grid), 0, True, 0

        reward_0 = self.score

        # Apply transitions and update state
        if action == 0:
            self.move_left()
        if action == 1:
            self.move_right()
        if action == 2:
            self.move_down()
            self.update_score(0, 1)
        if action == 3:
            self.rotate()
        if action == 4:
            pass # do noting
        
        self.step_count += 1

        if self.step_count >= self.interval and not self.game_over:
            self.move_down()
            self.step_count = 0  # Reset the step counter after each automatic move
            
        if self.game_over:
            self.update_score(0, -100)

        timestep = timestep + 1

        # Calculate reward added by taking `action`
        delta_reward = self.score - reward_0
        return np.array(self.features_to_vector(self.extract_features())), delta_reward, self.game_over, timestep