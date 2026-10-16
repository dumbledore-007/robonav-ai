
import numpy as np


class RobotEnvironment:
    def __init__(self, size=8):
        self.size = size

        # Robot starts at the top-left corner
        self.start = (0, 0)

        # Goal is at the bottom-right corner
        self.goal = (7, 7)

        # Obstacles the robot must avoid
        self.obstacles = {
            (1, 1), (1, 2), (1, 3),
            (3, 1), (3, 2), (3, 3),
            (4, 5), (5, 5), (6, 5),
            (5, 2), (6, 2)
        }

        self.reset()

    def reset(self):
        """Return the robot to its starting position."""
        self.position = self.start
        return self.position

    def step(self, action):
        """
        Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right
        """
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        row, col = self.position
        dr, dc = moves[action]

        new_position = (row + dr, col + dc)

        # Penalty for trying to leave the grid
        if not (0 <= new_position[0] < self.size and
                0 <= new_position[1] < self.size):
            return self.position, -5, False

        # Penalty for hitting an obstacle
        if new_position in self.obstacles:
            return self.position, -5, False

        self.position = new_position

        # Reward for reaching the goal
        if self.position == self.goal:
            return self.position, 100, True

        # Small penalty for every movement
        return self.position, -1, False

    def render(self):
        """Display the grid in the terminal."""
        grid = np.full((self.size, self.size), ".")

        for row, col in self.obstacles:
            grid[row, col] = "#"

        grid[self.goal] = "G"
        grid[self.position] = "R"

        print("\n".join(" ".join(row) for row in grid))
        print("\nR = Robot | G = Goal | # = Obstacle | . = Empty")
