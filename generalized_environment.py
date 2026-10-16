
import random
from collections import deque


class GeneralizedRobotEnvironment:
    def __init__(self, size=8, obstacle_count=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacle_count = obstacle_count
        self.obstacles = set()
        self.position = self.start

    def is_valid(self, position):
        r, c = position
        return (
            0 <= r < self.size
            and 0 <= c < self.size
            and position not in self.obstacles
        )

    def has_path(self):
        queue = deque([self.start])
        visited = {self.start}

        while queue:
            r, c = queue.popleft()

            if (r, c) == self.goal:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (r + dr, c + dc)

                if self.is_valid(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return False

    def reset(self):
        # Generate a new map with a valid route to the goal.
        cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in (self.start, self.goal)
        ]

        while True:
            self.obstacles = set(
                random.sample(cells, self.obstacle_count)
            )
            if self.has_path():
                break

        self.position = self.start
        return self.get_state()

    def get_state(self):
        r, c = self.position

        # Four-bit mask: up, down, left, right.
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        mask = 0

        for i, (dr, dc) in enumerate(directions):
            nxt = (r + dr, c + dc)

            if not self.is_valid(nxt):
                mask |= (1 << i)

        # State includes position and nearby obstacles.
        return r, c, mask

    def step(self, action):
        directions = [
            (-1, 0),  # 0: Up
            (1, 0),   # 1: Down
            (0, -1),  # 2: Left
            (0, 1)    # 3: Right
        ]

        r, c = self.position
        dr, dc = directions[action]
        nxt = (r + dr, c + dc)

        if not self.is_valid(nxt):
            return self.get_state(), -5, False

        self.position = nxt

        if self.position == self.goal:
            return self.get_state(), 100, True

        return self.get_state(), -1, False
