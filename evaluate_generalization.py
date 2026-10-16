
import numpy as np
import pandas as pd
import random

from environment import RobotEnvironment

# Load the trained Q-learning agent
q_table = np.load("q_table.npy")

TEST_EPISODES = 200
MAX_STEPS = 200
SIZE = 8

results = []

# Generate different obstacle layouts
for episode in range(TEST_EPISODES):

    env = RobotEnvironment()

    # Keep start and goal clear
    start = (0, 0)
    goal = (SIZE - 1, SIZE - 1)

    # Generate a new random obstacle layout
    all_cells = [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if (r, c) not in (start, goal)
    ]

    obstacles = set(random.sample(all_cells, 10))
    env.obstacles = obstacles

    state = env.reset()
    total_reward = 0
    collisions = 0
    success = False
    steps = 0

    for step in range(MAX_STEPS):
        row, col = state

        # Choose the best learned action
        action = int(np.argmax(q_table[row, col]))

        next_state, reward, done = env.step(action)

        if next_state == state and reward == -5:
            collisions += 1

        total_reward += reward
        steps += 1
        state = next_state

        if done:
            success = True
            break

    results.append({
        "episode": episode + 1,
        "success": success,
        "steps": steps,
        "reward": total_reward,
        "collisions": collisions
    })

df = pd.DataFrame(results)
df.to_csv("generalization_results.csv", index=False)

print("\n===== RO BONAV AI: NEW MAP TEST =====")
print("Test episodes:", TEST_EPISODES)
print(f"Success rate: {df['success'].mean() * 100:.2f}%")
print(f"Average reward: {df['reward'].mean():.2f}")
print(f"Average collision attempts: {df['collisions'].mean():.2f}")
print("Results saved to generalization_results.csv")
