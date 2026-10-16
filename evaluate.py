
import random
import numpy as np
import pandas as pd

from environment import RobotEnvironment

# Load the already-trained Q-learning agent
env = RobotEnvironment()
q_table = np.load("q_table.npy")

TEST_EPISODES = 500
MAX_STEPS = 200

results = []

for episode in range(TEST_EPISODES):
    state = env.reset()
    total_reward = 0
    collisions = 0
    reached_goal = False
    steps_taken = 0

    for step in range(MAX_STEPS):
        row, col = state

        # Greedy policy: select the best learned action
        action = int(np.argmax(q_table[row, col]))

        next_state, reward, done = env.step(action)

        # Detect an attempted move that was blocked
        if next_state == state and reward == -5:
            collisions += 1

        total_reward += reward
        steps_taken += 1
        state = next_state

        if done:
            reached_goal = True
            break

    results.append({
        "episode": episode + 1,
        "success": reached_goal,
        "steps": steps_taken,
        "reward": total_reward,
        "collision_attempts": collisions
    })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

success_rate = df["success"].mean() * 100
successful_runs = df[df["success"]]

print("\n===== ROBONAV AI EVALUATION =====")
print(f"Independent test episodes: {TEST_EPISODES}")
print(f"Successful runs: {len(successful_runs)}/{TEST_EPISODES}")
print(f"Success rate: {success_rate:.2f}%")
print(f"Average reward: {df['reward'].mean():.2f}")
print(f"Average collision attempts per run: "
      f"{df['collision_attempts'].mean():.2f}")

if len(successful_runs) > 0:
    print(f"Average steps on successful runs: "
          f"{successful_runs['steps'].mean():.2f}")
else:
    print("No successful runs, so average successful steps is unavailable.")

print("\nResults saved to evaluation_results.csv")
print("\nFirst five test results:")
print(df.head().to_string(index=False))
