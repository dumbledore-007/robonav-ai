
import random
import numpy as np
import pandas as pd
from generalized_environment import GeneralizedRobotEnvironment

random.seed(2026)
np.random.seed(2026)

env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)
q_table = np.load("q_table_generalized.npy")

EPISODES = 500
MAX_STEPS = 100
results = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    collisions = 0
    success = False
    steps_taken = 0

    for step in range(MAX_STEPS):
        r, c, mask = state

        # Greedy action: use the trained Q-table
        action = int(np.argmax(q_table[r, c, mask]))

        old_position = env.position
        next_state, reward, done = env.step(action)

        if env.position == old_position and reward == -5:
            collisions += 1

        total_reward += reward
        state = next_state
        steps_taken = step + 1

        if done:
            success = True
            break

    results.append({
        "episode": episode + 1,
        "success": success,
        "steps": steps_taken,
        "reward": total_reward,
        "collision_attempts": collisions
    })

df = pd.DataFrame(results)
df.to_csv("generalized_evaluation_results.csv", index=False)

print("\n===== GENERALIZED ROBONAV AI EVALUATION =====")
print(f"Independent test episodes: {EPISODES}")
print(f"Successful runs: {df['success'].sum()}/{EPISODES}")
print(f"Success rate: {df['success'].mean() * 100:.2f}%")
print(f"Average reward: {df['reward'].mean():.2f}")
print(f"Average collision attempts: {df['collision_attempts'].mean():.2f}")

successful = df[df["success"]]
if not successful.empty:
    print(
        "Average steps on successful runs: "
        f"{successful['steps'].mean():.2f}"
    )

print("\nResults saved to generalized_evaluation_results.csv")
print("\nFirst five test results:")
print(df.head())
