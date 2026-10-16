
import numpy as np
from environment import RobotEnvironment

env = RobotEnvironment()
q_table = np.load("q_table.npy")

state = env.reset()
path = [state]
total_reward = 0

print("Robot's starting position:")
env.render()

for step in range(100):
    row, col = state

    # Choose the action with the highest learned Q-value
    action = int(np.argmax(q_table[row, col]))

    next_state, reward, done = env.step(action)
    total_reward += reward

    # Stop if the robot cannot make progress
    if next_state == state and not done:
        print("Robot is stuck. Stopping test.")
        break

    state = next_state
    path.append(state)

    if done:
        break

print("\nLearned path:")
print(" -> ".join(str(position) for position in path))

print("\nFinal robot position:")
env.render()

print(f"\nSteps taken: {len(path) - 1}")
print(f"Total reward: {total_reward}")
print(f"Reached goal: {state == env.goal}")
