
import random
import numpy as np
from environment import RobotEnvironment


# Create the robot environment
env = RobotEnvironment()

# Four possible actions: up, down, left, right
actions = 4

# Q-table stores the value of each action at each position
q_table = np.zeros((env.size, env.size, actions))

# Learning settings
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 3000
max_steps = 200

rewards_history = []
successes = 0

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        row, col = state

        # Explore randomly or choose the best-known action
        if random.random() < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = int(np.argmax(q_table[row, col]))

        next_state, reward, done = env.step(action)
        next_row, next_col = next_state

        # Q-learning update rule
        old_value = q_table[row, col, action]
        next_max = np.max(q_table[next_row, next_col])

        new_value = old_value + learning_rate * (
            reward + discount_factor * next_max - old_value
        )

        q_table[row, col, action] = new_value

        state = next_state
        total_reward += reward

        if done:
            successes += 1
            break

    # Gradually reduce random exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_history.append(total_reward)

    if (episode + 1) % 500 == 0:
        recent = rewards_history[-100:]
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Average reward (last 100): {np.mean(recent):.2f} | "
            f"Exploration: {epsilon:.3f}"
        )

# Save the learned Q-table
np.save("q_table.npy", q_table)

print("\nTraining complete!")
print(f"Successful episodes: {successes}/{episodes}")
print("Learned Q-table saved to q_table.npy")
