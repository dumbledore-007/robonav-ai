
import random
import numpy as np
from generalized_environment import GeneralizedRobotEnvironment

# Reproducible training
random.seed(42)
np.random.seed(42)

env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)

# State: row, column, nearby-obstacle mask (0 to 15)
q_table = np.zeros((8, 8, 16, 4))

EPISODES = 20000
MAX_STEPS = 100
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY = 0.9995

recent_successes = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    success = False

    for step in range(MAX_STEPS):
        r, c, mask = state

        # Explore or use the best-known action
        if random.random() < EPSILON:
            action = random.randrange(4)
        else:
            action = int(np.argmax(q_table[r, c, mask]))

        next_state, reward, done = env.step(action)
        nr, nc, nmask = next_state

        # Q-learning update
        old_value = q_table[r, c, mask, action]
        next_best = np.max(q_table[nr, nc, nmask])

        target = reward
        if not done:
            target += DISCOUNT * next_best

        q_table[r, c, mask, action] = (
            old_value
            + LEARNING_RATE * (target - old_value)
        )

        state = next_state
        total_reward += reward

        if done:
            success = True
            break

    recent_successes.append(int(success))

    if len(recent_successes) > 500:
        recent_successes.pop(0)

    EPSILON = max(MIN_EPSILON, EPSILON * DECAY)

    if (episode + 1) % 1000 == 0:
        rate = np.mean(recent_successes) * 100
        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Recent success rate: {rate:.1f}% | "
            f"Epsilon: {EPSILON:.3f}"
        )

np.save("q_table_generalized.npy", q_table)

print("\nTraining complete!")
print("Saved as q_table_generalized.npy")
