
import random
import numpy as np
from generalized_environment import GeneralizedRobotEnvironment
from path_planner import find_shortest_path

random.seed(2028)
np.random.seed(2028)

EPISODES = 500
RL_STEP_LIMIT = 20

env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)
q_table = np.load("q_table_generalized.npy")

successes = 0
total_steps = 0
fallbacks = 0
rl_successes = 0
path_lengths = []

for episode in range(EPISODES):
    env.reset()
    obstacles = set(env.obstacles)
    env.position = env.start

    steps = 0
    reached_goal = False

    # First, try the learned Q-learning policy.
    for _ in range(RL_STEP_LIMIT):
        r, c, mask = env.get_state()
        action = int(np.argmax(q_table[r, c, mask]))

        _, reward, done = env.step(action)
        steps += 1

        if done:
            reached_goal = True
            rl_successes += 1
            break

    # If learning did not reach the goal, use BFS.
    if not reached_goal:
        fallbacks += 1
        env.obstacles = set(obstacles)
        env.position = env.start

        path = find_shortest_path(env)

        if path is not None:
            steps += len(path)
            reached_goal = True

    successes += int(reached_goal)
    total_steps += steps

    if reached_goal:
        path_lengths.append(steps)

print("\n===== FAST HYBRID EVALUATION =====")
print(f"Test maps: {EPISODES}")
print(f"Overall success rate: {successes / EPISODES * 100:.2f}%")
print(f"Q-learning successes: {rl_successes}/{EPISODES}")
print(f"BFS fallbacks: {fallbacks}/{EPISODES}")
print(f"Average total steps per map: {total_steps / EPISODES:.2f}")
print(f"Average steps on successful runs: {np.mean(path_lengths):.2f}")
