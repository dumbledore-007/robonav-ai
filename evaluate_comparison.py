
import random
import numpy as np
import csv
from generalized_environment import GeneralizedRobotEnvironment
from path_planner import find_shortest_path

random.seed(2027)
np.random.seed(2027)

EPISODES = 500
MAX_STEPS = 100

q_table = np.load("q_table_generalized.npy")
env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)

results = []

for episode in range(EPISODES):
    # Generate one solvable map shared by all three methods.
    env.reset()
    obstacles = set(env.obstacles)

    # ---------- Q-LEARNING ----------
    env.obstacles = set(obstacles)
    env.position = env.start
    q_steps = 0
    q_collisions = 0
    q_success = False

    for _ in range(MAX_STEPS):
        r, c, mask = env.get_state()
        action = int(np.argmax(q_table[r, c, mask]))

        old_position = env.position
        _, reward, done = env.step(action)
        q_steps += 1

        if env.position == old_position and reward == -5:
            q_collisions += 1

        if done:
            q_success = True
            break

    # ---------- BFS ----------
    env.obstacles = set(obstacles)
    env.position = env.start
    bfs_path = find_shortest_path(env)
    bfs_steps = len(bfs_path) if bfs_path is not None else 0
    bfs_success = bfs_path is not None

    # ---------- HYBRID ----------
    # Try Q-learning first. If it fails, restart and use BFS.
    hybrid_steps = q_steps
    hybrid_success = q_success
    used_fallback = False

    if not q_success:
        used_fallback = True
        env.obstacles = set(obstacles)
        env.position = env.start

        fallback_path = find_shortest_path(env)

        if fallback_path is not None:
            hybrid_steps += len(fallback_path)
            hybrid_success = True
        else:
            hybrid_success = False

    results.append({
        "episode": episode + 1,
        "q_success": q_success,
        "q_steps": q_steps,
        "q_collisions": q_collisions,
        "bfs_success": bfs_success,
        "bfs_steps": bfs_steps,
        "hybrid_success": hybrid_success,
        "hybrid_steps": hybrid_steps,
        "used_bfs_fallback": used_fallback
    })

# Save detailed results.
with open("comparison_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n===== ROBONAV AI: ALGORITHM COMPARISON =====")
print(f"Shared solvable test maps: {EPISODES}\n")

for name, success_key, steps_key in [
    ("Q-learning", "q_success", "q_steps"),
    ("BFS", "bfs_success", "bfs_steps"),
    ("Hybrid", "hybrid_success", "hybrid_steps"),
]:
    successes = sum(row[success_key] for row in results)
    success_rate = successes / EPISODES * 100
    successful_steps = [
        row[steps_key] for row in results if row[success_key]
    ]

    average_steps = (
        np.mean(successful_steps) if successful_steps else 0
    )

    print(f"{name}:")
    print(f"  Success rate: {success_rate:.2f}%")
    print(f"  Average steps on successful runs: {average_steps:.2f}")

fallbacks = sum(row["used_bfs_fallback"] for row in results)
print(f"\nHybrid BFS fallbacks: {fallbacks}/{EPISODES}")
print("Detailed results saved to comparison_results.csv")
