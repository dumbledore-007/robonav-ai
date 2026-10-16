
from collections import deque


# Action mapping matches generalized_environment.py
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}


def find_shortest_path(env):
    """Return a shortest list of actions, or None if no path exists."""

    start = env.start
    goal = env.goal

    queue = deque([start])
    parent = {start: None}
    action_used = {}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for action, (dr, dc) in ACTIONS.items():
            r, c = current
            neighbor = (r + dr, c + dc)

            if neighbor in parent:
                continue

            if not env.is_valid(neighbor):
                continue

            parent[neighbor] = current
            action_used[neighbor] = action
            queue.append(neighbor)

    if goal not in parent:
        return None

    # Reconstruct the route from goal to start.
    path = []
    current = goal

    while current != start:
        path.append(action_used[current])
        current = parent[current]

    path.reverse()
    return path


if __name__ == "__main__":
    from generalized_environment import GeneralizedRobotEnvironment

    env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)
    env.reset()

    path = find_shortest_path(env)

    print("Start:", env.start)
    print("Goal:", env.goal)
    print("Obstacles:", sorted(env.obstacles))

    if path is None:
        print("No valid route exists.")
    else:
        print("Route found!")
        print("Number of moves:", len(path))
        print("Actions:", path)
