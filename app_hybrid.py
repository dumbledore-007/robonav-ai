
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from generalized_environment import GeneralizedRobotEnvironment
from path_planner import find_shortest_path

st.set_page_config(
    page_title="RoboNav AI | Hybrid Navigation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RoboNav AI")
st.subheader("Reinforcement Learning + Reliable Path Planning")
st.write(
    "Explore how a Q-learning robot navigates unfamiliar maps "
    "and how BFS provides a reliable fallback."
)

MODEL_PATH = "q_table_generalized.npy"
RL_LIMIT = 20

try:
    q_table = np.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Generalized model not found. Run python train_generalized.py first.")
    st.stop()


def build_demo():
    env = GeneralizedRobotEnvironment(size=8, obstacle_count=10)
    env.reset()

    # Preserve the randomly generated map.
    obstacles = set(env.obstacles)

    # Test the learned policy on this map.
    env.position = env.start
    q_path = [env.position]
    q_success = False
    q_steps = 0

    for _ in range(RL_LIMIT):
        r, c, mask = env.get_state()
        action = int(np.argmax(q_table[r, c, mask]))

        _, reward, done = env.step(action)
        q_steps += 1
        q_path.append(env.position)

        if done:
            q_success = True
            break

    # Find a shortest route on the same map.
    env.obstacles = set(obstacles)
    env.position = env.start
    bfs_actions = find_shortest_path(env)

    bfs_path = [env.start]
    if bfs_actions is not None:
        for action in bfs_actions:
            dr, dc = {
                0: (-1, 0),
                1: (1, 0),
                2: (0, -1),
                3: (0, 1)
            }[action]

            r, c = bfs_path[-1]
            bfs_path.append((r + dr, c + dc))

    return {
        "obstacles": obstacles,
        "start": env.start,
        "goal": env.goal,
        "q_path": q_path,
        "q_success": q_success,
        "q_steps": q_steps,
        "bfs_path": bfs_path,
        "bfs_success": bfs_actions is not None,
    }


if "demo" not in st.session_state:
    st.session_state.demo = build_demo()
    st.session_state.current_step = 0

if st.button("🗺️ Generate New Map"):
    st.session_state.demo = build_demo()
    st.session_state.current_step = 0
    st.rerun()

demo = st.session_state.demo

algorithm = st.selectbox(
    "Choose navigation strategy",
    ["Hybrid", "Q-learning", "BFS"]
)

if algorithm == "Q-learning":
    path = demo["q_path"]
    reached_goal = demo["q_success"]
    explanation = "The learned Q-table controls every move."
    total_steps = demo["q_steps"]

elif algorithm == "BFS":
    path = demo["bfs_path"]
    reached_goal = demo["bfs_success"]
    explanation = "BFS finds a shortest path on the current map."
    total_steps = len(path) - 1

else:
    if demo["q_success"]:
        path = demo["q_path"]
        reached_goal = True
        total_steps = demo["q_steps"]
        explanation = "Q-learning reached the goal without a fallback."
    else:
        # The learned attempt is capped. BFS supplies the final route.
        path = demo["bfs_path"]
        reached_goal = demo["bfs_success"]
        total_steps = demo["q_steps"] + len(path) - 1
        explanation = (
            f"Q-learning used {demo['q_steps']} moves without reaching "
            "the goal. The robot then restarts from the start and uses BFS. "
            "The grid shows the final BFS route."
        )

if not path:
    st.error("No route is available for this map.")
    st.stop()

last_step = len(path) - 1
st.session_state.current_step = min(
    st.session_state.current_step, last_step
)
step = st.session_state.current_step
robot = path[step]
visited = set(path[:step + 1])


def create_grid(position, visited_cells):
    fig = go.Figure()

    for row in range(8):
        for col in range(8):
            cell = (row, col)

            if cell in demo["obstacles"]:
                fill, label = "#263238", "✕"
            elif cell == position:
                fill, label = "#60A5FA", "🤖"
            elif cell == demo["goal"]:
                fill, label = "#FDE68A", "🏁"
            elif cell == demo["start"]:
                fill, label = "#BBF7D0", "S"
            elif cell in visited_cells:
                fill, label = "#BFDBFE", "•"
            else:
                fill, label = "#FFFFFF", ""

            fig.add_shape(
                type="rect",
                x0=col - 0.47, x1=col + 0.47,
                y0=row - 0.47, y1=row + 0.47,
                line=dict(color="#94A3B8", width=1),
                fillcolor=fill
            )

            fig.add_annotation(
                x=col, y=row, text=label,
                showarrow=False,
                font=dict(
                    size=21 if cell == position else 16,
                    color="white" if cell in demo["obstacles"]
                    else "#172033"
                )
            )

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#E2E8F0",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Column",
            tickmode="linear",
            dtick=1,
            range=[-0.6, 7.6],
            side="top",
            fixedrange=True
        ),
        yaxis=dict(
            title="Row",
            tickmode="linear",
            dtick=1,
            range=[7.6, -0.6],
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        )
    )
    return fig


c1, c2, c3 = st.columns(3)
c1.metric("Navigation strategy", algorithm)
c2.metric("Moves in displayed route", len(path) - 1)
c3.metric("Goal reached", "Yes" if reached_goal else "Not yet")

st.info(explanation)

st.subheader("Live Navigation")
st.caption("Green = Start | Yellow = Goal | Dark = Obstacle | Blue = Robot / visited path")

st.plotly_chart(
    create_grid(robot, visited),
    width="stretch",
    key=f"grid_{algorithm}_{step}"
)

st.write(f"**Robot position:** {robot}")
st.write(f"**Step:** {step} / {last_step}")

a, b, c = st.columns(3)

with a:
    if st.button("➡️ Next Step", disabled=step >= last_step,
                 width="stretch"):
        st.session_state.current_step += 1
        st.rerun()

with b:
    if st.button("▶ Auto Run", disabled=step >= last_step,
                 type="primary", width="stretch"):
        for next_step in range(step + 1, last_step + 1):
            st.session_state.current_step = next_step
            st.plotly_chart(
                create_grid(path[next_step], set(path[:next_step + 1])),
                width="stretch",
                key=f"auto_{algorithm}_{next_step}"
            )
            time.sleep(0.25)

with c:
    if st.button("🔄 Restart Route", width="stretch"):
        st.session_state.current_step = 0
        st.rerun()

with st.expander("View route coordinates"):
    st.write(" → ".join(map(str, path)))

st.divider()
st.subheader("Model and Evaluation Notes")
st.write(
    "- Q-learning uses a learned Q-table and local obstacle observations."
)
st.write(
    "- BFS is a classical shortest-path search, not a learned model."
)
st.write(
    "- Hybrid navigation attempts Q-learning for at most 20 moves, "
    "then restarts from the start and uses BFS if needed."
)
st.write(
    "- The displayed route shows the selected/final route. In hybrid mode, "
    "the total move metric includes the failed learned attempt when fallback "
    "is required."
)
st.caption(
    "Results depend on the map distribution. Test performance is not "
    "a guarantee for every possible obstacle layout."
)
