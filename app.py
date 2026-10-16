
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from environment import RobotEnvironment

st.set_page_config(
    page_title="RoboNav AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RoboNav AI")
st.subheader("Autonomous Robot Navigation using Reinforcement Learning")
st.write("Watch a trained Q-learning robot navigate a virtual grid.")

try:
    q_table = np.load("q_table.npy")
except FileNotFoundError:
    st.error("Model not found. Run python train.py first.")
    st.stop()

env = RobotEnvironment()


def get_path():
    state = env.reset()
    path = [state]

    for _ in range(100):
        row, col = state
        action = int(np.argmax(q_table[row, col]))
        next_state, reward, done = env.step(action)

        if next_state == state and not done:
            break

        state = next_state
        path.append(state)

        if done:
            break

    return path


if "robot_path" not in st.session_state:
    st.session_state.robot_path = get_path()
    st.session_state.current_step = 0

path = st.session_state.robot_path
current_step = st.session_state.current_step
robot_position = path[current_step]
visited = set(path[:current_step + 1])
last_step = len(path) - 1


def create_grid(step):
    robot = path[step]
    visited_cells = set(path[:step + 1])

    fig = go.Figure()

    # Draw each grid cell as a separate square
    for row in range(env.size):
        for col in range(env.size):
            cell = (row, col)

            if cell in env.obstacles:
                fill = "#263238"
            elif cell == robot:
                fill = "#60A5FA"
            elif cell == env.goal:
                fill = "#FDE68A"
            elif cell == env.start:
                fill = "#BBF7D0"
            elif cell in visited_cells:
                fill = "#BFDBFE"
            else:
                fill = "#FFFFFF"

            fig.add_shape(
                type="rect",
                x0=col - 0.47,
                x1=col + 0.47,
                y0=row - 0.47,
                y1=row + 0.47,
                line=dict(color="#94A3B8", width=1.5),
                fillcolor=fill
            )

            # Keep start and goal labels visible
            if cell == robot and cell == env.start:
                label = "🤖<br><b>S</b>"
            elif cell == robot and cell == env.goal:
                label = "🤖<br><b>G</b>"
            elif cell == robot:
                label = "🤖"
            elif cell == env.start:
                label = "<b>S</b>"
            elif cell == env.goal:
                label = "🏁<br><b>G</b>"
            elif cell in env.obstacles:
                label = "✕"
            elif cell in visited_cells:
                label = "•"
            else:
                label = ""

            fig.add_annotation(
                x=col,
                y=row,
                text=label,
                showarrow=False,
                font=dict(
                    size=22 if cell == robot else 17,
                    color="white" if cell in env.obstacles
                    else "#172033"
                ),
                align="center"
            )

    fig.update_layout(
        height=560,
        margin=dict(l=25, r=25, t=20, b=20),
        plot_bgcolor="#E2E8F0",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Column",
            tickmode="array",
            tickvals=list(range(env.size)),
            range=[-0.6, env.size - 0.4],
            side="top",
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="Row",
            tickmode="array",
            tickvals=list(range(env.size)),
            range=[env.size - 0.4, -0.6],
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        )
    )

    return fig


st.divider()

# Performance metrics
c1, c2, c3 = st.columns(3)
c1.metric("Training Episodes", "3,000")
c2.metric("Training Success Rate", "99.6%")
c3.metric("Algorithm", "Q-learning")

st.header("🗺️ Live Robot Navigation")

st.markdown(
    "**S = Start** 🟢 &nbsp;&nbsp; "
    "**G = Goal** 🏁 &nbsp;&nbsp; "
    "**✕ = Obstacle** ⬛ &nbsp;&nbsp; "
    "**Blue = Visited path**",
    unsafe_allow_html=True
)

speed = st.slider(
    "Animation speed (seconds per step)",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05
)

status = st.empty()
grid_area = st.empty()


def display_step(step):
    position = path[step]

    status.metric(
        "Robot status",
        "Goal reached! 🎉" if position == env.goal
        else "Moving to goal...",
        delta=f"Step {step} of {last_step}"
    )

    grid_area.plotly_chart(
        create_grid(step),
        width="stretch",
        key=f"robot_grid_{step}"
    )


display_step(current_step)

st.caption(
    f"Current position: {robot_position} | "
    f"Start: {env.start} | Goal: {env.goal}"
)

a, b, c = st.columns(3)

with a:
    if st.button(
        "➡️ Next Step",
        disabled=current_step >= last_step,
        width="stretch"
    ):
        st.session_state.current_step += 1
        st.rerun()

with b:
    if st.button(
        "▶ Auto Run",
        disabled=current_step >= last_step,
        type="primary",
        width="stretch"
    ):
        for step in range(current_step + 1, last_step + 1):
            st.session_state.current_step = step
            display_step(step)
            time.sleep(speed)

with c:
    if st.button("🔄 Reset", width="stretch"):
        st.session_state.current_step = 0
        st.rerun()

with st.expander("View the complete learned route"):
    st.write(" → ".join(map(str, path)))
    st.write(f"Total movements: {last_step}")
    st.write(f"Reached goal: {path[-1] == env.goal}")

st.divider()
st.header("🧠 Learned Q-values")

best_values = np.max(q_table, axis=2)

import plotly.express as px

fig = px.imshow(
    best_values,
    text_auto=".1f",
    aspect="equal",
    color_continuous_scale="Viridis",
    labels={
        "x": "Column",
        "y": "Row",
        "color": "Q-value"
    },
    title="Best Estimated Action Value at Each Position"
)

st.plotly_chart(fig, width="stretch")

st.caption("RoboNav AI | Reinforcement Learning Project")


# Independent evaluation results
st.divider()
st.header("📊 Performance Evaluation")

try:
    results = pd.read_csv("evaluation_results.csv")

    success_rate = results["success"].mean() * 100
    avg_reward = results["reward"].mean()
    avg_collisions = results["collision_attempts"].mean()
    successful = results[results["success"] == True]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Episodes", len(results))
    m2.metric("Success Rate", f"{success_rate:.1f}%")
    m3.metric("Average Reward", f"{avg_reward:.1f}")
    m4.metric("Avg. Collision Attempts", f"{avg_collisions:.2f}")

    if len(successful) > 0:
        st.metric(
            "Average Steps on Successful Runs",
            f"{successful['steps'].mean():.1f}"
        )

    chart_tab1, chart_tab2 = st.tabs(
        ["Rewards", "Steps per Episode"]
    )

    with chart_tab1:
        reward_fig = px.line(
            results,
            x="episode",
            y="reward",
            title="Reward per Evaluation Episode",
            labels={
                "episode": "Episode",
                "reward": "Total reward"
            }
        )
        st.plotly_chart(reward_fig, width="stretch")

    with chart_tab2:
        steps_fig = px.histogram(
            results,
            x="steps",
            color="success",
            title="Distribution of Steps per Episode",
            labels={"steps": "Movement steps"},
            nbins=20
        )
        st.plotly_chart(steps_fig, width="stretch")

    with st.expander("View evaluation data"):
        st.dataframe(results, width="stretch")

except FileNotFoundError:
    st.warning(
        "Evaluation results not found. Run `python evaluate.py` first."
    )
