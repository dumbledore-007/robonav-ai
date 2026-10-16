# RoboNav AI — Autonomous Robot Navigation Simulator

An interactive robot navigation project combining **Q-learning reinforcement learning** with **Breadth-First Search (BFS)** path planning. Built with Python and Streamlit, RoboNav AI simulates a robot navigating obstacle-filled grids without requiring physical hardware.

## Project Overview

RoboNav AI explores how a robot can learn to navigate an environment and how classical path planning can improve reliability when the learned policy struggles on unfamiliar maps.

The project includes:
- Q-learning on a fixed 8×8 grid.
- Generalized Q-learning experiments on randomly generated maps.
- BFS shortest-path planning.
- Hybrid navigation with a BFS fallback.
- An interactive Streamlit visualization.
- Evaluation scripts and CSV results for algorithm comparison.

## Features

- Interactive 8×8 grid visualization.
- Randomly generated maps containing 10 obstacles.
- Robot start and goal positions.
- Animated robot navigation and route visualization.
- Q-learning, BFS, and hybrid navigation modes.
- Training and evaluation metrics.
- CSV files containing experiment results.

## System Architecture

```text
Randomly Generated Environment
             |
             v
    Robot State Observation
             |
             v
       Navigation Mode
       /      |       \
      /       |        \
 Q-learning   BFS    Hybrid
      \       |        /
       \      |       /
             v
      Navigation Results
             |
             v
     Streamlit Visualization
             |
             v
       Evaluation Metrics
```

## Algorithms

### 1. Q-learning

Q-learning is a model-free reinforcement learning algorithm. The agent learns action values through interaction with the environment.

The generalized model uses the robot's row, column, and a four-direction local obstacle mask as its state representation.

### 2. Breadth-First Search (BFS)

BFS explores the grid level by level. Since every move has equal cost, it finds a shortest valid route whenever one exists.

BFS is used as a classical path-planning baseline and is not itself a machine learning algorithm.

### 3. Hybrid Navigation

The hybrid strategy first attempts navigation using the trained Q-learning policy for up to 20 moves. If the robot does not reach the goal, it restarts from the start and uses BFS to calculate a route.

This improves reliability on the tested maps, but a failed learning attempt adds extra movement cost.

## Evaluation Results

The following results were measured during development.

### Q-learning vs. BFS vs. Hybrid

| Metric | Q-learning | BFS | Hybrid |
|---|---:|---:|---:|
| Test maps | 500 | 500 | 500 |
| Success rate | 65.2% | 100% | 100% |
| Average steps on successful runs | 14.08 | 14.00 | — |
| Average total steps per map | — | — | 20.60 in the fast-hybrid experiment |

The comparison experiment tested all three methods on the same 500 solvable maps. The fast-hybrid experiment used its own reproducible map-generation seed and a 20-step Q-learning attempt limit.

Additional measured results:
- Generalized Q-learning training: 20,000 episodes.
- Recent training success rate: 87.6% over the final 500 training episodes.
- Earlier independent generalized evaluation: 64% success over 500 maps.
- Fast-hybrid experiment: 100% success over 500 maps, with 163 BFS fallbacks.

These results apply to the tested environment and map-generation procedure. They do not guarantee success on every possible obstacle layout.

## Project Structure

```text
robonav-ai/
├── app.py
├── app_hybrid.py
├── environment.py
├── generalized_environment.py
├── train.py
├── train_generalized.py
├── test_robot.py
├── evaluate.py
├── evaluate_generalization.py
├── evaluate_generalized.py
├── evaluate_comparison.py
├── evaluate_hybrid_fast.py
├── path_planner.py
├── q_table.npy
├── q_table_generalized.npy
├── evaluation_results.csv
├── generalization_results.csv
├── generalized_evaluation_results.csv
└── comparison_results.csv
```

## Installation

### Requirements

- Python 3.10 or newer
- pip
- Streamlit
- NumPy
- Pandas
- Plotly

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/robonav-ai.git
cd robonav-ai
```

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository URL.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the hybrid application

```bash
streamlit run app_hybrid.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The original fixed-map application can be launched with:

```bash
streamlit run app.py
```

## Training

To train the original fixed-map model:

```bash
python train.py
```

To train the generalized Q-learning model:

```bash
python train_generalized.py
```

Training the generalized model overwrites `q_table_generalized.npy`. Keep a backup if you want to preserve the current evaluated model.

## Evaluation

Run the following commands to reproduce the experiments:

```bash
python evaluate.py
python evaluate_generalized.py
python evaluate_comparison.py
python evaluate_hybrid_fast.py
```

The evaluation scripts save results to CSV files where applicable.

## Limitations

- The generalized Q-learning state includes only local obstacle observations, not the complete map.
- Q-learning does not reliably solve every unseen map.
- BFS requires access to the map's obstacle layout and is not a learned navigation policy.
- The hybrid fallback improves reliability but can increase the total number of moves.
- Evaluation results depend on the map-generation process and test sample.

## Future Improvements

- Compare additional reinforcement learning algorithms.
- Improve state representation and obstacle awareness.
- Add execution-time and path-efficiency measurements.
- Add downloadable evaluation reports.
- Improve the visualization of the learning-to-fallback transition.
- Evaluate on larger grids and different obstacle densities.

## Technologies Used

Python · NumPy · Pandas · Plotly · Streamlit · Reinforcement Learning · Q-learning · Breadth-First Search

## Author

Developed as an AI and Data Science portfolio project exploring reinforcement learning, path planning, simulation, and algorithm evaluation.
