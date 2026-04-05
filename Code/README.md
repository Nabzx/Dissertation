# Multi-Agent Resource-Scarcity Simulation

A 2D grid world simulation for studying multi-agent behavior in resource-scarce environments. This project implements a PettingZoo-based parallel environment where two agents compete to collect non-renewable resources.

## Project Structure

```
.
├── env/
│   ├── gridworld_env.py      # PettingZoo ParallelEnv implementation
│   └── utils.py              # Visualisation utilities
├── agents/
│   └── heuristic_agent.py    # Greedy heuristic agents
├── train/
│   ├── run_simulation.py     # Simulation runner
│   └── generate_preliminary_results.py  # Results analysis
├── logs/
│   ├── episodes/             # Episode JSON data
│   ├── screenshots/          # Grid state screenshots
│   ├── heatmaps/             # Agent movement heatmaps
│   └── resources/            # Resource distribution plots
├── results/                  # Generated analysis results
├── main.py                   # Main entry point
└── README.md                 # This file
```

## Environment Description

### Grid World
- **Size**: 15×15 grid
- **Agents**: 2 agents (agent_0 and agent_1)
- **Resources**: Non-renewable resources randomly placed at episode start
- **No collisions**: Multiple agents can occupy the same cell
- **No combat**: No health, decay, or combat mechanics

### Observations
Each agent receives a 15×15 grid observation with:
- `0` = Empty cell
- `1` = Resource
- `2` = Agent 0
- `3` = Agent 1

### Actions
Discrete action space (5 actions):
- `0` = Stay
- `1` = Move Up
- `2` = Move Down
- `3` = Move Left
- `4` = Move Right

### Rewards
- `+1` for collecting a resource
- `0` otherwise

### Episode Termination
An episode ends when:
- All resources are collected, OR
- Maximum steps (default: 200) is reached

## Agent Description

### Heuristic Agents
The current implementation uses simple greedy heuristic agents:
- **Strategy**: Scan for nearest resource using Manhattan distance
- **Movement**: Move greedily toward the nearest resource (one step per turn)
- **Fallback**: If no resources remain, move randomly
- **Identical Logic**: Both agents use the same heuristic strategy

These agents serve as a baseline for preliminary results before implementing reinforcement learning.

## Installation

### Requirements
- Python 3.10+
- NumPy
- PettingZoo
- Matplotlib
- Gymnasium

### Setup

1. Install dependencies:
```bash
pip install numpy pettingzoo matplotlib gymnasium
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations

Run the main script to execute simulations and generate results:

```bash
python main.py
```

This will:
1. Run 20 episodes of simulation
2. Save episode data, screenshots, and heatmaps to `logs/`
3. Generate preliminary results and analysis in `results/`

### Running Individual Components

#### Run Simulations Only
```bash
python train/run_simulation.py
```

#### Generate Results Only
```bash
python train/generate_preliminary_results.py
```

### Configuration

Edit `main.py` to adjust simulation parameters:
- `num_episodes`: Number of episodes to run (default: 20)
- `grid_size`: Size of the grid (default: 15)
- `num_resources`: Number of resources per episode (default: 10)
- `max_steps`: Maximum steps per episode (default: 200)

## Output Files

### Logs Directory (`logs/`)
- **episodes/**: JSON files with episode summaries including:
  - Resource collection counts
  - Survival status
  - Step counts
  - Initial resource positions
  
- **screenshots/**: Final grid state visualisations for each episode

- **heatmaps/**: Movement heatmaps showing where each agent visited

- **resources/**: Initial resource distribution plots

### Results Directory (`results/`)
- **preliminary_results_summary.json**: Summary statistics including:
  - Survival rates
  - Cooperation scores
  - Resource collection averages
  
- **resource_distribution_heatmap.png**: Heatmap showing where resources were spawned across all episodes

- **screenshot_montage.png**: Grid of sample episode final states

- **reward_curves/reward_curve.png**: Episode reward curves for each agent

### Additional Visualisations

The simulation generates several additional visualisations to provide deeper insights into agent behavior:

#### Movement Heatmaps (Aggregated)
- **Location**: `logs/heatmaps/agent_0_heatmap.png` and `logs/heatmaps/agent_1_heatmap.png`
- **Description**: Aggregated movement density heatmaps showing where each agent visited across all episodes
- **Purpose**: Reveals spatial exploration patterns, preferred regions, and movement strategies
- **Visualisation**: Uses "hot" colormap where brighter colors indicate higher visit frequencies
- **Why it matters**: Helps identify if agents develop consistent movement patterns, whether they explore uniformly or focus on specific areas, and if there are spatial preferences that emerge over multiple episodes

#### Trajectory Plots
- **Location**: `results/trajectories/episode_0_trajectory.png` and `results/trajectories/episode_1_trajectory.png`
- **Description**: Visual representation of agent paths for the first two episodes
- **Features**:
  - Different colors for each agent (blue for agent_0, red for agent_1)
  - Arrows indicating movement direction
  - Starting positions marked with circles
  - Ending positions marked with squares
- **Purpose**: Shows the actual paths agents take during an episode
- **Why it matters**: Reveals navigation strategies, whether agents take direct paths to resources or explore more, and how agents interact spatially (e.g., do they avoid each other or converge on the same resources?)

#### Reward Curves
- **Location**: `results/reward_curves/reward_curve.png`
- **Description**: Line plot showing total reward per episode for each agent over all episodes
- **Features**:
  - Separate lines for each agent
  - Episode number on x-axis
  - Total reward (resources collected) on y-axis
- **Purpose**: Tracks performance trends across episodes
- **Why it matters**: Reveals if agent performance is consistent, improving, or degrading over time. Helps identify learning trends (if RL is added later) and performance stability. Useful for comparing agent strategies and understanding reward distribution patterns.

## Metrics

### Survival Rate
Percentage of episodes where an agent collected at least 1 resource:
- Agent 0 survival rate
- Agent 1 survival rate
- Both agents survival rate

### Cooperation Score
A simple measure of resource sharing:
```
cooperation = min(agent1_resources, agent2_resources) / max(1, total_resources_spawned)
```
- Range: [0, 1]
- Higher values indicate more balanced resource collection

## Example Output

After running simulations, you'll see output like:

```
============================================================
PRELIMINARY RESULTS SUMMARY
============================================================
Total Episodes: 20

Survival Rates:
  Agent 0: 95.00% (19/20)
  Agent 1: 90.00% (18/20)
  Both: 85.00% (17/20)

Cooperation Score:
  Average: 0.3245
  Std Dev: 0.1234
  Range: [0.1000, 0.6000]

Resource Statistics:
  Avg Resources/Episode: 10.00
  Avg Agent 0 Collected: 5.20
  Avg Agent 1 Collected: 4.80
============================================================
```

## Roadmap

### Phase 1: Current (Preliminary Results) ✅
- [x] Basic grid world environment
- [x] Heuristic agents
- [x] Logging and visualisation pipeline
- [x] Preliminary metrics (survival, cooperation)
- [x] Aggregated movement heatmaps
- [x] Trajectory visualisation
- [x] Reward curve analysis

### Phase 2: Reinforcement Learning (Next)
- [ ] Implement PPO/A3C for multi-agent RL
- [ ] Add communication channels between agents
- [ ] Study emergent cooperation strategies
- [ ] Compare RL agents vs. heuristic baselines

### Phase 3: Advanced Features
- [ ] Variable resource scarcity levels
- [ ] Dynamic resource spawning
- [ ] Agent heterogeneity (different capabilities)
- [ ] Communication protocols and analysis

## Development Notes

- The environment uses PettingZoo's `ParallelEnv` for efficient parallel agent execution
- All visualisation uses Matplotlib for consistency
- Episode data is saved as JSON for easy analysis and reproducibility
- Random seeds can be set for reproducible experiments

## License

This project is part of my university final-year dissertation project.



