# Offline RL Trajectory Reanalysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

An advanced research framework studying **data augmentation for Offline Reinforcement Learning** through synthetic trajectory reanalysis. Built around **Implicit Q-Learning (IQL)** as a shared training backbone, this project compares multiple world-model-based data augmentation strategies and introduces a **Critic-Guided Reanalysis Pipeline** that selectively reanalyzes only the most promising trajectories.

---

## 💡 Motivation & Key Ideas

Offline RL algorithms often suffer from severe out-of-distribution (OOD) action extrapolation and restricted data coverage. While augmenting datasets with synthetic transitions from a learned world model expands buffer diversity, naive augmentation can introduce compounding errors and value overestimation.

This project addresses these challenges by introducing three core mechanisms:

* **IQL Shared Backbone**: Standardized Implicit Q-Learning trainer across all experiments to isolate and measure the exact impact of each augmentation strategy.
* **Probabilistic Ensemble World Model**: Ensemble of MLPs predicting state deltas $(\Delta s = s' - s)$ and rewards. Uses ensemble disagreement as an explicit **uncertainty metric**.
* **Uncertainty Calibration**: A calibrated threshold ($90^{\text{th}}$ percentile of ensemble variance) filters out unreliable synthetic transitions.
* **Critic-Guided Reanalysis**: A warm-up critic computes advantages $A(s,a) = Q(s,a) - V(s)$ over the dataset. Monte Carlo Tree Search (MCTS) reanalyzes **only the top-K most under-exploited states**, populating an enriched replay buffer $\mathcal{D}'$.

---

## 🧪 Augmentation Strategies Compared

| Strategy | Description | Key Focus |
| :--- | :--- | :--- |
| **Baseline** | Standard Implicit Q-Learning (IQL) on raw dataset. | Reference benchmark |
| **Vine Sampling** | Localized action branch exploration around pivot trajectory states. | Local action perturbations |
| **MCTS Augmentation** | World-model-driven Monte Carlo Tree Search trajectory planning. | High-value trajectory synthesis |
| **VAE Augmentation** | Conditional Variational Autoencoder modeling state-action transitions. | Generative transition density |
| **Critic-Guided Reanalysis** | Top-K advantage selection combined with targeted MCTS rollouts. | Selective data enrichment |

---

## 📊 Benchmarks & Benchmark Performance

Evaluated on standard **D4RL** continuous control environments (`gym_mujoco_v2`):

| Environment | Observation Dim | Action Dim |
| :--- | :---: | :---: |
| `hopper-medium-v2` | 11 | 3 |
| `halfcheetah-medium-v2` | 17 | 6 |
| `walker2d-medium-v2` | 17 | 6 |

Full logs, CSV metrics, and publication-ready plots (acceptance rates, trajectory divergence, Q-loss comparisons, and stability heatmaps) are saved to [`results/`](results/).

---

## 📁 Repository Architecture

```text
offline_rl_trajectory_reanalysis/
├── checkpoints/          # Model weights and training checkpoints
├── data/                 # Offline D4RL datasets (HDF5 format)
├── logs/                 # Training logs and execution traces
├── results/              # Evaluation outputs, CSV logs, HTML reports, and plots
│   └── plots/            # Diagnostic plots (Q-loss, acceptance rates, heatmaps)
├── src/                  # Core source code
│   ├── critic_guided.py   # Critic-guided top-K selection & MCTS reanalysis
│   ├── data_loader.py     # HDF5 loader and replay buffer manager
│   ├── evaluate.py        # Analytics suite, metric computation, report generator
│   ├── interfaces.py      # Abstract base contracts & standardized hyperparameters
│   ├── iql_trainer.py     # Implicit Q-Learning (IQL) trainer
│   ├── mcts_augment.py    # MCTS trajectory augmentation planner
│   ├── run_all.py         # End-to-end multi-strategy pipeline runner
│   ├── sensitivity_vine.py# Ablation suite for uncertainty percentiles
│   ├── vae_augment.py     # VAE transition generator
│   ├── vine_augment.py    # Vine branching rollout sampler
│   └── world_model.py     # Ensemble dynamics network & uncertainty estimator
├── requirements.txt      # Dependency list
├── setup_data.sh         # Automated dataset download script
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.10 or higher
* CUDA-compatible GPU (Optional, recommended for training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/offline_rl_trajectory_reanalysis.git
   cd offline_rl_trajectory_reanalysis
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch D4RL Datasets**:
   ```bash
   chmod +x setup_data.sh
   ./setup_data.sh
   ```

---

## 🚀 Execution & Usage

### 1. Run Baseline & Augmentation Comparisons
Run full pipeline experiments across a specified environment or all environments:
```bash
# Single environment
python src/run_all.py --env hopper-medium-v2 --steps 100000

# Run across all benchmark environments
python src/run_all.py --env all --steps 100000
```

### 2. Execute Critic-Guided Trajectory Reanalysis
Run advantage-targeted top-K selection with MCTS reanalysis:
```bash
python src/critic_guided.py --env hopper-medium-v2 --steps 50000
```

### 3. Run Uncertainty Sensitivity Analysis
Ablation study on uncertainty filtering thresholds (e.g., $p_{50}, p_{75}, p_{90}$):
```bash
python src/sensitivity_vine.py --steps 50000
```

### 4. Generate Reports & Diagnostic Plots
Compute normalized scores and generate the interactive summary report (`results/evaluate_report.html`):
```bash
python src/evaluate.py --results_dir results/
```

---

## 🔧 Implementation Highlights

* **Standalone HDF5 Reader**: Direct dataset parsing without hard dependencies on `gym` or `d4rl` setup packages.
* **Delta State Formulation**: Predicts state changes $(\Delta s = s' - s)$ instead of raw next states for improved convergence.
* **Truncation Handling**: Correctly tags terminal states ($terminal = 0$ on step limits) to prevent value function corruption.
* **Centralized Calibration**: Uncertainty bounds are calibrated per-dataset and shared across all augmentation methods for consistent evaluation.

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).  
Copyright (c) 2026 **ECHALH MANAL**.
