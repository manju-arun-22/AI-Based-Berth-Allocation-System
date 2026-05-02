# AI-Driven Berth Allocation Optimization
**MSc AI Dissertation | University of Hull | Student: Manju Arun | ID: 202433718**

## Project Overview
A hybrid predictive-prescriptive AI framework for maritime berth allocation optimization.
Combines ML-based vessel arrival forecasting (LSTM, XGBoost, Random Forest ensemble)
with optimization algorithms (Genetic Algorithm + Deep Reinforcement Learning/PPO).

## Project Structure
```
berth_allocation/
├── data/
│   ├── raw/              # Raw datasets (real or downloaded)
│   ├── processed/        # Cleaned and feature-engineered data
│   └── synthetic/        # Generated synthetic port data
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/             # Data loading, cleaning, feature engineering
│   ├── models/           # ML models (LSTM, XGBoost, RF, Ensemble)
│   ├── optimization/     # GA and PPO berth allocation optimizers
│   ├── simulation/       # SimPy discrete-event simulation
│   ├── visualization/    # Plots, dashboards, SHAP explanations
│   └── utils/            # Helpers, config, logging
├── tests/                # Unit tests
├── outputs/
│   ├── models/           # Saved trained models
│   ├── results/          # Experiment results (CSV, JSON)
│   └── figures/          # Generated plots
└── docs/                 # Documentation
```

## Phases
- **Phase 1:** Data generation & preprocessing
- **Phase 2:** ML forecasting (vessel arrival prediction)
- **Phase 3:** Berth allocation optimization (GA + PPO)
- **Phase 4:** Simulation validation & evaluation

## Quick Start
```bash
pip install -r requirements.txt
python src/data/synthetic_generator.py   # Generate dataset
python src/models/train_forecasting.py   # Train ML models
python src/optimization/run_ga.py        # Run Genetic Algorithm
python src/optimization/run_ppo.py       # Run PPO optimizer
python src/simulation/run_simulation.py  # Validate with SimPy
```
