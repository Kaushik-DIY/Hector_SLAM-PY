# Hector SLAM (Python, No ROS) with Loop Closure

This repository contains a **from-scratch Python implementation of Hector SLAM**
(without ROS), extended with **pose-graph loop closure** and evaluated on the
**Freiburg (CARMEN) dataset**.

The project includes:
- Occupancy-grid mapping
- Multi-resolution scan-to-map matching (Gauss–Newton)
- Offline pose-graph optimization (PGO)
- Quantitative evaluation using Freiburg relative pose constraints

---

## Features

- Pure Python (NumPy + SciPy)
- No odometry required (Hector-style)
- Offline loop closure using pose graphs
- Robust optimization with sparse solvers
- Clean visualization and evaluation pipeline

---

## Folder Structure

```text
Hector_slam/
├── slam/           # Core SLAM implementation
├── eval/           # Evaluation & loop-closure scripts
├── viz/            # Visualization utilities
├── dataio/         # Dataset loaders
├── config.py       # Parameters
├── main.py         # Baseline Hector SLAM run
├── requirements.txt
└── README.md
