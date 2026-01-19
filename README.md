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

---

## Reproducible Excution Pipeline

**Run Hector SLAM (Baseline - No Loop Closure)**
"python main.py"

**Apply Pose-Graph Loop Closure**
"python -m eval.pose_graph_lc \
  --traj outputs/fr079_no_lc/trajectory.txt \
  --stamps outputs/fr079_no_lc/timestamps.txt \
  --rels datasets/freiburg/fr079.relations \
  --out_traj outputs/fr079_with_lc/trajectory.txt \
  --out_overlay outputs/overlay_no_lc_vs_lc.png \
  --max_dt 2.0 \
  --lc_sig_th_deg 0.3 \
  --pgo_iters 40
"

**Quantitative Evaluation (RMSE)**
**Without Loop Closure**
"python eval/relations_eval.py \
  --traj outputs/fr079_no_lc/trajectory.txt \
  --rels datasets/freiburg/fr079.relations \
  --stamps outputs/fr079_no_lc/timestamps.txt \
  --mode time --convention ij
"
**With Loop Closure**
"python eval/relations_eval.py \
  --traj outputs/fr079_with_lc/trajectory.txt \
  --rels datasets/freiburg/fr079.relations \
  --stamps outputs/fr079_no_lc/timestamps.txt \
  --mode time --convention ij
"

**Rebuild Final Maps (Visualization)**
**No Loop Closure**
"python -m eval.rebuild_map_from_traj \
  --traj outputs/fr079_no_lc/trajectory.txt \
  --log datasets/freiburg/fr079.clf \
  --out_dir outputs/fr079_maps \
  --tag no_lc
"

**With Loop Closure**
"python -m eval.rebuild_map_from_traj \
  --traj outputs/fr079_with_lc/trajectory.txt \
  --log datasets/freiburg/fr079.clf \
  --out_dir outputs/fr079_maps \
  --tag with_lc
"