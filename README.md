# 🚁 Drone Tracking

> Advanced drone tracking methods using **Kalman Filters**, **Reinforcement Learning**, and **Machine Learning**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![MATLAB](https://img.shields.io/badge/matlab-compatible-success.svg)](https://www.mathworks.com/products/matlab.html)

## 📋 Overview

This project implements state-of-the-art drone tracking algorithms combining classical filtering techniques with modern machine learning approaches. It includes implementations of Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), Interactive Multiple Model (IMM), and optimized RL-enhanced filtering methods.

## ✨ Features

### 1. **Trajectory Analysis**
   - Comprehensive trajectory function implementations
   - Mathematical documentation and theoretical analysis
   - PDF reference guide included

### 2. **Reinforcement Learning Integration**
   - RL-enhanced spiral EKF implementation
   - Full RL methodology and workflow
   - Optimization results and performance analysis
   - Complete optimization pipeline with outputs

### 3. **MATLAB GUI**
   - Full-featured GUI for filter visualization
   - Implementations of IMM, UKF, EKF, and KF algorithms
   - Support for spiral and constant acceleration motion models
   - Recommended Q-matrix element values (post-RL optimization)
   - Interactive parameter tuning

### 4. **Classification & Error Estimation**
   - Pole Classification error estimator
   - Jupyter notebook implementation
   - Mathematical derivations
   - Complete error analysis documentation

### 5. **Advanced Filtering**
   - Full IMM (Interactive Multiple Model) implementation
   - Comprehensive algorithmic analysis
   - Complete MATLAB implementation (.m files)

## 📁 Project Structure

```
Drone-tracking/
├── README.md
├── trajectory/                  # Trajectory analysis
│   ├── trajectory_functions.py
│   └── trajectory_analysis.pdf
├── rl_optimization/             # RL-based optimization
│   ├── spiral_ekf_rl.py
│   ├── rl_methodology.pdf
│   └── optimization_results.pdf
├── matlab_implementation/       # MATLAB GUI and filters
│   ├── gui/
│   ├── imm_filter.m
│   ├── ukf_filter.m
│   ├── ekf_filter.m
│   └── kalman_filter.m
├── pole_classification/         # Classification error estimation
│   ├── classifier.ipynb
│   └── error_analysis.pdf
└── docs/                        # Documentation
    └── mathematical_foundations.md
```

## 📚 Documentation

- **Mathematical Foundations**: See `docs/mathematical_foundations.md`
- **Kalman Filter Theory**: Referenced in trajectory_analysis.pdf
- **RL Optimization**: See `rl_optimization/rl_methodology.pdf`
- **Error Analysis**: See `pole_classification/error_analysis.pdf`

## 🎯 Key Results

- **RL Optimization**: Significant improvement in tracking accuracy through reinforcement learning
- **Q-Matrix Tuning**: Empirically derived optimal Q-matrix values for different motion models
- **Pole Classification**: Robust error classification with detailed statistical analysis

## 📊 Performance Metrics

See `rl_optimization/optimization_results.pdf` for detailed performance comparisons and benchmarks.

## 👤 Author

**MataKalu** - [GitHub Profile](https://github.com/MataKalu)

## 📧 Contact & Support

For questions or issues, please open a [GitHub Issue](https://github.com/MataKalu/Drone-tracking/issues).

## 🙏 Acknowledgments

- Kalman Filter family methodologies
- Reinforcement Learning optimization techniques
- Open-source scientific computing community

---

**Last Updated:** June 2026  
**Status:** Active Development
