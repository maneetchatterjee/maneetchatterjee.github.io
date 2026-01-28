# Experimental Protocol Documentation

## Complete Methodology for Rigorous Evaluation

This document specifies the exact experimental protocol for fair comparison of quantum-inspired and classical graph neural networks for multi-robot coordination.

---

## 1. Experimental Design

### 1.1 Between-Subjects Factors
- **Model Type**: 4 levels (Classical GNN, Classical Transformer, QIGNN, QIGNN-Ablated)
- **Random Seed**: 5 levels (42, 123, 456, 789, 2024)

### 1.2 Within-Subjects Factors
- **Episode**: 50 evaluation episodes per seed
- **Environment**: PyBullet 3D (primary), 2D (sanity check only)

### 1.3 Total Experimental Conditions
- 4 models × 5 seeds × 50 episodes = **1000 evaluation trials**
- Plus 600 training episodes (30 per model per seed)

---

## 2. Randomization Protocol

### 2.1 Seed Assignment
```python
EXPERIMENT_SEEDS = [42, 123, 456, 789, 2024]

for seed in EXPERIMENT_SEEDS:
    # Set all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment seed
    env.seed(seed)
    
    # Initialize model with seed
    model = Model(seed=seed)
```

### 2.2 What Seeds Control
- Robot initial positions (sampled from workspace)
- Obstacle positions and velocities
- Network weight initialization
- Training batch sampling
- Evaluation episode ordering

### 2.3 What Seeds Do NOT Control
- Hyperparameters (fixed across all seeds)
- Network architecture (identical per model type)
- Training budget (30 episodes for all)

---

## 3. Training Protocol

### 3.1 Hyperparameters (LOCKED - Not Tuned Per Model)

```python
TRAINING_CONFIG = {
    "optimizer": "Adam",
    "learning_rate": 3e-4,
    "batch_size": 32,
    "episodes": 30,
    "max_steps_per_episode": 200,
    "gamma": 0.99,  # Discount factor
    "tau": 0.005,  # Soft update
    "buffer_size": 100000,
    "update_frequency": 1,
    "warmup_episodes": 5,
}
```

### 3.2 Training Procedure

```
For each model M in [Classical_GNN, Classical_Transformer, QIGNN, QIGNN_Ablated]:
    For each seed S in [42, 123, 456, 789, 2024]:
        1. Set random seed S
        2. Initialize environment(seed=S)
        3. Initialize model M(seed=S)
        4. Initialize optimizer(lr=3e-4)
        
        For episode E in range(30):
            5. Reset environment
            6. Collect trajectory until done or 200 steps
            7. Update model M using collected experience
            8. Log training metrics
            9. NO early stopping, NO validation checks
        
        10. Save model M trained on seed S
        11. Proceed to evaluation
```

### 3.3 Training Termination
- Fixed at 30 episodes for ALL models
- No early stopping based on performance
- No model-specific adjustments

---

## 4. Evaluation Protocol

### 4.1 Evaluation Procedure

```
For each trained model M_S (model M trained on seed S):
    For evaluation episode E in range(50):
        1. Set evaluation seed: S_eval = S + E
        2. Reset environment(seed=S_eval)
        3. Run episode with model M_S (deterministic policy)
        4. Record all metrics
        5. Save trajectory data
```

### 4.2 Metrics Collected (Per Episode)

**Primary Metrics:**
- `formation_error`: Mean L2 distance from robots to target formation
- `success_rate`: Binary (1 if formation_error < 0.5m at episode end, else 0)
- `collision_rate`: Number of collisions / total steps
- `episode_reward`: Cumulative reward over episode

**Secondary Metrics:**
- `convergence_time`: Steps until formation_error < 0.5m (if achieved)
- `final_formation_error`: Formation error at episode termination
- `robot_robot_collisions`: Count of robot-robot collisions
- `robot_obstacle_collisions`: Count of robot-obstacle collisions
- `mean_velocity`: Average robot velocity over episode
- `trajectory_smoothness`: Variance in velocity changes

### 4.3 Data Logging

All metrics saved to:
```
results/
├── classical_gnn/
│   ├── seed_42/
│   │   ├── training_log.json
│   │   ├── eval_episode_000.json
│   │   ├── eval_episode_001.json
│   │   └── ...
│   ├── seed_123/
│   └── ...
├── classical_transformer/
├── qignn/
└── qignn_ablated/
```

---

## 5. Statistical Analysis Plan (Pre-Registered)

### 5.1 Primary Research Question
Does QIGNN show different behavior from matched classical baselines under identical conditions?

### 5.2 Null Hypotheses

**H0_1**: Distribution of formation errors is identical across all 4 models
**H0_2**: Distribution of success rates is identical across all 4 models
**H0_3**: Distribution of collision rates is identical across all 4 models

### 5.3 Statistical Tests

**Step 1: Omnibus Test (Kruskal-Wallis)**
```python
from scipy.stats import kruskal

# Formation error across all 4 models
formation_errors = {
    'Classical_GNN': [...],  # 250 values (50 episodes × 5 seeds)
    'Classical_Transformer': [...],
    'QIGNN': [...],
    'QIGNN_Ablated': [...]
}

H, p_value = kruskal(*formation_errors.values())

if p_value < 0.05:
    print("Significant difference detected, proceed to post-hoc")
else:
    print("No significant difference across models")
```

**Step 2: Post-Hoc Pairwise Tests (Mann-Whitney U)**
```python
from scipy.stats import mannwhitneyu

# Bonferroni correction for 3 comparisons:
# QIGNN vs Classical_GNN
# QIGNN vs Classical_Transformer
# QIGNN vs QIGNN_Ablated
alpha_corrected = 0.05 / 3  # = 0.0167

comparisons = [
    ('QIGNN', 'Classical_GNN'),
    ('QIGNN', 'Classical_Transformer'),
    ('QIGNN', 'QIGNN_Ablated'),
]

for model_a, model_b in comparisons:
    U, p = mannwhitneyu(
        formation_errors[model_a],
        formation_errors[model_b],
        alternative='two-sided'
    )
    significant = "YES" if p < alpha_corrected else "NO"
    print(f"{model_a} vs {model_b}: p={p:.4f}, Significant: {significant}")
```

**Step 3: Effect Size (Cohen's d)**
```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d_gnn = cohens_d(formation_errors['QIGNN'], formation_errors['Classical_GNN'])
d_transformer = cohens_d(formation_errors['QIGNN'], formation_errors['Classical_Transformer'])
d_ablated = cohens_d(formation_errors['QIGNN'], formation_errors['QIGNN_Ablated'])

# Interpret:
# |d| < 0.5: Small effect
# 0.5 ≤ |d| < 0.8: Medium effect
# |d| ≥ 0.8: Large effect
```

**Step 4: Confidence Intervals (Bootstrapping)**
```python
from scipy.stats import bootstrap

def median_func(data, axis):
    return np.median(data, axis=axis)

# 95% CI for median formation error
for model in formation_errors.keys():
    data = (formation_errors[model],)
    res = bootstrap(data, median_func, n_resamples=10000, confidence_level=0.95)
    ci_low, ci_high = res.confidence_interval
    median = np.median(formation_errors[model])
    print(f"{model}: Median={median:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")
```

### 5.4 Multiple Comparison Corrections

**Family-Wise Error Rate (FWER) Control:**
- Bonferroni correction applied to all pairwise comparisons
- α_corrected = α / number_of_comparisons
- For 3 comparisons: α_corrected = 0.05 / 3 = 0.0167

**What This Means:**
- We only reject H0 if p < 0.0167 (not p < 0.05)
- This controls false positive rate across all tests
- More conservative but avoids Type I errors

### 5.5 Power Analysis

**Expected Effect Size:**
- Based on pilot studies: d ≈ 0.4 (small-medium effect)

**Achieved Power:**
```python
from statsmodels.stats.power import tt_ind_solve_power

power = tt_ind_solve_power(
    effect_size=0.4,
    nobs1=250,  # 50 episodes × 5 seeds per model
    alpha=0.0167,  # Bonferroni corrected
    ratio=1.0,
    alternative='two-sided'
)
print(f"Achieved power: {power:.2f}")
# Typically want power ≥ 0.80
```

---

## 6. Data Quality Checks

### 6.1 Training Convergence Checks

For each model-seed combination, verify:
- Loss decreases over training episodes
- No NaN or Inf values in gradients
- Model parameters update (not frozen)
- Episode rewards show learning trend

If any checks fail:
- Log the failure
- Re-run that specific model-seed combination
- If failure persists, report in limitations

### 6.2 Evaluation Data Checks

For each evaluation episode, verify:
- Episode completes (no crashes)
- Metrics are within valid ranges
- No duplicate episode IDs
- Seed reproducibility (re-running same seed gives same result)

### 6.3 Outlier Handling

**Rule**: Do NOT remove outliers unless:
- Episode crashed (environment error)
- Metrics are physically impossible (e.g., negative distance)

**All legitimate outliers**: Kept in analysis (important for distribution)

---

## 7. Reproducibility Requirements

### 7.1 Code Availability
- All code public on GitHub
- Requirements.txt with exact versions
- README with step-by-step instructions
- Docker container for environment (optional but recommended)

### 7.2 Data Availability
- All raw episode logs saved and available
- Aggregated metrics in CSV format
- Training checkpoints saved

### 7.3 Hardware Specifications
Document:
- CPU model and core count
- GPU model (if used) and VRAM
- RAM amount
- OS version
- Python version
- PyTorch version
- PennyLane version

### 7.4 Replication Instructions

```bash
# 1. Clone repository
git clone https://github.com/maneetchatterjee/quantum-robotics-gnn.git
cd quantum-robotics-gnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments (will take ~52 hours on single GPU)
python experiments/rigorous_experiments.py --full

# 4. Analyze results
python src/analysis/conservative_statistics.py

# 5. Generate report
python src/analysis/generate_report.py
```

---

## 8. Deviation Log

Any deviations from this protocol must be:
1. Documented immediately
2. Justified scientifically
3. Reported in final results
4. Labeled clearly as post-hoc

**No deviations are currently planned.**

---

## 9. Timeline

### Estimated Duration
- **Setup**: 2 hours (environment, models, logging)
- **Training**: 20 hours (600 training runs)
- **Evaluation**: 17 hours (1000 evaluation runs)
- **Ablations**: 15 hours (additional variants)
- **Statistical Analysis**: 3 hours (tests, plots, report)
- **Total**: ~57 hours

### Parallelization Opportunities
- Train 5 seeds in parallel: reduces to ~11 hours
- Use multiple GPUs: further speedup possible
- Batch evaluation episodes: minor speedup

---

## 10. Success Criteria

This protocol is successful if:
1. ✅ All 1000 evaluation episodes complete without crashes
2. ✅ Data quality checks pass for all episodes
3. ✅ Statistical tests executed as pre-registered
4. ✅ Results reported with full transparency
5. ✅ No overclaims or forbidden statements made

**This protocol does NOT require:**
- QIGNN to outperform baselines
- Statistical significance in any particular direction
- Publication acceptance

**Goal**: Fair, rigorous comparison that answers the research question honestly.

---

Last Updated: 2025-12-24
Protocol Version: 1.0
