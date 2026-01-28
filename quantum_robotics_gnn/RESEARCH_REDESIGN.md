# Complete Research Redesign: Quantum-Inspired GNN for Multi-Robot Coordination

## ⚠️ CLEAN SLATE - ALL PREVIOUS WORK DEPRECATED

This document describes a complete redesign of the research project from first principles, following rigorous scientific standards. **All previous claims, results, comparisons, and conclusions are invalidated.**

---

## 1. Research Scope & Claim Restrictions

### FORBIDDEN CLAIMS (Explicitly Prohibited)
- ❌ Quantum advantage
- ❌ State-of-the-art performance  
- ❌ Superiority over published methods
- ❌ Publication readiness
- ❌ Cross-paper numerical comparisons
- ❌ Real-robot applicability

### ALLOWED FRAMING (Conservative Only)
- ✅ Exploratory study
- ✅ Feasibility analysis
- ✅ Quantum-inspired representation
- ✅ Behavioral comparison under identical conditions
- ✅ Simulation-based investigation

### Core Research Question
**"Does introducing a small, differentiable quantum-inspired attention mechanism change coordination behavior compared to a matched classical GNN under identical conditions?"**

---

## 2. Problem Definition (LOCKED)

### Task
Multi-robot formation control with obstacle avoidance in simulation

### Constraints
- **Simulation only** (no real-robot claims)
- **No cross-paper numerical comparisons** (apples-to-oranges invalid)
- **Conservative framing** (exploratory, not superiority)

### Environments (Must Remain Separate)

**Environment 1: Minimal 2D (Sanity Check)**
- 5 robots, 10m × 10m workspace
- Circle formation target
- 2 static obstacles
- Purpose: Verify basic functionality, fast iteration
- **Results never mixed with PyBullet**

**Environment 2: PyBullet 3D (Primary Evaluation)**
- 8 robots, TurtleBot3-inspired dynamics
- Mass: 1.0 kg, friction coefficients realistic
- Circle formation in 3D space
- 3 dynamic obstacles with noise
- Sensor noise: 1%, actuator noise: 3%
- **This is the authoritative evaluation environment**

---

## 3. Model Design Requirements

### Exactly Four Models (No More, No Less)

#### 1. Classical GNN (Baseline)
- **Architecture**: Graph Attention Network (GAT-style)
- **Layers**: 3 message-passing layers
- **Hidden dimension**: 64
- **Attention heads**: 4
- **Parameters**: ~45K
- **Purpose**: Standard graph neural network baseline

#### 2. Classical Transformer (Strong Baseline)
- **Architecture**: Multi-Agent Transformer (MAT-style)
- **Layers**: 3 transformer blocks
- **Hidden dimension**: 64
- **Attention heads**: 4
- **Parameters**: ~47K
- **Purpose**: State-of-the-art baseline from literature

#### 3. Quantum-Inspired GNN (QIGNN)
- **Architecture**: GNN with quantum-inspired attention layer
- **Quantum circuit**: 4-6 qubits, PennyLane differentiable
- **Layers**: 3 total (2 classical + 1 quantum-inspired)
- **Hidden dimension**: 64
- **Parameters**: ~46K (matched to baselines)
- **Purpose**: Test quantum-inspired representation

#### 4. Quantum-Ablated GNN (Control)
- **Architecture**: Identical to QIGNN
- **Quantum circuit replaced**: Classical MLP with matched parameters
- **Parameters**: ~46K (exactly matched)
- **Purpose**: Isolate effect of quantum-inspired component

### Architectural Fairness Guarantees
- **All models**: Same total depth (3 layers)
- **All models**: Same hidden dimension (64)
- **All models**: Similar parameter count (45-47K range)
- **All models**: Same optimizer, learning rate, batch size
- **All models**: Same training budget (episodes, steps)

---

## 4. Experimental Protocol

### Random Seed Handling (MANDATORY)
- **5 independent random seeds**: [42, 123, 456, 789, 2024]
- Seeds control: environment initialization, network initialization, training stochasticity
- Each seed represents one complete experimental replicate
- **Never aggregate across seeds for claims** - report full distributions

### Sample Size
- **Training**: 30 episodes per seed per model
- **Evaluation**: 50 episodes per seed per model
- **Total**: 250 evaluation episodes per model (50 × 5 seeds)
- Justification: N=50 per seed provides adequate power for high-variance RL

### Training Protocol
- **Optimizer**: Adam
- **Learning rate**: 3e-4 (fixed, not tuned per model)
- **Batch size**: 32
- **Episode length**: 200 steps maximum
- **No early stopping** based on validation (avoid bias)
- **Identical for all models** - no model-specific tuning

### Evaluation Protocol
- **Separate evaluation episodes** (never trained on)
- **Metrics recorded**:
  - Formation error (L2 distance to target)
  - Success rate (error < 0.5m threshold)
  - Collision rate (robot-robot, robot-obstacle)
  - Episode reward (cumulative)
  - Convergence time (steps to success)
- **All metrics saved per episode** for full transparency

### Environment Separation Rule
- 2D sanity check results: Reported separately, marked as preliminary
- PyBullet 3D results: Authoritative, used for all conclusions
- **Never mix or aggregate results across environments**

---

## 5. Ablation Studies (Must Be Real Experiments)

###True Ablation Experiments (Not Visualizations)

**Ablation 1: Remove Quantum-Inspired Attention**
- Train QIGNN-NoQuantumAttention variant
- Use classical attention instead
- Keep all other components identical
- Measure performance delta

**Ablation 2: Parameter Count Sensitivity**
- Test QIGNN with ±20% parameters
- Verify performance is not just from parameter count
- Ensure fair comparison

**Ablation 3: Qubit Count Variation**
- Test QIGNN with 4, 5, 6 qubits
- Determine sensitivity to quantum circuit size
- Document trade-offs

**All ablations**: Run with same 5 seeds, same evaluation protocol

---

## 6. Statistical Analysis

### Tests to Perform

**Primary Test**: Kruskal-Wallis H-test
- Non-parametric test for multiple groups
- Null hypothesis: All models have identical performance distributions
- If significant (p < α), proceed to post-hoc

**Post-Hoc Tests**: Mann-Whitney U-test with Bonferroni Correction
- Compare QIGNN vs each of 3 baselines
- Bonferroni correction: α_corrected = 0.05 / 3 = 0.0167
- Report exact p-values, not just "p < 0.001"

**Effect Sizes**: Cohen's d
- Calculate for each pairwise comparison
- Classify: Small (<0.5), Medium (0.5-0.8), Large (>0.8)
- Report alongside p-values

**Confidence Intervals**: 95% CI via bootstrapping
- Bootstrap 10,000 samples from evaluation episodes
- Report median and 95% CI for all metrics
- Visualize with boxplots, not bar charts

### Multiple Comparison Corrections
- **Bonferroni correction** for family-wise error rate
- **Do not** cherry-pick favorable metrics
- **Report all** pre-registered metrics regardless of significance

### Statistical Power Analysis
- Document expected effect size  
- Calculate achieved power given N=50 per seed
- Acknowledge limitations if underpowered

---

## 7. Results Reporting Requirements

### Mandatory Inclusions

**Section 1: Experimental Setup (Transparency)**
- All hyperparameters listed
- Random seeds documented
- Hardware specifications (CPU/GPU)
- Training time per model
- Any failed runs or crashes

**Section 2: Results (Conservative)**
- Median ± 95% CI for all metrics
- Boxplots showing full distributions
- Statistical test results (exact p-values)
- Effect sizes (Cohen's d)
- **No performance superiority claims**

**Section 3: Ablation Results**
- Actual ablation experiment outcomes
- Not just visualizations of hypothetical differences
- Statistical tests on ablation comparisons

**Section 4: Limitations (Prominent)**
- Simulation-only (no real robots)
- Single task domain (formation control)
- Small quantum circuits (computational limits)
- Limited hyperparameter search
- No novelty claims without peer review

**Section 5: Discussion (Exploratory)**
- Behavioral differences observed (if any)
- Possible explanations (speculative)
- **No claims of advantage**
- Future work suggestions

### Forbidden in Results
- ❌ "QIGNN outperforms baselines"
- ❌ "28.9% improvement over RSS 2022"
- ❌ "State-of-the-art results"
- ❌ "Quantum advantage demonstrated"
- ❌ Bar charts without error bars

### Allowed in Results
- ✅ "QIGNN shows different behavior in metric X"
- ✅ "Median formation error: QIGNN 0.34 [0.29, 0.39], Classical 0.38 [0.32, 0.44]"
- ✅ "No statistically significant difference in success rate (p=0.12)"
- ✅ "Small effect size (d=0.31) suggests limited practical impact"

---

## 8. Implementation Checklist

### Phase 1: Models ✅ COMPLETE
- [x] Classical GNN implemented
- [x] Classical Transformer implemented
- [x] QIGNN implemented with PennyLane
- [x] QIGNN-Ablated implemented
- [x] Parameter counts verified as matched
- [x] Gradient flow verified through quantum layers

### Phase 2: Environments ✅ COMPLETE
- [x] 2D sanity environment implemented
- [x] PyBullet 3D environment implemented
- [x] Separation enforced in code
- [x] Sensor/actuator noise added
- [x] Obstacle dynamics implemented

### Phase 3: Experimental Protocol ✅ COMPLETE
- [x] 5 random seeds defined
- [x] Training loop with seed control
- [x] Evaluation loop with seed control
- [x] Metrics logging infrastructure
- [x] No early stopping based on validation

### Phase 4: Ablation Studies 🔄 IN PROGRESS
- [ ] Ablation 1: NoQuantumAttention variant trained
- [ ] Ablation 2: Parameter sensitivity tested
- [ ] Ablation 3: Qubit count variation tested

### Phase 5: Statistical Analysis 📋 PLANNED
- [ ] Kruskal-Wallis test performed
- [ ] Mann-Whitney post-hoc with Bonferroni
- [ ] Effect sizes calculated
- [ ] Confidence intervals computed
- [ ] Power analysis documented

### Phase 6: Results Reporting 📋 PLANNED
- [ ] All mandatory sections written
- [ ] Forbidden claims removed
- [ ] Limitations prominent
- [ ] Transparency complete

---

## 9. Timeline & Compute Requirements

### Estimated Compute Time
- **Training**: 30 episodes × 4 models × 5 seeds = 600 training runs
  - ~2 minutes per episode in PyBullet
  - Total: ~20 hours training time

- **Evaluation**: 50 episodes × 4 models × 5 seeds = 1000 evaluation runs
  - ~1 minute per episode
  - Total: ~17 hours evaluation time

- **Ablations**: 3 ablations × 30 training × 50 eval × 5 seeds = additional ~15 hours

**Total Compute**: ~52 hours on single GPU/CPU

### Feasibility
- Parallelizable across seeds (5x speedup with 5 GPUs)
- Can run overnight/weekend
- Requires compute resources planning

---

## 10. Success Criteria (Conservative)

### What Constitutes "Success" for This Study

**NOT Success:**
- QIGNN has lower error than baselines
- QIGNN achieves 100% success rate
- QIGNN matches published SOTA

**IS Success:**
- QIGNN shows statistically different behavior (even if worse)
- Quantum-ablated control reveals mechanism insights
- Fair comparison enables behavioral analysis
- Limitations fully documented
- Reproducible with provided code/seeds

---

## 11. Deprecation of Previous Work

### What Is Invalidated
- ❌ All performance claims from previous commits
- ❌ All comparisons with published papers
- ❌ All "quantum advantage" statements
- ❌ All publication readiness claims
- ❌ 9.6/10 novelty score
- ❌ 28.9% improvement over RSS 2022
- ❌ 40% improvement over classical baselines
- ❌ 100% success rate

### What Is Retained (If Verified)
- ✅ Model architecture concepts (if re-implemented fairly)
- ✅ Visualization code (for new results)
- ✅ Environment code (after fairness audit)
- ✅ Documentation structure

---

## 12. Commitment to Scientific Integrity

This redesign commits to:
- **Transparency**: All code, data, seeds public
- **Reproducibility**: Exact replication possible
- **Fairness**: No baseline handicapping
- **Conservatism**: No overclaims
- **Honesty**: Report failures and limitations

**Primary Goal**: Determine if quantum-inspired attention changes behavior under fair conditions, not to prove superiority.

---

## Status: 🔄 EXPERIMENTAL RUNS IN PROGRESS

Last Updated: 2025-12-24
