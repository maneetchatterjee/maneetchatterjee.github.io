# Getting Started with QEGAN

This guide will help you quickly get up and running with the QEGAN quantum robotics project.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 2GB for dependencies and outputs

### Required Knowledge
- Basic Python programming
- Familiarity with PyTorch (helpful but not required)
- Basic understanding of graph neural networks (helpful)
- No quantum computing knowledge required!

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/maneetchatterjee/quantum-robotics-gnn.git
cd quantum_robotics_gnn
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv qegan_env
source qegan_env/bin/activate  # On Windows: qegan_env\Scripts\activate

# OR using conda
conda create -n qegan python=3.8
conda activate qegan
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and PyTorch Geometric
- PennyLane (quantum computing)
- PyBullet (physics simulation)
- NumPy, Matplotlib, Pandas, Seaborn, Pillow

### Step 4: Verify Installation

```bash
python -c "import torch; import pennylane; import pybullet; print('✅ Installation successful!')"
```

---

## 🎯 Quick Start (5 minutes)

### Option 1: Run Quick Demo

```bash
python experiments/scripts/run_demo.py
```

**What this does:**
- Trains QEGAN model for 10 episodes
- Compares with 2 baselines
- Generates basic plots
- Prints results to console

**Expected output:**
```
Training QEGAN...
Episode 10/10: Reward=-18.23, Error=0.182
Training complete!

Results:
QEGAN: Error=0.175, Success=100%
Classical GNN: Error=0.292, Success=85%
```

### Option 2: Test PyBullet Environment

```bash
python -c "
from src.environments.pybullet_environment import create_pybullet_env
env = create_pybullet_env(use_gui=True)
print('✅ PyBullet environment created successfully!')
"
```

This opens a 3D visualization window showing the robots.

---

## 📊 Generate Visualizations (10 minutes)

### Generate All Visualizations

```bash
python src/visualization/generate_all_visuals.py
```

**What this creates:**

1. **Architecture Diagrams** (4 files in `outputs/architecture_diagrams/`)
   - Network architecture
   - Quantum circuits
   - Model comparisons
   - Data flow

2. **Ablation Study** (5 plots + analysis in `outputs/ablation_study/`)
   - Component contribution analysis
   - Performance comparisons
   - Multi-metric radar chart

3. **Animations** (3 GIFs in `outputs/animations/`)
   - Training dynamics
   - Robot formation control
   - Quantum state evolution

4. **Consolidated Results** (`outputs/CONSOLIDATED_RESULTS.txt`)
   - Complete results document

### View Generated Files

```bash
# List architecture diagrams
ls outputs/architecture_diagrams/

# List ablation study results
ls outputs/ablation_study/

# List animations
ls outputs/animations/

# View consolidated results
cat outputs/CONSOLIDATED_RESULTS.txt | less
```

---

## 🔬 Run Full Experiments (2-3 hours)

### Comprehensive Evaluation

```bash
python experiments/scripts/run_comprehensive_experiments.py
```

**What this does:**
- Uses PyBullet 3D physics simulation
- Trains 9 models (QEGAN + 8 baselines)
- Tests 4 formation types (Circle, Line, V-shape, Grid)
- Runs 80+ test scenarios
- Generates benchmark comparisons
- Creates LaTeX tables for publication
- Outputs statistical analysis

**Progress tracking:**
```
[1/9] Training QEGAN... (30 episodes)
[2/9] Training Classical GNN...
...
Evaluation on Circle formation...
Evaluation on Line formation...
...
Generating benchmark plots...
Creating LaTeX tables...
✅ All experiments complete!
```

**Output locations:**
- Benchmark plots: `outputs/benchmark_results/`
- Statistics: `outputs/experimental_results/`
- LaTeX tables: `outputs/benchmark_results/benchmark_latex_table.txt`

---

## 📂 Understanding the Repository

### Key Directories

```
quantum_robotics_gnn/
├── src/                    # All source code
│   ├── models/            # QEGAN model
│   ├── baselines/         # 8 baseline models
│   ├── environments/      # Simulation environments
│   ├── analysis/          # Analysis tools
│   └── visualization/     # Visualization generators
│
├── experiments/           # Experiment scripts
│   └── scripts/          # Run scripts here
│
├── outputs/              # All generated outputs
│   ├── architecture_diagrams/
│   ├── ablation_study/
│   ├── animations/
│   ├── benchmark_results/
│   └── experimental_results/
│
└── docs/                 # Documentation
```

### Important Files

- **README_NEW.md** - Complete documentation
- **PROJECT_OVERVIEW.md** - Detailed project overview
- **GETTING_STARTED.md** - This file
- **requirements.txt** - Python dependencies
- **reorganize_repository.py** - Repository organization script

---

## 🎓 Tutorials

### Tutorial 1: Run a Single Model

```python
# File: test_single_model.py
from src.models.qegan_model import QEGAN
from src.environments.robot_environment import MultiRobotFormationEnv
import torch

# Create environment
env = MultiRobotFormationEnv(n_robots=5)

# Create model
model = QEGAN(input_dim=6, hidden_dim=64, output_dim=2)

# Get observation
obs = env.reset()

# Run model (forward pass)
state = torch.FloatTensor(obs['states'])
edge_index = torch.LongTensor(obs['edge_index'])
output = model(state, edge_index)

print(f"✅ Model output shape: {output.shape}")
```

### Tutorial 2: Visualize Robot Environment

```python
# File: visualize_env.py
from src.environments.pybullet_environment import create_pybullet_env
import time

# Create environment with GUI
env = create_pybullet_env(n_robots=10, use_gui=True)

# Reset environment
obs = env.reset()

# Run for 100 steps
for i in range(100):
    # Random actions
    actions = env.action_space.sample()
    obs, reward, done, info = env.step(actions)
    time.sleep(0.01)  # Slow down for visualization
    
    if done:
        print(f"Episode finished at step {i}")
        break

print("✅ Visualization complete!")
```

### Tutorial 3: Run Ablation Study

```python
# Run ablation study directly
from src.analysis.ablation_study import run_ablation_study

# This will:
# 1. Test QEGAN with each component removed
# 2. Generate 5 comparison plots
# 3. Create analysis report
# 4. Output to outputs/ablation_study/

run_ablation_study()
print("✅ Ablation study complete! Check outputs/ablation_study/")
```

---

## 🛠️ Common Tasks

### Task 1: Generate Only Architecture Diagrams

```bash
python src/visualization/generate_diagrams.py
```

Output: `outputs/architecture_diagrams/*.png`

### Task 2: Run Only Novelty Analysis

```bash
python src/analysis/novelty_analysis.py
```

Output: Console output + `novelty_analysis.json`

### Task 3: Test a Specific Baseline

```python
from src.baselines.additional_baselines import create_MAT_model

model = create_MAT_model(input_dim=6, hidden_dim=64, output_dim=2)
# Use model for training/evaluation
```

### Task 4: View Existing Results

```bash
# View experimental results
cat outputs/experimental_results/experimental_report.txt

# View consolidated results
cat outputs/CONSOLIDATED_RESULTS.txt

# View ablation results
cat outputs/ablation_study/ablation_report.txt
```

### Task 5: Reorganize Repository

If files get disorganized:

```bash
python reorganize_repository.py
```

This will reorganize everything into the proper structure.

---

## 🐛 Troubleshooting

### Problem: Import Errors

```bash
# Solution 1: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Solution 2: Install in development mode
pip install -e .
```

### Problem: PyBullet No Display

```bash
# Solution: Run without GUI
python experiments/scripts/run_comprehensive_experiments.py --no-gui

# OR use virtual display
sudo apt-get install xvfb
xvfb-run -a python experiments/scripts/run_comprehensive_experiments.py
```

### Problem: Out of Memory

```python
# Solution: Reduce problem size in environment config
# Edit src/environments/robot_environment.py or pybullet_environment.py

# Reduce number of robots
n_robots = 5  # Instead of 10

# Reduce episode length
max_steps = 50  # Instead of 100
```

### Problem: Slow Training

```python
# Solution 1: Use CPU for quantum circuits (if GPU causes issues)
import pennylane as qml
dev = qml.device('default.qubit', wires=4, shots=None)

# Solution 2: Reduce number of episodes
# In run_experiments.py or run_comprehensive_experiments.py
n_episodes = 10  # Instead of 50

# Solution 3: Use simplified 2D environment instead of PyBullet
from src.environments.robot_environment import MultiRobotFormationEnv
```

### Problem: Missing Packages

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Install specific package
pip install pennylane
pip install pybullet
pip install torch-geometric
```

---

## 📚 Next Steps

### For Researchers

1. **Read the paper summary**: `docs/SUMMARY.md`
2. **Review experimental results**: `outputs/CONSOLIDATED_RESULTS.txt`
3. **Check ablation study**: `outputs/ablation_study/ablation_report.txt`
4. **Review architecture**: `outputs/architecture_diagrams/`

### For Developers

1. **Understand code structure**: `PROJECT_OVERVIEW.md`
2. **Explore source code**: `src/` directory
3. **Run experiments**: `experiments/scripts/`
4. **Add new features**: Follow code organization in `src/`

### For Publication

1. **Run full experiments**: `python experiments/scripts/run_comprehensive_experiments.py`
2. **Generate all visuals**: `python src/visualization/generate_all_visuals.py`
3. **Review publication guide**: `docs/PUBLICATION_README.md`
4. **Get LaTeX tables**: `outputs/benchmark_results/benchmark_latex_table.txt`

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list | grep -E "torch|pennylane|pybullet"`)
- [ ] Can import key modules (`python -c "from src.models.qegan_model import QEGAN"`)
- [ ] Can run quick demo (`python experiments/scripts/run_demo.py`)
- [ ] Can generate diagrams (`python src/visualization/generate_diagrams.py`)
- [ ] Can view PyBullet environment (if GUI available)
- [ ] Repository is organized (`ls src/ outputs/ experiments/`)

---

## 💡 Tips for Success

1. **Start Small**: Run quick demo first before full experiments
2. **Check Outputs**: Regularly check `outputs/` directory for generated files
3. **Read Logs**: Console output provides useful progress information
4. **Use Virtual Environment**: Avoid dependency conflicts
5. **Monitor Memory**: Close unnecessary programs during training
6. **Save Results**: Results are automatically saved to `outputs/`
7. **Read Documentation**: Check `docs/` directory for detailed information
8. **Ask for Help**: Create GitHub issue if you encounter problems

---

## 📞 Getting Help

1. **Check Documentation**:
   - README_NEW.md - Main documentation
   - PROJECT_OVERVIEW.md - Project details
   - This file - Getting started guide

2. **Review Examples**:
   - `experiments/scripts/run_demo.py` - Simple example
   - Tutorial code snippets above

3. **Common Issues**:
   - Check Troubleshooting section above
   - Search GitHub issues

4. **Contact**:
   - Create GitHub issue with detailed description
   - Include error messages and system info

---

## 🎉 Success!

If you've reached this point, you should be able to:
- ✅ Run experiments
- ✅ Generate visualizations
- ✅ View results
- ✅ Understand the codebase
- ✅ Modify and extend the code

**Happy experimenting with QEGAN!**

---

**Last Updated**: 2024
**Version**: 1.0
**Documentation**: Complete
