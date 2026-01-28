#!/usr/bin/env python3
"""
Reorganization Script for QEGAN Repository

This script reorganizes the quantum robotics GNN repository into a clean,
professional structure suitable for publication and sharing.

Directory Structure:
    src/                    - Source code organized by functionality
    outputs/                - All generated outputs (plots, animations, results)
    docs/                   - Documentation files
    experiments/            - Experiment configurations and scripts
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the organized directory structure."""
    dirs = [
        # Source code organization
        'src/models',
        'src/environments',
        'src/baselines',
        'src/experiments',
        'src/analysis',
        'src/visualization',
        
        # Output organization
        'outputs/architecture_diagrams',
        'outputs/ablation_study',
        'outputs/animations',
        'outputs/benchmark_results',
        'outputs/experimental_results',
        'outputs/visualizations',
        
        # Documentation
        'docs',
        
        # Experiments
        'experiments/configs',
        'experiments/scripts',
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created")

def organize_source_files():
    """Organize source code files into appropriate directories."""
    
    # Model files
    model_files = {
        'qegan_model.py': 'src/models/qegan_model.py',
        'baseline_models.py': 'src/baselines/baseline_models.py',
        'additional_baselines.py': 'src/baselines/additional_baselines.py',
    }
    
    # Environment files
    env_files = {
        'robot_environment.py': 'src/environments/robot_environment.py',
        'pybullet_environment.py': 'src/environments/pybullet_environment.py',
    }
    
    # Analysis files
    analysis_files = {
        'novelty_analysis.py': 'src/analysis/novelty_analysis.py',
        'benchmark_comparison.py': 'src/analysis/benchmark_comparison.py',
        'ablation_study.py': 'src/analysis/ablation_study.py',
    }
    
    # Visualization files
    viz_files = {
        'generate_diagrams.py': 'src/visualization/generate_diagrams.py',
        'generate_animations.py': 'src/visualization/generate_animations.py',
        'generate_consolidated_results.py': 'src/visualization/generate_consolidated_results.py',
        'generate_all_visuals.py': 'src/visualization/generate_all_visuals.py',
    }
    
    # Experiment files
    exp_files = {
        'run_experiments.py': 'experiments/scripts/run_experiments.py',
        'run_demo.py': 'experiments/scripts/run_demo.py',
        'run_comprehensive_experiments.py': 'experiments/scripts/run_comprehensive_experiments.py',
    }
    
    all_files = {**model_files, **env_files, **analysis_files, **viz_files, **exp_files}
    
    for src, dst in all_files.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
    
    print("✅ Source files organized")

def organize_output_files():
    """Organize output files into appropriate directories."""
    
    # Move existing results
    result_mappings = {
        'results/training_rewards.png': 'outputs/experimental_results/training_rewards.png',
        'results/formation_error.png': 'outputs/experimental_results/formation_error.png',
        'results/success_rate.png': 'outputs/experimental_results/success_rate.png',
        'results/comprehensive_comparison.png': 'outputs/experimental_results/comprehensive_comparison.png',
        'results/training_results.json': 'outputs/experimental_results/training_results.json',
        'results/evaluation_results.json': 'outputs/experimental_results/evaluation_results.json',
        'results/statistics.json': 'outputs/experimental_results/statistics.json',
        'results/experimental_report.txt': 'outputs/experimental_results/experimental_report.txt',
        'results/novelty_report.txt': 'outputs/experimental_results/novelty_report.txt',
        'results/CONSOLIDATED_RESULTS.txt': 'outputs/CONSOLIDATED_RESULTS.txt',
        'novelty_analysis.json': 'outputs/experimental_results/novelty_analysis.json',
    }
    
    for src, dst in result_mappings.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
    
    print("✅ Output files organized")

def organize_documentation():
    """Organize documentation files."""
    
    doc_files = {
        'README.md': 'README.md',  # Keep at root
        'PUBLICATION_README.md': 'docs/PUBLICATION_README.md',
        'SUMMARY.md': 'docs/SUMMARY.md',
        'RESULTS_SUMMARY.md': 'docs/RESULTS_SUMMARY.md',
    }
    
    for src, dst in doc_files.items():
        if os.path.exists(src):
            if src != dst:  # Don't copy if same location
                shutil.copy2(src, dst)
                print(f"  Copied {src} -> {dst}")
    
    print("✅ Documentation organized")

def create_init_files():
    """Create __init__.py files for Python packages."""
    
    init_dirs = [
        'src',
        'src/models',
        'src/environments',
        'src/baselines',
        'src/experiments',
        'src/analysis',
        'src/visualization',
    ]
    
    for d in init_dirs:
        init_file = Path(d) / '__init__.py'
        init_file.write_text('"""QEGAN Package"""\n')
    
    print("✅ Package __init__.py files created")

def create_placeholder_outputs():
    """Create placeholder README files in output directories."""
    
    placeholders = {
        'outputs/architecture_diagrams/README.md': 
            '# Architecture Diagrams\n\nRun `python src/visualization/generate_diagrams.py` to generate architecture diagrams.\n\nExpected outputs:\n- qegan_architecture.png\n- quantum_circuits.png\n- architecture_comparison.png\n- data_flow_diagram.png\n',
        
        'outputs/ablation_study/README.md':
            '# Ablation Study Results\n\nRun `python src/analysis/ablation_study.py` to generate ablation study results.\n\nExpected outputs:\n- ablation_formation_error.png\n- ablation_success_rate.png\n- ablation_component_analysis.png\n- ablation_multi_metric.png\n- ablation_relative_performance.png\n- ablation_results.json\n- ablation_report.txt\n',
        
        'outputs/animations/README.md':
            '# Animations\n\nRun `python src/visualization/generate_animations.py` to generate animations.\n\nExpected outputs:\n- training_dynamics.gif\n- robot_formation.gif\n- quantum_evolution.gif\n',
        
        'outputs/benchmark_results/README.md':
            '# Benchmark Results\n\nRun `python src/analysis/benchmark_comparison.py` to generate benchmark comparison results.\n\nExpected outputs:\n- benchmark_comparison_*.png\n- benchmark_statistics.json\n- benchmark_latex_table.txt\n',
        
        'outputs/visualizations/README.md':
            '# General Visualizations\n\nAdditional visualizations and plots generated during experiments.\n',
    }
    
    for path, content in placeholders.items():
        Path(path).write_text(content)
    
    print("✅ Placeholder README files created")

def main():
    """Main reorganization function."""
    print("=" * 80)
    print("QEGAN Repository Reorganization")
    print("=" * 80)
    print()
    
    print("Step 1: Creating directory structure...")
    create_directory_structure()
    print()
    
    print("Step 2: Organizing source files...")
    organize_source_files()
    print()
    
    print("Step 3: Organizing output files...")
    organize_output_files()
    print()
    
    print("Step 4: Organizing documentation...")
    organize_documentation()
    print()
    
    print("Step 5: Creating __init__.py files...")
    create_init_files()
    print()
    
    print("Step 6: Creating placeholder README files...")
    create_placeholder_outputs()
    print()
    
    print("=" * 80)
    print("✅ REORGANIZATION COMPLETE!")
    print("=" * 80)
    print()
    print("New Repository Structure:")
    print("  src/              - Source code organized by functionality")
    print("  outputs/          - All generated outputs")
    print("  docs/             - Documentation")
    print("  experiments/      - Experiment scripts and configs")
    print()
    print("Next steps:")
    print("  1. Review the new structure")
    print("  2. Run experiments: python experiments/scripts/run_comprehensive_experiments.py")
    print("  3. Generate visualizations: python src/visualization/generate_all_visuals.py")
    print("  4. Check outputs/ directory for all results")

if __name__ == '__main__':
    main()
