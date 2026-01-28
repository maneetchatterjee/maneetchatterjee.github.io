"""
Master script to generate all visualizations, diagrams, ablations, and animations.

Runs all visualization scripts in proper order:
1. Architecture diagrams
2. Ablation study
3. Animations
4. Consolidated results
"""

import os
import sys


def run_module(module_name, description):
    """Run a Python module and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    try:
        if module_name == 'diagrams':
            from generate_diagrams import generate_all_diagrams
            generate_all_diagrams()
        elif module_name == 'ablation':
            from ablation_study import run_ablation_study
            run_ablation_study()
        elif module_name == 'animations':
            from generate_animations import generate_all_animations
            generate_all_animations()
        elif module_name == 'consolidated':
            from generate_consolidated_results import save_consolidated_results
            save_consolidated_results()
        
        print(f"✓ {description} completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in {description}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all visualization generation scripts."""
    print("\n" + "="*80)
    print("QEGAN: COMPREHENSIVE VISUALIZATION GENERATION")
    print("="*80)
    print("\nThis script will generate:")
    print("  • Architecture diagrams (4 files)")
    print("  • Ablation study plots (5 files)")
    print("  • Animations (3 files)")
    print("  • Consolidated results document")
    print("\nTotal output: 12+ files")
    print("="*80)
    
    # Create output directories
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/ablation', exist_ok=True)
    os.makedirs('results/animations', exist_ok=True)
    
    # Run all modules
    modules = [
        ('diagrams', 'Architecture Diagrams Generation'),
        ('ablation', 'Ablation Study Analysis'),
        ('animations', 'Animation Generation'),
        ('consolidated', 'Consolidated Results Document'),
    ]
    
    results = {}
    for module_name, description in modules:
        results[module_name] = run_module(module_name, description)
    
    # Summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nResults: {successful}/{total} modules completed successfully")
    
    if successful == total:
        print("\n✓ All visualizations generated successfully!")
        print("\nOutput locations:")
        print("  • Architecture diagrams: results/visualizations/")
        print("  • Ablation study: results/ablation/")
        print("  • Animations: results/animations/")
        print("  • Consolidated document: results/CONSOLIDATED_RESULTS.txt")
    else:
        print("\n⚠ Some modules failed. Check error messages above.")
    
    print("="*80)
    
    # List all generated files
    print("\nGenerated Files:")
    print("-"*80)
    for root, dirs, files in os.walk('results'):
        for file in files:
            if file.endswith(('.png', '.gif', '.txt', '.json')):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                print(f"  {filepath:60s} ({size:>8,} bytes)")
    
    print("="*80)


if __name__ == '__main__':
    main()
