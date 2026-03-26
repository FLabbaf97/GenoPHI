#!/usr/bin/env python3
"""
Generate baseline outputs for GenoPHI test suite.

This script runs the GenoPHI workflows with test data to generate baseline
outputs that can be used for comparison in regression tests.

Usage:
    python scripts/generate_test_baselines.py --output-dir baselines/

The script will:
1. Create test datasets (subsets of full test data)
2. Run protein family workflow with standard parameters
3. Save model performance metrics as baseline files
4. Generate summary report of baseline metrics
"""

import argparse
import shutil
import pandas as pd
from pathlib import Path
import logging
import sys

# Add parent directory to path to import genophi
sys.path.insert(0, str(Path(__file__).parent.parent))

from genophi.workflows.protein_family_workflow import run_protein_family_workflow


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_subset_dataset(source_dir, output_dir, n_samples, suffix='faa'):
    """
    Create a subset of FASTA files for testing.

    Args:
        source_dir: Path to source directory with FASTA files
        output_dir: Path to output directory for subset
        n_samples: Number of samples to include
        suffix: File extension for FASTA files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(Path(source_dir).glob(f'*.{suffix}'))[:n_samples]

    logger.info(f"Creating subset with {len(source_files)} files")

    for source_file in source_files:
        dest_file = output_dir / source_file.name
        shutil.copy2(source_file, dest_file)

    return output_dir


def create_subset_interaction_matrix(source_matrix, output_file, strain_names, phage_names=None):
    """
    Create a subset interaction matrix matching the test dataset.

    Args:
        source_matrix: Path to source interaction matrix
        output_file: Path to output subset matrix
        strain_names: List of strain names to include
        phage_names: List of phage names to include (optional)
    """
    df = pd.read_csv(source_matrix)

    # Filter to matching strains
    if 'strain' in df.columns:
        df = df[df['strain'].isin(strain_names)]

    if phage_names is not None and 'phage' in df.columns:
        df = df[df['phage'].isin(phage_names)]

    df.to_csv(output_file, index=False)
    logger.info(f"Created subset interaction matrix with {len(df)} rows")

    return output_file


def generate_quick_baseline(test_data_dir, baseline_dir):
    """
    Generate quick baseline (5 strains, reduced iterations).

    Args:
        test_data_dir: Path to test data directory
        baseline_dir: Path to baseline output directory
    """
    logger.info("=" * 80)
    logger.info("Generating QUICK baseline (5 strains, 5 iterations)")
    logger.info("=" * 80)

    # Setup directories
    test_data_dir = Path(test_data_dir)
    baseline_dir = Path(baseline_dir)
    quick_dir = baseline_dir / 'quick'
    quick_dir.mkdir(parents=True, exist_ok=True)

    # Create subset datasets
    subset_dir = quick_dir / 'test_data'
    strain_subset = create_subset_dataset(
        test_data_dir / 'strain_AAs',
        subset_dir / 'strain_AAs',
        n_samples=5
    )

    # Get strain names for interaction matrix filtering
    strain_files = list(strain_subset.glob('*.faa'))
    strain_names = [f.stem for f in strain_files]

    # Create subset interaction matrix
    interaction_subset = create_subset_interaction_matrix(
        test_data_dir / 'ecoli_test_interaction_matrix.csv',
        subset_dir / 'interaction_matrix.csv',
        strain_names=strain_names
    )

    # Run workflow
    output_dir = quick_dir / 'output'

    logger.info("Running protein family workflow (this may take 15-20 minutes)...")

    try:
        run_protein_family_workflow(
            input_path_strain=str(strain_subset),
            input_path_phage=None,
            phenotype_matrix=str(interaction_subset),
            output_dir=str(output_dir),
            tmp_dir=str(quick_dir / 'tmp'),
            phenotype_column='interaction',
            sample_column='strain',
            task_type='classification',
            num_runs_fs=5,
            num_runs_modeling=5,
            num_features=20,
            min_features=2,
            min_seq_id=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=4,
            clear_tmp=True
        )

        # Copy metrics to baseline location
        metrics_source = output_dir / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
        metrics_dest = baseline_dir / 'baseline_metrics_quick.csv'

        if metrics_source.exists():
            shutil.copy2(metrics_source, metrics_dest)
            logger.info(f"Baseline metrics saved to: {metrics_dest}")

            # Print summary
            df = pd.read_csv(metrics_dest)
            logger.info("\nBaseline metrics summary (top 5 runs):")
            logger.info(df.head().to_string())

            # Clean up intermediate files - keep only the baseline CSV
            logger.info("Cleaning up intermediate files...")
            if quick_dir.exists():
                # Remove everything except the baseline CSV we just saved
                for item in quick_dir.rglob('*'):
                    if item != metrics_dest:
                        if item.is_symlink():
                            item.unlink()  # Remove symlinks
                        elif item.is_file():
                            item.unlink()  # Remove regular files
                # Remove empty directories
                for item in sorted(quick_dir.rglob('*'), reverse=True):
                    if item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
            logger.info("Cleanup complete")

            return metrics_dest
        else:
            logger.error(f"Metrics file not found: {metrics_source}")
            return None

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


def generate_standard_baseline(test_data_dir, baseline_dir):
    """
    Generate standard baseline (10 strains, standard iterations).

    Args:
        test_data_dir: Path to test data directory
        baseline_dir: Path to baseline output directory
    """
    logger.info("=" * 80)
    logger.info("Generating STANDARD baseline (10 strains, 10 iterations)")
    logger.info("=" * 80)

    # Setup directories
    test_data_dir = Path(test_data_dir)
    baseline_dir = Path(baseline_dir)
    standard_dir = baseline_dir / 'standard'
    standard_dir.mkdir(parents=True, exist_ok=True)

    # Create subset datasets
    subset_dir = standard_dir / 'test_data'
    strain_subset = create_subset_dataset(
        test_data_dir / 'strain_AAs',
        subset_dir / 'strain_AAs',
        n_samples=10
    )

    # Get strain names
    strain_files = list(strain_subset.glob('*.faa'))
    strain_names = [f.stem for f in strain_files]

    # Create subset interaction matrix
    interaction_subset = create_subset_interaction_matrix(
        test_data_dir / 'ecoli_test_interaction_matrix.csv',
        subset_dir / 'interaction_matrix.csv',
        strain_names=strain_names
    )

    # Run workflow
    output_dir = standard_dir / 'output'

    logger.info("Running protein family workflow (this may take 30-45 minutes)...")

    try:
        run_protein_family_workflow(
            input_path_strain=str(strain_subset),
            input_path_phage=None,
            phenotype_matrix=str(interaction_subset),
            output_dir=str(output_dir),
            tmp_dir=str(standard_dir / 'tmp'),
            phenotype_column='interaction',
            sample_column='strain',
            task_type='classification',
            num_runs_fs=10,
            num_runs_modeling=10,
            num_features=30,
            min_features=5,
            min_seq_id=0.4,
            coverage=0.8,
            sensitivity=7.5,
            threads=4,
            clear_tmp=True
        )

        # Copy metrics to baseline location
        metrics_source = output_dir / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
        metrics_dest = baseline_dir / 'baseline_metrics_standard.csv'

        if metrics_source.exists():
            shutil.copy2(metrics_source, metrics_dest)
            logger.info(f"Baseline metrics saved to: {metrics_dest}")

            # Print summary
            df = pd.read_csv(metrics_dest)
            logger.info("\nBaseline metrics summary (top 5 runs):")
            logger.info(df.head().to_string())

            # Clean up intermediate files - keep only the baseline CSV
            logger.info("Cleaning up intermediate files...")
            if standard_dir.exists():
                # Remove everything except the baseline CSV we just saved
                for item in standard_dir.rglob('*'):
                    if item != metrics_dest:
                        if item.is_symlink():
                            item.unlink()  # Remove symlinks
                        elif item.is_file():
                            item.unlink()  # Remove regular files
                # Remove empty directories
                for item in sorted(standard_dir.rglob('*'), reverse=True):
                    if item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
            logger.info("Cleanup complete")

            return metrics_dest
        else:
            logger.error(f"Metrics file not found: {metrics_source}")
            return None

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate baseline outputs for GenoPHI test suite'
    )
    parser.add_argument(
        '--test-data-dir',
        type=str,
        default='/usr2/people/anoonan/BRaVE/machine_learning/genophi/data/test_data',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/baselines',
        help='Path to baseline output directory'
    )
    parser.add_argument(
        '--quick-only',
        action='store_true',
        help='Generate only quick baseline (5 strains)'
    )
    parser.add_argument(
        '--standard-only',
        action='store_true',
        help='Generate only standard baseline (10 strains)'
    )

    args = parser.parse_args()

    # Validate test data directory
    test_data_dir = Path(args.test_data_dir)
    if not test_data_dir.exists():
        logger.error(f"Test data directory not found: {test_data_dir}")
        sys.exit(1)

    baseline_dir = Path(args.output_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Generate baselines
    generated = []

    if not args.standard_only:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Quick Baseline")
        logger.info("=" * 80 + "\n")
        try:
            quick_baseline = generate_quick_baseline(test_data_dir, baseline_dir)
            if quick_baseline:
                generated.append(('Quick', quick_baseline))
        except Exception as e:
            logger.error(f"Quick baseline generation failed: {e}")

    if not args.quick_only:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Standard Baseline")
        logger.info("=" * 80 + "\n")
        try:
            standard_baseline = generate_standard_baseline(test_data_dir, baseline_dir)
            if standard_baseline:
                generated.append(('Standard', standard_baseline))
        except Exception as e:
            logger.error(f"Standard baseline generation failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info("=" * 80)

    if generated:
        logger.info("\nGenerated baselines:")
        for name, path in generated:
            logger.info(f"  - {name}: {path}")

        logger.info("\nYou can now run tests with baseline comparison:")
        logger.info("  pytest tests/e2e/ -v")
    else:
        logger.error("\nNo baselines were generated successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
