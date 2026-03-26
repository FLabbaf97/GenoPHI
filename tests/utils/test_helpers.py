"""
Test helper functions and validation utilities.

This module provides utilities for validating GenoPHI workflow outputs,
checking file structures, and comparing results against baselines.
"""

import os
import pandas as pd
import logging
from pathlib import Path


def validate_output_structure(output_dir, workflow_type='protein_family'):
    """
    Validate that expected output directory structure exists.

    Args:
        output_dir (Path or str): Output directory to validate
        workflow_type (str): Type of workflow ('protein_family', 'kmer', 'full')

    Returns:
        dict: Validation results with keys: valid, errors
    """
    output_dir = Path(output_dir)

    required_files = {
        'protein_family': [
            'strain/presence_absence_matrix.csv',
            'strain/clusters.tsv',
            'strain/features/feature_table.csv',
            'feature_selection/filtered_feature_tables',
            'modeling_results/model_performance/model_performance_metrics.csv',
            'workflow_report.txt'
        ],
        'kmer': [
            'strain_combined.faa',
            'strain_proteins.csv',
            'modeling/modeling_results/model_performance/model_performance_metrics.csv',
            'kmer_workflow.log'
        ],
        'full': [
            'strain/presence_absence_matrix.csv',
            'modeling_results/model_performance/model_performance_metrics.csv',
            'kmer_modeling',
            'workflow_section_metrics.csv'
        ]
    }

    missing_paths = []
    for file_path in required_files.get(workflow_type, []):
        full_path = output_dir / file_path
        if not full_path.exists():
            missing_paths.append(str(file_path))

    return {
        'valid': len(missing_paths) == 0,
        'errors': missing_paths
    }


def validate_presence_absence_matrix(csv_path, expected_genomes=None):
    """
    Validate presence/absence matrix structure.

    Args:
        csv_path (Path or str): Path to presence/absence matrix CSV
        expected_genomes (int, optional): Expected number of genomes

    Returns:
        dict: Validation results with keys: valid, errors, warnings
    """
    results = {'valid': True, 'errors': [], 'warnings': []}

    try:
        df = pd.read_csv(csv_path)

        # Check for Genome column
        if 'Genome' not in df.columns:
            results['errors'].append("Missing 'Genome' column")
            results['valid'] = False

        # Check binary values in feature columns
        feature_cols = [c for c in df.columns if c != 'Genome']
        for col in feature_cols:
            if not df[col].isin([0, 1]).all():
                results['errors'].append(f"Non-binary values in column: {col}")
                results['valid'] = False
                break  # Don't report all columns, just first error

        # Check genome count
        if expected_genomes and len(df) != expected_genomes:
            results['warnings'].append(
                f"Expected {expected_genomes} genomes, got {len(df)}"
            )

        # Check for empty matrix
        if len(feature_cols) == 0:
            results['errors'].append("No feature columns found")
            results['valid'] = False

    except Exception as e:
        results['errors'].append(f"Failed to read CSV: {str(e)}")
        results['valid'] = False

    return results


def validate_model_performance_metrics(csv_path):
    """
    Validate model performance metrics CSV structure and value ranges.

    Args:
        csv_path (Path or str): Path to model performance metrics CSV

    Returns:
        dict: Validation results with keys: valid, errors, warnings
    """
    results = {'valid': True, 'errors': [], 'warnings': []}

    try:
        df = pd.read_csv(csv_path)

        # Check required columns
        required_columns = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'cut_off']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['valid'] = False
            return results

        # Validate metric ranges
        if not df['AUC'].between(0, 1).all():
            results['errors'].append("AUC values out of range [0, 1]")
            results['valid'] = False

        if not df['Accuracy'].between(0, 1).all():
            results['errors'].append("Accuracy values out of range [0, 1]")
            results['valid'] = False

        if not df['MCC'].between(-1, 1).all():
            results['errors'].append("MCC values out of range [-1, 1]")
            results['valid'] = False

        # Check if sorted (should be sorted by performance)
        if not (df['MCC'].is_monotonic_decreasing or df['AUC'].is_monotonic_decreasing):
            results['warnings'].append("Metrics not sorted by performance")

        # Check for NaN values
        if df.isnull().any().any():
            results['warnings'].append("NaN values found in metrics")

    except Exception as e:
        results['errors'].append(f"Failed to read CSV: {str(e)}")
        results['valid'] = False

    return results


def validate_feature_table(csv_path, min_features=1, max_features=None,
                           expected_samples=None):
    """
    Validate feature table structure.

    Args:
        csv_path (Path or str): Path to feature table CSV
        min_features (int): Minimum number of features expected
        max_features (int, optional): Maximum number of features expected
        expected_samples (int, optional): Expected number of samples

    Returns:
        dict: Validation results
    """
    results = {'valid': True, 'errors': [], 'warnings': [], 'feature_count': 0, 'sample_count': 0}

    try:
        df = pd.read_csv(csv_path)

        # Check for sample ID column (strain, Genome, or phage)
        sample_col_options = ['strain', 'Genome', 'phage']
        sample_col = None
        for col in sample_col_options:
            if col in df.columns:
                sample_col = col
                break

        if sample_col is None:
            results['errors'].append("Missing sample ID column (expected: strain, Genome, or phage)")
            results['valid'] = False
            return results

        # Count feature columns (exclude metadata columns)
        metadata_cols = ['strain', 'phage', 'interaction', 'Genome']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        results['feature_count'] = len(feature_cols)
        results['sample_count'] = len(df)

        # Check feature count
        if len(feature_cols) < min_features:
            results['errors'].append(
                f"Too few features: {len(feature_cols)} < {min_features}"
            )
            results['valid'] = False

        if max_features and len(feature_cols) > max_features:
            results['warnings'].append(
                f"More features than expected: {len(feature_cols)} > {max_features}"
            )

        # Check sample count
        if expected_samples and len(df) != expected_samples:
            results['warnings'].append(
                f"Expected {expected_samples} samples, got {len(df)}"
            )

        # Check for NaN values
        if df.isnull().any().any():
            results['errors'].append("NaN values found in feature table")
            results['valid'] = False

    except Exception as e:
        results['errors'].append(f"Failed to read CSV: {str(e)}")
        results['valid'] = False

    return results


def compare_to_baseline(results_csv, baseline_csv, tolerance=0.05):
    """
    Compare model performance metrics to baseline.

    Args:
        results_csv (Path or str): Path to results metrics CSV
        baseline_csv (Path or str): Path to baseline metrics CSV
        tolerance (float): Acceptable deviation from baseline (default: 0.05 = 5%)

    Returns:
        dict: Comparison results with pass/fail for each metric
    """
    try:
        results = pd.read_csv(results_csv)
        baseline = pd.read_csv(baseline_csv)

        # Compare metrics for best performing cutoff (first row)
        result_best = results.iloc[0]
        baseline_best = baseline.iloc[0]

        comparisons = {}
        for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']:
            if metric not in result_best or metric not in baseline_best:
                comparisons[metric] = {
                    'result': None,
                    'baseline': None,
                    'diff': None,
                    'pass': False,
                    'error': f'Missing {metric} in results or baseline'
                }
                continue

            result_val = result_best[metric]
            baseline_val = baseline_best[metric]
            diff = abs(result_val - baseline_val)
            within_tolerance = diff <= tolerance

            comparisons[metric] = {
                'result': result_val,
                'baseline': baseline_val,
                'diff': diff,
                'tolerance': tolerance,
                'pass': within_tolerance
            }

        # Overall pass if all metrics pass
        overall_pass = all(c.get('pass', False) for c in comparisons.values())

        return {
            'overall_pass': overall_pass,
            'metrics': comparisons,
            'tolerance': tolerance
        }

    except Exception as e:
        return {
            'overall_pass': False,
            'error': str(e),
            'metrics': {}
        }


def validate_fasta_file(fasta_path, min_sequences=1):
    """
    Validate FASTA file structure.

    Args:
        fasta_path (Path or str): Path to FASTA file
        min_sequences (int): Minimum number of sequences expected

    Returns:
        dict: Validation results
    """
    results = {'valid': True, 'errors': [], 'warnings': [], 'sequence_count': 0}

    try:
        from Bio import SeqIO

        if not os.path.exists(fasta_path):
            results['errors'].append(f"FASTA file not found: {fasta_path}")
            results['valid'] = False
            return results

        sequences = list(SeqIO.parse(fasta_path, 'fasta'))
        results['sequence_count'] = len(sequences)

        if len(sequences) < min_sequences:
            results['errors'].append(
                f"Too few sequences: {len(sequences)} < {min_sequences}"
            )
            results['valid'] = False

        # Check for duplicate IDs
        ids = [seq.id for seq in sequences]
        if len(ids) != len(set(ids)):
            results['errors'].append("Duplicate sequence IDs found")
            results['valid'] = False

        # Check for empty sequences
        if any(len(seq.seq) == 0 for seq in sequences):
            results['errors'].append("Empty sequences found")
            results['valid'] = False

    except ImportError:
        results['errors'].append("BioPython not installed, cannot validate FASTA")
        results['valid'] = False
    except Exception as e:
        results['errors'].append(f"Failed to parse FASTA: {str(e)}")
        results['valid'] = False

    return results


def log_validation_results(name, results, logger=None):
    """
    Log validation results in a readable format.

    Args:
        name (str): Name/description of what is being validated
        results (dict): Validation results dictionary
        logger (logging.Logger, optional): Logger to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if results.get('valid', False):
        logger.info(f"✓ {name}: Validation passed")
    else:
        logger.error(f"✗ {name}: Validation failed")

    for error in results.get('errors', []):
        logger.error(f"  Error: {error}")

    for warning in results.get('warnings', []):
        logger.warning(f"  Warning: {warning}")
