"""
Pytest configuration and shared fixtures for GenoPHI tests.

This module provides fixtures for test data paths, temporary directories,
and common test utilities used across integration and E2E tests.
"""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path


@pytest.fixture(scope='session')
def test_data_root():
    """Root directory for test data."""
    root = Path('/usr2/people/anoonan/BRaVE/machine_learning/genophi/data/test_data')
    if not root.exists():
        pytest.skip(f"Test data directory not found: {root}")
    return root


@pytest.fixture(scope='session')
def test_data_strains(test_data_root):
    """Path to strain FASTA directory."""
    strain_dir = test_data_root / 'strain_AAs'
    if not strain_dir.exists():
        pytest.skip(f"Strain data directory not found: {strain_dir}")
    return strain_dir


@pytest.fixture(scope='session')
def test_data_phages(test_data_root):
    """Path to phage FASTA directory."""
    phage_dir = test_data_root / 'phage_AAs'
    if not phage_dir.exists():
        pytest.skip(f"Phage data directory not found: {phage_dir}")
    return phage_dir


@pytest.fixture(scope='session')
def test_interaction_matrix(test_data_root):
    """Path to interaction matrix CSV."""
    matrix_path = test_data_root / 'ecoli_test_interaction_matrix.csv'
    if not matrix_path.exists():
        pytest.skip(f"Interaction matrix not found: {matrix_path}")
    return matrix_path


@pytest.fixture(scope='session')
def baseline_dir(test_data_root):
    """Directory containing baseline outputs for comparison."""
    baseline = test_data_root / 'baseline_outputs'
    # Don't skip if baseline doesn't exist - tests will handle gracefully
    return baseline


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for test runs."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    yield output_dir
    # Cleanup handled automatically by tmp_path


@pytest.fixture(scope='function')
def small_test_dataset(test_data_strains, test_data_phages, test_interaction_matrix, tmp_path):
    """
    Create small subset of test data for quick tests.

    Creates:
    - 5 strain FASTA files
    - 5 phage FASTA files
    - Subset interaction matrix matching these samples
    """
    # Create directories
    small_strain_dir = tmp_path / 'strain_AAs_small'
    small_phage_dir = tmp_path / 'phage_AAs_small'
    small_strain_dir.mkdir()
    small_phage_dir.mkdir()

    # Copy first 5 strain and phage files
    strain_files = sorted(test_data_strains.glob('*.faa'))[:5]
    phage_files = sorted(test_data_phages.glob('*.faa'))[:5]

    for f in strain_files:
        shutil.copy(f, small_strain_dir / f.name)
    for f in phage_files:
        shutil.copy(f, small_phage_dir / f.name)

    # Create subset interaction matrix
    full_matrix = pd.read_csv(test_interaction_matrix)
    strain_names = [f.stem for f in strain_files]
    phage_names = [f.stem for f in phage_files]

    small_matrix = full_matrix[
        full_matrix['strain'].isin(strain_names) &
        full_matrix['phage'].isin(phage_names)
    ]

    matrix_path = tmp_path / 'interaction_matrix_small.csv'
    small_matrix.to_csv(matrix_path, index=False)

    return {
        'strain_dir': small_strain_dir,
        'phage_dir': small_phage_dir,
        'matrix': matrix_path,
        'output_dir': tmp_path / 'output',
        'strain_count': len(strain_files),
        'phage_count': len(phage_files),
        'interaction_count': len(small_matrix)
    }


@pytest.fixture(scope='function')
def medium_test_dataset(test_data_strains, test_data_phages, test_interaction_matrix, tmp_path):
    """
    Create medium-sized subset of test data for standard E2E tests.

    Creates:
    - 10 strain FASTA files
    - 10 phage FASTA files
    - Subset interaction matrix matching these samples
    """
    # Create directories
    med_strain_dir = tmp_path / 'strain_AAs_medium'
    med_phage_dir = tmp_path / 'phage_AAs_medium'
    med_strain_dir.mkdir()
    med_phage_dir.mkdir()

    # Copy first 10 strain and phage files
    strain_files = sorted(test_data_strains.glob('*.faa'))[:10]
    phage_files = sorted(test_data_phages.glob('*.faa'))[:10]

    for f in strain_files:
        shutil.copy(f, med_strain_dir / f.name)
    for f in phage_files:
        shutil.copy(f, med_phage_dir / f.name)

    # Create subset interaction matrix
    full_matrix = pd.read_csv(test_interaction_matrix)
    strain_names = [f.stem for f in strain_files]
    phage_names = [f.stem for f in phage_files]

    med_matrix = full_matrix[
        full_matrix['strain'].isin(strain_names) &
        full_matrix['phage'].isin(phage_names)
    ]

    matrix_path = tmp_path / 'interaction_matrix_medium.csv'
    med_matrix.to_csv(matrix_path, index=False)

    return {
        'strain_dir': med_strain_dir,
        'phage_dir': med_phage_dir,
        'matrix': matrix_path,
        'output_dir': tmp_path / 'output',
        'strain_count': len(strain_files),
        'phage_count': len(phage_files),
        'interaction_count': len(med_matrix)
    }


@pytest.fixture
def baseline_metrics_standard(baseline_dir):
    """Load standard baseline metrics if available."""
    metrics_path = baseline_dir / 'standard' / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


@pytest.fixture
def baseline_metrics_quick(baseline_dir):
    """Load quick baseline metrics if available."""
    metrics_path = baseline_dir / 'quick' / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None
