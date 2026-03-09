"""
Smoke tests for GenoPHI package.

Quick tests to verify package installation and basic functionality.
These tests should run in seconds and catch major installation issues.
"""

import pytest
import subprocess
import sys


@pytest.mark.smoke
def test_package_import():
    """Test that the main package can be imported."""
    import genophi
    assert genophi is not None


@pytest.mark.smoke
def test_workflow_imports():
    """Test that workflow modules can be imported."""
    from genophi.workflows import protein_family_workflow
    from genophi.workflows import kmer_full_workflow
    from genophi.workflows import full_workflow

    assert hasattr(protein_family_workflow, 'run_protein_family_workflow')
    assert hasattr(kmer_full_workflow, 'run_kmer_workflow')
    assert hasattr(full_workflow, 'run_full_workflow')


@pytest.mark.smoke
def test_core_module_imports():
    """Test that core processing modules can be imported."""
    from genophi import mmseqs2_clustering
    from genophi import feature_selection
    from genophi import select_feature_modeling
    from genophi import feature_annotations
    from genophi import utils

    assert hasattr(mmseqs2_clustering, 'run_clustering_workflow')
    assert hasattr(feature_selection, 'run_feature_selection_iterations')
    assert hasattr(select_feature_modeling, 'model_testing_select_MCC')
    assert hasattr(feature_annotations, 'get_predictive_features')
    assert hasattr(utils, 'validate_phenotype_task_type')


@pytest.mark.smoke
def test_cli_module_import():
    """Test that CLI module can be imported."""
    from genophi import cli
    assert hasattr(cli, 'main')


@pytest.mark.smoke
def test_cli_executable():
    """Test that genophi CLI is available as executable."""
    result = subprocess.run(
        [sys.executable, '-m', 'genophi.cli', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert 'genophi' in result.stdout.lower() or 'usage' in result.stdout.lower()


@pytest.mark.smoke
def test_required_dependencies():
    """Test that all required dependencies are importable."""
    dependencies = [
        'pandas',
        'numpy',
        'sklearn',
        'Bio',  # biopython
        'catboost',
        'matplotlib',
        'seaborn',
        'tqdm',
        'joblib',
        'plotnine',
        'shap',
        'psutil',
        'scipy',
        'hdbscan',
    ]

    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    assert not missing, f"Missing required dependencies: {missing}"


@pytest.mark.smoke
def test_validation_functions():
    """Test that validation utilities work with simple data."""
    from genophi.utils import validate_phenotype_task_type
    import pandas as pd

    # Test classification validation (should pass)
    y_classification = pd.Series([0, 1, 0, 1, 1, 0])
    validate_phenotype_task_type(y_classification, 'classification')

    # Test regression validation (should pass)
    y_regression = pd.Series([1.5, 2.3, 4.7, 3.2, 5.1])
    validate_phenotype_task_type(y_regression, 'regression')

    # Test that invalid task_type raises error
    with pytest.raises(ValueError, match="Invalid task_type"):
        validate_phenotype_task_type(y_classification, 'invalid')


@pytest.mark.smoke
def test_feature_annotation_functions():
    """Test that feature annotation functions work with simple data."""
    from genophi.feature_annotations import get_predictive_features
    import pandas as pd
    import tempfile

    # Create simple test feature table
    df = pd.DataFrame({
        'strain': ['s1', 's2', 's3'],
        'interaction': [0, 1, 0],
        'sc_0': [1, 0, 1],
        'sc_1': [0, 1, 0],
        'pc_0': [1, 1, 0],
    })

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)

    try:
        # Test with strain features only (default)
        strain_features = get_predictive_features(
            temp_path,
            sample_column='strain',
            phenotype_column='interaction',
            feature_type='strain'
        )

        assert 'sc_0' in strain_features
        assert 'sc_1' in strain_features
        assert 'pc_0' not in strain_features  # pc features filtered out when feature_type='strain'

        # Test with all features
        all_features = get_predictive_features(
            temp_path,
            sample_column='strain',
            phenotype_column='interaction',
            feature_type='all'
        )

        assert 'sc_0' in all_features
        assert 'sc_1' in all_features
        assert 'pc_0' in all_features
        assert 'strain' not in all_features
        assert 'interaction' not in all_features
    finally:
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
