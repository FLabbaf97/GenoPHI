"""
Integration tests for clustering → feature assignment pipeline.

Tests the integration between MMSeqs2 clustering and feature assignment,
validating that clustering outputs can be correctly converted to feature tables.
"""

import pytest
import pandas as pd
from pathlib import Path
from genophi.mmseqs2_clustering import run_clustering_workflow, run_feature_assignment
from tests.utils.test_helpers import (
    validate_presence_absence_matrix,
    validate_feature_table,
    log_validation_results
)


@pytest.mark.integration
@pytest.mark.requires_mmseqs2
def test_clustering_creates_required_outputs(small_test_dataset, temp_output_dir):
    """
    Test that MMSeqs2 clustering produces all required output files.

    This test validates:
    - Clustering completes without errors
    - presence_absence_matrix.csv is created
    - clusters.tsv is created
    - Matrix has correct structure (binary values, expected dimensions)
    """
    # Setup paths
    strain_output = temp_output_dir / 'strain'
    tmp_dir = temp_output_dir / 'tmp'
    strain_output.mkdir()
    tmp_dir.mkdir()

    # Run clustering
    run_clustering_workflow(
        input_path=str(small_test_dataset['strain_dir']),
        output_dir=str(strain_output),
        tmp_dir=str(tmp_dir),
        min_seq_id=0.6,
        coverage=0.8,
        sensitivity=7.5,
        suffix='faa',
        threads=2,
        strain_list='none',
        strain_column='strain',
        compare=False,
        bootstrapping=False,
        clear_tmp=False
    )

    # Validate outputs exist
    assert (strain_output / 'presence_absence_matrix.csv').exists(), \
        "presence_absence_matrix.csv not created"
    assert (strain_output / 'clusters.tsv').exists(), \
        "clusters.tsv not created"

    # Validate matrix structure
    matrix_path = strain_output / 'presence_absence_matrix.csv'
    results = validate_presence_absence_matrix(
        matrix_path,
        expected_genomes=small_test_dataset['strain_count']
    )

    assert results['valid'], f"Matrix validation failed: {results['errors']}"

    # Additional checks on matrix content
    matrix = pd.read_csv(matrix_path)
    assert len(matrix) == small_test_dataset['strain_count'], \
        f"Expected {small_test_dataset['strain_count']} strains, got {len(matrix)}"

    feature_cols = [c for c in matrix.columns if c != 'Genome']
    assert len(feature_cols) > 0, "No feature columns found"

    # Check that all values are binary
    assert all(matrix[col].isin([0, 1]).all() for col in feature_cols), \
        "Non-binary values found in feature columns"


@pytest.mark.integration
@pytest.mark.requires_mmseqs2
def test_feature_assignment_from_clustering(small_test_dataset, temp_output_dir):
    """
    Test feature assignment produces correct feature table from clustering results.

    This test validates:
    - Feature assignment completes without errors
    - feature_table.csv is created
    - Feature table has expected structure
    - Feature columns match clusters from clustering
    """
    # First run clustering (or reuse if previous test ran)
    strain_output = temp_output_dir / 'strain'
    tmp_dir = temp_output_dir / 'tmp'
    strain_output.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    # Run clustering
    run_clustering_workflow(
        input_path=str(small_test_dataset['strain_dir']),
        output_dir=str(strain_output),
        tmp_dir=str(tmp_dir),
        min_seq_id=0.6,
        coverage=0.8,
        sensitivity=7.5,
        suffix='faa',
        threads=2,
        strain_list='none',
        strain_column='strain',
        compare=False,
        bootstrapping=False,
        clear_tmp=False
    )

    # Run feature assignment
    features_output = strain_output / 'features'
    features_output.mkdir(exist_ok=True)

    run_feature_assignment(
        input_file=str(strain_output / 'presence_absence_matrix.csv'),
        output_dir=str(features_output),
        source='strain',
        select='none',
        select_column='strain',
        input_type='file',
        max_ram=8,
        threads=2
    )

    # Validate feature table exists
    feature_table_path = features_output / 'feature_table.csv'
    assert feature_table_path.exists(), "feature_table.csv not created"

    # Validate feature table structure
    results = validate_feature_table(
        feature_table_path,
        min_features=1,
        expected_samples=small_test_dataset['strain_count']
    )

    assert results['valid'], f"Feature table validation failed: {results['errors']}"
    assert results['feature_count'] > 0, "No features in feature table"

    # Check that feature table has expected columns
    feature_table = pd.read_csv(feature_table_path)
    assert 'strain' in feature_table.columns or 'Genome' in feature_table.columns, \
        "Missing sample column in feature table"

    # Check feature column naming (should be sc_* for strain)
    feature_cols = [c for c in feature_table.columns
                   if c not in ['strain', 'Genome', 'phage', 'interaction']]
    assert all(c.startswith('sc_') for c in feature_cols), \
        "Feature columns don't follow expected naming convention (sc_*)"

    # Validate no duplicate feature columns
    assert len(feature_cols) == len(set(feature_cols)), \
        "Duplicate feature columns found"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_mmseqs2
def test_clustering_with_phage_data(small_test_dataset, temp_output_dir):
    """
    Test clustering works correctly with both strain and phage data.

    This test validates:
    - Both strain and phage clustering complete
    - Separate outputs created for each
    - Feature prefixes are correct (sc_ for strain, pc_ for phage)
    """
    strain_output = temp_output_dir / 'strain'
    phage_output = temp_output_dir / 'phage'
    tmp_dir_strain = temp_output_dir / 'tmp_strain'
    tmp_dir_phage = temp_output_dir / 'tmp_phage'

    for output_dir in [strain_output, phage_output, tmp_dir_strain, tmp_dir_phage]:
        output_dir.mkdir(exist_ok=True)

    # Run strain clustering
    run_clustering_workflow(
        input_path=str(small_test_dataset['strain_dir']),
        output_dir=str(strain_output),
        tmp_dir=str(tmp_dir_strain),
        min_seq_id=0.6,
        coverage=0.8,
        sensitivity=7.5,
        suffix='faa',
        threads=2,
        strain_list='none',
        strain_column='strain',
        compare=False,
        bootstrapping=False,
        clear_tmp=True
    )

    # Run phage clustering
    run_clustering_workflow(
        input_path=str(small_test_dataset['phage_dir']),
        output_dir=str(phage_output),
        tmp_dir=str(tmp_dir_phage),
        min_seq_id=0.6,
        coverage=0.8,
        sensitivity=7.5,
        suffix='faa',
        threads=2,
        strain_list='none',
        strain_column='phage',
        compare=False,
        bootstrapping=False,
        clear_tmp=True
    )

    # Validate both outputs
    for output_dir, name in [(strain_output, 'strain'), (phage_output, 'phage')]:
        assert (output_dir / 'presence_absence_matrix.csv').exists(), \
            f"{name} presence_absence_matrix.csv not created"

        # Run feature assignment
        features_dir = output_dir / 'features'
        features_dir.mkdir(exist_ok=True)

        run_feature_assignment(
            input_file=str(output_dir / 'presence_absence_matrix.csv'),
            output_dir=str(features_dir),
            source=name,
            select='none',
            select_column=name,
            input_type='file',
            max_ram=8,
            threads=2
        )

        # Check feature naming
        feature_table = pd.read_csv(features_dir / 'feature_table.csv')
        feature_cols = [c for c in feature_table.columns
                       if c not in [name, 'Genome', 'strain', 'phage', 'interaction']]

        expected_prefix = 'sc_' if name == 'strain' else 'pc_'
        assert all(c.startswith(expected_prefix) for c in feature_cols), \
            f"{name} features don't have correct prefix ({expected_prefix})"
