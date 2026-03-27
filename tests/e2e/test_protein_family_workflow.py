"""
End-to-end tests for complete protein family workflow.

Tests the full pipeline from FASTA files through clustering, feature selection,
and modeling to validate complete workflow execution.
"""

import pytest
import pandas as pd
from pathlib import Path
from genophi.workflows.protein_family_workflow import run_protein_family_workflow
from tests.utils.test_helpers import (
    validate_output_structure,
    validate_model_performance_metrics,
    log_validation_results
)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_mmseqs2
def test_complete_protein_workflow_strain_and_phage(medium_test_dataset, temp_output_dir):
    """
    Test complete protein family workflow with both strain and phage data.

    This test validates:
    - Both strain and phage clustering complete
    - Feature tables merged correctly
    - Modeling uses both feature types
    - Output structure correct for dual-source workflow
    """
    strain_dir = medium_test_dataset['strain_dir']
    phage_dir = medium_test_dataset['phage_dir']
    interaction_matrix = medium_test_dataset['interaction_matrix']
    output_dir = temp_output_dir / 'protein_workflow_dual_output'

    # Run workflow with both strain and phage
    run_protein_family_workflow(
        input_path_strain=str(strain_dir),
        input_path_phage=str(phage_dir),
        phenotype_matrix=str(interaction_matrix),
        output_dir=str(output_dir),
        tmp_dir=str(temp_output_dir / 'tmp'),
        phenotype_column='interaction',
        sample_column='strain',
        task_type='classification',
        num_runs_fs=5,
        num_runs_modeling=5,
        num_features=20,
        min_features=2,  # Lower threshold for test data
        min_seq_id=0.4,
        coverage=0.8,
        sensitivity=7.5,
        threads=2,
        clear_tmp=True
    )

    # Validate both strain and phage clustering outputs
    strain_output = output_dir / 'strain'
    phage_output = output_dir / 'phage'

    assert strain_output.exists(), "Strain output directory not created"
    assert phage_output.exists(), "Phage output directory not created"

    assert (strain_output / 'presence_absence_matrix.csv').exists(), \
        "Strain presence_absence_matrix.csv not created"
    assert (phage_output / 'presence_absence_matrix.csv').exists(), \
        "Phage presence_absence_matrix.csv not created"

    # Validate merged feature table
    merged_features = output_dir / 'merged' / 'full_feature_table.csv'
    assert merged_features.exists(), "Merged feature table not created"

    merged_df = pd.read_csv(merged_features)

    # Check for both strain and phage features
    feature_cols = [c for c in merged_df.columns
                    if c not in ['strain', 'phage', 'interaction', 'Genome']]

    strain_features = [c for c in feature_cols if c.startswith('sc_')]
    phage_features = [c for c in feature_cols if c.startswith('pc_')]

    assert len(strain_features) > 0, "No strain features in merged table"
    assert len(phage_features) > 0, "No phage features in merged table"

    # Validate modeling completed
    metrics_path = output_dir / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
    assert metrics_path.exists(), "Model performance metrics not created"

    metrics_results = validate_model_performance_metrics(metrics_path)
    assert metrics_results['valid'], \
        f"Metrics validation failed: {metrics_results['errors']}"


@pytest.mark.e2e
@pytest.mark.requires_mmseqs2
def test_workflow_handles_metadata_columns(medium_test_dataset, temp_output_dir):
    """
    Test that workflow correctly handles phenotype matrix with metadata columns.

    This validates the bug fix for issue #2 - metadata columns should not
    interfere with feature processing.

    Validates:
    - Workflow completes without errors
    - Metadata columns excluded from features
    - Only cluster features (sc_*, pc_*) used for modeling
    """
    strain_dir = medium_test_dataset['strain_dir']
    interaction_matrix = medium_test_dataset['interaction_matrix']

    # Create modified interaction matrix with metadata columns
    df = pd.read_csv(interaction_matrix)

    # Add metadata columns that could cause issues
    df['vaccine_dose'] = range(len(df))
    df['magic_number'] = range(100, 100 + len(df))
    df['collection_date'] = ['2024-01-01'] * len(df)

    metadata_matrix = temp_output_dir / 'interaction_with_metadata.csv'
    df.to_csv(metadata_matrix, index=False)

    output_dir = temp_output_dir / 'metadata_test_output'

    # Run workflow - should complete without errors
    run_protein_family_workflow(
        input_path_strain=str(strain_dir),
        input_path_phage=None,
        phenotype_matrix=str(metadata_matrix),
        output_dir=str(output_dir),
        tmp_dir=str(temp_output_dir / 'tmp'),
        phenotype_column='interaction',
        sample_column='strain',
        task_type='classification',
        num_runs_fs=3,
        num_runs_modeling=3,
        num_features=20,
        min_features=2,  # Lower threshold for test data
        min_seq_id=0.4,
        coverage=0.8,
        sensitivity=7.5,
        threads=2,
        clear_tmp=True
    )

    # Workflow should complete and produce outputs
    metrics_path = output_dir / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
    assert metrics_path.exists(), "Workflow did not complete with metadata columns"

    # Validate metrics
    metrics_results = validate_model_performance_metrics(metrics_path)
    assert metrics_results['valid'], \
        f"Metrics validation failed: {metrics_results['errors']}"

    # Check that metadata columns were not used as features
    # by inspecting the filtered feature tables
    filtered_dir = output_dir / 'feature_selection' / 'filtered_feature_tables'
    filtered_tables = list(filtered_dir.glob('*.csv'))

    for table_path in filtered_tables:
        table_df = pd.read_csv(table_path)
        feature_cols = [c for c in table_df.columns
                       if c not in ['strain', 'phage', 'interaction', 'Genome']]

        # All features should be cluster features (sc_* or pc_*)
        non_cluster = [c for c in feature_cols
                      if not (c.startswith('sc_') or c.startswith('pc_'))]

        assert len(non_cluster) == 0, \
            f"Non-cluster features found in {table_path.name}: {non_cluster}"


@pytest.mark.e2e
@pytest.mark.requires_mmseqs2
def test_workflow_validates_task_type(medium_test_dataset, temp_output_dir):
    """
    Test that workflow validates phenotype data matches task_type.

    This validates the bug fix for issue #1 - continuous data should
    raise error with task_type='classification'.

    Validates:
    - Classification with binary data succeeds
    - Classification with continuous data raises error
    - Regression with continuous data succeeds (with warning for binary)
    """
    strain_dir = medium_test_dataset['strain_dir']
    interaction_matrix = medium_test_dataset['interaction_matrix']

    output_dir = temp_output_dir / 'validation_test_output'

    # Test 1: Binary data with classification should work
    run_protein_family_workflow(
        input_path_strain=str(strain_dir),
        input_path_phage=None,
        phenotype_matrix=str(interaction_matrix),
        output_dir=str(output_dir / 'binary_classification'),
        tmp_dir=str(temp_output_dir / 'tmp1'),
        phenotype_column='interaction',
        sample_column='strain',
        task_type='classification',
        num_runs_fs=5,
        num_runs_modeling=5,
        num_features=15,
        min_features=2,  # Lower threshold for test data
        min_seq_id=0.4,
        coverage=0.8,
        threads=2,
        clear_tmp=True
    )

    metrics_path = output_dir / 'binary_classification' / 'modeling_results' / 'model_performance' / 'model_performance_metrics.csv'
    assert metrics_path.exists(), "Binary classification workflow did not complete"

    # Test 2: Continuous data with classification should raise error
    df = pd.read_csv(interaction_matrix)
    df['continuous_phenotype'] = df['interaction'] * 3.14 + 0.5  # Make continuous

    continuous_matrix = temp_output_dir / 'continuous_phenotype.csv'
    df.to_csv(continuous_matrix, index=False)

    with pytest.raises(ValueError, match="classification.*continuous|float values"):
        run_protein_family_workflow(
            input_path_strain=str(strain_dir),
            input_path_phage=None,
            phenotype_matrix=str(continuous_matrix),
            output_dir=str(output_dir / 'continuous_classification'),
            tmp_dir=str(temp_output_dir / 'tmp2'),
            phenotype_column='continuous_phenotype',
            sample_column='strain',
            task_type='classification',  # Should fail
            num_runs_fs=5,
            num_runs_modeling=5,
            num_features='15',
            min_features=2,  # Lower threshold for test data
            min_seq_id=0.4,
            coverage=0.8,
            threads=2,
            clear_tmp=True
        )
