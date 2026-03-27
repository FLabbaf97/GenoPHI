"""
Integration tests for k-mer feature tables.

Tests k-mer feature table structure validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from tests.utils.test_helpers import validate_feature_table


@pytest.mark.integration
def test_kmer_feature_table_structure(temp_output_dir):
    """
    Test that k-mer feature tables have correct structure.

    This validates binary presence/absence format for k-mer features.
    """
    # Create test k-mer feature table
    kmer_table_path = temp_output_dir / 'kmer_features.csv'

    data = {
        'strain': [f'genome_{i}' for i in range(5)],
        'interaction': [0, 1, 0, 1, 1],
        'AAKL': [0, 1, 0, 1, 1],
        'AGLV': [1, 0, 1, 0, 0],
        'CDEF': [0, 0, 1, 1, 0],
        'WXYZ': [1, 1, 0, 0, 1]
    }

    df = pd.DataFrame(data)
    df.to_csv(kmer_table_path, index=False)

    # Validate structure
    results = validate_feature_table(
        kmer_table_path,
        min_features=1,
        expected_samples=5
    )

    assert results['valid'], f"K-mer table validation failed: {results['errors']}"
    assert results['feature_count'] == 4

    # Validate binary values
    df_check = pd.read_csv(kmer_table_path)
    feature_cols = [c for c in df_check.columns if c not in ['strain', 'interaction']]

    for col in feature_cols:
        unique_vals = set(df_check[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has non-binary values"
