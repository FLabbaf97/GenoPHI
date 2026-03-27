"""
Integration tests for feature table → feature selection pipeline.

Tests data loading and basic feature table validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from tests.utils.test_helpers import validate_feature_table


@pytest.mark.integration
def test_feature_table_structure(small_test_dataset, temp_output_dir):
    """
    Test that feature tables have correct structure.

    This validates basic CSV structure without calling internal functions.
    """
    # Create a simple test feature table
    feature_table_path = temp_output_dir / 'test_feature_table.csv'

    data = {
        'strain': [f'strain_{i}' for i in range(5)],
        'interaction': [0, 1, 0, 1, 1],
        'sc_0': [0, 1, 0, 1, 0],
        'sc_1': [1, 0, 1, 0, 1],
        'sc_2': [0, 0, 1, 1, 0]
    }

    df = pd.DataFrame(data)
    df.to_csv(feature_table_path, index=False)

    # Validate structure
    results = validate_feature_table(
        feature_table_path,
        min_features=1,
        expected_samples=5
    )

    assert results['valid'], f"Feature table validation failed: {results['errors']}"
    assert results['feature_count'] == 3
