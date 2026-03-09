"""
Integration tests for modeling outputs.

Tests model performance metrics file structure and validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from tests.utils.test_helpers import validate_model_performance_metrics


@pytest.mark.integration
def test_model_metrics_structure(temp_output_dir):
    """
    Test that model performance metrics have correct structure.

    This validates the expected CSV format for model outputs.
    """
    # Create a test metrics file
    metrics_path = temp_output_dir / 'model_metrics.csv'

    data = {
        'run': [0, 1, 2],
        'features': [20, 20, 20],
        'model': ['random_forest', 'random_forest', 'random_forest'],
        'AUC': [0.85, 0.82, 0.88],
        'Accuracy': [0.80, 0.78, 0.83],
        'Precision': [0.75, 0.72, 0.78],
        'Recall': [0.70, 0.68, 0.73],
        'F1': [0.72, 0.70, 0.75],
        'MCC': [0.60, 0.55, 0.65],
        'cut_off': ['cutoff_1', 'cutoff_1', 'cutoff_1']
    }

    df = pd.DataFrame(data)
    df.to_csv(metrics_path, index=False)

    # Validate structure
    results = validate_model_performance_metrics(metrics_path)

    assert results['valid'], f"Metrics validation failed: {results['errors']}"
