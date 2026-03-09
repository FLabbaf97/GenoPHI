# Test Baselines

This directory contains baseline metrics for regression testing.

## Files to Commit

Only the following baseline CSV files should be committed to the repository:

- `baseline_metrics_quick.csv` - Quick test baseline (5 strains, 5 iterations)
- `baseline_metrics_standard.csv` - Standard test baseline (10 strains, 10 iterations)

## Files to Ignore

All intermediate workflow outputs are automatically cleaned up by the baseline generation script and should not be committed:

- `quick/` directory - Temporary files for quick baseline generation
- `standard/` directory - Temporary files for standard baseline generation

These directories are in `.gitignore`.

## Generating Baselines

To regenerate baseline files:

```bash
# Generate both quick and standard baselines
python scripts/generate_test_baselines.py

# Or generate only one:
python scripts/generate_test_baselines.py --quick-only
python scripts/generate_test_baselines.py --standard-only
```

The script will:
1. Create test datasets (subsets of full test data)
2. Run the protein family workflow
3. Extract model performance metrics
4. Save only the baseline CSV files
5. Clean up all intermediate files automatically

## Baseline Structure

Each baseline CSV contains model performance metrics with columns:
- `run` - Feature selection run number
- `features` - Number of features used
- `model` - Model type used
- `AUC` - Area under ROC curve
- `Accuracy` - Classification accuracy
- `MCC` - Matthews correlation coefficient
- Additional performance metrics

Tests compare current runs against these baselines with a tolerance (typically 5-8%) to catch performance regressions.
