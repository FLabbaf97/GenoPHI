# GenoPHI Test Suite

This directory contains the test suite for the GenoPHI package, organized into multiple tiers for different testing scenarios.

## Test Organization

### Test Tiers

**Smoke Tests** (`test_smoke.py`)
- Quick package installation verification
- Runtime: <5 seconds
- Run with: `pytest -m smoke`

**Unit Tests**
- Test individual functions with simple/mock data
- Runtime: seconds
- Run with: `pytest -m unit`

**Integration Tests** (`tests/integration/`)
- Test module-to-module interactions
- Use small test datasets (5-10 samples)
- Runtime: ~30-45 minutes
- Run with: `pytest -m integration`

**End-to-End Tests** (`tests/e2e/`)
- Test complete workflows with reduced parameters
- Runtime: ~60-90 minutes
- Run with: `pytest -m e2e`

**Full Regression Tests**
- Complete workflows with production parameters
- Runtime: ~2-3 hours
- Run with: `pytest -m full`

## Running Tests

### Quick Validation (Recommended for development)
```bash
# Run smoke tests only
pytest -m smoke -v

# Run smoke + unit tests
pytest -m "smoke or unit" -v
```

### Standard Testing (Pre-commit/PR)
```bash
# Run integration tests (excludes slow tests)
pytest -m "integration and not slow" -v

# Run all integration tests
pytest tests/integration/ -v
```

### Full Testing (Pre-release)
```bash
# Run all tests
pytest -v

# Run only tests that require MMSeqs2
pytest -m requires_mmseqs2 -v

# Run integration + E2E (skip full regression)
pytest -m "integration or e2e" -v
```

### Selective Testing
```bash
# Skip tests that require MMSeqs2 (useful for quick checks)
pytest -m "not requires_mmseqs2" -v

# Run only slow tests
pytest -m slow -v

# Run specific test file
pytest tests/test_smoke.py -v

# Run specific test function
pytest tests/test_smoke.py::test_package_import -v
```

## Test Markers

- `smoke`: Quick installation verification (<5 sec)
- `unit`: Unit tests for individual functions
- `integration`: Module-to-module interaction tests (~30-45 min)
- `e2e`: End-to-end workflow tests (~60-90 min)
- `slow`: Long-running tests (>30 min)
- `full`: Full regression tests (~2-3 hours)
- `requires_mmseqs2`: Tests requiring MMSeqs2 installation

## Test Data

Test data is located in `/data/test_data/`:
- `strain_AAs/`: 25 E. coli strain proteomes (FASTA)
- `phage_AAs/`: 25 phage proteomes (FASTA)
- `ecoli_test_interaction_matrix.csv`: Interaction matrix (625 interactions)

Tests use subsets of this data:
- Small dataset: 5 strains/phages
- Medium dataset: 10 strains/phages
- Full dataset: All 25 strains/phages

## Baseline Metrics

Baseline performance metrics for regression testing are stored in `tests/baselines/`:
- `baseline_metrics_quick.csv`: 5 strains, 5 iterations
- `baseline_metrics_standard.csv`: 10 strains, 10 iterations

To regenerate baselines:
```bash
python scripts/generate_test_baselines.py --output-dir tests/baselines/
```

## Continuous Integration Recommendations

### On Every Commit
```bash
pytest -m smoke
```

### On Pull Requests
```bash
pytest -m "integration and not slow"
```

### On Release
```bash
pytest -v
```

## Test Development Guidelines

1. **Write smoke tests** for new modules/workflows
2. **Add integration tests** for module interactions
3. **Mark tests appropriately** with pytest markers
4. **Use fixtures** from `conftest.py` for test data
5. **Keep tests isolated** - each test should be independent
6. **Document test purpose** in docstrings

## Troubleshooting

**Tests fail with "MMSeqs2 not found"**
- Ensure MMSeqs2 is installed and in PATH
- Or skip these tests: `pytest -m "not requires_mmseqs2"`

**Tests timeout**
- Increase timeout in `pytest.ini`
- Or run with: `pytest --timeout=1200`

**Baseline comparison fails**
- Regenerate baselines with current code
- Or adjust tolerance in test parameters

## Structure

```
tests/
├── README.md                    # This file
├── pytest.ini                   # Pytest configuration
├── conftest.py                  # Shared fixtures
├── test_smoke.py               # Smoke tests
├── baselines/                   # Baseline metrics
│   ├── baseline_metrics_quick.csv
│   └── baseline_metrics_standard.csv
├── integration/                 # Integration tests
│   ├── test_clustering_to_features.py
│   ├── test_features_to_selection.py
│   ├── test_selection_to_modeling.py
│   └── test_kmer_pipeline.py
├── e2e/                        # End-to-end tests
│   └── test_protein_family_workflow.py
└── utils/                      # Test utilities
    └── test_helpers.py
```
