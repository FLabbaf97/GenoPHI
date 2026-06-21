# GenoPHI: Practical Guide to Training, Validation, and External Testing

This guide explains how to use GenoPHI for phage–host interaction prediction when you need:

1. **Training and internal validation** on one set of bacteria (with anti-leakage splits).
2. **External testing / inference** on held-out bacteria or new phage–strain pairs.
3. Either a **pre-built merged feature table** or the **full protein-family pipeline** from FASTA files.

It complements the main [README](../README.md) with workflow-focused detail and points to the exact code modules that implement each behavior.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Map: Modules and Commands](#2-project-map-modules-and-commands)
3. [Three Different Meanings of "Clustering"](#3-three-different-meanings-of-clustering)
4. [Training vs Validation vs External Testing](#4-training-vs-validation-vs-external-testing)
5. [Input Data Requirements](#5-input-data-requirements)
6. [Splitting and Anti-Leakage Configuration](#6-splitting-and-anti-leakage-configuration)
   - [6.5 Phage Lytic-Rate Imbalance and Dynamic Sample Weights](#65-phage-lytic-rate-imbalance-and-dynamic-sample-weights)
7. [Workflow A: Pre-Merged Feature Table](#7-workflow-a-pre-merged-feature-table)
8. [Workflow B: Full Protein-Family Pipeline](#8-workflow-b-full-protein-family-pipeline)
9. [Workflow C: External Inference (Saved Models)](#9-workflow-c-external-inference-saved-models)
   - [9.5 Evaluate saved models on labeled merged features](#95-evaluate-saved-models-on-labeled-merged-features)
10. [Configuration Reference](#10-configuration-reference)
11. [Reading Results](#11-reading-results)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick Decision Tree](#13-quick-decision-tree)

---

## 1. Environment Setup

Always activate the conda environment before running any command:

```bash
conda activate genophi
```

Verify installation:

```bash
genophi --version
mmseqs version
genophi --help
```

**External dependency:** MMseqs2 is required for protein-family workflows and for assigning features to new genomes from `.faa` files. It is **not** used during train/test splitting when you already have a merged feature table.

---

## 2. Project Map: Modules and Commands

### Core Python modules

| Module | Responsibility |
|--------|----------------|
| `genophi/cli.py` | Unified CLI entry point (`genophi <command>`) |
| `genophi/mmseqs2_clustering.py` | MMseqs2 protein clustering, feature tables, `merge_feature_tables()`, `cluster_and_filter_features()` |
| `genophi/feature_selection.py` | `load_and_prepare_data()`, `filter_data()` (train/test splits), feature selection methods, CatBoost grid search |
| `genophi/select_feature_modeling.py` | Multi-run experiments, performance plots, cutoff comparison |
| `genophi/workflows/protein_family_workflow.py` | End-to-end: cluster → merge → select → train |
| `genophi/workflows/select_and_model_workflow.py` | Feature selection + training from an existing feature table |
| `genophi/workflows/assign_predict_workflow.py` | Assign protein families to new FASTAs → predict |
| `genophi/workflows/prediction_workflow.py` | Predict from pre-built feature tables |

### CLI commands you will use most

| Command | When to use |
|---------|-------------|
| `genophi select-and-train` | You already have a merged feature CSV |
| `genophi protein-family-workflow` | You have `.faa` files and want the full pipeline |
| `genophi assign-predict` | New genomes (FASTA) + saved MMseqs DB + saved models |
| `genophi predict` | Pre-computed feature rows for new pairs (no FASTA assignment) |
| `scripts/evaluate_saved_model.py` | **Evaluate** saved models on a **labeled** merged feature table (metrics + plots) |

---

## 3. Three Different Meanings of "Clustering"

GenoPHI uses the word "clustering" in three **independent** ways. Confusing them is the most common source of misconfiguration.

### 3.1 Protein-family clustering (MMseqs2)

- **What:** Groups similar **protein sequences** into families from `.faa` files.
- **When:** Feature extraction step only.
- **Code:** `genophi/mmseqs2_clustering.py` → `run_clustering_workflow()`
- **CLI:** `genophi cluster`, or the first step of `genophi protein-family-workflow`
- **Reuse prior results:** `--clustering_dir /path/to/previous/output/`
  - Symlinks `strain/`, `phage/`, and `tmp/` (including `mmseqs_db`) from a prior run.
  - Implemented in `genophi/workflows/protein_family_workflow.py` (lines ~205–225).
- **This is NOT** train/test validation clustering.

### 3.2 Sample clustering for train/test splits (`--use_clustering`)

- **What:** Clusters **strains** or **phages** by their genomic **feature vectors** (`sc_*` or `pc_*` columns) before splitting data into train and test.
- **When:** Feature selection and model training.
- **Code:** `genophi/feature_selection.py` → `filter_data()`
- **Uses:** Numeric features from the merged table — **not** `.faa` files.
- **Purpose:** Prevent data leakage between phylogenetically similar strains that would otherwise land on opposite sides of a random split.

**Feature columns used:**

| `filter_type` | Columns clustered | One row per |
|---------------|-------------------|-------------|
| `strain` | `sc_*` (strain protein-family features) | Unique strain ID |
| `phage` | `pc_*` (phage protein-family features) | Unique phage ID |

**`n_clusters` requirement:**

| `cluster_method` | Need to set `n_clusters`? |
|------------------|---------------------------|
| `hierarchical` | Optional; default **20** (auto-capped to `n_samples - 1`) |
| `hdbscan` | No; automatic. Falls back to hierarchical with 5 clusters if &lt;5 groups found |

### 3.3 Pre-processing feature filtering (`--use_feature_clustering`)

- **What:** Before modeling, clusters strains (or phages) by `sc_*` / `pc_*` features and **removes protein-family features** that appear in too few genome-clusters.
- **When:** At merge time in the full protein-family workflow.
- **Code:** `genophi/mmseqs2_clustering.py` → `cluster_and_filter_features()`, called from `merge_feature_tables()`
- **Parameters:**
  - `--feature_n_clusters` (default: 20): how many genome-groups to form.
  - `--feature_min_cluster_presence` (default: 2): a feature must be present in at least this many groups to be kept.
- **This is NOT** the train/test split. It reduces the feature space before ML.

---

## 4. Training vs Validation vs External Testing

GenoPHI does **not** ship a single "train / val / test" flag. You implement three distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAINING SET (you control via CSV filtering)                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Internal validation (automatic, repeated)                        │  │
│  │  • num_runs_fs iterations with different random seeds             │  │
│  │  • num_runs_modeling train/test splits per cutoff table           │  │
│  │  • filter_type=strain + use_clustering = anti-leakage CV          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  → Outputs: models in modeling_results/cutoff_N/run_*/best_model.pkl    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL TEST / INFERENCE (held-out bacteria or new pairs)             │
│  • Strains NEVER seen during training                                   │
│  • genophi assign-predict (from FASTA + MMseqs DB)                     │
│    OR genophi predict (from pre-built feature rows)                     │
│    OR evaluate_saved_model.py (labeled merged features → metrics)    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Internal validation (built into GenoPHI)

- Each **feature selection run** (`num_runs_fs`) performs a train/test split, selects features on train only, and evaluates briefly.
- Each **modeling run** (`num_runs_modeling`) performs another split, grid-searches CatBoost hyperparameters, and saves metrics.
- **Model selection within a run:** best hyperparameters chosen by **MCC** (classification) or **R²** (regression) on the test fold.
- **Cutoff selection across runs:** compares feature-selection occurrence thresholds (cutoff_3, cutoff_10, …) in `model_performance_metrics.csv`.

### External testing (your responsibility)

GenoPHI has **no `--holdout_strains` flag**. You must:

1. **Exclude** external bacteria from the training phenotype matrix and merged feature table.
2. After training, run inference separately on external strains using `assign-predict` or `predict`.
3. If you have **ground-truth labels** for external pairs in a merged feature table, run `scripts/evaluate_saved_model.py` to score performance (see [§9.5](#95-evaluate-saved-models-on-labeled-merged-features)).

**Important:** If external strain rows are included in the training CSV, the model may have seen those strains during internal CV (depending on split luck) or feature selection — this is **not** a clean external test.

---

## 5. Input Data Requirements

### 5.1 Phenotype / interaction matrix (CSV)

Phage–host binary classification:

```csv
strain,phage,interaction
Strain_001,Phage_A,1
Strain_001,Phage_B,0
Strain_002,Phage_A,1
```

Map column names with `--strain_column`, `--phage_column`, `--phenotype_column`.

### 5.2 Protein FASTAs (`.faa`)

- One file per genome; filename or internal IDs should match the phenotype matrix.
- Protein IDs must be unique across all genomes (GenoPHI prefixes duplicates automatically).

### 5.3 Merged feature table (CSV)

Required columns:

| Column type | Naming | Example |
|-------------|--------|---------|
| Metadata | `strain`, `phage`, phenotype | `interaction` |
| Strain features | `sc_*` | `sc_42` |
| Phage features | `pc_*` | `pc_17` |

Loaded by `load_and_prepare_data()` in `genophi/feature_selection.py`.

### 5.4 Subsetting genomes

| Method | Flag | Notes |
|--------|------|-------|
| Filter phenotype CSV manually | — | **Recommended** for train vs external split |
| Strain/phage list files | `--strain_list`, `--phage_list` | Only in `protein-family-workflow` / `cluster` |

---

## 6. Splitting and Anti-Leakage Configuration

All split logic lives in `filter_data()` (`genophi/feature_selection.py`).

### 6.1 `filter_type` — who is kept together?

| Value | Split unit | Use case |
|-------|------------|----------|
| `strain` | Strain ID (or strain cluster if `use_clustering`) | **Phage–host: predict on unseen bacteria** |
| `phage` | Phage ID (or phage cluster) | Generalize to unseen phages |
| `none` | Random 80/20 **rows** | Single-strain phenotypes; **not** recommended for phage–host |

**Why `strain` for phage–host:** Each row is a (strain, phage) pair. Random row splits put different phages for the **same strain** in train and test. The model can memorize strain-specific patterns instead of learning transferable host features.

### 6.2 `use_clustering` — phylogenetic anti-leakage

| `filter_type` | `use_clustering` | Behavior |
|---------------|------------------|----------|
| `none` | any | Random row split; clustering ignored |
| `strain` | OFF | All rows for strain A together; split by strain ID |
| `strain` | ON | Cluster strains by `sc_*` features → split **clusters** (~80/20) |
| `phage` | ON | Cluster phages by `pc_*` features → split clusters |

With clustering ON, closely related strains tend to stay on the same side of the split — stronger protection than strain-ID grouping alone.

### 6.3 Recommended settings for phage–host

```bash
--filter_type strain \
--use_clustering \
--cluster_method hierarchical \
--n_clusters 20 \
--check_feature_presence \
--filter_by_cluster_presence \
--min_cluster_presence 2
```

### 6.4 CLI default traps

| Command | `filter_type` default | `use_clustering` default |
|---------|----------------------|--------------------------|
| `protein-family-workflow` | `none` | **OFF** (must pass `--use_clustering`) |
| `select-and-train` | `strain` | **ON** (pass `--no-clustering` to disable) |
| `train` | `strain` (via `--set_filter`) | **ON** |

For phage–host work via `protein-family-workflow`, you **must** explicitly set `--filter_type strain --use_clustering`.

### 6.5 Phage Lytic-Rate Imbalance and Dynamic Sample Weights

In phage–host datasets, **phages differ strongly in how often they lyse hosts**: some infect many strains (high positive rate), others infect very few (low positive rate). If the model treats every row equally, it can overfit to phages with many positives and under-learn patterns for rarely lytic phages.

GenoPHI addresses this with **phage-aware sample weights** during training (feature selection and modeling), controlled by:

| Flag | Default | Effect |
|------|---------|--------|
| `--use_dynamic_weights` | OFF | When set, computes per-phage weights and passes them to CatBoost as `sample_weight` |
| `--weights_method` | `log10` (`protein-family-workflow`, `select-and-train`) | Formula used to up-weight positive interactions for low-lytic phages |

**Code:** `compute_phage_weights()` and `build_row_weights()` in `genophi/feature_selection.py` (~lines 530–604). Applied in:

- `perform_rfe()` — weights during RFE feature selection (~668–683)
- `train_and_evaluate()` — weights during CatBoost training (~977–994)
- `grid_search()` — weights in every hyperparameter combination (~1104–1116)

Weights are computed **per training fold** from `(phage, interaction)` counts in that fold only — they do not leak test-set information.

#### How weights are calculated

For each phage in the **training fold**, GenoPHI counts positive (`interaction=1`) and negative (`interaction=0`) rows, then assigns a weight to **positive** rows (negative rows always get weight `1.0`):

| `weights_method` | Formula for positive-class weight | Intuition |
|------------------|-----------------------------------|-----------|
| `log10` | `max(1.0, log10(neg_count / (pos_count + 1)) + 1)` | Moderate correction; default in `protein-family-workflow` |
| `inverse_frequency` | `total_count / (pos_count + 1)` | Stronger up-weighting for rarely lytic phages; good when lytic rates vary widely |
| `balanced` | `(total_count - pos_count) / (pos_count + 1)` | Sklearn-style balance ratio per phage |

If a phage has **zero** positive interactions in the training fold, positive weight is set to a small smoothing value (`1.0`) to avoid division-by-zero.

**Example:** Phage A has 90 negatives and 10 positives; Phage B has 50 negatives and 50 positives. With `inverse_frequency`, a positive row for Phage A gets a higher weight than a positive row for Phage B, because lytic events for Phage A are rarer relative to its panel.

#### When to enable

| Situation | Recommendation |
|-----------|----------------|
| Phage–host matrix with uneven lytic rates across phages | `--use_dynamic_weights` **ON** |
| Single-strain phenotype (no `phage` column) | Leave **OFF** (weights require `phage_column`) |
| All phages have similar positive rates | Optional; may have little effect |
| Large panels (e.g. Vibrio, Klebsiella) | **ON** with `--weights_method inverse_frequency` |

#### CLI defaults differ by command

| Command | `use_dynamic_weights` default | `weights_method` default |
|---------|------------------------------|--------------------------|
| `protein-family-workflow` | OFF | `log10` |
| `select-and-train` | OFF | `log10` |
| `train` / `select-features` | OFF | `inverse_frequency` (CLI) |

You must **explicitly pass** `--use_dynamic_weights` to activate weighting.

#### Example

```bash
genophi select-and-train \
    --full_feature_table /path/to/train_merged_features.csv \
    --output /path/to/training_output/ \
    --filter_type strain \
    --use_dynamic_weights \
    --weights_method inverse_frequency \
    --phage_column phage \
    --phenotype_column interaction \
    ...
```

#### Outputs and limitations

- Per-run weight tables: `feature_selection/run_*/phage_weights.csv` and `modeling_results/cutoff_N/run_M/phage_weights.csv` (when weights are enabled).
- **Not applied at inference:** `genophi predict` and `genophi assign-predict` do not re-weight samples; they use trained models as-is.
- **Classification only:** weights are designed for binary `interaction` labels (0/1).
- **Requires a `phage` column** in the merged feature table and `--phage_column` matching that name.

> **Note:** The README sometimes describes `use_dynamic_weights` in the context of feature occurrence imbalance. In the current codebase, `--use_dynamic_weights` and `--weights_method` specifically implement **phage lytic-rate sample weighting**, not feature-frequency weighting. Feature rarity is handled separately via `--filter_by_cluster_presence` and `--use_feature_clustering`.

---

## 7. Workflow A: Pre-Merged Feature Table

**You have:** a single CSV with `strain`, `phage`, `interaction`, and all `sc_*` / `pc_*` features for training bacteria. MMseqs DBs exist for later inference on new strains.

### Step A1 — Split training vs external data

Create two files from your master table:

```bash
# Example: keep only training strains in the training file
# (do this with pandas, or manually edit CSV)

# train_merged_features.csv   → strains in TRAIN_SET only
# external_merged_features.csv → held-out strains + interaction labels + sc_*/pc_* features
```

**Rule:** `train_merged_features.csv` must not contain any external-test strain IDs.

### Step A2 — Feature selection + training

```bash
conda activate genophi

genophi select-and-train \
    --full_feature_table /path/to/train_merged_features.csv \
    --output /path/to/training_output/ \
    \
    --sample_column strain \
    --phage_column phage \
    --phenotype_column interaction \
    --task_type classification \
    \
    --filter_type strain \
    --cluster_method hierarchical \
    --n_clusters 20 \
    --min_cluster_size 5 \
    \
    --method rfe \
    --num_features 100 \
    --num_runs_fs 25 \
    --check_feature_presence \
    --filter_by_cluster_presence \
    --min_cluster_presence 2 \
    \
    --num_runs_modeling 50 \
    --use_dynamic_weights \
    --weights_method inverse_frequency \
    --use_shap \
    \
    --threads 8 \
    --max_ram 16 \
    --verbose
```

**Python API equivalent:**

```python
from genophi.workflows.select_and_model_workflow import run_modeling_workflow_from_feature_table

run_modeling_workflow_from_feature_table(
    full_feature_table="/path/to/train_merged_features.csv",
    output_dir="/path/to/training_output/",
    sample_column="strain",
    phage_column="phage",
    phenotype_column="interaction",
    task_type="classification",
    filter_type="strain",
    use_clustering=True,
    cluster_method="hierarchical",
    n_clusters=20,
    method="rfe",
    num_features=100,
    num_runs_fs=25,
    num_runs_modeling=50,
    check_feature_presence=True,
    filter_by_cluster_presence=True,
    min_cluster_presence=2,
    use_dynamic_weights=True,
    weights_method="inverse_frequency",
    use_shap=True,
    threads=8,
    max_ram=16,
)
```

### Step A3 — Choose the best feature cutoff

```bash
cat /path/to/training_output/modeling_results/model_performance/model_performance_metrics.csv
```

Pick the top `cut_off` (e.g. `cutoff_10`). Models live at:

```
training_output/modeling_results/cutoff_10/run_0/best_model.pkl
training_output/modeling_results/cutoff_10/run_1/best_model.pkl
...
```

Feature list for that cutoff:

```
training_output/feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv
```

### Step A4 — External inference

**Option 1: Pre-computed feature rows** (if you already merged features for external pairs):

```bash
mkdir -p /path/to/external_predict/input
# Place strain_feature_table.csv (or equivalent) in input/

genophi predict \
    --input_dir /path/to/external_predict/input/ \
    --phage_feature_table /path/to/phage_features.csv \
    --feature_table /path/to/training_output/feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --output_dir /path/to/external_predictions/ \
    --strain_source strain \
    --phage_source phage \
    --threads 8
```

**Option 2: New strain FASTAs** → see [Workflow C](#9-workflow-c-external-inference-saved-models).

**Option 3: Evaluate on labeled external merged features** (recommended when you already have `strain`, `phage`, `interaction`, and `sc_*` / `pc_*` columns for held-out bacteria):

```bash
python scripts/evaluate_saved_model.py \
    --feature_table /path/to/external_merged_features.csv \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --output_dir /path/to/external_evaluation/ \
    --phenotype_column interaction \
    --strain_column strain \
    --phage_column phage \
    --threads 8
```

See [§9.5](#95-evaluate-saved-models-on-labeled-merged-features) for required inputs, outputs, and how this differs from `genophi predict`.

### Workflow A output layout

```
training_output/
├── feature_table_modeling_report.csv   # all CLI parameters + input/final feature counts
├── feature_table_modeling_report.txt   # human-readable summary
├── feature_table_modeling_workflow.log
├── feature_selection/
│   ├── features_occurrence.csv
│   └── filtered_feature_tables/
│       └── select_feature_table_cutoff_10.csv
├── modeling_results/
│   ├── cutoff_10/
│   │   ├── run_0/
│   │   │   ├── best_model.pkl
│   │   │   ├── best_model_predictions.csv
│   │   │   └── model_performance.csv
│   │   └── top_models_summary.csv
│   └── model_performance/
│       └── model_performance_metrics.csv
└── protein_family_workflow.log   # if logging enabled
```

---

## 8. Workflow B: Full Protein-Family Pipeline

**You have:** directories of `.faa` files, an interaction matrix, and optionally pre-computed MMseqs2 results.

### Step B1 — Prepare training-only interaction matrix

Same rule as Workflow A: remove external-test strains from `train_interactions.csv`.

### Step B2 — Run the full pipeline

```bash
conda activate genophi

genophi protein-family-workflow \
    --input_path_strain /path/to/strain_fastas/ \
    --input_path_phage /path/to/phage_fastas/ \
    --phenotype_matrix /path/to/train_interactions.csv \
    --output_dir /path/to/full_pipeline_output/ \
    \
    --clustering_dir /path/to/existing_mmseq_output/ \
    \
    --min_seq_id 0.4 \
    --coverage 0.8 \
    --sensitivity 7.5 \
    --suffix faa \
    \
    --strain_column strain \
    --phage_column phage \
    --sample_column strain \
    --phenotype_column interaction \
    --task_type classification \
    \
    --use_feature_clustering \
    --feature_cluster_method hierarchical \
    --feature_n_clusters 20 \
    --feature_min_cluster_presence 2 \
    \
    --filter_type strain \
    --use_clustering \
    --cluster_method hierarchical \
    --n_clusters 20 \
    --min_cluster_size 5 \
    --check_feature_presence \
    --filter_by_cluster_presence \
    --min_cluster_presence 2 \
    \
    --method rfe \
    --num_features 100 \
    --num_runs_fs 25 \
    --num_runs_modeling 50 \
    --use_dynamic_weights \
    --weights_method inverse_frequency \
    --use_shap \
    \
    --threads 8 \
    --max_ram 16 \
    --tmp_dir tmp \
    --clear_tmp \
    --verbose
```

Omit `--clustering_dir` to run MMseqs2 from scratch on the FASTA directories.

### Step B3 — Pipeline stages (code path)

Implemented in `genophi/workflows/protein_family_workflow.py`:

| Step | Action | Output |
|------|--------|--------|
| 1 | MMseqs2 clustering (strain, phage) | `strain/`, `phage/`, `tmp/` |
| 2 | Merge features + phenotype | `merged/full_feature_table.csv` |
| 3 | Feature selection (`num_runs_fs`) | `feature_selection/` |
| 4 | Generate cutoff tables | `feature_selection/filtered_feature_tables/` |
| 5 | Modeling (`num_runs_modeling`) | `modeling_results/cutoff_*/` |
| 6 | Best cutoff + predictive proteins | `modeling_results/model_performance/` |

### Step B4 — Re-run modeling without re-clustering proteins

If you only change the phenotype matrix or ML parameters:

```bash
genophi protein-family-workflow \
    --clustering_dir /path/to/full_pipeline_output/ \
    --input_path_strain /path/to/strain_fastas/ \
    --input_path_phage /path/to/phage_fastas/ \
    --phenotype_matrix /path/to/new_train_interactions.csv \
    --output_dir /path/to/new_modeling_run/ \
    --filter_type strain \
    --use_clustering \
    ... # other ML flags
```

`--clustering_dir` reuses MMseqs2 protein-family results only. Train/test splitting still runs fresh on the new phenotype data.

### Modular alternative (step by step)

```bash
# 1. Clustering only
genophi cluster \
    --input_strain /path/to/strain_fastas/ \
    --input_phage /path/to/phage_fastas/ \
    --phenotype_matrix /path/to/train_interactions.csv \
    --output /path/to/clustering/

# 2+3. Feature selection + training (from merged table)
genophi select-and-train \
    --full_feature_table /path/to/clustering/merged/full_feature_table.csv \
    --output /path/to/modeling/ \
    --filter_type strain \
    ... # ML flags
```

---

## 9. Workflow C: External Inference (Saved Models)

After Workflow A or B, retain these artifacts for inference:

```
training_output/
├── tmp/strain/mmseqs_db
├── tmp/phage/mmseqs_db                    # if predicting new phages
├── strain/clusters.tsv
├── strain/features/selected_features.csv
├── phage/features/feature_table.csv
├── modeling_results/cutoff_10/            # best cutoff
└── feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv
```

### C1 — New bacterial strains (known phage panel)

Assigns new proteins to existing protein families via MMseqs, merges with known phage features, runs all ensemble models.

```bash
conda activate genophi

genophi assign-predict \
    --input_dir /path/to/new_strain_fastas/ \
    --mmseqs_db /path/to/training_output/tmp/strain/mmseqs_db \
    --clusters_tsv /path/to/training_output/strain/clusters.tsv \
    --feature_map /path/to/training_output/strain/features/selected_features.csv \
    --tmp_dir /path/to/tmp_assign/ \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --phage_feature_table /path/to/training_output/phage/features/feature_table.csv \
    --feature_table /path/to/training_output/feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv \
    --output_dir /path/to/predictions_new_strains/ \
    --genome_type strain \
    --min_seq_id 0.4 \
    --coverage 0.8 \
    --sensitivity 7.5 \
    --suffix faa \
    --threads 8
```

Code: `genophi/workflows/assign_predict_workflow.py`

### C2 — New phages (known strain panel)

```bash
genophi assign-predict \
    --input_dir /path/to/new_phage_fastas/ \
    --mmseqs_db /path/to/training_output/tmp/phage/mmseqs_db \
    --clusters_tsv /path/to/training_output/phage/clusters.tsv \
    --feature_map /path/to/training_output/phage/features/selected_features.csv \
    --tmp_dir /path/to/tmp_assign_phage/ \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --strain_feature_table /path/to/training_output/strain/features/feature_table.csv \
    --feature_table /path/to/training_output/feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv \
    --output_dir /path/to/predictions_new_phages/ \
    --genome_type phage \
    --threads 8
```

### C3 — Both strain and phage are new

GenoPHI does not provide a single command for this. Recommended approach:

1. Run `assign-features` (or `assign-predict --genome_type strain`) for new strains.
2. Run `assign-features` for new phages (if needed).
3. Build the cross-product feature table or use `genophi predict` with both feature tables.

### C4 — Predict from pre-merged external feature rows

When features are already computed (no FASTA assignment):

```bash
genophi predict \
    --input_dir /path/to/external_strain_features/ \
    --phage_feature_table /path/to/phage_features.csv \
    --feature_table /path/to/training_output/feature_selection/filtered_feature_tables/select_feature_table_cutoff_10.csv \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --output_dir /path/to/external_predictions/ \
    --threads 8
```

Code: `genophi/workflows/prediction_workflow.py`

- Loads all `run_*/best_model.pkl` models in `model_dir`.
- Aggregates median confidence per (strain, phage) pair.
- Output: `{strain}_median_predictions.csv` with `Final_Prediction` and `Confidence`.
- **No built-in metrics:** `genophi predict` does not compare predictions to ground truth. Use [§9.5](#95-evaluate-saved-models-on-labeled-merged-features) when labels are available.

### 9.5 Evaluate saved models on labeled merged features

Use `scripts/evaluate_saved_model.py` when your goal is to **test a trained model on new data** and **measure performance** — not just generate predictions.

This is the right tool when you already have a **single merged feature CSV** for external (held-out) bacteria with:

| Requirement | Details |
|-------------|---------|
| Metadata | `strain`, `phage`, and a ground-truth label column (default: `interaction`, 0/1) |
| Features | `sc_*` and `pc_*` columns (same protein-family naming as training) |
| Holdout | Strain IDs **not** present in the training merged table |
| Models | All `run_*/best_model.pkl` files under `modeling_results/cutoff_N/` |

You do **not** need to pass `--feature_table` from feature selection separately: each saved CatBoost model carries its own `feature_names_`. The script (via `predict_interactions`) keeps only the features each model expects. Extra columns in your CSV are ignored; **missing** required features raise an error.

#### `evaluate_saved_model.py` vs `genophi predict`

| | `scripts/evaluate_saved_model.py` | `genophi predict` |
|---|-----------------------------------|-------------------|
| Input | One **merged** table (strain + phage + features + labels) | Separate strain/phage feature tables merged at predict time |
| Labels required | Yes (for metrics) | No |
| Ensemble | Median probability across all `run_*/best_model.pkl` | Same |
| Outputs | Predictions + global metrics + hit@k / precision@k + plots | Prediction CSVs only |
| Typical use | External test set with known interactions | Deployment / unlabeled screening |

#### Step-by-step workflow

1. **Train** on training-only data (`genophi select-and-train` or `protein-family-workflow`). Pick the best cutoff from `model_performance_metrics.csv` (e.g. `cutoff_3`).
2. **Build** `external_merged_features.csv` for held-out strains — same column layout as `train_merged_features.csv`, including true `interaction` labels. You can produce this by merging external strain/phage feature tables the same way as in training, or by subsetting a master table after excluding training strain IDs.
3. **Evaluate** with the script (from the GenoPHI repository root):

```bash
conda activate genophi

python scripts/evaluate_saved_model.py \
    --feature_table /path/to/external_merged_features.csv \
    --model_dir /path/to/training_output/modeling_results/cutoff_10/ \
    --output_dir /path/to/external_evaluation/ \
    --phenotype_column interaction \
    --strain_column strain \
    --phage_column phage \
    --threads 8 \
    --threshold 0.5 \
    --verbose
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `--feature_table` / `-i` | Yes | Labeled merged CSV or parquet |
| `--model_dir` / `-m` | Yes | e.g. `modeling_results/cutoff_10/` (contains `run_0/`, `run_1/`, …) |
| `--output_dir` / `-o` | Yes | Where predictions, metrics, and plots are written |
| `--phenotype_column` | No | Ground-truth column (default: `interaction`) |
| `--strain_column` | No | Default: `strain` |
| `--phage_column` | No | Default: `phage` |
| `--threads` | No | CatBoost prediction threads (default: 4) |
| `--threshold` | No | Probability cutoff for `Final_Prediction` (default: 0.5) |
| `--verbose` / `-v` | No | Debug logging |

#### Outputs in `output_dir/`

| File | Description |
|------|-------------|
| `all_run_predictions.csv` | Per-model probabilities for every (strain, phage) pair |
| `median_predictions.csv` | Ensemble median `Confidence` and thresholded `Final_Prediction` |
| `evaluation_pairs.csv` | Predictions joined with ground-truth labels |
| `evaluation_metrics.json` / `.csv` | Global classification + averaged ranking metrics |
| `per_strain_ranking_metrics.csv` | hit@k and precision@k for each strain |
| `predicted_probability_distribution.png` | Histogram of probabilities (overall and by true class) |

**Global metrics:** accuracy, precision (PPV), recall (sensitivity), specificity, NPV, F1, MCC, AUC-ROC, AUC-PRC (average precision), % predicted positive, confusion matrix counts.

**Per-strain ranking metrics (averaged over eligible strains):**

| Metric | Definition | Eligibility |
|--------|------------|-------------|
| `hit@k` (k=1…5) | Fraction of strains where ≥1 true positive appears in the top-k phages ranked by predicted probability | Strains with **≥1** true positive in the dataset |
| `precision@k` (k=1…5) | Mean of `(true positives in top-k) / k` per strain | Strains with **≥k** total true positives |

#### Example (Klebsiella-style layout)

```bash
python scripts/evaluate_saved_model.py \
    --feature_table /path/to/Klebsiella_external_merged_features.csv \
    --model_dir results/Klebsiella_output_clustered/modeling_results/cutoff_3/ \
    --output_dir results/Klebsiella_external_eval/ \
    --phenotype_column interaction \
    --threads 8
```

Review `external_evaluation/evaluation_metrics.csv` for a one-row summary; use `evaluation_pairs.csv` for per-pair error analysis.

For **prediction only** (no labels), use `genophi predict` as in [§9.4](#c4--predict-from-pre-merged-external-feature-rows) above.

---

## 10. Configuration Reference

### 10.1 Anti-leakage and splitting

| Parameter | Default (`protein-family-workflow`) | Default (`select-and-train`) | Recommended (phage–host) | Code |
|-----------|-------------------------------------|------------------------------|--------------------------|------|
| `filter_type` | `none` | `strain` | `strain` | `feature_selection.py` |
| `use_clustering` | OFF | ON | ON | `feature_selection.py` |
| `cluster_method` | `hierarchical` | `hierarchical` | `hierarchical` | `feature_selection.py` |
| `n_clusters` | 20 | 20 | 10–30 | `feature_selection.py` |
| `check_feature_presence` | OFF | OFF | ON | `feature_selection.py` |
| `filter_by_cluster_presence` | OFF | OFF | ON | `feature_selection.py` |
| `min_cluster_presence` | 2 | 2 | 2–3 | `feature_selection.py` |

### 10.2 Pre-merge feature filtering (optional)

| Parameter | Default | Suggested | Code |
|-----------|---------|-----------|------|
| `use_feature_clustering` | OFF | ON for large diverse panels | `mmseqs2_clustering.py` |
| `feature_n_clusters` | 20 | 5–30 | `cluster_and_filter_features()` |
| `feature_min_cluster_presence` | 2 | 2–3 | `cluster_and_filter_features()` |

### 10.3 Feature selection configuration

| Parameter | Default | Suggested | Notes | Code |
|-----------|---------|-----------|-------|------|
| `method` | `rfe` | `rfe` | See method table below | `feature_selection.py` |
| `num_features` | `none` (auto) | 50–150 | Auto in `protein-family-workflow`: 50 if &lt;500 rows, 100 if &lt;2000, else `rows/20` | `protein_family_workflow.py` ~344–350 |
| `num_runs_fs` | 25 | 25–50 | Independent FS iterations with different random seeds | `run_feature_selection_iterations()` |
| `max_features` | `none` | — | Cap features in cutoff tables | `generate_feature_tables()` |
| `min_features` | `none` (auto) | — | Minimum features required for a cutoff to be kept | `generate_feature_tables()` |
| `binary_data` | OFF (`select-and-train`) / ON (`protein-family-workflow`) | ON for presence/absence | Converts feature values to 0/1 in cutoff tables | `generate_feature_tables()` |
| `check_feature_presence` | OFF | ON for phage–host | Drops features absent from train or test fold | `filter_data()` |
| `filter_by_cluster_presence` | OFF | ON for phage–host | Drops features rare across strain/phage groups | `filter_data()` |
| `min_cluster_presence` | 2 | 2–3 | Min groups/clusters a feature must appear in | `filter_data()` |

**Feature selection methods** (`--method`):

| Method | Speed | Best for | Notes |
|--------|-------|----------|-------|
| `rfe` | Medium | General use (recommended) | Recursive Feature Elimination with CatBoost |
| `shap_rfe` | Slow, high RAM | Model-agnostic importance | RFE driven by SHAP values |
| `select_k_best` | Fast | Quick screening | ANOVA F-test |
| `chi_squared` | Fast | Classification only | χ² test |
| `lasso` | Fast | Sparse models | L1 regularization |
| `shap` | Slow, high RAM | Direct importance ranking | Select top features by SHAP |

### 10.4 Modeling configuration

| Parameter | Default | Suggested | Notes | Code |
|-----------|---------|-----------|-------|------|
| `num_runs_modeling` | 50 | 50–100 | Repeated train/test splits + grid search per cutoff table | `run_experiments()` |
| `task_type` | `classification` | `classification` | `regression` for continuous phenotypes (optimizes R²) | `grid_search()` / `grid_search_regressor()` |
| `phenotype_column` | `interaction` | your column name | Target variable; dropped from features | `load_and_prepare_data()` |
| `sample_column` | `strain` | `strain` | Sample ID metadata column | `load_and_prepare_data()` |
| `phage_column` | `phage` | `phage` | Required for phage–host; used in dynamic weights | `compute_phage_weights()` |
| `use_dynamic_weights` | OFF | ON for uneven phage lytic rates | See [§6.5](#65-phage-lytic-rate-imbalance-and-dynamic-sample-weights) | `feature_selection.py` |
| `weights_method` | `log10` | `inverse_frequency` | `log10`, `inverse_frequency`, `balanced` | `compute_phage_weights()` |
| `use_shap` | OFF | ON for interpretability | Saves SHAP plots per modeling run; slower | `select_feature_modeling.py` |
| `set_filter` | `strain` (`train` CLI) | `strain` | Same as `filter_type`; controls train/test split in `train` command | `filter_data()` |

**CatBoost training behavior** (not CLI flags — fixed in code unless you edit source):

| Setting | Value | Location |
|---------|-------|----------|
| Hyperparameter grid | `iterations` [500, 1000], `learning_rate` [0.05, 0.1], `depth` [4, 6] | `grid_search()` in `feature_selection.py` |
| Best model selection | Highest **MCC** (classification) or **R²** (regression) on test fold | `grid_search()` |
| Early stopping | 100 rounds on test fold (`eval_set`) | `train_and_evaluate()` ~997 |
| Loss function | `Logloss` (classification), `RMSE` (regression) | `model_testing_select_MCC()` |

### 10.5 Compute and system resources

| Parameter | Default | Suggested | What it controls |
|-----------|---------|-----------|------------------|
| `--threads` | 4 | 8–16 (match CPU cores) | CatBoost `thread_count`; MMseqs2 parallelism in clustering/assignment; prediction `thread_count` |
| `--max_ram` | 8 GB (`protein-family-workflow`), 16 GB (`select-and-train`, `cluster`) | 16–32 GB for large panels | CatBoost `used_ram_limit` (e.g. `16gb`); caps memory during FS and modeling |
| `--verbose` / `-v` | OFF | ON while debugging | Enables DEBUG logging in CLI |
| `--clear_tmp` | OFF | ON if disk is tight | Removes MMseqs2 temp files after workflow (`protein-family-workflow` only) |
| `--tmp_dir` | `tmp` | fast local disk | MMseqs2 intermediate files |

**Sizing guidance:**

| Dataset scale | `threads` | `max_ram` | `num_runs_fs` | `num_runs_modeling` |
|---------------|-----------|-----------|---------------|---------------------|
| Demo / &lt;500 interactions | 4 | 8 | 5–10 | 10–25 |
| Medium (e.g. Klebsiella) | 8 | 16 | 25 | 50 |
| Large (e.g. Vibrio ~60k rows) | 12–16 | 32+ | 25–50 | 50–100 |

SHAP-based methods (`shap`, `shap_rfe`) and `--use_shap` add substantial RAM and runtime beyond `--max_ram` alone.

### 10.6 MMseqs2 protein clustering

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_seq_id` | 0.4 | Minimum sequence identity for protein families |
| `coverage` | 0.8 | Minimum alignment coverage |
| `sensitivity` | 7.5 | Search sensitivity (higher = slower, more sensitive) |
| `clustering_dir` | none | Reuse prior MMseqs2 outputs |

### 10.7 Hyperparameter grid (advanced)

To change CatBoost search space, edit `param_grid` in:

- `genophi/feature_selection.py` → `grid_search()` (~line 1106)
- `genophi/select_feature_modeling.py` → `model_testing_select_MCC()` (~line 139)

Default grid: `iterations` [500, 1000], `learning_rate` [0.05, 0.1], `depth` [4, 6].

### 10.8 Feature cutoff list (advanced)

Cutoff thresholds (3, 4, 5, … 50) are hardcoded in:

- `genophi/workflows/protein_family_workflow.py` (~line 392)
- `genophi/workflows/select_and_model_workflow.py` (~line 155)

Edit these lists to add or remove cutoff values.

---

## 11. Reading Results

### Internal validation metrics

| File | Content |
|------|---------|
| `modeling_results/cutoff_N/run_M/model_performance.csv` | All grid-search combinations for one run |
| `modeling_results/cutoff_N/run_M/best_model_predictions.csv` | Test-fold predictions for best model |
| `modeling_results/cutoff_N/top_models_summary.csv` | Best MCC/R² per run |
| `modeling_results/model_performance/model_performance_metrics.csv` | **Compare cutoffs** — start here |
| `feature_selection/features_occurrence.csv` | How often each feature was selected |

### Classification metrics

- **Primary optimization metric:** MCC (Matthews Correlation Coefficient)
- **Also reported:** AUC-ROC, Precision, Recall, F1, Accuracy
- **Plots per run:** confusion matrix, ROC, precision–recall (`modeling_results/cutoff_N/run_M/`)

### Regression metrics

- **Primary optimization metric:** R²
- **Also reported:** RMSE, MAE

### Ensemble prediction

`assign-predict`, `predict`, and `evaluate_saved_model.py` all load **all** `run_*/best_model.pkl` files and use **median probability** per (strain, phage) pair. This mirrors the internal multi-run design. Only `evaluate_saved_model.py` compares those predictions to ground-truth labels and writes full metric reports.

---

## 12. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Optimistic performance on phage–host | `filter_type=none` or clustering off | Set `--filter_type strain --use_clustering` |
| External test strains in training CSV | No manual holdout | Filter training CSV before `select-and-train` |
| `filter_type must be a column` error | Missing `strain` column | Ensure merged table has `strain` column |
| No `sc_*` columns / clustering fallback | Wrong feature prefixes | Strain features must be `sc_*`, phage `pc_*` |
| Missing features at predict time | Feature name mismatch | Ensure external merged table has same `sc_*` / `pc_*` names as training; models use `feature_names_` |
| Missing features at evaluate time | Same as predict | All features in `best_model.pkl` must exist in `--feature_table` |
| MMseqs2 not found | PATH issue | `conda activate genophi && which mmseqs` |
| Out of memory | Large feature table | Reduce `--num_features`, increase `--max_ram`, use fewer SHAP runs, or subset data |
| `protein-family-workflow` ignores clustering | Default is OFF | Explicitly pass `--use_clustering` |
| Model biased toward high-lytic phages | Equal row weights | Enable `--use_dynamic_weights --weights_method inverse_frequency` |
| Dynamic weights have no effect | No `phage` column or weights off | Ensure phage–host mode and pass `--use_dynamic_weights` |

---

## 13. Quick Decision Tree

```
Do you already have a merged feature CSV?
├── YES → genophi select-and-train (Workflow A)
│         └── External test
│             ├── Have labels in merged CSV → evaluate_saved_model.py (§9.5)
│             ├── Feature rows only → genophi predict
│             └── New FASTAs → genophi assign-predict
└── NO  → Do you have .faa files?
          ├── YES → genophi protein-family-workflow (Workflow B)
          │         └── Reuse MMseqs? → add --clustering_dir
          └── NO  → Build features first (cluster workflow or external pipeline)

Phage–host prediction?
└── Always: --filter_type strain --use_clustering
    Exclude external strains from training CSV manually

New genomes at inference?
├── Have FASTA → genophi assign-predict + saved mmseqs_db
└── Have feature rows
    ├── Labeled merged table → evaluate_saved_model.py
    └── Unlabeled → genophi predict
```

---

## Summary Checklist

Before training:

- [ ] `conda activate genophi`
- [ ] Training CSV contains **only** training strains/phages
- [ ] External strains stored separately for inference
- [ ] `--filter_type strain` and `--use_clustering` set (especially for `protein-family-workflow`)
- [ ] `--use_dynamic_weights` set if phage lytic rates are uneven (see [§6.5](#65-phage-lytic-rate-imbalance-and-dynamic-sample-weights))
- [ ] `--threads` and `--max_ram` sized for your dataset
- [ ] Phenotype column names match CLI flags

After training:

- [ ] Review `model_performance_metrics.csv` and pick best cutoff
- [ ] Save `tmp/`, `strain/`, `phage/`, `modeling_results/cutoff_N/`, and cutoff feature table paths
- [ ] Run external inference: `assign-predict` or `predict` (unlabeled), or `evaluate_saved_model.py` (labeled merged features)
- [ ] Review `evaluation_metrics.csv` and `evaluation_pairs.csv` for held-out performance

---

*For general installation, demo data, and API documentation, see the main [README](../README.md).*
