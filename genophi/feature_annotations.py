import os
import logging
import pandas as pd
import numpy as np
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _read_table_file(filepath):
    """Read DataFrame from parquet or CSV."""
    from genophi.mmseqs2_clustering import _read_table, _resolve_table_path
    return _read_table(_resolve_table_path(filepath))

def load_annotation_table(annotation_table_path):
    """
    Loads an annotation table, automatically detecting whether it is CSV or TSV based on its extension or content.

    Args:
        annotation_table_path (str): Path to the annotation table (CSV or TSV format).

    Returns:
        annotation_df (DataFrame): Loaded annotation table as a DataFrame.
    """
    file_extension = os.path.splitext(annotation_table_path)[-1].lower()

    # Load based on file extension
    if file_extension == '.tsv':
        return pd.read_csv(annotation_table_path, sep='\t')
    elif file_extension == '.csv':
        return pd.read_csv(annotation_table_path)
    else:
        # Check content to guess the delimiter
        with open(annotation_table_path, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                return pd.read_csv(annotation_table_path, sep='\t')
            elif ',' in first_line:
                return pd.read_csv(annotation_table_path)
            else:
                raise ValueError("Unsupported file format and unable to detect delimiter. Please provide a valid CSV or TSV file.")

def get_predictive_features(feature_file_path, sample_column='strain', phenotype_column='interaction', feature_type='strain'):
    """
    Loads the feature table from a CSV file and extracts the predictive features, excluding the specified sample and phenotype columns.

    Args:
        feature_file_path (str): Path to the CSV file containing feature data.
        sample_column (str): The name of the column representing the sample identifier.
        phenotype_column (str): The name of the column representing the phenotype or target variable.

    Returns:
        predictive_features (list): A list of predictive feature names excluding the sample and phenotype columns.
    """
    # Load the feature table (CSV or parquet)
    feature_df = _read_table_file(feature_file_path)

    # Identify predictive feature groups
    # Only consider columns that match cluster naming pattern: <single_char>c_<number>
    # Pattern examples: 'sc_0', 'pc_1', 'bc_2' (source[0] + 'c_' + cluster_index)
    # This filters out metadata columns that don't follow the cluster naming convention
    all_columns = [x for x in feature_df.columns if x not in {sample_column, phenotype_column, 'phage', 'strain'}]

    # Filter to only valid cluster feature columns
    # Must match pattern: single lowercase letter + 'c_' (e.g., 'sc_', 'pc_', 'bc_')
    import re
    cluster_pattern = re.compile(r'^[a-z]c_')
    predictive_features = [x for x in all_columns if cluster_pattern.match(x)]

    if not predictive_features:
        logging.error(f"No cluster feature columns found (expected format: 'sc_*' or 'pc_*').")
        logging.info(f"Available columns after excluding metadata: {all_columns}")
        logging.info(f"Note: Cluster features should contain 'c_' (e.g., 'sc_cluster1', 'pc_cluster2')")
        return []

    # Extract feature group prefixes (e.g., 'sc', 'pc')
    predictive_feature_groups = list(set(x.split('c_')[0] + 'c' for x in predictive_features))
    logging.info(f"Predictive feature groups detected: {predictive_feature_groups}")

    # Log any columns that were filtered out as non-cluster columns
    non_cluster_columns = [x for x in all_columns if x not in predictive_features]
    if non_cluster_columns:
        logging.info(f"Non-cluster columns (excluded from feature analysis): {non_cluster_columns}")

    # Check the number of detected groups and classify features
    if len(predictive_feature_groups) > 2:
        logging.warning(f"More than two feature groups detected: {predictive_feature_groups}. Filtering to standard 'sc' and 'pc' groups only.")
        # When >2 groups detected, fall back to standard sc/pc patterns
        strain_features = [x for x in predictive_features if x.startswith('sc_')]
        phage_features = [x for x in predictive_features if x.startswith('pc_')]

        if not strain_features and not phage_features:
            logging.error("No standard 'sc_' or 'pc_' prefixed features found among detected groups.")
            logging.info(f"Detected feature groups: {predictive_feature_groups}")
            logging.info(f"Sample features: {predictive_features[:10]}")
            strain_features = []
            phage_features = []
    elif len(predictive_feature_groups) == 2:
        if 'sc' in predictive_feature_groups and 'pc' in predictive_feature_groups:
            logging.info("Strain features with 'sc' ID detected.")
            strain_features = [x for x in predictive_features if 'sc_' in x]
            phage_features = [x for x in predictive_features if 'pc_' in x]
        elif 'sc' in predictive_feature_groups:
            logging.info("Only strain features with 'sc' ID detected.")
            strain_features = [x for x in predictive_features if 'sc_' in x]
            phage_group = [x for x in predictive_feature_groups if x != 'sc'][0]
            logging.info(f"Phage group detected: {phage_group}")
            phage_features = [x for x in predictive_features if phage_group in x]
        elif 'pc' in predictive_feature_groups:
            logging.info("Only phage features with 'pc' ID detected.")
            phage_features = [x for x in predictive_features if 'pc_' in x]
            strain_group = [x for x in predictive_feature_groups if x != 'pc'][0]
            logging.info(f"Strain group detected: {strain_group}")
            strain_features = [x for x in predictive_features if strain_group in x]
        else:
            logging.error("No valid feature groups detected.")
            strain_features = []
            phage_features = []
    elif len(predictive_feature_groups) == 1:
        if 'sc' in predictive_feature_groups:
            logging.info("Only strain features with 'sc' ID detected.")
            strain_features = [x for x in predictive_features if 'sc_' in x]
            phage_features = []
        elif 'pc' in predictive_feature_groups:
            logging.info("Only phage features with 'pc' ID detected.")
            phage_features = [x for x in predictive_features if 'pc_' in x]
            strain_features = []
        else:
            logging.error("No valid feature groups detected.")
            strain_features = []
            phage_features = []
    else:
        logging.error("No valid feature groups detected.")
        strain_features = []
        phage_features = []

    if feature_type == 'strain':
        predictive_features = strain_features
    elif feature_type == 'phage':
        predictive_features = phage_features
    else:
        predictive_features = predictive_features

    logging.info(f"Loaded {len(predictive_features)} predictive total features.")
    logging.info(f"Loaded {len(strain_features)} predictive strain features.")
    logging.info(f"Loaded {len(phage_features)} predictive phage features.")

    return predictive_features

def get_predictive_proteins(select_features, feature2cluster_path, cluster2protein_path):
    """
    Retrieves predictive proteins based on selected features from the feature and cluster mappings.

    Args:
        select_features (list): List of selected feature names.
        feature2cluster_path (str): Path to the file containing the mapping of features to clusters.
        cluster2protein_path (str): Path to the file containing the mapping of clusters to proteins.

    Returns:
        filtered_proteins (DataFrame): DataFrame with predictive proteins and their associated clusters.
    """
    logging.info("Loading predictive proteins based on selected features.")

    # Load mappings (feature2cluster can be parquet or CSV)
    feature2cluster_df = _read_table_file(feature2cluster_path)
    if 'Cluster_Label' in feature2cluster_df.columns:
        feature2cluster_df = feature2cluster_df.rename(columns={'Cluster_Label': 'cluster'})
    elif 'cluster' not in feature2cluster_df.columns:
        feature2cluster_df.columns = ['Feature', 'cluster']
    cluster2protein_df = pd.read_csv(cluster2protein_path, sep='\t', names=['cluster', 'protein_ID'])

    # Ensure compatibility between protein IDs in the cluster file
    # if '|' in cluster2protein_df['protein_ID'].iloc[0]:
    #     cluster2protein_df['protein_ID'] = cluster2protein_df['protein_ID'].str.split('|').str[0]

    filtered_feature2cluster_df = feature2cluster_df[feature2cluster_df['Feature'].isin(select_features)]
    filtered_proteins = filtered_feature2cluster_df.merge(cluster2protein_df, on='cluster', how='left')

    # Count unique clusters per feature
    filtered_proteins['unique_clusters'] = filtered_proteins.groupby('Feature')['cluster'].transform('nunique')

    # Count unique proteins per feature
    filtered_proteins['unique_proteins'] = filtered_proteins.groupby('Feature')['protein_ID'].transform('nunique')

    logging.info(f"Retrieved {filtered_proteins.shape[0]} predictive proteins.")
    return filtered_proteins

def output_predictive_feature_overview(predictive_proteins, feature_assignments_df, genome_protein_df, strain_column='strain', output_dir='.'):
    """
    Generates an overview of predictive features, including Feature, cluster, protein_ID, and strain.

    Args:
        predictive_proteins (DataFrame): DataFrame with predictive proteins, including clusters and protein IDs.
        feature_assignments_df (DataFrame): DataFrame containing 'Feature' and strain information.
        genome_protein_df (DataFrame): DataFrame mapping protein_IDs to genomes.
        strain_column (str): Column name in feature_assignments_df and genome_protein_df for strain.
        output_dir (str): Directory where the output CSV will be saved.

    Returns:
        overview_df (DataFrame): DataFrame with the merged overview of predictive features.
    """
    logging.info("Generating predictive feature overview.")

    # Verify required columns in feature_assignments_df
    if 'Feature' not in feature_assignments_df.columns:
        raise ValueError(f"'Feature' column not found in feature_assignments_df.")

    # Ensure strain_column is present, or rename "Genome" if available
    if strain_column not in feature_assignments_df.columns:
        if 'Genome' in feature_assignments_df.columns:
            feature_assignments_df.rename(columns={'Genome': strain_column}, inplace=True)
            logging.info(f"'Genome' column found and renamed to '{strain_column}'.")
        else:
            raise ValueError(f"'{strain_column}' column not found in feature_assignments_df or genome_protein_df.")

    # Merge predictive_proteins with genome_protein_df on 'protein_ID'
    predictive_proteins = predictive_proteins.merge(genome_protein_df, on='protein_ID', how='inner')
    predictive_proteins[strain_column] = predictive_proteins[strain_column].astype(str)
    feature_assignments_df[strain_column] = feature_assignments_df[strain_column].astype(str)

    # Merge predictive_proteins with the feature assignments on 'Feature' and strain
    overview_df = feature_assignments_df.merge(predictive_proteins, on=[strain_column, 'Feature'], how='inner')
    overview_df = overview_df[[strain_column, 'Feature', 'cluster', 'protein_ID', 'unique_clusters', 'unique_proteins']]

    # Output the merged overview to a CSV file
    output_file_path = os.path.join(output_dir, f'{strain_column}_predictive_feature_overview.csv')
    overview_df.to_csv(output_file_path, index=False)
    
    logging.info(f"Saved predictive feature overview to {output_file_path} with {len(overview_df)} entries.")
    
    return overview_df

def parse_feature_information(modeling_dir, output_dir="."):
    """
    Parses feature importance and SHAP importance information from modeling results across multiple runs.

    Args:
        modeling_dir (str): Path to the directory containing modeling runs (e.g., 'run_0', 'run_1', etc.).
        output_dir (str): Path to the directory where the final feature importance CSV will be saved.

    Returns:
        full_feature_importance_df (DataFrame): DataFrame containing feature importance metrics including:
            - Importance: Mean CatBoost feature importance across runs
            - SHAP_mean: Mean of absolute SHAP values
            - SHAP_median: Median of absolute SHAP values
            - SHAP_median_present: Median SHAP value when feature is present (value > 0)
            - SHAP_median_absent: Median SHAP value when feature is absent (value = 0)
            - SHAP_importance: Legacy column (same as SHAP_mean) for backward compatibility
    """
    logging.info("Parsing feature importance and SHAP values from modeling results.")

    full_feature_importance_path = os.path.join(output_dir, "full_feature_importances.csv")
    if os.path.exists(full_feature_importance_path):
        logging.info(f"Feature importance file already exists at {full_feature_importance_path}.")
        return pd.read_csv(full_feature_importance_path)
    else:
        logging.info(f"Feature importance file not found at {full_feature_importance_path}. Parsing feature importance data.")

        run_dirs = os.listdir(modeling_dir)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.startswith("run_")]

        feature_importance_df = pd.DataFrame()
        shap_importance_df = pd.DataFrame()
        for run_dir in run_dirs:
            feature_importance_file = os.path.join(
                modeling_dir, run_dir, "feature_importances.csv"
            )
            if os.path.exists(feature_importance_file):
                feature_importance_run = pd.read_csv(feature_importance_file)
                feature_importance_run["run"] = run_dir
                feature_importance_df = pd.concat([feature_importance_df, feature_importance_run], ignore_index=True)
            else:
                logging.warning(f"Feature importance file not found in {run_dir}")

            shap_importance_file = os.path.join(
                modeling_dir, run_dir, "shap_importances.csv"
            )
            if os.path.exists(shap_importance_file):
                shap_importance_run = pd.read_csv(shap_importance_file)
                shap_importance_run = shap_importance_run.rename(columns={"feature": "Feature"})
                shap_importance_run["run"] = run_dir
                shap_importance_df = pd.concat([shap_importance_df, shap_importance_run], ignore_index=True)
            else:
                logging.warning(f"SHAP importance file not found in {run_dir}")

        if not feature_importance_df.empty:
            feature_importance_df = feature_importance_df.groupby("Feature")[["Importance"]].mean().reset_index()
        else:
            logging.error("No feature importance data found.")
            return pd.DataFrame()

        if not shap_importance_df.empty:
            # Calculate absolute SHAP values for mean and median
            shap_importance_df['SHAP_abs'] = np.abs(shap_importance_df['shap_value'])

            # Calculate mean and median of absolute SHAP values
            shap_agg_overall = shap_importance_df.groupby("Feature").agg({
                'SHAP_abs': ['mean', 'median']
            }).reset_index()
            shap_agg_overall.columns = ['Feature', 'SHAP_mean', 'SHAP_median']

            # Calculate median SHAP for present (value=1) and absent (value=0)
            # First, ensure 'value' column exists
            if 'value' in shap_importance_df.columns:
                # For binary features, separate by presence/absence
                # Handle cases where some features may not have both present and absent values
                shap_present = shap_importance_df[shap_importance_df['value'] > 0].groupby("Feature", as_index=False)['shap_value'].median()
                shap_present.columns = ['Feature', 'SHAP_median_present']

                shap_absent = shap_importance_df[shap_importance_df['value'] == 0].groupby("Feature", as_index=False)['shap_value'].median()
                shap_absent.columns = ['Feature', 'SHAP_median_absent']

                # Merge all SHAP metrics (left join to preserve all features)
                shap_importance_df = shap_agg_overall.merge(shap_present, on='Feature', how='left')
                shap_importance_df = shap_importance_df.merge(shap_absent, on='Feature', how='left')

                # Log if any features are missing present/absent values
                n_missing_present = shap_importance_df['SHAP_median_present'].isna().sum()
                n_missing_absent = shap_importance_df['SHAP_median_absent'].isna().sum()
                if n_missing_present > 0:
                    logging.warning(f"{n_missing_present} features have no 'present' (value > 0) samples.")
                if n_missing_absent > 0:
                    logging.warning(f"{n_missing_absent} features have no 'absent' (value = 0) samples.")
            else:
                # If no 'value' column, just use overall aggregations
                logging.warning("'value' column not found in SHAP data. Only calculating overall SHAP metrics.")
                shap_importance_df = shap_agg_overall

            # Keep legacy SHAP_importance column for backward compatibility (same as SHAP_mean)
            shap_importance_df['SHAP_importance'] = shap_importance_df['SHAP_mean']
        else:
            logging.error("No SHAP importance data found.")

        if not shap_importance_df.empty:
            full_feature_importance_df = pd.merge(
                feature_importance_df, shap_importance_df, on="Feature", how="inner"
            )
        else:
            full_feature_importance_df = feature_importance_df

        full_feature_importance_df = full_feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        full_feature_importance_df.to_csv(full_feature_importance_path, index=False)
        logging.info(f"Saved full feature importances to {full_feature_importance_path}.")

        logging.info("Parsed and combined feature importance data.")
        return full_feature_importance_df

def merge_importance_table(predictive_proteins, feature_importance_df, output_dir = '.', protein_id_col="protein_ID", file_type='check', prefix='strain'):
    """
    Merges the predictive proteins DataFrame with an annotation table based on protein IDs and combines it with feature importance data.

    Args:
        predictive_proteins (DataFrame): DataFrame with predictive proteins and protein IDs.
        feature_importance_df (DataFrame): DataFrame containing feature importance and SHAP importance.
        protein_id_col (str): Column name for protein IDs in the predictive_proteins DataFrame (default: 'protein_ID').

    Returns:
        merged_df (DataFrame): DataFrame with merged annotation and feature importance information.
    """
    logging.info(f"Merging predictive proteins with feature importance data.")

    # Ensure necessary columns are present
    if protein_id_col not in predictive_proteins.columns:
        raise ValueError(f"Column '{protein_id_col}' not found in predictive_proteins DataFrame.")

    # Merge feature importance with predictive proteins
    merged_df = feature_importance_df.merge(predictive_proteins, on='Feature', how='inner')

    predictive_protein_info_path = os.path.join(output_dir, f'{prefix}_predictive_protein_info.csv')
    merged_df.to_csv(predictive_protein_info_path, index=False)
    logging.info(f"Saved predictive proteins overview to {predictive_protein_info_path}.")

    return merged_df

def merge_annotation_table(annotation_table_path, merged_df, output_dir = '.', protein_id_col="protein_ID", file_type='check', prefix='strain'):
    """
    Merges the predictive proteins DataFrame with an annotation table based on protein IDs and combines it with feature importance data.

    Args:
        annotation_table_path (str): Path to the annotation table (CSV/TSV format).
        predictive_proteins (DataFrame): DataFrame with predictive proteins and protein IDs.
        feature_importance_df (DataFrame): DataFrame containing feature importance and SHAP importance.
        protein_id_col (str): Column name for protein IDs in the predictive_proteins DataFrame (default: 'protein_ID').

    Returns:
        merged_df (DataFrame): DataFrame with merged annotation and feature importance information.
    """
    logging.info(f"Merging predictive proteins with annotation table from {annotation_table_path}.")

    # Load the annotation table using the helper function
    annotation_df = load_annotation_table(annotation_table_path)

    merged_df_annotations = merged_df.merge(annotation_df, left_on='protein_ID', right_on=protein_id_col, how='inner')

    prediction_protein_annotations_path = os.path.join(output_dir, f'{prefix}_predictive_protein_annotations.csv')
    merged_df_annotations.to_csv(prediction_protein_annotations_path, index=False)
    logging.info(f"Saved predictive protein annotations to {prediction_protein_annotations_path}.")

    logging.info(f"Merged {merged_df_annotations.shape[0]} rows with annotation information.")

def parse_and_filter_aa_sequences(fasta_dir_or_file, filtered_proteins, output_dir, protein_id_col="protein_ID", output_fasta="predictive_AA_seqs.faa", prefix='strain'):
    """
    Parses and filters AA sequences from either a single FASTA file or multiple files in a directory.
    Outputs the filtered sequences into a new FASTA file and saves genome-protein mappings to a CSV file.

    Args:
        fasta_dir_or_file (str): Path to a FASTA file or directory containing FASTA files.
        filtered_proteins (DataFrame): DataFrame containing predictive protein IDs.
        output_dir (str): Directory to save outputs.
        protein_id_col (str): Column name in filtered_proteins DataFrame containing protein IDs.
        output_fasta (str): Name of the output FASTA file for filtered sequences.
        prefix (str): Prefix for naming output files based on feature type, e.g., 'strain' or 'phage'.

    Raises:
        FileNotFoundError: If the specified file or directory does not exist.
    """
    logging.info("Starting to parse and filter AA sequences.")

    # Extract predictive protein IDs
    predictive_protein_ids = set(filtered_proteins[protein_id_col])
    logging.info(f"Filtering for {len(predictive_protein_ids)} protein IDs.")

    # Initialize lists for storing filtered sequences and genome-protein mappings
    filtered_seqs = []
    genome_protein_df = pd.DataFrame()

    # Determine if input is a directory or a single file
    if os.path.isdir(fasta_dir_or_file):
        fasta_files = [os.path.join(fasta_dir_or_file, f) for f in os.listdir(fasta_dir_or_file) if f.endswith('.faa')]
        input_type = 'directory'
        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in directory {fasta_dir_or_file}")
    else:
        input_type = 'file'
        if not os.path.exists(fasta_dir_or_file):
            raise FileNotFoundError(f"FASTA file {fasta_dir_or_file} does not exist")
        fasta_files = [fasta_dir_or_file]

    # Parse and filter sequences
    for fasta_file in fasta_files:
        logging.info(f"Parsing {fasta_file}.")
        filtered_ids = []
        if input_type == 'directory':
            genome_id = '.'.join(os.path.basename(fasta_file).split('.')[:-1])
            logging.info(f"Detected genome ID: {genome_id} from {fasta_file}")
        else:
            genome_id = None

        for record in SeqIO.parse(fasta_file, "fasta"):
            protein_id = record.id  # Assuming protein ID is directly in record.id
            if protein_id in predictive_protein_ids:
                filtered_seqs.append(record)
                filtered_ids.append(protein_id)

        if not filtered_ids:
            logging.info(f"No matching sequences found in {fasta_file}.")
        
        protein_ids_df_temp = pd.DataFrame(filtered_ids, columns=[protein_id_col])
        if input_type == 'directory':
            protein_ids_df_temp[prefix] = genome_id
        else:
            protein_ids_df_temp[prefix] = protein_ids_df_temp[protein_id_col].str.split('_').str[:-1].str.join('_')
        genome_protein_df = pd.concat([genome_protein_df, protein_ids_df_temp])

    # Save the genome-protein mapping to a CSV file
    genome_protein_ids_path = os.path.join(output_dir, f'{prefix}_protein_ids.csv')
    genome_protein_df = genome_protein_df[[prefix, protein_id_col]]
    genome_protein_df.to_csv(genome_protein_ids_path, index=False)
    logging.info(f"Genome-protein ID map saved to {genome_protein_ids_path}")

    # Write filtered sequences to output FASTA file
    output_fasta = f"{prefix}_{output_fasta}"
    output_fasta_path = os.path.join(output_dir, output_fasta)
    with open(output_fasta_path, "w") as output_handle:
        SeqIO.write(filtered_seqs, output_handle, "fasta")

    logging.info(f"Filtered AA sequences saved to {output_fasta_path}.")

    return genome_protein_df
