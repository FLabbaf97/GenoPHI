import os
import pandas as pd
import argparse
import logging
import time
import psutil
from genophi.mmseqs2_clustering import run_clustering_workflow, run_feature_assignment

def setup_logging(output_dir, log_filename="extract_features_workflow.log"):
    """
    Set up logging to both console and file if logging is not already configured.

    Args:
        output_dir (str): Directory where the log file will be saved.
        log_filename (str): Name of the log file. Default is "extract_features_workflow.log".
    """
    if not logging.getLogger().hasHandlers():
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, log_filename)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),  # Overwrite log file
                logging.StreamHandler()
            ]
        )
        logging.info("Logging initialized. Logs will be written to: %s", log_file)
    else:
        logging.info("Logging is already configured by the calling workflow.")

def write_report(output_dir, start_time, end_time, ram_usage, avg_cpu_usage, max_cpu_usage, 
                 input_genomes, protein_families, features, source_type):
    """
    Writes a detailed workflow report to a text file.

    Args:
        output_dir (str): Directory where the report file will be saved.
        start_time (float): Workflow start time.
        end_time (float): Workflow end time.
        ram_usage (int): Maximum RAM usage in bytes.
        avg_cpu_usage (float): Average CPU usage during workflow.
        max_cpu_usage (float): Maximum CPU usage during workflow.
        input_genomes (int): Number of input genomes.
        protein_families (int): Number of protein families identified.
        features (int): Number of features in the final table.
        source_type (str): Type of source ('strain', 'phage', etc.).
    """
    report_file = os.path.join(output_dir, "extract_features_report.txt")
    with open(report_file, "w") as report:
        report.write("Feature Extraction Workflow Report\n")
        report.write("=" * 50 + "\n")
        report.write(f"Source Type: {source_type}\n")
        report.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        report.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        report.write(f"Total Runtime: {end_time - start_time:.2f} seconds\n")
        report.write(f"Max RAM Usage: {ram_usage / (1024 ** 3):.2f} GB\n")
        report.write(f"Average CPU Usage: {avg_cpu_usage:.2f}%\n")
        report.write(f"Max CPU Usage: {max_cpu_usage:.2f}%\n")
        report.write(f"Input Genomes: {input_genomes}\n")
        report.write(f"Protein Families Identified: {protein_families}\n")
        report.write(f"Features in Final Table: {features}\n")
    logging.info(f"Report saved to: {report_file}")

def extract_features_workflow(
    input_path,
    output_dir,
    tmp_dir="tmp",
    source='strain',
    min_seq_id=0.6,
    coverage=0.8,
    sensitivity=7.5,
    suffix='faa',
    threads=4,
    genome_list='none',
    genome_column='strain',
    compare=False,
    max_ram=8,
    bootstrapping=False,
    clear_tmp=False,
    clustering_dir=None
):
    """
    Extract protein features from a folder containing .faa files.
    
    This workflow:
    1. Runs MMseqs2 clustering to identify protein families
    2. Assigns features to genomes
    3. Generates a feature table
    
    Does NOT require a phenotype matrix and does NOT run modeling.
    
    Args:
        input_path (str): Path to the input directory or file containing .faa files.
        output_dir (str): Directory to save all results.
        tmp_dir (str): Directory for temporary files (default: "tmp").
        source (str): Prefix for naming features (default: 'strain', use 'phage' for phage data).
        min_seq_id (float): Minimum sequence identity for clustering (default: 0.6).
        coverage (float): Minimum coverage for clustering (default: 0.8).
        sensitivity (float): Sensitivity for clustering (default: 7.5).
        suffix (str): Suffix for input FASTA files (default: 'faa').
        threads (int): Number of threads to use (default: 4).
        genome_list (str): Path to genome list file for filtering or 'none' (default: 'none').
        genome_column (str): Column name for genome identifiers (default: 'strain').
        compare (bool): Whether to compare original clusters with assigned clusters (default: False).
        max_ram (int): Maximum RAM to use in GB (default: 8).
        bootstrapping (bool): Whether to use bootstrapping (default: False).
        clear_tmp (bool): Whether to clear temporary files after each step (default: False).
        clustering_dir (str, optional): Path to existing clustering results to reuse.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    # Track time and resource usage
    start_time = time.time()
    ram_monitor = psutil.Process()
    cpu_usage_points = []
    max_ram_usage = 0

    # Initialize counters for report
    input_genomes = protein_families = features = 0

    try:
        logging.info(f"Step 1: Running clustering workflow for {source} genomes...")
        
        if clustering_dir:
            # Use existing clustering results via symlink
            old_clustering_dir = os.path.abspath(clustering_dir)
            output_clustering_dir = os.path.abspath(output_dir)
            
            logging.info(f"Using existing clustering results from: {old_clustering_dir}")
            
            if not os.path.exists(output_clustering_dir):
                os.symlink(old_clustering_dir, output_clustering_dir, target_is_directory=True)
            
            # Verify symlink was created and is valid
            logging.info(f"Checking symlink exists: {os.path.exists(output_clustering_dir)}")
            
            old_tmp_dir = os.path.abspath(os.path.join(clustering_dir, "..", "tmp", source))
            tmp_output_dir = os.path.abspath(os.path.join(output_dir, "..", "tmp", source))
            if os.path.exists(old_tmp_dir) and not os.path.exists(tmp_output_dir):
                os.makedirs(os.path.dirname(tmp_output_dir), exist_ok=True)
                os.symlink(old_tmp_dir, tmp_output_dir, target_is_directory=True)
            
            features_path = os.path.join(output_dir, "features", "feature_table.csv")
            
        elif os.path.exists(os.path.join(output_dir, "features", "feature_table.csv")):
            logging.info(f"Using existing {source} clustering results...")
            features_path = os.path.join(output_dir, "features", "feature_table.csv")
            
        else:
            # Run normal clustering workflow
            tmp_output_dir = os.path.join(output_dir, tmp_dir, source)
            
            run_clustering_workflow(
                input_path, 
                output_dir, 
                tmp_output_dir, 
                min_seq_id, 
                coverage, 
                sensitivity, 
                suffix, 
                threads, 
                genome_list, 
                genome_column, 
                compare, 
                bootstrapping, 
                clear_tmp
            )
            
            features_path = os.path.join(output_dir, "features", "feature_table.csv")
        
        # Run feature assignment if feature table doesn't exist
        if not os.path.exists(features_path):
            logging.info(f"Step 2: Running feature assignment for {source} genomes...")
            presence_absence_matrix = os.path.join(output_dir, "presence_absence_matrix.csv")
            
            if not os.path.exists(presence_absence_matrix):
                raise FileNotFoundError(f"Presence-absence matrix not found at {presence_absence_matrix}")
            
            run_feature_assignment(
                input_file=presence_absence_matrix,
                output_dir=os.path.join(output_dir, "features"),
                source=source,
                select=genome_list,
                select_column=genome_column,
                max_ram=max_ram,
                threads=threads
            )
        
        # Count genomes and protein families
        presence_absence_matrix = os.path.join(output_dir, "presence_absence_matrix.csv")
        if os.path.exists(presence_absence_matrix):
            matrix_df = pd.read_csv(presence_absence_matrix)
            input_genomes = len(matrix_df['Genome'].unique())
            protein_families = len(matrix_df.columns) - 1  # Exclude 'Genome' column
        
        # Count features
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            features = len(features_df.columns) - 2  # Exclude genome column and feature column
        
        logging.info(f"Feature extraction completed!")
        logging.info(f"  - Input genomes: {input_genomes}")
        logging.info(f"  - Protein families: {protein_families}")
        logging.info(f"  - Features: {features}")
        logging.info(f"  - Feature table saved to: {features_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        end_time = time.time()
        max_ram_usage = max(max_ram_usage, ram_monitor.memory_info().rss)
        cpu_usage_points.append(psutil.cpu_percent(interval=None))

        avg_cpu_usage = sum(cpu_usage_points) / len(cpu_usage_points) if cpu_usage_points else 0
        max_cpu_usage = max(cpu_usage_points) if cpu_usage_points else 0

        # Write report
        write_report(
            output_dir, 
            start_time, 
            end_time, 
            max_ram_usage, 
            avg_cpu_usage, 
            max_cpu_usage, 
            input_genomes, 
            protein_families, 
            features,
            source
        )


# Main function for CLI
def main():
    parser = argparse.ArgumentParser(
        description='Extract protein features from a folder containing .faa files. '
                    'Runs clustering and feature assignment without requiring a phenotype matrix or modeling.'
    )
    
    # Input data
    input_group = parser.add_argument_group('Input data')
    input_group.add_argument('-i', '--input', type=str, required=True, 
                            help='Path to the input directory or file containing .faa files.')
    input_group.add_argument('-o', '--output', type=str, required=True, 
                            help='Output directory to save results.')
    
    # Optional input arguments
    optional_input_group = parser.add_argument_group('Optional input arguments')
    optional_input_group.add_argument('--clustering_dir', type=str, 
                                     help='Path to an existing clustering directory to reuse.')
    optional_input_group.add_argument('--source', type=str, default='strain', 
                                     help='Prefix for naming features (default: strain, use "phage" for phage data).')
    optional_input_group.add_argument('--suffix', type=str, default='faa', 
                                     help='Suffix for input FASTA files (default: faa).')
    optional_input_group.add_argument('--genome_list', type=str, default='none', 
                                     help='Path to a genome list file for filtering (default: none).')
    optional_input_group.add_argument('--genome_column', type=str, default='strain', 
                                     help='Column in the genome list containing genome names (default: strain).')
    
    # Clustering parameters
    clustering_group = parser.add_argument_group('Clustering')
    clustering_group.add_argument('--min_seq_id', type=float, default=0.6, 
                                 help='Minimum sequence identity for clustering (default: 0.6).')
    clustering_group.add_argument('--coverage', type=float, default=0.8, 
                                 help='Minimum coverage for clustering (default: 0.8).')
    clustering_group.add_argument('--sensitivity', type=float, default=7.5, 
                                 help='Sensitivity for clustering (default: 7.5).')
    clustering_group.add_argument('--compare', action='store_true', 
                                 help='Compare original clusters with assigned clusters.')
    
    # General parameters
    general_group = parser.add_argument_group('General')
    general_group.add_argument('--tmp', type=str, default="tmp", 
                              help='Temporary directory for intermediate files (default: tmp).')
    general_group.add_argument('--threads', type=int, default=4, 
                              help='Number of threads to use (default: 4).')
    general_group.add_argument('--max_ram', type=float, default=8, 
                              help='Maximum RAM usage in GB (default: 8).')
    general_group.add_argument('--bootstrapping', action='store_true', 
                              help='Use bootstrapping (default: False).')
    general_group.add_argument('--clear_tmp', action='store_true', 
                              help='Clear temporary files after each step (default: False).')
    
    args = parser.parse_args()

    # Run the feature extraction workflow
    extract_features_workflow(
        input_path=args.input,
        output_dir=args.output,
        tmp_dir=args.tmp,
        source=args.source,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        sensitivity=args.sensitivity,
        suffix=args.suffix,
        threads=args.threads,
        genome_list=args.genome_list,
        genome_column=args.genome_column,
        compare=args.compare,
        max_ram=args.max_ram,
        bootstrapping=args.bootstrapping,
        clear_tmp=args.clear_tmp,
        clustering_dir=args.clustering_dir
    )


if __name__ == "__main__":
    main()
