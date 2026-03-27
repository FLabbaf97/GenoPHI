import os
import pandas as pd
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from genophi.kmer_modeling_analysis import (
    load_aa_sequences,
    get_predictive_kmers,
    merge_kmers_with_families,
    cluster_sequences_mmseqs,
    align_sequences,
    find_kmer_indices,
    calculate_coverage,
    identify_segments,
    plot_segments,
    aggregate_shap_values
)

def kmer_analysis_workflow(
    aa_sequence_file,
    feature_file_path,
    feature2cluster_path,
    protein_families_file,
    output_dir,
    feature_type='strain',
    annotation_file=None,
    model_output_dir=None,
    quick_run=False,
    ignore_families=False,
    genome_mapping_file=None
):
    """
    Workflow that supports both standard protein family alignment 
    AND 'ignore_families' mode which uses MMseqs2 clustering.
    """
    logging.info("Starting k-mer analysis workflow...")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    type_output_dir = os.path.join(output_dir, feature_type)
    if not os.path.exists(type_output_dir): os.makedirs(type_output_dir)

    # 1. Load Sequences
    aa_sequences_df = load_aa_sequences(aa_sequence_file)
    # SAVE OUTPUT
    aa_sequences_df.to_csv(os.path.join(type_output_dir, 'aa_sequences_df.csv'), index=False)

    # 2. Extract K-mers
    filtered_kmers = get_predictive_kmers(feature_file_path, feature2cluster_path, feature_type, ignore_families)
    if filtered_kmers.empty: return
    # SAVE OUTPUT
    filtered_kmers.to_csv(os.path.join(type_output_dir, 'filtered_kmers.csv'), index=False)

    # 3. Merge / Search
    full_df = merge_kmers_with_families(
        protein_families_file, 
        aa_sequences_df, 
        feature_type, 
        ignore_families=ignore_families, 
        filtered_kmers=filtered_kmers
    )
    if full_df.empty: return

    # 3.5. Ensure Feature/Kmer Data is present in Standard Mode
    if not ignore_families:
        logging.info("Standard Mode: Merging predictive K-mers into protein dataframe...")
        full_df = full_df.merge(
            filtered_kmers[['protein_family', 'kmer', 'Feature']], 
            on='protein_family', 
            how='inner'
        )
        if full_df.empty:
            logging.warning("No intersection between protein families and predictive k-mers.")
            return
            
    # SAVE OUTPUT (This corresponds to protein_families_df.csv)
    full_df.to_csv(os.path.join(type_output_dir, 'protein_families_df.csv'), index=False)

    # Save protein sequences as FASTA (Useful for external tools)
    logging.info("Saving protein sequences to FASTA...")
    protein_seqs_df = full_df[['protein_ID', 'sequence']].drop_duplicates()
    seqrecords = [SeqRecord(Seq(row['sequence']), id=row['protein_ID'], description='') for _, row in protein_seqs_df.iterrows()]
    SeqIO.write(seqrecords, os.path.join(type_output_dir, f'{feature_type}_protein_sequences.faa'), 'fasta')

    # 4. Define Groups (Clustering vs Family)
    if ignore_families:
        # CLUSTERING MODE
        logging.info("ignore_families=True: Running MMseqs2 clustering...")
        clusters_df = cluster_sequences_mmseqs(full_df, type_output_dir)
        
        if clusters_df.empty:
            logging.error("Clustering failed.")
            return
        
        # SAVE OUTPUT
        clusters_df.to_csv(os.path.join(type_output_dir, 'mmseqs_clusters.csv'), index=False)
            
        # Merge cluster IDs back
        full_df = full_df.merge(clusters_df, on='protein_ID', how='inner')
        grouping_col = 'cluster_id'
        
    else:
        # STANDARD MODE
        logging.info("Standard Mode: Grouping by original Protein Family.")
        grouping_col = 'protein_family'
        full_df['cluster_id'] = full_df['protein_family']

    if quick_run: 
        logging.info("Quick run selected. Skipping alignment and coverage calculation.")
        return

    # 5. Alignment
    logging.info(f"Aligning sequences grouped by {grouping_col}...")
    aligned_dfs = []
    
    for group_id, group_data in full_df.groupby(grouping_col):
        seqs_for_group = group_data[['protein_ID', 'sequence']].drop_duplicates()
        seq_tuples = [(row['protein_ID'], row['sequence']) for _, row in seqs_for_group.iterrows()]
        
        if len(seq_tuples) < 2:
            continue
            
        aln_df = align_sequences(seq_tuples, type_output_dir, f"group_{group_id}")
        
        if not aln_df.empty:
            merged_aln = aln_df.merge(
                group_data[['protein_ID', 'cluster_id', 'kmer', 'Feature']].drop_duplicates(),
                on='protein_ID',
                how='inner'
            )
            aligned_dfs.append(merged_aln)

    if not aligned_dfs:
        logging.warning("No successful alignments.")
        return

    final_aligned_df = pd.concat(aligned_dfs, ignore_index=True)
    
    # 6. Indices & Coverage
    logging.info("Finding k-mer indices in aligned sequences...")
    final_aligned_df[['start_indices', 'stop_indices']] = final_aligned_df.apply(find_kmer_indices, axis=1)
    
    # SAVE OUTPUT
    final_aligned_df.to_csv(os.path.join(type_output_dir, 'aligned_df.csv'), index=False)
    
    coverage_df = calculate_coverage(final_aligned_df)
    
    # 7. Segments & Plotting
    segments_df = identify_segments(coverage_df)

    # SAVE OUTPUT
    segments_df.to_csv(os.path.join(type_output_dir, 'segments_df.csv'), index=False)

    plot_dir = os.path.join(type_output_dir, 'plots')
    plot_segments(segments_df, plot_dir)

    # 7.5. Extract segment sequences for covered segments
    logging.info("Extracting segment sequences for covered segments...")
    covered_segments = segments_df[segments_df['segment_type'] == 1].copy()

    if not covered_segments.empty:
        # Merge with aligned sequences
        covered_segments = covered_segments.merge(
            final_aligned_df[['protein_ID', 'aln_sequence']].drop_duplicates(),
            on='protein_ID',
            how='left'
        )

        # Extract segment sequences (with and without gaps)
        def extract_segment_seq(row):
            if pd.isna(row['aln_sequence']):
                return pd.Series(['', '', 0, 0, 0, 0])

            aln_seq = row['aln_sequence']
            start_aln = row['start']
            stop_aln = row['stop']

            # Extract segment with gaps
            segment_with_gaps = aln_seq[start_aln:stop_aln+1]
            segment_no_gaps = segment_with_gaps.replace('-', '')
            segment_length = stop_aln - start_aln + 1
            aa_count = len(segment_no_gaps)

            # Calculate ungapped positions (positions in the original gene, not alignment)
            # Count non-gap characters before the start and stop positions
            start_ungapped = sum(1 for c in aln_seq[:start_aln] if c != '-')
            stop_ungapped = sum(1 for c in aln_seq[:stop_aln+1] if c != '-') - 1

            return pd.Series([segment_with_gaps, segment_no_gaps, segment_length, aa_count, start_ungapped, stop_ungapped])

        covered_segments[['segment_sequence', 'segment_sequence_nogaps', 'segment_length', 'aa_count', 'start_ungapped', 'stop_ungapped']] = \
            covered_segments.apply(extract_segment_seq, axis=1)

        # Add full ungapped protein sequences
        covered_segments = covered_segments.merge(
            aa_sequences_df[['protein_ID', 'sequence']].drop_duplicates().rename(columns={'sequence': 'full_sequence'}),
            on='protein_ID',
            how='left'
        )

        # Add genome/strain information (if available)
        genome_col_added = False

        if feature_type in full_df.columns:
            # Standard mode: feature_type already in full_df (from protein_families_file)
            covered_segments = covered_segments.merge(
                full_df[['protein_ID', feature_type]].drop_duplicates(),
                on='protein_ID',
                how='left'
            )
            covered_segments = covered_segments.rename(columns={feature_type: 'genome'})
            genome_col_added = True

        elif genome_mapping_file and os.path.exists(genome_mapping_file):
            # Ignore_families mode: load from separate genome_mapping_file
            logging.info(f"Loading genome/strain mapping from {genome_mapping_file}")
            try:
                mapping_df = pd.read_csv(genome_mapping_file)

                # Look for genome column with flexible naming
                possible_cols = ['phage', 'genome', 'strain', 'species', 'genus', feature_type]
                found_col = None
                for col in possible_cols:
                    if col in mapping_df.columns:
                        found_col = col
                        break

                if found_col and 'protein_ID' in mapping_df.columns:
                    genome_mapping = mapping_df[['protein_ID', found_col]].drop_duplicates()
                    covered_segments = covered_segments.merge(genome_mapping, on='protein_ID', how='left')
                    covered_segments = covered_segments.rename(columns={found_col: 'genome'})
                    genome_col_added = True
                    logging.info(f"Added genome information from column '{found_col}'")
                else:
                    logging.warning(f"No suitable genome column found in {genome_mapping_file}. Skipping.")
            except Exception as e:
                logging.warning(f"Failed to load genome mapping: {e}")

        else:
            logging.info(f"No genome/strain information available. Skipping this column in output.")

        # Select and reorder columns (conditionally include 'genome' if available)
        if genome_col_added:
            output_cols = ['Feature', 'genome', 'protein_ID', 'cluster_id',
                           'segment_sequence_nogaps', 'segment_sequence',
                           'start_ungapped', 'stop_ungapped', 'aa_count',
                           'full_sequence',
                           'start', 'stop', 'segment_length']
            sort_cols = ['Feature', 'genome', 'protein_ID', 'start_ungapped']
        else:
            output_cols = ['Feature', 'protein_ID', 'cluster_id',
                           'segment_sequence_nogaps', 'segment_sequence',
                           'start_ungapped', 'stop_ungapped', 'aa_count',
                           'full_sequence',
                           'start', 'stop', 'segment_length']
            sort_cols = ['Feature', 'protein_ID', 'start_ungapped']

        output_cols = [col for col in output_cols if col in covered_segments.columns]
        covered_segments_output = covered_segments[output_cols].copy()

        # Drop duplicates
        n_before = len(covered_segments_output)
        covered_segments_output = covered_segments_output.drop_duplicates()
        n_after = len(covered_segments_output)
        if n_before > n_after:
            logging.info(f"Removed {n_before - n_after} duplicate covered segments")

        # Sort for readability
        covered_segments_output = covered_segments_output.sort_values(sort_cols)

        # SAVE OUTPUT
        covered_segments_output.to_csv(
            os.path.join(type_output_dir, 'covered_segments_with_sequences.csv'),
            index=False
        )
        logging.info(f"Saved {len(covered_segments_output)} covered segments with sequences")
    else:
        logging.warning("No covered segments found to extract sequences from")

    # 8. Optional SHAP
    if model_output_dir:
        shap_df = aggregate_shap_values(model_output_dir)
        if not shap_df.empty:
            shap_df.to_csv(os.path.join(output_dir, 'full_SHAP_values.csv'), index=False)

    logging.info("Workflow complete.")