import os
import glob
import sqlite3
import csv

def get_matrices_with_reordering_results(results_dir="./results"):
    """
    Get all matrix names that have results containing the specified reordering prefixes.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        set: Set of matrix names that have results with the specified prefixes
    """
    # Reordering prefixes to look for
    reordering_prefixes = [
        "reordered_louvain_global_rcm_res0.7_", 
        "reordered_louvain_global_rcm_res1.0_", 
        "reordered_louvain_global_rcm_res1.3_", 
        "reordered_louvain_local_rcm_res1.3_",
        "reordered_louvain_local_rcm_res1.0_", 
        "reordered_louvain_local_rcm_res0.7_",
        "reordered_metis_local_rcm_", 
        "reordered_metis_global_rcm_", 
        "reordered_rcm_"
    ]
    
    matrices_with_results = set()
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return matrices_with_results
    
    # Iterate through each matrix directory
    for matrix_dir in os.listdir(results_dir):
        matrix_path = os.path.join(results_dir, matrix_dir)
        
        # Skip if not a directory
        if not os.path.isdir(matrix_path):
            continue
            
        matrix_name = matrix_dir
        
        # Check for any files with the specified prefixes
        for prefix in reordering_prefixes:
            # Look for files starting with the prefix and containing the matrix name
            pattern = os.path.join(matrix_path, f"{prefix}*{matrix_name}*")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                matrices_with_results.add(matrix_name)
                break  # Found at least one match, no need to check other prefixes
    
    return matrices_with_results

def get_complete_matrices():
    """Return matrices that have all 4 types of results (original + 3 reordering variants)."""
    results_dir = "./results"
    complete_matrices = []
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return complete_matrices
    
    # Get all matrix directories
    for matrix_dir in os.listdir(results_dir):
        matrix_path = os.path.join(results_dir, matrix_dir)
        
        # Skip if not a directory
        if not os.path.isdir(matrix_path):
            continue
            
        matrix_name = matrix_dir
        
        # Check for all 4 types
        original_pattern = os.path.join(matrix_path, f"{matrix_name}_sqlite")
        original_exists = os.path.exists(original_pattern)
        
        louvain_pattern = os.path.join(matrix_path, "reordered_louvain*_sqlite")
        louvain_files = glob.glob(louvain_pattern)
        louvain_exists = len(louvain_files) > 0
        
        metis_pattern = os.path.join(matrix_path, "reordered_metis*_sqlite")
        metis_files = glob.glob(metis_pattern)
        metis_exists = len(metis_files) > 0
        
        rcm_pattern = os.path.join(matrix_path, "reordered_rcm*_sqlite")
        rcm_files = glob.glob(rcm_pattern)
        rcm_exists = len(rcm_files) > 0
        
        # Only include if all 4 types exist
        if original_exists and louvain_exists and metis_exists and rcm_exists:
            complete_matrices.append(matrix_name)
    
    return sorted(complete_matrices)

def extract_nsys_metrics(sqlite_file):
    """Extract performance metrics from nsys SQLite file."""
    if not os.path.exists(sqlite_file):
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        
        # Initialize metrics dictionary
        metrics = {
            'avg_time_ms': 0,
            'memory_usage_mb': 0,
            'occupancy_pct': 0,
            'cache_hit_pct': 0,
            'utilization_pct': 0
        }
        
        # Query for kernel execution times
        time_query = """
        SELECT AVG(dur) as avg_duration_ns 
        FROM CUPTI_ACTIVITY_KIND_KERNEL 
        WHERE dur > 0
        """
        try:
            result = conn.execute(time_query).fetchone()
            if result and result[0]:
                metrics['avg_time_ms'] = result[0] / 1_000_000  # Convert ns to ms
        except:
            pass
        
        # Query for memory usage
        memory_query = """
        SELECT AVG(bytes) as avg_bytes 
        FROM CUPTI_ACTIVITY_KIND_MEMCPY 
        WHERE bytes > 0
        """
        try:
            result = conn.execute(memory_query).fetchone()
            if result and result[0]:
                metrics['memory_usage_mb'] = result[0] / (1024 * 1024)  # Convert to MB
        except:
            pass
        
        # Query for occupancy (theoretical vs achieved)
        occupancy_query = """
        SELECT AVG(CAST(achievedOccupancy AS FLOAT) / CAST(theoreticalOccupancy AS FLOAT) * 100) as avg_occupancy
        FROM CUPTI_ACTIVITY_KIND_KERNEL 
        WHERE theoreticalOccupancy > 0 AND achievedOccupancy >= 0
        """
        try:
            result = conn.execute(occupancy_query).fetchone()
            if result and result[0]:
                metrics['occupancy_pct'] = result[0]
        except:
            pass
        
        # Query for grid and block utilization
        util_query = """
        SELECT AVG(CAST(gridX * gridY * gridZ AS FLOAT) / 
                  (CAST(blockX * blockY * blockZ AS FLOAT) * registersPerThread) * 100) as utilization
        FROM CUPTI_ACTIVITY_KIND_KERNEL 
        WHERE blockX > 0 AND blockY > 0 AND blockZ > 0 AND registersPerThread > 0
        """
        try:
            result = conn.execute(util_query).fetchone()
            if result and result[0]:
                metrics['utilization_pct'] = result[0]
        except:
            pass
        
        conn.close()
        return metrics
        
    except Exception as e:
        print(f"Error processing {sqlite_file}: {e}")
        return None

def create_metrics_csv():
    """Create CSV with performance metrics for matrices with complete results."""
    complete_matrices = get_complete_matrices()
    
    if not complete_matrices:
        print("No matrices found with complete results")
        return
    
    print(f"Found {len(complete_matrices)} matrices with complete results")
    
    results_dir = "./results"
    all_data = []
    
    for matrix_name in complete_matrices:
        print(f"Processing {matrix_name}...")
        matrix_path = os.path.join(results_dir, matrix_name)
        
        # Process each variant
        variants = [
            (matrix_name, "original"),
            ("reordered_louvain*", "reordered_louvain"),
            ("reordered_metis*", "reordered_metis"), 
            ("reordered_rcm*", "reordered_rcm")
        ]
        
        for pattern, variant_name in variants:
            if variant_name == "original":
                sqlite_file = os.path.join(matrix_path, f"{matrix_name}_sqlite")
            else:
                sqlite_pattern = os.path.join(matrix_path, f"{pattern}_sqlite")
                sqlite_files = glob.glob(sqlite_pattern)
                if sqlite_files:
                    sqlite_file = sqlite_files[0]  # Take first match
                else:
                    continue
            
            metrics = extract_nsys_metrics(sqlite_file)
            if metrics:
                row_data = {
                    'matrix_name': matrix_name,
                    'variant': variant_name,
                    'avg_time_ms': metrics['avg_time_ms'],
                    'memory_usage_mb': metrics['memory_usage_mb'],
                    'occupancy_pct': metrics['occupancy_pct'],
                    'cache_hit_pct': metrics['cache_hit_pct'],
                    'utilization_pct': metrics['utilization_pct']
                }
                all_data.append(row_data)
    
    # Create DataFrame and save to CSV (only complete results)
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Filter to only include matrices with all 4 variants
        matrix_variant_counts = df.groupby('matrix_name')['variant'].count()
        complete_matrices_only = matrix_variant_counts[matrix_variant_counts == 4].index
        df_complete = df[df['matrix_name'].isin(complete_matrices_only)]
        
        if len(df_complete) > 0:
            output_file = "nsys_performance_metrics_complete.csv"
            df_complete.to_csv(output_file, index=False)
            print(f"Saved complete metrics to {output_file}")
            print(f"Total rows: {len(df_complete)}")
            print(f"Matrices with complete results: {len(complete_matrices_only)}")
            return df_complete
        else:
            print("No matrices found with complete results (all 4 variants)")
            return None
    else:
        print("No data collected")
        return None

def create_comparison_plots(df):
    """Create bar plots comparing the 4 variants for each metric."""
    if df is None or len(df) == 0:
        print("No data available for plotting")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Define metrics to plot
    metrics = ['avg_time_ms', 'memory_usage_mb', 'occupancy_pct', 'utilization_pct']
    metric_labels = {
        'avg_time_ms': 'Average Time (ms)',
        'memory_usage_mb': 'Memory Usage (MB)', 
        'occupancy_pct': 'Occupancy (%)',
        'utilization_pct': 'Utilization (%)'
    }
    
    # Get unique matrices
    matrices = df['matrix_name'].unique()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        print(f"Creating plot for {metric}...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for plotting
        pivot_data = df.pivot(index='matrix_name', columns='variant', values=metric)
        
        # Reorder columns to have a consistent order
        column_order = ['original', 'reordered_louvain', 'reordered_metis', 'reordered_rcm']
        pivot_data = pivot_data.reindex(columns=column_order)
        
        # Create bar plot
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        
        # Customize plot
        ax.set_title(f'{metric_labels[metric]} Comparison Across Matrix Variants', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Matrix Name', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        
        # Customize legend
        ax.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        filename = f"{metric}_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        
        plt.close()
    
    # Create a summary plot with all metrics
    print("Creating summary comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        ax = axes[i]
        
        # Prepare data
        pivot_data = df.pivot(index='matrix_name', columns='variant', values=metric)
        pivot_data = pivot_data.reindex(columns=column_order)
        
        # Create bar plot
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        
        # Customize subplot
        ax.set_title(f'{metric_labels[metric]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Matrix Name', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=8)
        ax.legend(title='Variant', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Performance Metrics Comparison Across All Variants', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = "all_metrics_comparison.png"
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_filename}")
    
    plt.close()
    
    print("All plots created successfully!")

if __name__ == "__main__":
    print("Creating metrics CSV for complete results...")
    df = create_metrics_csv()
    
    if df is not None:
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nCreating comparison plots...")
        create_comparison_plots(df)
    else:
        print("No metrics data generated")