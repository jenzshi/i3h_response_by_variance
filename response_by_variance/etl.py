import polars as pl
import numpy as np
from response_by_variance.optimize import find_best_combos
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def filter_by_group(
    df: pl.DataFrame, by_filter_columns: dict[str, str]
) -> pl.DataFrame:
    for column, value in by_filter_columns.items():
        df = df.filter(pl.col(column) == value)
    return df


def filter_by_group_negate(
    df: pl.DataFrame, by_filter_columns: dict[str, str]
) -> pl.DataFrame:
    for column, value in by_filter_columns.items():
        df = df.filter(pl.col(column) != value)
    return df


def filter_data(
    df: pl.DataFrame, initial_filters: dict[str, str], value_column="value"
) -> pl.DataFrame:
    return filter_by_group(df.drop_nans(value_column), initial_filters)


def remove_outliers(
    df: pl.DataFrame,
    by_grouping_columns: list[str],
    num_std_dev: int,
    value_column="value",
) -> pl.DataFrame:
    grouped_std_var = (
        df.group_by(by_grouping_columns)
        .agg(pl.col(value_column).std())
        .rename({value_column: "std"})
    )

    return df.join(grouped_std_var, how="inner", on=by_grouping_columns).filter(
        (pl.col(value_column) <= pl.col("std") * num_std_dev)
        & (pl.col(value_column) >= -pl.col("std") * num_std_dev)
    )


def normalize_by_basal(
    df: pl.DataFrame,
    basal_filters: dict[str, str],
    normalization_join: list[str],
    value_column="value",
) -> pl.DataFrame:
    base = (
        filter_by_group(df, basal_filters)
        .rename({value_column: "basal_value"})
        .drop("Condition")
    )

    non_base = filter_by_group_negate(df, basal_filters)

    return base.join(non_base, how="inner", on=normalization_join).with_columns(
        (pl.col(value_column) - pl.col("basal_value")).alias("normalized_value"),
    )


def group_by_and_agg(df: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    med = (
        df.group_by(group_by)
        .agg(pl.col("normalized_value").median())
        .rename({"normalized_value": "median"})
    )
    var = (
        df.group_by(group_by)
        .agg(pl.col("normalized_value").var())
        .rename({"normalized_value": "variance"})
    )
    return med.join(var, how="inner", on=group_by)


def avg_across_cell_populations(
    df: pl.DataFrame, pivot_value_column: str, output_value_column: str
) -> pl.DataFrame:
    unique_populations = df.select(pl.col("population")).unique().to_series().to_list()
    medpivot = df.pivot(
        on="population",
        index=["reagent", "Condition"],
        values=pivot_value_column,
        aggregate_function="mean"
    ).with_columns(pl.mean_horizontal(unique_populations).alias(output_value_column))
    return medpivot.drop(unique_populations)


# TODO this needs to be validated
def summary_score(df: pl.DataFrame) -> pl.DataFrame:
    medpop = avg_across_cell_populations(df, "median", "average_cell_response")
    varpop = avg_across_cell_populations(df, "variance", "average_celltype_variance")

    response_weight = 1
    variance_weight = 0.5

    return (
        medpop.join(varpop, on=["reagent", "Condition"], how="inner")
        .with_columns(
            (
                (pl.col("average_cell_response") * response_weight)
                + (pl.col("average_celltype_variance") * variance_weight)
            ).alias("cross_celltype_summary_score")
        )
        .drop(["average_cell_response", "average_celltype_variance"])
    )


def correlation_transform(
    df: pl.DataFrame,
    correlation_columns: list[str],
    value_column: str,
) -> pl.DataFrame:
    print("Starting correlation transform...")
    responses = (
        df.group_by(correlation_columns)
        .agg(pl.col(value_column).alias("values"))
        .with_columns(
            pl.concat_str(correlation_columns, separator=",").alias("group_key"),
            pl.col("values").list.len().alias("values_size"),
        )
        .drop(correlation_columns)
        .sort("values_size", descending=False)
    )
    print(f"Grouped data into {len(responses)} unique combinations")

    # Convert to numpy array for faster correlation calculation
    values = [arr.to_list() for arr in responses.get_column("values")]
    min_len = min(len(v) for v in values)
    truncated = [v[:min_len] for v in values]
    print(f"Truncated vectors to length {min_len}")

    # Use numpy's optimized correlation calculation
    print("Calculating correlation matrix...")
    matrix = np.array(truncated)
    corr_matrix = np.corrcoef(matrix)
    print("Correlation matrix calculated")

    # Convert back to polars DataFrame
    keys = responses.get_column("group_key").to_list()  # Convert to list
    corr = pl.DataFrame(
        data=corr_matrix,
        schema=keys,  # Use the list of keys as schema
    )
    print("Correlation transform complete")
    return corr


def find_correlated_features(corr_df: pl.DataFrame, threshold: float = 0.7) -> dict[str, list[str]]:
    """Find features that are highly correlated with each combination.
    
    Args:
        corr_df: Correlation matrix DataFrame
        threshold: Correlation threshold (default 0.7)
        
    Returns:
        Dictionary mapping each feature to list of correlated features
    """
    correlated_features = {}
    
    for col in corr_df.columns:
        # Get correlations for this column
        correlations = corr_df.get_column(col)
        # Find indices where correlation exceeds threshold
        correlated_idx = [i for i, v in enumerate(correlations) if abs(v) >= threshold and v != 1.0]
        # Map indices back to feature names
        correlated = [corr_df.columns[i] for i in correlated_idx]
        if correlated:
            correlated_features[col] = correlated
            
    return correlated_features


def preprocess(
    input_frame: pl.DataFrame,
    initial_filters: dict[str, str],
    basal_filters: dict[str, str],
    normalization_join: list[str],
    keep_columns: list[str],
    aggregation_columns: list[str],
    std_dev_count: int,
    value_column: str = "value",
):
    df = filter_data(input_frame, initial_filters, value_column)
    df = normalize_by_basal(df, basal_filters, normalization_join, value_column)
    df = remove_outliers(df, aggregation_columns, num_std_dev=std_dev_count)
    return df


def analyze_correlation_thresholds(corr_df: pl.DataFrame) -> pl.DataFrame:
    """Analyze correlation patterns at different correlation thresholds.
    
    For each threshold level (0.5-0.9), calculates:
    1. Raw count of correlated pairs (absolute number)
    2. Percentage of all possible pairs (relative prevalence)
    3. Mean correlation value (average strength)
    4. Min/max correlations (range of strengths)
    
    Note: Self-correlations excluded from all calculations
    
    Args:
        corr_df: Correlation matrix as a Polars DataFrame
        
    Returns:
        DataFrame with rows for each threshold (0.5-0.9) and columns:
        - threshold: The correlation cutoff value
        - num_correlations: Raw count of correlated pairs
        - percent_of_total: What percentage of all possible pairs
        - mean_correlation: Average correlation strength
        - max/min_correlation: Strongest/weakest correlations found
    """
    print("Analyzing correlation thresholds...")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    stats = []
    
    # Total possible pairs = n^2 - n (exclude self-correlations)
    n = len(corr_df.columns)
    total_possible = (n * n - n)
    
    for threshold in thresholds:
        print(f"Processing threshold {threshold}...")
        correlations = []
        for col in corr_df.columns:
            values = corr_df.get_column(col)
            # Only include correlations above threshold, excluding self (1.0)
            high_corr = [v for v in values if abs(v) >= threshold and v != 1.0]
            correlations.extend(high_corr)
            
        if correlations:
            stats.append({
                "threshold": threshold,
                "num_correlations": len(correlations),
                "percent_of_total": (len(correlations) / total_possible) * 100,
                "mean_correlation": sum(abs(c) for c in correlations) / len(correlations),
                "max_correlation": max(abs(c) for c in correlations),
                "min_correlation": min(abs(c) for c in correlations)
            })
    
    print("Threshold analysis complete")
    return pl.DataFrame(stats)


def get_correlation_statistics(corr_df: pl.DataFrame, feature: str) -> dict:
    """Get detailed correlation statistics for a feature."""
    correlations = corr_df.get_column(feature)
    # Remove self-correlation
    correlations = [v for v in correlations if v != 1.0]
    
    return {
        "feature": feature,
        "mean_correlation": sum(abs(c) for c in correlations) / len(correlations),
        "std_correlation": pl.Series(correlations).std(),
        "num_high_correlations": sum(1 for c in correlations if abs(c) >= 0.7),
        "max_correlation": max(abs(c) for c in correlations),
        "min_correlation": min(abs(c) for c in correlations)
    }


def plot_correlation_analysis(threshold_stats: pl.DataFrame, output_dir: str = "."):
    """Generate three plots visualizing correlation analysis results.
    
    Creates a figure with three subplots:
    1. Number of Pairs (BLUE) - Shows absolute count of correlations
       - X: correlation threshold
       - Y: number of correlated pairs
       - Labels: formatted with commas (e.g., "123,456")
    
    2. Percentage of Pairs (RED) - Shows relative prevalence
       - X: correlation threshold
       - Y: percentage of all possible pairs
       - Labels: formatted with 2 decimals and % (e.g., "10.55%")
    
    3. Mean Correlation (GREEN) - Shows average strength
       - X: correlation threshold
       - Y: mean correlation value
       - Labels: 3 decimal places (e.g., "0.763")
    
    All plots include:
    - Data points as circles
    - Connected line between points
    - Value labels above each point
    - Clear titles and axis labels
    
    Args:
        threshold_stats: DataFrame with correlation statistics
        output_dir: Where to save the output PNG file
    """
    # Set style and create 3-panel figure
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    df = threshold_stats.to_pandas()
    
    # 1. Number of Pairs (BLUE)
    sns.lineplot(data=df, 
                x="threshold", 
                y="num_correlations",
                marker='o',
                color='blue',
                ax=ax1)
    for x, y in zip(df['threshold'], df['num_correlations']):
        ax1.annotate(f'{int(y):,}',  # Add commas for readability
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    ax1.set_title("Number of Correlated Feature Pairs")
    ax1.set_xlabel("Correlation Threshold")
    ax1.set_ylabel("Number of Pairs")
    
    # 2. Percentage of Pairs (RED)
    sns.lineplot(data=df,
                x="threshold",
                y="percent_of_total",
                marker='o',
                color='red',
                ax=ax2)
    for x, y in zip(df['threshold'], df['percent_of_total']):
        ax2.annotate(f'{y:.2f}%',  # 2 decimal places with %
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    ax2.set_title("Percentage of Correlated Feature Pairs")
    ax2.set_xlabel("Correlation Threshold")
    ax2.set_ylabel("Percentage of All Possible Pairs")
    
    # 3. Mean Correlation (GREEN)
    sns.lineplot(data=df,
                x="threshold",
                y="mean_correlation",
                marker='o',
                color='green',
                ax=ax3)
    for x, y in zip(df['threshold'], df['mean_correlation']):
        ax3.annotate(f'{y:.3f}',  # 3 decimal places
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    ax3.set_title("Mean Correlation vs Threshold")
    ax3.set_xlabel("Correlation Threshold")
    ax3.set_ylabel("Mean Correlation")
    
    # Save high-resolution figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_threshold_analysis.png"), 
                dpi=300,  # High resolution for clear labels
                bbox_inches='tight'  # Prevent label cutoff
    )
    plt.close()


def response_and_variance_transform(
    input_frame: pl.DataFrame,
    initial_filters: dict[str, str],
    basal_filters: dict[str, str],
    normalization_join: list[str],
    keep_columns: list[str],
    aggregation_columns: list[str],
    std_dev_count: int,
    value_column: str = "value",
):
    print(f"Input shape: {input_frame.shape}")
    preprocessed = preprocess(
        input_frame,
        initial_filters,
        basal_filters,
        normalization_join,
        keep_columns,
        aggregation_columns,
        std_dev_count,
        value_column,
    )
    print(f"After preprocess shape: {preprocessed.shape}")
    
    aggregated = group_by_and_agg(preprocessed, aggregation_columns).select(
        keep_columns
    )
    print(f"After aggregation shape: {aggregated.shape}")

    print("\nStarting correlation analysis...")
    cdf = correlation_transform(
        preprocessed,
        [
            "population",
            "reagent",
            "Condition",
        ],
        "normalized_value",
    )

    # Ensure output directory exists
    output_dir = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving correlation matrix...")
    cdf.write_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    
    print("Analyzing correlation thresholds...")
    threshold_stats = analyze_correlation_thresholds(cdf)
    threshold_stats.write_csv(os.path.join(output_dir, "correlation_threshold_analysis.csv"))
    
    print("Creating visualization plots...")
    plot_correlation_analysis(threshold_stats, output_dir)
    
    print("Calculating correlation statistics...")
    correlation_stats = []
    for feature in cdf.columns:
        stats = get_correlation_statistics(cdf, feature)
        correlation_stats.append(stats)
    pl.DataFrame(correlation_stats).write_csv(os.path.join(output_dir, "correlation_statistics.csv"))
    
    print("\nCorrelation Analysis Summary:")
    print(f"Total features analyzed: {len(cdf.columns)}")
    print(f"Correlation matrix shape: {cdf.shape}")
    print("\nThreshold Analysis:")
    print(threshold_stats)
    
    print("\nFinding correlated features...")
    correlated_features_dict = find_correlated_features(cdf)
    
    print("Adding correlated features to output...")
    aggregated = aggregated.with_columns([
        pl.Series(
            name="correlated_features",
            values=[
                "|".join(correlated_features_dict.get(f"{row['population']},{row['reagent']},{row['Condition']}", []))
                for row in aggregated.iter_rows(named=True)
            ]
        )
    ])
    print(f"After adding correlated features shape: {aggregated.shape}")
    print("Final columns:", aggregated.columns)

    return aggregated

##############################################################################################################################
### adding experimental featues that incorporates indexing idea and more complex optimizations###############################
##############################################################################################################################






def process_data(
    input_frame: pl.DataFrame,
    initial_filters: dict[str, str],
    basal_filters: dict[str, str],
    normalization_join: list[str],
    std_dev_count: int,
    value_column: str = "value",
) -> pl.DataFrame:
    """
    Process raw data for output into processed_data.csv.
    
    This function performs data cleaning, normalization, and outlier removal,
    but preserves individual data points (doesn't aggregate).
    
    Args:
        input_frame: Input data frame
        initial_filters: Initial data filters
        basal_filters: Filters for basal/baseline conditions
        normalization_join: Columns to join on for normalization
        std_dev_count: Number of standard deviations for outlier removal
        value_column: Name of the value column
        
    Returns:
        Processed DataFrame ready for output
    """
    print("Processing data for output...")
    
    # Filter and normalize data
    processed = preprocess(
        input_frame,
        initial_filters,
        basal_filters,
        normalization_join,
        keep_columns=[], # No columns to drop at this stage
        aggregation_columns=["population", "reagent", "Condition"],
        std_dev_count=std_dev_count,
        value_column=value_column,
    )
    
    # Add metadata for tracking processing steps
    processed = processed.with_columns([
        pl.lit("Applied initial filters: " + ", ".join([f"{k}={v}" for k, v in initial_filters.items()])).alias("processing_info"),
        pl.lit(f"Normalized against {list(basal_filters.values())[0]} condition").alias("normalization"),
        pl.lit(f"Removed outliers beyond {std_dev_count} standard deviations").alias("outlier_handling")
    ])
    
    # Select columns for output
    # Keep original values alongside normalized values for reference
    final_columns = [col for col in processed.columns if col not in [
        "basal_value", "std", "processing_info", "normalization", "outlier_handling"
    ]]
    
    # Move metadata columns to the end
    final_columns.extend(["processing_info", "normalization", "outlier_handling"])
    
    print(f"Processed data shape: {processed.shape}")
    return processed.select(final_columns)


# def create_heatmap_from_pivot(pivot_df: pl.DataFrame, title: str, output_path: str) -> None:
#     """
#     Create a heatmap visualization from a pivot table.
    
#     Args:
#         pivot_df: Pivot table DataFrame
#         title: Title for the heatmap
#         output_path: Path to save the heatmap image
#     """
#     # Convert to pandas for visualization
#     pd_df = pivot_df.to_pandas()
    
#     # We need to ensure all data is numeric for the heatmap
#     # First, save the original index and columns
#     index_labels = pd_df.index.tolist()
#     column_labels = pd_df.columns.tolist()
    
#     # Convert data to numeric, replacing non-numeric with NaN
#     numeric_data = pd_df.apply(pd.to_numeric, errors='coerce')
    
#     # Create figure with appropriate size
#     plt.figure(figsize=(max(12, len(column_labels) * 0.4), max(8, len(index_labels) * 0.4)))
    
#     # Create heatmap using imshow
#     im = plt.imshow(numeric_data.values, cmap='viridis', aspect='auto')
    
#     # Add colorbar
#     plt.colorbar(im)
    
#     # Set tick labels
#     plt.yticks(range(len(index_labels)), index_labels)
#     plt.xticks(range(len(column_labels)), column_labels, rotation=90)
    
#     # Add labels and title
#     plt.title(title)
#     plt.ylabel("Population")
#     plt.xlabel("Reagent/Condition")
    
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
    
#     print(f"Created heatmap visualization: {output_path}")


# def create_clustered_heatmap(pivot_df: pl.DataFrame, title: str, output_path: str) -> None:
#     """
#     Create a clustered heatmap visualization from a pivot table with hierarchical clustering.
    
#     Args:
#         pivot_df: Pivot table DataFrame
#         title: Title for the heatmap
#         output_path: Path to save the heatmap image
#     """
#     try:
#         # Convert to pandas for visualization
#         pd_df = pivot_df.to_pandas()
        
#         # Convert data to numeric, replacing non-numeric with NaN
#         numeric_data = pd_df.apply(pd.to_numeric, errors='coerce')
        
#         # Fill NaN values with the mean for visualization purposes
#         numeric_data = numeric_data.fillna(numeric_data.mean().mean())
        
#         # Create clustered heatmap with hierarchical clustering
#         g = sns.clustermap(
#             numeric_data,
#             cmap="viridis",
#             figsize=(max(14, len(pd_df.columns) * 0.4), max(10, len(pd_df) * 0.4)),
#             linewidths=0.5,
#             dendrogram_ratio=0.1,
#             cbar_pos=(0.02, 0.8, 0.05, 0.18),
#         )
        
#         # Add title
#         g.fig.suptitle(title)
        
#         # Rotate x-axis labels for better readability
#         plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # Save figure
#         g.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
#         plt.savefig(output_path, dpi=300)
#         plt.close()
        
#         print(f"Created clustered heatmap visualization: {output_path}")
#     except Exception as e:
#         print(f"Warning: Could not create clustered heatmap for {title}: {str(e)}")
#         # Fall back to regular heatmap if clustering fails
#         create_heatmap_from_pivot(pivot_df, f"{title} (Without Clustering)", output_path)


# def generate_all_pivot_visualizations(pivot_tables: dict[str, pl.DataFrame], output_dir: str) -> None:
#     """
#     Generate visualizations for all pivot tables.
    
#     Args:
#         pivot_tables: Dictionary mapping pivot table names to DataFrames
#         output_dir: Directory to save visualization files
#     """
#     print("\nGenerating visualization plots...")
    
#     # Create visualization directory
#     viz_dir = os.path.join(output_dir, "visualizations")
#     os.makedirs(viz_dir, exist_ok=True)
    
#     # Create visualizations for each pivot table
#     for name, pivot_df in pivot_tables.items():
#         # Skip if pivot table is empty or has no data to visualize
#         if pivot_df is None or len(pivot_df) == 0 or len(pivot_df.columns) <= 1:
#             print(f"Skipping visualization for {name} - insufficient data")
#             continue
            
#         # Create standard heatmap
#         title = f"{name.replace('_', ' ').title()} Heatmap"
#         output_path = os.path.join(viz_dir, f"{name}_heatmap.png")
#         create_heatmap_from_pivot(pivot_df, title, output_path)
        
#         # Create clustered heatmap
#         title = f"{name.replace('_', ' ').title()} Clustered Heatmap"
#         output_path = os.path.join(viz_dir, f"{name}_clustered_heatmap.png")
#         create_clustered_heatmap(pivot_df, title, output_path)
    
#     print("Visualization generation complete")
