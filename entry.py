from response_by_variance.etl import (
    response_and_variance_transform,
    avg_across_cell_populations,
    process_data,
    correlation_transform,
    analyze_correlation_thresholds,
    get_correlation_statistics,
)
from response_by_variance.optimize import (
    find_best_combos,
    calculate_variance_index,
    calculate_coverage_index,
    calculate_response_index,
    calculate_reagent_impact,
    calculate_channel_efficiency,
    optimize_by_variance,
    optimize_by_coverage,
    optimize_by_response,
    optimize_panel,
)
import polars as pl
import os


def save_results(dataframes: dict[str, pl.DataFrame], output_dir: str) -> None:
    """
    Save analysis results to output files.
    
    Args:
        dataframes: Dictionary mapping output filenames to DataFrames
        output_dir: Directory to save output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each DataFrame to CSV
    for filename, df in dataframes.items():
        output_path = os.path.join(output_dir, filename)
        print(f"Saving {filename}...")
        df.write_csv(output_path)
        print(f"Saved to {output_path} ({df.shape[0]} rows, {df.shape[1]} columns)")


def main():
    input_filepath = os.environ["INPUT_DIR"]
    output_filepath = os.environ["OUTPUT_DIR"]

    # we will use these columns to filter the initial data
    # in our case we are only interested in human data
    initial_filters = {
        "Species": "Human",
    }

    # we will use these columns to filter the basal data (no stimulus applied)
    basal_filters = {
        "Condition": "Basal",
    }

    # we will use these columns to join the basal and non-basal data
    normalization_join = [
        "population",
        "reagent",
        "Donor",
    ]

    # only these columns will be output
    keep_columns = [
        "population",
        "reagent",
        "Condition",
        "median",
        "variance",
    ]

    # we will use these columns to group the data before calculating the median and variance
    aggregation_columns = ["population", "reagent", "Condition"]

    input_frame = pl.read_csv(f"{input_filepath}/human.csv")
    
    # Process data and generate the main results
    # This is the existing functionality, preserved
    output_frame = response_and_variance_transform(
        input_frame,
        initial_filters,
        basal_filters,
        normalization_join,
        keep_columns,
        aggregation_columns,
        std_dev_count=4,
    )
    output_frame.write_csv(f"{output_filepath}/output.csv")
    
    # New functionality: run additional analyses
    print("\n========= Running Additional Analyses =========\n")
    
    # Get preprocessed data (non-aggregated)
    print("Preprocessing data for analysis...")
    preprocessed = process_data(
        input_frame,
        initial_filters,
        basal_filters,
        normalization_join,
        std_dev_count=4,
    )
    
    # Generate correlation matrix
    print("Generating correlation matrix...")
    correlation_matrix = correlation_transform(
        preprocessed,
        ["population", "reagent", "Condition"],
        "normalized_value",
    )
    
    # Calculate variance index
    print("Calculating variance index...")
    variance_index = calculate_variance_index(preprocessed)
    
    # Calculate coverage index
    print("Calculating coverage index...")
    coverage_index = calculate_coverage_index(preprocessed)
    
    # Calculate response index
    print("Calculating response index...")
    response_index = calculate_response_index(preprocessed)
    
    # Calculate reagent impact
    print("Calculating reagent impact...")
    reagent_impact = calculate_reagent_impact(response_index)
    
    # Calculate channel efficiency
    print("Calculating channel efficiency...")
    channel_efficiency = calculate_channel_efficiency(
        variance_index, coverage_index, response_index
    )
    
    # Generate optimal panels
    print("Generating optimal panels...")
    variance_panels = optimize_by_variance(variance_index, correlation_matrix)
    coverage_panels = optimize_by_coverage(coverage_index)
    response_panels = optimize_by_response(response_index)
    complex_panels = optimize_panel(
        variance_index, coverage_index, response_index, correlation_matrix, channel_efficiency
    )
    
    # Calculate correlation statistics
    print("Calculating correlation statistics...")
    correlation_stats = []
    for feature in correlation_matrix.columns:
        stats = get_correlation_statistics(correlation_matrix, feature)
        correlation_stats.append(stats)
    correlation_statistics = pl.DataFrame(correlation_stats)
    
    # Calculate threshold analysis
    print("Analyzing correlation thresholds...")
    threshold_stats = analyze_correlation_thresholds(correlation_matrix)
    
    # Save all results
    print("\n========= Saving Results =========\n")
    results = {
        "processed_data.csv": preprocessed,
        "correlation_matrix.csv": correlation_matrix,
        "correlation_statistics.csv": correlation_statistics,
        "correlation_threshold_analysis.csv": threshold_stats,
        "variance_index.csv": variance_index,
        "coverage_index.csv": coverage_index,
        "response_index.csv": response_index,
        "reagent_impact_index.csv": reagent_impact,
        "channel_efficiency.csv": channel_efficiency,
        "variance_optimal_panels.csv": variance_panels,
        "coverage_optimal_panels.csv": coverage_panels,
        "response_optimal_panels.csv": response_panels,
        "complex_optimal_panels.csv": complex_panels,
        "output.csv": output_frame,  # Include original output for compatibility
    }
    
    save_results(results, output_filepath)
    print("\n========= Analysis Complete =========\n")


if __name__ == "__main__":
    main()
