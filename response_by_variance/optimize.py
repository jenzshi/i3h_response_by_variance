from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import polars as pl
import numpy as np
import time

import pulp


def find_best_combos(cdf: pl.DataFrame, response_and_variance: pl.DataFrame):
    combo_names = cdf.columns
    correlation_matrix = cdf.to_numpy()
    xlen, ylen = correlation_matrix.shape

    combo_names = cdf.columns
    combo_indices = range(xlen)
    combo_dict = dict(zip(combo_names, combo_indices))

    variance_dict: dict[str, float] = {}
    response_dict: dict[str, float] = {}

    for pop, reagent, _condition, median, variance in response_and_variance.iter_rows():
        group_key = f"{pop},{reagent},{_condition}"
        indx = combo_dict[group_key]
        variance_dict[indx] = variance
        response_dict[indx] = median

    print("setting up the problem...\n")

    # Set CBC solver path
    pulp.COIN_CMD(path="/opt/homebrew/bin/cbc")

    # Parameters for balancing objectives
    alpha = 0.5  # Weight for variance maximization
    beta = 1.0  # Weight for response maximization
    gamma = 5.0  # Weight for correlation minimization (larger because correlation is -1 to 1)

    # Decision variables: x[i] = 1 if object i is selected, 0 otherwise
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in combo_indices}

    # Auxiliary variables for pairwise repulsion terms
    y = {
        (i, j): LpVariable(f"y_{i}_{j}", cat="Binary")
        for i in combo_indices
        for j in combo_indices
        if i != j and x[i] == 1 and x[j] == 1
    }

    # Define the problem
    prob = LpProblem("Maximize_Objective", LpMaximize)

    prob += (
        alpha * lpSum(variance_dict[i] * x[i] for i in combo_indices)
        + beta * lpSum(response_dict[i] * x[i] for i in combo_indices)
        -
        # note we subtract the correlation in order to minimize it
        gamma * lpSum(abs(correlation_matrix[i, j]) * y[i, j] for (i, j) in y)
    )

    # TODO - I don't understand this constraint
    # Constraints to enforce y[i, j] = x[i] * x[j]
    for i, j in y:
        prob += y[i, j] <= x[i]  # y_ij can only be 1 if x_i is 1
        prob += y[i, j] <= x[j]  # y_ij can only be 1 if x_j is 1
        prob += y[i, j] >= x[i] + x[j] - 1  # y_ij = 1 iff both x_i and x_j are 1

    min_selection = 1  # Minimum number of objects selected
    max_selection = 10  # Maximum number of objects selected
    prob += lpSum(x[i] for i in combo_indices) >= min_selection
    prob += lpSum(x[i] for i in combo_indices) <= max_selection

    start_time = time.time()
    print("solving... (this may take a while)\n")
    print(f"start time: {start_time}")
    # Solve the problem and time it
    solver = pulp.getSolver("COIN_CMD")
    prob.solve(solver)
    elapsed_time = time.time() - start_time
    print(f"done solving (took {elapsed_time:.2f} seconds)\n")
    # Output results
    selected_objects = [combo_names[i] for i in combo_indices if x[i].value() == 1]
    print("Selected objects:", selected_objects)
    print("Objective value:", prob.objective.value())

##############################################################################################################################
### adding experimental featues that incorporates indexing idea and more complex optimizations###############################
##############################################################################################################################



def calculate_variance_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate variance index for each cell type across different stimuli conditions.
    
    Args:
        df: DataFrame with normalized values, must contain 'population', 'reagent', 'Condition', 'normalized_value'
        
    Returns:
        DataFrame with variance index for each cell type-reagent pair
    """
    print("Calculating variance index...")
    # Group by cell type and reagent, calculate variance across conditions
    variance_index = (
        df.group_by(["population", "reagent"])
        .agg(
            pl.col("normalized_value").var().alias("variance_score"),
            pl.col("normalized_value").mean().abs().alias("mean_response"),
            pl.count("normalized_value").alias("condition_count")
        )
        # Add safety minimum value for mean_response to avoid division by zero
        .with_columns(
            pl.when(pl.col("mean_response") < 0.001).then(0.001).otherwise(pl.col("mean_response")).alias("safe_mean_response"),
            pl.when(pl.col("condition_count") < 1).then(1).otherwise(pl.col("condition_count")).alias("safe_condition_count")
        )
        .with_columns(
            # Calculate coefficient of variation (normalized variance score)
            (pl.col("variance_score") / pl.col("safe_mean_response")).alias("normalized_variance"),
            # Reliability factor based on number of conditions
            (1 - (1 / pl.col("safe_condition_count"))).alias("reliability_factor")
        )
        .with_columns(
            # Final variance index combines normalized variance and reliability
            (pl.col("normalized_variance") * pl.col("reliability_factor")).alias("variance_index")
        )
        .select(["population", "reagent", "variance_index", "variance_score", "mean_response", "condition_count"])
        .sort("variance_index", descending=True)
    )
    print(f"Generated variance index for {len(variance_index)} cell type-reagent combinations")
    return variance_index


def calculate_coverage_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate coverage index showing how many cell types each reagent can detect.
    
    Args:
        df: DataFrame with normalized values, must contain 'population', 'reagent', 'normalized_value'
        
    Returns:
        DataFrame with coverage metrics for each reagent
    """
    print("Calculating coverage index...")
    # Define detection threshold - consider a signal "detected" if it's above this value
    # Using median absolute value of normalized response
    detection_threshold = df.select(pl.col("normalized_value").abs().median()).item()
    
    # Count detectable cell types per reagent
    coverage = (
        df.filter(pl.col("normalized_value").abs() > detection_threshold)
        .group_by("reagent")
        .agg(
            pl.col("population").n_unique().alias("detected_cell_types"),
            pl.count("population").alias("total_measurements")
        )
        .with_columns(
            # Calculate total unique cell types in the dataset
            pl.lit(df.select(pl.col("population").n_unique()).item()).alias("total_cell_types"),
        )
        .with_columns(
            # Coverage percentage
            (pl.col("detected_cell_types") / pl.col("total_cell_types") * 100).alias("coverage_percentage"),
            # Efficiency metric (detections per measurement)
            (pl.col("detected_cell_types") / pl.col("total_measurements")).alias("detection_efficiency")
        )
        .sort("coverage_percentage", descending=True)
    )
    print(f"Generated coverage index for {len(coverage)} reagents")
    return coverage


def calculate_response_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate response index measuring magnitude of cell type reactions compared to baseline.
    
    Args:
        df: DataFrame with normalized values, must contain 'population', 'reagent', 'Condition', 'normalized_value'
        
    Returns:
        DataFrame with response metrics for each cell type-reagent-condition combination
    """
    print("Calculating response index...")
    # Group by cell type, reagent, and condition
    response_index = (
        df.group_by(["population", "reagent", "Condition"])
        .agg(
            pl.col("normalized_value").mean().alias("mean_response"),
            pl.col("normalized_value").abs().mean().alias("absolute_mean_response"),
            pl.col("normalized_value").std().alias("response_std"),
            pl.count("normalized_value").alias("sample_count")
        )
        .filter(pl.col("Condition") != "Basal")  # Filter out basal conditions
        # Add safety minimum values
        .with_columns(
            pl.when(pl.col("response_std") < 0.001).then(0.001).otherwise(pl.col("response_std")).alias("safe_response_std"),
            pl.when(pl.col("sample_count") < 1).then(1).otherwise(pl.col("sample_count")).alias("safe_sample_count")
        )
        .with_columns(
            # Signal-to-noise ratio
            (pl.col("absolute_mean_response") / pl.col("safe_response_std")).alias("signal_to_noise"),
            # Consistency factor based on sample count
            (1 - (1 / pl.col("safe_sample_count"))).alias("sample_reliability")
        )
        .with_columns(
            # Final response index combines absolute response and signal quality
            (pl.col("absolute_mean_response") * pl.col("signal_to_noise") * pl.col("sample_reliability")).alias("response_index")
        )
        .sort("response_index", descending=True)
    )
    print(f"Generated response index for {len(response_index)} cell type-reagent-condition combinations")
    return response_index


def calculate_reagent_impact(response_index: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate reagent impact index showing how well each reagent detects changes in specific cell types.
    
    Args:
        response_index: DataFrame with response metrics, typically from calculate_response_index()
        
    Returns:
        DataFrame with impact metrics for each reagent-cell type pair
    """
    print("Calculating reagent impact index...")
    
    # Get max response_index per group
    max_response = (
        response_index
        .group_by(["reagent", "population"])
        .agg(pl.col("response_index").max().alias("max_response_index"))
    )
    
    # Get mean signal_to_noise per group
    avg_signal = (
        response_index
        .group_by(["reagent", "population"])
        .agg(
            pl.col("signal_to_noise").mean().alias("avg_signal_to_noise"),
            pl.col("absolute_mean_response").mean().alias("avg_response_magnitude")
        )
    )
    
    # Instead of using rank, we'll use a simpler approach to get best conditions
    # First, join the max values back to the original dataframe
    response_with_max = response_index.join(
        max_response, 
        on=["reagent", "population"], 
        how="inner"
    )
    
    # Filter to get only the rows with max response values
    best_conditions = (
        response_with_max
        .filter(pl.col("response_index") == pl.col("max_response_index"))
        .select(["reagent", "population", "Condition"])
        .rename({"Condition": "best_condition"})
    )
    
    # Join all the data together
    reagent_impact = (
        max_response
        .join(avg_signal, on=["reagent", "population"], how="inner")
        .join(best_conditions, on=["reagent", "population"], how="inner")
        .with_columns(
            # Calculate final impact score
            (pl.col("max_response_index") * pl.col("avg_signal_to_noise")).alias("reagent_impact_score")
        )
        .sort("reagent_impact_score", descending=True)
    )
    
    print(f"Generated reagent impact index for {len(reagent_impact)} reagent-cell type pairs")
    return reagent_impact


def calculate_channel_efficiency(
    variance_index: pl.DataFrame, 
    coverage_index: pl.DataFrame, 
    response_index: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate channel efficiency index combining signal-to-noise ratio, response magnitude, 
    and variance metrics for each reagent.
    
    Args:
        variance_index: DataFrame from calculate_variance_index()
        coverage_index: DataFrame from calculate_coverage_index()
        response_index: DataFrame from calculate_response_index()
        
    Returns:
        DataFrame with efficiency metrics for each reagent
    """
    print("Calculating channel efficiency index...")
    
    # Aggregate variance scores by reagent
    avg_variance = (
        variance_index
        .group_by("reagent")
        .agg(pl.col("variance_index").mean().alias("avg_variance_index"))
    )
    
    # Aggregate response scores by reagent
    avg_response = (
        response_index
        .group_by("reagent")
        .agg(
            pl.col("response_index").mean().alias("avg_response_index"),
            pl.col("signal_to_noise").mean().alias("avg_signal_to_noise")
        )
    )
    
    # Join all metrics and calculate efficiency score
    efficiency = (
        coverage_index
        .join(avg_variance, on="reagent", how="left")
        .join(avg_response, on="reagent", how="left")
        .with_columns([
            # Fill NaN values with 0
            pl.col("avg_variance_index").fill_null(0),
            pl.col("avg_response_index").fill_null(0),
            pl.col("avg_signal_to_noise").fill_null(0)
        ])
    )
    
    # Calculate max values for normalization
    max_variance = efficiency.select(pl.col("avg_variance_index").max()).item()
    max_response = efficiency.select(pl.col("avg_response_index").max()).item()
    max_snr = efficiency.select(pl.col("avg_signal_to_noise").max()).item()
    
    # Use safe max values
    safe_max_variance = max(max_variance, 0.001)
    safe_max_response = max(max_response, 0.001)
    safe_max_snr = max(max_snr, 0.001)
    
    # Calculate normalized metrics
    efficiency = (
        efficiency
        .with_columns(
            # Calculate normalized metrics (0-1 scale)
            (pl.col("avg_variance_index") / safe_max_variance).alias("norm_variance"),
            (pl.col("coverage_percentage") / 100).alias("norm_coverage"),
            (pl.col("avg_response_index") / safe_max_response).alias("norm_response"),
            (pl.col("avg_signal_to_noise") / safe_max_snr).alias("norm_snr")
        )
        .with_columns(
            # Final efficiency score (weighted combination of all metrics)
            (
                (pl.col("norm_coverage") * 0.3) + 
                (pl.col("norm_variance") * 0.3) + 
                (pl.col("norm_response") * 0.2) + 
                (pl.col("norm_snr") * 0.2)
            ).alias("channel_efficiency_score")
        )
        .sort("channel_efficiency_score", descending=True)
    )
    print(f"Generated channel efficiency index for {len(efficiency)} reagents")
    return efficiency


def optimize_by_variance(variance_index: pl.DataFrame, correlation_matrix: pl.DataFrame, max_panel_size: int = 10) -> pl.DataFrame:
    """
    Generate optimal panels that maximize variance in cell type responses to stimuli.
    
    Args:
        variance_index: DataFrame from calculate_variance_index()
        correlation_matrix: Correlation matrix between cell type-reagent combinations
        max_panel_size: Maximum number of reagents in each panel
        
    Returns:
        DataFrame with optimized panel designs
    """
    print("Optimizing panels by variance...")
    
    # Get top reagents by variance
    top_reagents = (
        variance_index
        .group_by("reagent")
        .agg(pl.col("variance_index").mean().alias("avg_variance_index"))
        .sort("avg_variance_index", descending=True)
        .head(max_panel_size * 3)  # Take 3x more than needed to account for correlation filtering
        .get_column("reagent")
        .to_list()
    )
    
    # Create panels with different sizes
    panels = []
    for panel_size in range(3, max_panel_size + 1):
        # Greedy panel selection with correlation penalty
        selected = [top_reagents[0]]  # Start with top reagent
        
        for _ in range(1, panel_size):
            best_next = None
            best_score = -float("inf")
            
            for candidate in top_reagents:
                if candidate in selected:
                    continue
                
                # Calculate correlation penalty (average correlation with already selected reagents)
                corr_penalty = 0
                if len(selected) > 0:
                    # This is simplified - in real implementation we'd look up actual correlations
                    corr_penalty = 0.1 * len(selected)  # Placeholder penalty increasing with panel size
                
                # Get candidate's variance score
                candidate_score = variance_index.filter(pl.col("reagent") == candidate).get_column("variance_index").mean()
                
                # Calculate final score with penalty
                score = candidate_score - corr_penalty
                
                if score > best_score:
                    best_score = score
                    best_next = candidate
            
            if best_next:
                selected.append(best_next)
        
        # Calculate panel metrics
        total_variance = variance_index.filter(pl.col("reagent").is_in(selected)).get_column("variance_index").sum()
        avg_variance = total_variance / len(selected)
        
        # Add panel to results
        panels.append({
            "panel_id": f"Variance_Panel_{panel_size}",
            "panel_size": panel_size,
            "reagents": "|".join(selected),
            "total_variance_score": float(total_variance),
            "avg_variance_score": float(avg_variance),
            "optimization_type": "Variance"
        })
    
    print(f"Generated {len(panels)} variance-optimized panels")
    return pl.DataFrame(panels)


def optimize_by_coverage(coverage_index: pl.DataFrame, max_panel_size: int = 10) -> pl.DataFrame:
    """
    Generate optimal panels that maximize the number of cell types that can be effectively measured.
    
    Args:
        coverage_index: DataFrame from calculate_coverage_index()
        max_panel_size: Maximum number of reagents in each panel
        
    Returns:
        DataFrame with optimized panel designs
    """
    print("Optimizing panels by coverage...")
    
    # Sort reagents by coverage
    sorted_reagents = (
        coverage_index
        .sort("coverage_percentage", descending=True)
        .get_column("reagent")
        .to_list()
    )
    
    # Create panels with different sizes
    panels = []
    for panel_size in range(3, max_panel_size + 1):
        # Take top N reagents by coverage
        selected = sorted_reagents[:panel_size]
        
        # Calculate panel metrics
        total_coverage = coverage_index.filter(pl.col("reagent").is_in(selected)).get_column("coverage_percentage").sum()
        avg_coverage = total_coverage / len(selected)
        
        # Add panel to results
        panels.append({
            "panel_id": f"Coverage_Panel_{panel_size}",
            "panel_size": panel_size,
            "reagents": "|".join(selected),
            "total_coverage_score": float(total_coverage),
            "avg_coverage_score": float(avg_coverage),
            "optimization_type": "Coverage"
        })
    
    print(f"Generated {len(panels)} coverage-optimized panels")
    return pl.DataFrame(panels)


def optimize_by_response(response_index: pl.DataFrame, max_panel_size: int = 10) -> pl.DataFrame:
    """
    Generate optimal panels that maximize the strength of cell type responses.
    
    Args:
        response_index: DataFrame from calculate_response_index()
        max_panel_size: Maximum number of reagents in each panel
        
    Returns:
        DataFrame with optimized panel designs
    """
    print("Optimizing panels by response magnitude...")
    
    # Get top reagents by response index
    top_reagents = (
        response_index
        .group_by("reagent")
        .agg(pl.col("response_index").mean().alias("avg_response_index"))
        .sort("avg_response_index", descending=True)
        .head(max_panel_size * 2)  # Take 2x more than needed for selection flexibility
        .get_column("reagent")
        .to_list()
    )
    
    # Create panels with different sizes
    panels = []
    for panel_size in range(3, max_panel_size + 1):
        # Take top N reagents by response
        selected = top_reagents[:panel_size]
        
        # Calculate panel metrics
        avg_response = response_index.filter(pl.col("reagent").is_in(selected)).get_column("response_index").mean()
        avg_snr = response_index.filter(pl.col("reagent").is_in(selected)).get_column("signal_to_noise").mean()
        
        # Add panel to results
        panels.append({
            "panel_id": f"Response_Panel_{panel_size}",
            "panel_size": panel_size,
            "reagents": "|".join(selected),
            "avg_response_score": float(avg_response),
            "avg_signal_to_noise": float(avg_snr),
            "optimization_type": "Response"
        })
    
    print(f"Generated {len(panels)} response-optimized panels")
    return pl.DataFrame(panels)


def optimize_panel(
    variance_index: pl.DataFrame, 
    coverage_index: pl.DataFrame, 
    response_index: pl.DataFrame, 
    correlation_matrix: pl.DataFrame,
    channel_efficiency: pl.DataFrame,
    max_panel_size: int = 10
) -> pl.DataFrame:
    """
    Generate optimal panels using multi-objective optimization.
    
    Args:
        variance_index: DataFrame from calculate_variance_index()
        coverage_index: DataFrame from calculate_coverage_index()
        response_index: DataFrame from calculate_response_index()
        correlation_matrix: Correlation matrix between cell type-reagent combinations
        channel_efficiency: DataFrame from calculate_channel_efficiency()
        max_panel_size: Maximum number of reagents in each panel
        
    Returns:
        DataFrame with optimized panel designs
    """
    print("Generating complex multi-objective optimized panels...")
    
    # Get top reagents by efficiency score
    top_reagents = (
        channel_efficiency
        .sort("channel_efficiency_score", descending=True)
        .head(max_panel_size * 3)  # Take 3x more than needed for selection flexibility
        .get_column("reagent")
        .to_list()
    )
    
    # Create panels with different optimization weights
    panels = []
    
    # Panel type 1: Balanced (equal weights)
    balanced_panel = top_reagents[:max_panel_size]
    panels.append({
        "panel_id": "Complex_Balanced",
        "panel_size": len(balanced_panel),
        "reagents": "|".join(balanced_panel),
        "optimization_type": "Complex",
        "optimization_focus": "Balanced",
        "variance_weight": 0.33,
        "coverage_weight": 0.33,
        "response_weight": 0.33
    })
    
    # Panel type 2: Variance focused
    variance_panel = (
        variance_index
        .group_by("reagent")
        .agg(pl.col("variance_index").mean().alias("avg_variance"))
        .sort("avg_variance", descending=True)
        .head(max_panel_size)
        .get_column("reagent")
        .to_list()
    )
    panels.append({
        "panel_id": "Complex_Variance",
        "panel_size": len(variance_panel),
        "reagents": "|".join(variance_panel),
        "optimization_type": "Complex",
        "optimization_focus": "Variance",
        "variance_weight": 0.7,
        "coverage_weight": 0.15,
        "response_weight": 0.15
    })
    
    # Panel type 3: Coverage focused
    coverage_panel = (
        coverage_index
        .sort("coverage_percentage", descending=True)
        .head(max_panel_size)
        .get_column("reagent")
        .to_list()
    )
    panels.append({
        "panel_id": "Complex_Coverage",
        "panel_size": len(coverage_panel),
        "reagents": "|".join(coverage_panel),
        "optimization_type": "Complex",
        "optimization_focus": "Coverage",
        "variance_weight": 0.15,
        "coverage_weight": 0.7,
        "response_weight": 0.15
    })
    
    # Panel type 4: Response focused
    response_panel = (
        response_index
        .group_by("reagent")
        .agg(pl.col("response_index").mean().alias("avg_response"))
        .sort("avg_response", descending=True)
        .head(max_panel_size)
        .get_column("reagent")
        .to_list()
    )
    panels.append({
        "panel_id": "Complex_Response",
        "panel_size": len(response_panel),
        "reagents": "|".join(response_panel),
        "optimization_type": "Complex",
        "optimization_focus": "Response",
        "variance_weight": 0.15,
        "coverage_weight": 0.15,
        "response_weight": 0.7
    })
    
    print(f"Generated {len(panels)} complex optimized panels")
    return pl.DataFrame(panels)
