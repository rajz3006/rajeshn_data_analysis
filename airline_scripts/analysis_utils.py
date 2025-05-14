# airline_scripts/analysis_utils.py
import pandas as pd
import numpy as np
from . import config
from . import dq_utils 

DQ_PHASE_ANALYSIS = "Analysis"

def identify_busiest_round_trip_routes(abt_df, top_n=10):
    """
    Identifies the top N busiest round trip routes by number of round trip flights.
    A round trip route is e.g., JFK-ORD and ORD-JFK, grouped by CANONICAL_ROUTE_PAIR.
    Each flight record in ABT is one leg. Two legs make a "round trip flight".
    """
    step = f"Identify Top {top_n} Busiest Round Trip Routes"
    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "ABT", "Input Row Count for Busiest Routes", len(abt_df))

    if 'CANONICAL_ROUTE_PAIR' not in abt_df.columns:
        print("Error: 'CANONICAL_ROUTE_PAIR' column not found in ABT for busiest routes.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BusiestRoutes", "Error", "CANONICAL_ROUTE_PAIR missing")
        return pd.DataFrame()

    route_flight_counts = abt_df.groupby('CANONICAL_ROUTE_PAIR').size().reset_index(name='TOTAL_LEGS')
    route_flight_counts['ROUND_TRIP_FLIGHTS'] = route_flight_counts['TOTAL_LEGS'] // 2 
    busiest_routes = route_flight_counts.sort_values(by='ROUND_TRIP_FLIGHTS', ascending=False)
    top_busiest_routes = busiest_routes.head(top_n)

    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BusiestRoutes", f"Number of Top {top_n} Routes Found", len(top_busiest_routes))
    print(f"\n--- Top {top_n} Busiest Round Trip Routes ---")
    print(top_busiest_routes[['CANONICAL_ROUTE_PAIR', 'ROUND_TRIP_FLIGHTS']])
    return top_busiest_routes

def identify_most_profitable_round_trip_routes(abt_df, top_n=10):
    """
    Identifies the top N most profitable round trip routes (pre-airplane cost).
    Aggregates profit, revenue, cost, and other key components per CANONICAL_ROUTE_PAIR.
    """
    step = f"Identify Top {top_n} Most Profitable Round Trip Routes"
    
    required_cols = ['CANONICAL_ROUTE_PAIR', 'AVG_ROUTE_ITIN_FARE', 'TOTAL_FLIGHT_COST', 'TOTAL_FLIGHT_REVENUE', 'DISTANCE', 'DEP_DELAY', 'ARR_DELAY', 'OCCUPANCY_RATE', 'FL_DATE']
    if not all(col in abt_df.columns for col in required_cols):
        print(f"Error: ABT missing one or more required columns for profitability: {required_cols}")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "ProfitableRoutes", "Error", "Missing required columns in ABT")
        return pd.DataFrame()

    # Filter out flights where essential data for profit calculation is missing
    abt_profitable = abt_df.dropna(subset=['AVG_ROUTE_ITIN_FARE', 'TOTAL_FLIGHT_COST', 'TOTAL_FLIGHT_REVENUE']).copy()
    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "ABT_ForProfitability", "Input Row Count (Profit Essentials Not Null)", len(abt_profitable))
    
    if abt_profitable.empty:
        print("No flights with complete fare/cost information available for profitability analysis.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "ProfitableRoutes", "Error", "No data with fare/cost info.")
        return pd.DataFrame()

    route_summary = abt_profitable.groupby('CANONICAL_ROUTE_PAIR').agg(
        TOTAL_REVENUE=('TOTAL_FLIGHT_REVENUE', 'sum'),
        TOTAL_COST=('TOTAL_FLIGHT_COST', 'sum'),
        TOTAL_DISTANCE_MILES=('DISTANCE', 'sum'), 
        AVG_DEP_DELAY=('DEP_DELAY', 'mean'), 
        AVG_ARR_DELAY=('ARR_DELAY', 'mean'), 
        TOTAL_LEGS=('FL_DATE', 'count'), 
        AVG_OCCUPANCY=('OCCUPANCY_RATE', 'mean') 
    ).reset_index()

    route_summary['TOTAL_PROFIT'] = route_summary['TOTAL_REVENUE'] - route_summary['TOTAL_COST']
    route_summary['ROUND_TRIP_FLIGHTS'] = route_summary['TOTAL_LEGS'] // 2
    route_summary = route_summary[route_summary['ROUND_TRIP_FLIGHTS'] > 0] # Ensure valid round trips

    # Calculate Profit Per Flight for later use in scoring
    route_summary['PROFIT_PER_FLIGHT'] = route_summary.apply(
        lambda row: row['TOTAL_PROFIT'] / row['ROUND_TRIP_FLIGHTS'] if row['ROUND_TRIP_FLIGHTS'] > 0 else 0, axis=1
    )
    
    most_profitable_routes = route_summary.sort_values(by='TOTAL_PROFIT', ascending=False)
    top_profitable_routes = most_profitable_routes.head(top_n)
    
    cols_to_show = ['CANONICAL_ROUTE_PAIR', 'TOTAL_PROFIT', 'PROFIT_PER_FLIGHT', 'TOTAL_REVENUE', 'TOTAL_COST',
                    'ROUND_TRIP_FLIGHTS', 'TOTAL_DISTANCE_MILES', 
                    'AVG_DEP_DELAY', 'AVG_ARR_DELAY', 'AVG_OCCUPANCY']
    top_profitable_routes_display = top_profitable_routes[[col for col in cols_to_show if col in top_profitable_routes.columns]]


    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "ProfitableRoutes", f"Number of Top {top_n} Routes Found", len(top_profitable_routes_display))
    print(f"\n--- Top {top_n} Most Profitable Round Trip Routes (excluding upfront airplane cost) ---")
    print(top_profitable_routes_display)
    
    return top_profitable_routes # Return the full df with all calculated columns for scoring

def calculate_breakeven_flights(recommended_routes_df, profitable_routes_details_df):
    """
    Calculates the number of round trip flights to breakeven on airplane cost.
    Uses PROFIT_PER_FLIGHT from profitable_routes_details_df.
    """
    step = "Calculate Breakeven Flights"
    print(f"\n--- Calculating Breakeven Flights (Airplane Cost: ${config.AIRPLANE_UPFRONT_COST:,.0f}) ---")

    if recommended_routes_df.empty:
        print("No recommended routes to calculate breakeven for.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BreakevenAnalysis", "Error", "No recommended routes.")
        return pd.DataFrame()
        
    if profitable_routes_details_df.empty or 'PROFIT_PER_FLIGHT' not in profitable_routes_details_df.columns:
        print("No profitable route details with PROFIT_PER_FLIGHT available for breakeven calculation.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BreakevenAnalysis", "Error", "No profitable route details or PROFIT_PER_FLIGHT missing.")
        return pd.DataFrame()

    # Merge recommended routes with their detailed profit metrics
    breakeven_data = pd.merge(recommended_routes_df[['CANONICAL_ROUTE_PAIR']],
                              profitable_routes_details_df, # This should have PROFIT_PER_FLIGHT
                              on='CANONICAL_ROUTE_PAIR',
                              how='left')
    
    # Ensure we have profit per flight data and it's positive
    breakeven_data.dropna(subset=['PROFIT_PER_FLIGHT'], inplace=True)
    breakeven_data = breakeven_data[breakeven_data['PROFIT_PER_FLIGHT'] > 0]

    if breakeven_data.empty:
        print("No recommended routes with positive profit per round trip flight for breakeven calculation.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BreakevenAnalysis", "Error", "No routes with positive profit per flight.")
        return pd.DataFrame()
        
    breakeven_data['BREAKEVEN_ROUND_TRIP_FLIGHTS'] = np.ceil(
        config.AIRPLANE_UPFRONT_COST / breakeven_data['PROFIT_PER_FLIGHT']
    )
    
    cols_to_show_be = ['CANONICAL_ROUTE_PAIR', 'PROFIT_PER_FLIGHT',
                       'BREAKEVEN_ROUND_TRIP_FLIGHTS', 'TOTAL_PROFIT', 
                       'ROUND_TRIP_FLIGHTS', 'AVG_DEP_DELAY', 'AVG_ARR_DELAY']
    
    breakeven_summary = breakeven_data[[col for col in cols_to_show_be if col in breakeven_data.columns]]
    
    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "BreakevenAnalysis", "Number of Routes Analyzed for Breakeven", len(breakeven_summary))
    print(breakeven_summary)
    return breakeven_summary


def recommend_routes_advanced_scoring(
    all_route_metrics_df, # This should be a comprehensive df with all metrics per route
    num_recommendations=5
    ):
    """
    Recommends routes based on a multi-factored scoring model.

    Args:
        all_route_metrics_df (pd.DataFrame): DataFrame containing all necessary metrics per 
                                             CANONICAL_ROUTE_PAIR. Expected columns include:
                                             'TOTAL_PROFIT', 'PROFIT_PER_FLIGHT', 
                                             'ROUND_TRIP_FLIGHTS' (for volume),
                                             'AVG_DEP_DELAY', 'AVG_ARR_DELAY', 
                                             'AVG_OCCUPANCY', 'BREAKEVEN_ROUND_TRIP_FLIGHTS'.
        num_recommendations (int): Number of top routes to recommend.

    Returns:
        pd.DataFrame: Top N recommended routes with scores and justifications.
    """
    step = f"Recommend {num_recommendations} Routes (Advanced Scoring)"
    print(f"\n--- {step} ---")
    
    if all_route_metrics_df.empty:
        print("Input DataFrame 'all_route_metrics_df' is empty. Cannot make recommendations.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Error", "Input DF empty")
        return pd.DataFrame()

    df_scores = all_route_metrics_df.copy()

    # --- 1. Prepare Metrics for Scoring ---
    # Ensure all required metrics are present and handle missing values by dropping rows
    # This is a stricter approach; imputation could be used but adds complexity.
    metrics_for_scoring = {
        'TOTAL_PROFIT': 'higher_is_better',
        'PROFIT_PER_FLIGHT': 'higher_is_better',
        'ROUND_TRIP_FLIGHTS': 'higher_is_better', # Flight Volume
        'AVG_OCCUPANCY': 'higher_is_better',
        'AVG_TOTAL_DELAY': 'lower_is_better', # Will be created
        'BREAKEVEN_ROUND_TRIP_FLIGHTS': 'lower_is_better'
    }
    
    # Create AVG_TOTAL_DELAY
    if 'AVG_DEP_DELAY' in df_scores.columns and 'AVG_ARR_DELAY' in df_scores.columns:
        df_scores['AVG_TOTAL_DELAY'] = df_scores['AVG_DEP_DELAY'] + df_scores['AVG_ARR_DELAY']
    else:
        print("Warning: AVG_DEP_DELAY or AVG_ARR_DELAY missing. Cannot calculate AVG_TOTAL_DELAY.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Warning", "Delay columns missing for AVG_TOTAL_DELAY")
        # If AVG_TOTAL_DELAY is critical and missing, we might need to drop it from scoring or handle differently
        if 'AVG_TOTAL_DELAY' in metrics_for_scoring:
            del metrics_for_scoring['AVG_TOTAL_DELAY']


    # Check for presence of all metric columns and drop rows with NaNs in these critical scoring columns
    cols_to_check_for_nan = list(metrics_for_scoring.keys())
    # Remove AVG_TOTAL_DELAY if it couldn't be created
    if 'AVG_TOTAL_DELAY' not in df_scores.columns and 'AVG_TOTAL_DELAY' in cols_to_check_for_nan:
        cols_to_check_for_nan.remove('AVG_TOTAL_DELAY')
        
    initial_rows = len(df_scores)
    df_scores.dropna(subset=cols_to_check_for_nan, inplace=True)
    rows_dropped = initial_rows - len(df_scores)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} routes due to missing values in one of the critical scoring metrics.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Rows Dropped (NaN in scoring metrics)", rows_dropped)

    if df_scores.empty:
        print("No routes remaining after dropping NaNs in scoring metrics. Cannot proceed.")
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Error", "No routes after NaN drop")
        return pd.DataFrame()

    # --- 2. Normalize Metrics (Min-Max Scaling to 0-100) ---
    normalized_cols = {}
    for metric, preference in metrics_for_scoring.items():
        if metric not in df_scores.columns:
            print(f"Metric {metric} not found in DataFrame. Skipping normalization for it.")
            continue
        
        col_min = df_scores[metric].min()
        col_max = df_scores[metric].max()
        norm_col_name = f"NORM_{metric}"
        
        if col_min == col_max: # Avoid division by zero if all values are the same
            df_scores[norm_col_name] = 50.0 # Assign a neutral score, or 0 or 100 depending on preference
        elif preference == 'higher_is_better':
            df_scores[norm_col_name] = ((df_scores[metric] - col_min) / (col_max - col_min)) * 100
        elif preference == 'lower_is_better':
            df_scores[norm_col_name] = ((col_max - df_scores[metric]) / (col_max - col_min)) * 100
        normalized_cols[metric] = norm_col_name
        dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", f"Normalized Metric", norm_col_name)


    # --- 3. Apply Weights and Calculate Composite Score ---
    weights = {
        'TOTAL_PROFIT': 0.25,
        'PROFIT_PER_FLIGHT': 0.15,
        'ROUND_TRIP_FLIGHTS': 0.20,      # Market Demand
        'AVG_TOTAL_DELAY': 0.15,         # Punctuality
        'AVG_OCCUPANCY': 0.10,           # Operational Factor
        'BREAKEVEN_ROUND_TRIP_FLIGHTS': 0.15 # Investment Efficiency
    }
    
    df_scores['COMPOSITE_SCORE'] = 0.0
    final_score_calculation_summary = []

    for metric, weight in weights.items():
        norm_col_name = normalized_cols.get(metric)
        if norm_col_name and norm_col_name in df_scores.columns:
            df_scores['COMPOSITE_SCORE'] += df_scores[norm_col_name] * weight
            final_score_calculation_summary.append(f"({norm_col_name} * {weight})")
        else:
            print(f"Warning: Normalized column for metric '{metric}' not found. It will not be included in composite score.")
            dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Warning", f"Normalized {metric} missing for score")

    print(f"Composite Score calculated using: {' + '.join(final_score_calculation_summary)}")
    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", "Composite Score Calculated", True)

    # --- 4. Select Top N Routes ---
    recommended_df = df_scores.sort_values(by='COMPOSITE_SCORE', ascending=False)
    recommended_df = recommended_df.head(num_recommendations)
    recommended_df['RANK'] = range(1, len(recommended_df) + 1)

    print(f"\n--- Top {num_recommendations} Recommended Routes (Advanced Scoring) ---")
    # Select columns for display, including original metrics, normalized scores, and composite score
    display_cols = ['RANK', 'CANONICAL_ROUTE_PAIR', 'COMPOSITE_SCORE'] + \
                   list(metrics_for_scoring.keys()) + \
                   [normalized_cols[m] for m in metrics_for_scoring.keys() if m in normalized_cols and normalized_cols[m] in recommended_df.columns]
    
    # Ensure all display_cols exist in recommended_df
    display_cols = [col for col in display_cols if col in recommended_df.columns]
    print(recommended_df[display_cols])
    
    dq_utils.log_dq_metric(DQ_PHASE_ANALYSIS, step, "AdvancedRec", f"Number of Routes Recommended", len(recommended_df))
    return recommended_df


if __name__ == '__main__':
    print("Testing analysis_utils.py...")
