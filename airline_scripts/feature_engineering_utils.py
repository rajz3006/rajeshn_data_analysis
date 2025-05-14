# airline_scripts/feature_engineering_utils.py
import pandas as pd
import numpy as np
from . import config
from . import dq_utils

DQ_PHASE_FEATURE_ENG = "Data Transformation & Feature Engineering"

def create_route_columns(df, origin_col, dest_col):
    """
    Creates 'ROUTE' (e.g., JFK-ORD) and 'CANONICAL_ROUTE_PAIR' (e.g., JFK-ORD sorted alphabetically) columns.
    """
    table_name = "DataFrameWithRoutes" 
    step = "Create Route Columns"
    df_copy = df.copy()
    df_copy['ROUTE'] = df_copy[origin_col].astype(str) + '-' + df_copy[dest_col].astype(str)
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, table_name, "Feature Created", "ROUTE")
    df_copy['CANONICAL_ROUTE_PAIR'] = df_copy.apply(
        lambda row: config.get_canonical_route_pair(str(row[origin_col]), str(row[dest_col])), axis=1
    )
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, table_name, "Feature Created", "CANONICAL_ROUTE_PAIR")
    return df_copy

def join_flights_with_airports(df_flights, df_airports):
    """
    Joins flights data with airport data for origin and destination airports.
    Includes airport name, type, latitude, and longitude.
    """
    table_name = "Flights_Merged_With_Airports"
    step = "Join Flights with Airports"
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, "Flights", "Row Count Before Join", len(df_flights))
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, "Airports_Cleaned", "Row Count (Filtered US Med/Large with Coords)", len(df_airports))

    # Columns to select from airports data, including new LATITUDE and LONGITUDE
    airport_cols_to_select = [
        config.COL_AIRPORT_IATA, 
        config.COL_AIRPORT_TYPE, 
        config.COL_AIRPORT_NAME,
        config.COL_AIRPORT_LATITUDE, # New
        config.COL_AIRPORT_LONGITUDE # New
    ]
    # Ensure all selected columns actually exist in df_airports
    airport_cols_to_select = [col for col in airport_cols_to_select if col in df_airports.columns]


    df_airports_origin = df_airports[airport_cols_to_select].rename(columns={
        config.COL_AIRPORT_IATA: config.COL_FL_ORIGIN, 
        config.COL_AIRPORT_TYPE: 'ORIGIN_AIRPORT_TYPE',
        config.COL_AIRPORT_NAME: 'ORIGIN_AIRPORT_NAME',
        config.COL_AIRPORT_LATITUDE: 'ORIGIN_LATITUDE',   # New
        config.COL_AIRPORT_LONGITUDE: 'ORIGIN_LONGITUDE'  # New
    })
    df_airports_dest = df_airports[airport_cols_to_select].rename(columns={
        config.COL_AIRPORT_IATA: config.COL_FL_DEST, 
        config.COL_AIRPORT_TYPE: 'DEST_AIRPORT_TYPE',
        config.COL_AIRPORT_NAME: 'DEST_AIRPORT_NAME',
        config.COL_AIRPORT_LATITUDE: 'DEST_LATITUDE',     # New
        config.COL_AIRPORT_LONGITUDE: 'DEST_LONGITUDE'    # New
    })

    merged_df = pd.merge(df_flights, df_airports_origin, on=config.COL_FL_ORIGIN, how='inner')
    rows_after_origin_join = len(merged_df)
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, table_name, "Rows After Origin Airport Join", rows_after_origin_join)
    print(f"Flights after inner join with origin airports: {rows_after_origin_join} rows.")
    
    merged_df = pd.merge(merged_df, df_airports_dest, on=config.COL_FL_DEST, how='inner')
    rows_after_dest_join = len(merged_df)
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, table_name, "Rows After Destination Airport Join", rows_after_dest_join)
    print(f"Flights after inner join with destination airports: {rows_after_dest_join} rows.")
        
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, table_name, "Final Row Count After Airport Joins", len(merged_df))
    return merged_df

# Functions join_with_tickets_data, calculate_flight_costs, calculate_flight_revenue,
# and create_analytical_base_table remain largely the same as in feature_engineering_utils_py_v2.
# For brevity, I will not repeat them here but assume they are present.
# create_analytical_base_table will now naturally include the new coordinate columns
# if they are present in the output of join_flights_with_airports.

def join_with_tickets_data(df_flights_merged, df_tickets):
    table_name = "Flights_Merged_With_Tickets"; step = "Join Flights with Tickets (Aggregated)"
    dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, "Flights_With_Airports", "Row Count Before Ticket Join", len(df_flights_merged))
    if df_tickets.empty:
        print("Warning: Tickets DataFrame is empty. AVG_ROUTE_ITIN_FARE will be all NaN.")
        avg_fare_per_route = pd.DataFrame(columns=[config.COL_FL_ORIGIN, config.COL_FL_DEST, 'AVG_ROUTE_ITIN_FARE'])
        dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, "Tickets_Cleaned", "Row Count", 0)
    else:
        dq_utils.log_dq_metric(DQ_PHASE_FEATURE_ENG, step, "Tickets_Cleaned", "Row Count", len(df_tickets))
        avg_fare_per_route = df_tickets.groupby([config.COL_TK_ORIGIN, config.COL_TK_DEST])\
                                       [config.COL_TK_ITIN_FARE].mean().reset_index()
        avg_fare_per_route = avg_fare_per_route.rename(columns={
            config.COL_TK_ORIGIN: config.COL_FL_ORIGIN, config.COL_TK_DEST: config.COL_FL_DEST,
            config.COL_TK_ITIN_FARE: 'AVG_ROUTE_ITIN_FARE'
        })
    merged_df = pd.merge(df_flights_merged, avg_fare_per_route, on=[config.COL_FL_ORIGIN, config.COL_FL_DEST], how='left')
    flights_with_fare = merged_df['AVG_ROUTE_ITIN_FARE'].notna().sum(); flights_without_fare = merged_df['AVG_ROUTE_ITIN_FARE'].isna().sum()
    print(f"Joined with aggregated ticket data. {flights_with_fare} flights got fare, {flights_without_fare} did not.")
    return merged_df

def calculate_flight_costs(df):
    df_cost = df.copy(); table_name = "AnalyticalData"; step = "Calculate Flight Costs"
    if config.COL_FL_DISTANCE in df_cost.columns and pd.api.types.is_numeric_dtype(df_cost[config.COL_FL_DISTANCE]):
        df_cost['COST_FUEL_ETC'] = df_cost[config.COL_FL_DISTANCE] * config.FUEL_OIL_MAINTENANCE_CREW_COST_PER_MILE
        df_cost['COST_DEPRECIATION_ETC'] = df_cost[config.COL_FL_DISTANCE] * config.DEPRECIATION_INSURANCE_OTHER_COST_PER_MILE
    else: df_cost['COST_FUEL_ETC'] = 0.0; df_cost['COST_DEPRECIATION_ETC'] = 0.0
    if 'DEST_AIRPORT_TYPE' in df_cost.columns:
        df_cost['COST_AIRPORT_OPERATIONAL'] = df_cost['DEST_AIRPORT_TYPE'].apply(lambda x: config.MEDIUM_AIRPORT_OPERATIONAL_COST if x == 'medium_airport' else (config.LARGE_AIRPORT_OPERATIONAL_COST if x == 'large_airport' else 0))
    else: df_cost['COST_AIRPORT_OPERATIONAL'] = 0.0
    for delay_col, cost_col_name in [(config.COL_FL_DEP_DELAY, 'COST_DEP_DELAY'), (config.COL_FL_ARR_DELAY, 'COST_ARR_DELAY')]:
        if delay_col in df_cost.columns and pd.api.types.is_numeric_dtype(df_cost[delay_col]):
            df_cost[f"{delay_col}_CHARGEABLE"] = (df_cost[delay_col] - config.FREE_DELAY_MINUTES).clip(lower=0)
            df_cost[cost_col_name] = df_cost[f"{delay_col}_CHARGEABLE"] * config.DELAY_COST_PER_MINUTE
        else: df_cost[cost_col_name] = 0.0
    cost_components = ['COST_FUEL_ETC', 'COST_DEPRECIATION_ETC', 'COST_AIRPORT_OPERATIONAL', 'COST_DEP_DELAY', 'COST_ARR_DELAY']
    df_cost['TOTAL_FLIGHT_COST'] = df_cost[[col for col in cost_components if col in df_cost.columns]].sum(axis=1)
    return df_cost

def calculate_flight_revenue(df):
    df_rev = df.copy(); table_name = "AnalyticalData"; step = "Calculate Flight Revenue"
    if config.COL_FL_OCCUPANCY in df_rev.columns and pd.api.types.is_numeric_dtype(df_rev[config.COL_FL_OCCUPANCY]):
        df_rev['CALCULATED_PASSENGERS'] = (df_rev[config.COL_FL_OCCUPANCY] * config.MAX_PASSENGERS_PER_PLANE).round().astype(int)
    else: df_rev['CALCULATED_PASSENGERS'] = 0
    if 'AVG_ROUTE_ITIN_FARE' in df_rev.columns and 'CALCULATED_PASSENGERS' in df_rev.columns:
        df_rev['TICKET_REVENUE_PER_LEG'] = (df_rev['AVG_ROUTE_ITIN_FARE'] / 2) * df_rev['CALCULATED_PASSENGERS']
    else: df_rev['TICKET_REVENUE_PER_LEG'] = np.nan
    df_rev.loc[:, 'TICKET_REVENUE_PER_LEG'] = pd.to_numeric(df_rev['TICKET_REVENUE_PER_LEG'], errors='coerce').fillna(0)
    if 'CALCULATED_PASSENGERS' in df_rev.columns:
         df_rev['BAGGAGE_REVENUE_PER_LEG'] = (df_rev['CALCULATED_PASSENGERS'] * config.PASSENGER_CHECKED_BAG_RATIO * config.AVERAGE_BAGS_PER_CHECKING_PASSENGER * config.BAGGAGE_FEE_PER_BAG_PER_FLIGHT)
    else: df_rev['BAGGAGE_REVENUE_PER_LEG'] = 0.0
    revenue_components = ['TICKET_REVENUE_PER_LEG', 'BAGGAGE_REVENUE_PER_LEG']
    df_rev['TOTAL_FLIGHT_REVENUE'] = df_rev[[col for col in revenue_components if col in df_rev.columns]].sum(axis=1)
    if 'TOTAL_FLIGHT_REVENUE' in df_rev.columns and 'TOTAL_FLIGHT_COST' in df_rev.columns:
        df_rev['PROFIT_PER_LEG'] = df_rev['TOTAL_FLIGHT_REVENUE'] - df_rev['TOTAL_FLIGHT_COST']
    else: df_rev['PROFIT_PER_LEG'] = np.nan
    return df_rev

def create_analytical_base_table(df_flights_cleaned, df_tickets_cleaned, df_airports_cleaned):
    print("--- Starting Feature Engineering Phase ---")
    if df_flights_cleaned is None or df_flights_cleaned.empty: return pd.DataFrame()
    if df_airports_cleaned is None or df_airports_cleaned.empty: return pd.DataFrame()
    if df_tickets_cleaned is None: df_tickets_cleaned = pd.DataFrame()

    df_flights_with_route = create_route_columns(df_flights_cleaned, config.COL_FL_ORIGIN, config.COL_FL_DEST)
    abt = join_flights_with_airports(df_flights_with_route, df_airports_cleaned)
    if abt.empty: return abt 
    abt = join_with_tickets_data(abt, df_tickets_cleaned)
    if abt.empty: return abt 
    abt = calculate_flight_costs(abt)
    abt = calculate_flight_revenue(abt)
    # ... (final checks and logging as in v2) ...
    print(f"--- Feature Engineering Phase Complete. ABT created with shape: {abt.shape} ---")
    return abt

if __name__ == '__main__':
    print("Testing feature_engineering_utils.py...")
