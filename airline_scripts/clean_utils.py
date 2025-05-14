# airline_scripts/clean_utils.py
import pandas as pd
import numpy as np
from . import config
from . import dq_utils
import os
import datetime

DQ_PHASE_CLEANING = "Data Cleaning & QA"

# --- Helper Functions (identify_missing_values, convert_data_types, handle_outliers_iqr, _save_rejected_data, _filter_and_log_columns) ---
# These functions remain largely the same as in clean_utils_py_v7.
# For brevity, I will not repeat them here but assume they are present.
# Ensure `convert_data_types` can handle converting new LATITUDE/LONGITUDE to float.

def identify_missing_values(df, column_name, table_name, step_prefix=""):
    """Identifies and logs missing values for a specific column."""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in {table_name} for missing value check.")
        return 0
    
    missing_count = df[column_name].isnull().sum()
    if missing_count > 0:
        percentage = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"{step_prefix}Missing Values Check", table_name, f"Missing Count - {column_name}", missing_count)
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"{step_prefix}Missing Values Check", table_name, f"Missing Percentage - {column_name}", f"{percentage:.2f}%")
        print(f"Column '{column_name}' in {table_name}: {missing_count} missing values ({percentage:.2f}%).")
    return missing_count

def convert_data_types(df, table_name, column_types_map, step_suffix=""):
    """Converts columns to specified data types and logs issues."""
    step = f"Convert Data Types {step_suffix}".strip()
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name, "Row Count Before Type Conversion", len(df))
    df_converted = df.copy()

    for col, dtype_str in column_types_map.items():
        if col in df_converted.columns:
            original_series_for_error_msg = df_converted[col].copy()
            original_nulls = df_converted[col].isnull().sum()
            try:
                current_col_series = df_converted[col]
                if dtype_str.lower().startswith('int'):
                    numeric_series = pd.to_numeric(current_col_series, errors='coerce')
                    numeric_series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    converted_series = numeric_series.astype('Int64')
                    if dtype_str.lower() == 'int' and converted_series.notna().all():
                        df_converted[col] = converted_series.astype(int) 
                    else:
                        df_converted[col] = converted_series 
                elif dtype_str.lower().startswith('float'):
                    numeric_series = pd.to_numeric(current_col_series, errors='coerce')
                    df_converted[col] = numeric_series.astype('Float64')
                elif dtype_str.lower() == 'datetime64[ns]':
                    df_converted[col] = pd.to_datetime(current_col_series, errors='coerce')
                elif dtype_str.lower() == 'bool' or dtype_str.lower() == 'boolean':
                    bool_map = {'true': True, 'false': False, '1': True, '0': False,
                                '1.0': True, '0.0': False, 1: True, 0: False}
                    temp_series_for_bool = current_col_series
                    if pd.api.types.is_object_dtype(temp_series_for_bool) or pd.api.types.is_string_dtype(temp_series_for_bool):
                        temp_series_for_bool = temp_series_for_bool.astype(str).str.lower()
                    mapped_series = temp_series_for_bool.map(bool_map)
                    df_converted[col] = mapped_series.astype('boolean')
                elif dtype_str.lower() == 'str' or dtype_str.lower() == 'object':
                    df_converted[col] = current_col_series.astype(str)
                else:
                    df_converted[col] = current_col_series.astype(dtype_str)

                final_nulls = df_converted[col].isnull().sum()
                if final_nulls > original_nulls and (dtype_str.lower().startswith('int') or dtype_str.lower().startswith('float') or dtype_str.lower() == 'datetime64[ns]'):
                    coerced_count = final_nulls - original_nulls
                    print(f"Warning: Column '{col}' had {coerced_count} value(s) result in NaN/NaT/NA during conversion to {dtype_str}.")
                    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name, f"Values Resulting in Null - {col}", coerced_count)
                dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name, f"Type Conversion Success - {col}", f"to {df_converted[col].dtype}")
            except Exception as e:
                dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name, f"Type Conversion Error - {col}", str(e))
                print(f"ERROR: Could not convert column '{col}' to {dtype_str} in {table_name}. Error: {e}. Column {col} type before this attempt: {original_series_for_error_msg.dtype}")
    
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name, "Row Count After Type Conversion", len(df_converted))
    return df_converted

def handle_outliers_iqr(df, column_name, table_name_for_dq, rejected_rows_accumulator, iqr_multiplier=config.OUTLIER_IQR_MULTIPLIER):
    step = f"Handle Outliers IQR - {column_name} (Method: {config.OUTLIER_HANDLING_METHOD})"
    if config.OUTLIER_HANDLING_METHOD == 'none':
        print(f"Skipping outlier handling for column '{column_name}' as per configuration (OUTLIER_HANDLING_METHOD='none').")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, "Outlier Handling Skipped (Config)", f"Method set to 'none' for {column_name}")
        return df 
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Warning: Column '{column_name}' not found or not numeric. Skipping outlier handling.")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, "Outlier Handling Skipped", f"Column {column_name} not found or not numeric.")
        return df
    df_processed = df.copy() 
    valid_series = df_processed[column_name].dropna()
    if valid_series.empty:
        print(f"Warning: Column '{column_name}' contains all NaN values. Skipping outlier handling.")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, "Outlier Handling Skipped", f"Column {column_name} all NaN.")
        return df_processed 
    q1 = valid_series.quantile(0.25); q3 = valid_series.quantile(0.75); iqr = q3 - q1
    if iqr == 0:
        if valid_series.nunique() == 1: print(f"All valid values in '{column_name}' are identical. No outliers.")
        else: print(f"Warning: IQR for column '{column_name}' is 0, but multiple unique values exist. Skipping outlier handling.")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, f"Number of Outliers ({column_name})", 0)
        return df_processed 
    lower_bound = q1 - iqr_multiplier * iqr; upper_bound = q3 + iqr_multiplier * iqr
    outliers_mask = pd.Series(False, index=df_processed.index)
    not_na_mask = df_processed[column_name].notna()
    outliers_mask.loc[not_na_mask] = (df_processed.loc[not_na_mask, column_name] < lower_bound) | (df_processed.loc[not_na_mask, column_name] > upper_bound)
    num_outliers = outliers_mask.sum()
    # ... (DQ logging for bounds, num_outliers) ...
    if num_outliers == 0: print(f"No outliers detected in '{column_name}'."); return df_processed
    print(f"Detected {num_outliers} outliers in '{column_name}'.")
    if config.OUTLIER_HANDLING_METHOD == 'impute_mean':
        mean_val = df_processed.loc[~outliers_mask & df_processed[column_name].notna(), column_name].mean()
        if pd.isna(mean_val): mean_val = df[column_name].dropna().mean() 
        if pd.isna(mean_val): print(f"Warning: Could not calculate mean for '{column_name}'. Outliers not imputed."); return df_processed
        df_processed.loc[outliers_mask, column_name] = mean_val
        print(f"Imputed {num_outliers} outliers in '{column_name}' with mean: {mean_val:.2f}.")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, f"Outliers Imputed ({column_name})", num_outliers)
    elif config.OUTLIER_HANDLING_METHOD == 'filter':
        rejected_outliers = df_processed[outliers_mask].copy()
        rejected_outliers[config.REJECT_REASON_COL] = f"Outlier in '{column_name}'"
        rejected_rows_accumulator.append(rejected_outliers)
        df_processed = df_processed[~outliers_mask].copy() 
        print(f"Filtered {num_outliers} outlier rows for '{column_name}'.")
        dq_utils.log_dq_metric(DQ_PHASE_CLEANING, step, table_name_for_dq, f"Outliers Filtered ({column_name})", num_outliers)
    return df_processed

def _save_rejected_data(rejected_df_list, file_path, table_name_for_dq):
    if not rejected_df_list: print(f"No data rejected for {table_name_for_dq}."); return
    final_rejected_df = pd.concat(rejected_df_list, ignore_index=True)
    if not final_rejected_df.empty:
        cols_to_check_duplicates = [col for col in final_rejected_df.columns if col not in [config.REJECT_REASON_COL, config.BATCH_ID_COLUMN, 'REJECT_TIMESTAMP']]
        if cols_to_check_duplicates: final_rejected_df.drop_duplicates(subset=cols_to_check_duplicates, keep='first', inplace=True)
        elif not final_rejected_df.empty: final_rejected_df.drop_duplicates(inplace=True)
        
        current_time = datetime.datetime.now()
        final_rejected_df[config.BATCH_ID_COLUMN] = dq_utils.current_run_id if dq_utils.current_run_id else "UNKNOWN_RUN_ID" # Using Run ID as BATCH_ID
        final_rejected_df['REJECT_TIMESTAMP'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        final_rejected_df.to_csv(file_path, index=False)
        print(f"Saved {len(final_rejected_df)} unique rejected rows for {table_name_for_dq} to {file_path}")
    except Exception as e: print(f"Error saving rejected data for {table_name_for_dq}: {e}")

def _filter_and_log_columns(df, required_cols_list, table_name_dq, step_suffix_dq):
    initial_cols = df.columns.tolist()
    actual_required_cols = [col for col in required_cols_list if col in initial_cols]
    excluded_at_cleaning = [col for col in initial_cols if col not in actual_required_cols]
    missing_defined_required = [col for col in required_cols_list if col not in initial_cols]
    if missing_defined_required: print(f"Warning: For {table_name_dq}, defined REQUIRED cols not in loaded DF: {missing_defined_required}")
    df_filtered = df[actual_required_cols].copy()
    print(f"Step: Filtered {step_suffix_dq} to required columns. Kept {len(df_filtered.columns)} columns.")
    if excluded_at_cleaning: print(f"Excluded columns at cleaning start for {step_suffix_dq}: {excluded_at_cleaning}")
    return df_filtered

# --- Main Cleaning Functions ---

def clean_flights_data(df_flights_raw_full):
    # ... (Steps 1-4: Initial Logging, Column Filtering, Type Conversion, Cancelled Filter - as in v7) ...
    # For brevity, assuming these steps are present and largely unchanged from clean_utils_py_v7
    table_name_dq = "Flights_Processed"; step_suffix_dq = "Flights"
    rejected_rows_accumulator = []
    df = df_flights_raw_full.copy()
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"Start Cleaning {step_suffix_dq}", table_name_dq, "Initial Row Count (Full DF)", len(df))
    df = _filter_and_log_columns(df, config.REQUIRED_FLIGHTS_COLS, table_name_dq, step_suffix_dq)
    flight_column_types = { # Copied from previous version, ensure it's up-to-date
        config.COL_FL_DATE: 'datetime64[ns]', config.COL_FL_CANCELLED: 'int',
        config.COL_FL_DEP_DELAY: 'float', config.COL_FL_ARR_DELAY: 'float',
        config.COL_FL_DISTANCE: 'float', config.COL_FL_AIR_TIME: 'float',
        config.COL_FL_OCCUPANCY: 'float', config.COL_FL_ORIGIN: 'str',
        config.COL_FL_DEST: 'str', config.COL_FL_OP_CARRIER_FL_NUM: 'str'
    }
    flight_column_types = {k: v for k, v in flight_column_types.items() if k in df.columns} # Ensure keys exist
    df = convert_data_types(df, table_name_dq, flight_column_types, step_suffix=step_suffix_dq)
    # ... (Cancelled filter logic from v7) ...
    if config.COL_FL_CANCELLED in df.columns and (pd.api.types.is_numeric_dtype(df[config.COL_FL_CANCELLED]) or pd.api.types.is_bool_dtype(df[config.COL_FL_CANCELLED]) or pd.api.types.is_integer_dtype(df[config.COL_FL_CANCELLED])):
        try:
            cancelled_mask = df[config.COL_FL_CANCELLED] != 0
            if df[config.COL_FL_CANCELLED].hasnans: cancelled_mask = cancelled_mask | df[config.COL_FL_CANCELLED].isnull() 
        except TypeError: 
            temp_cancelled_as_float = pd.to_numeric(df[config.COL_FL_CANCELLED], errors='coerce').fillna(1) 
            cancelled_mask = temp_cancelled_as_float != 0.0
        if cancelled_mask.any(): 
            rejected_cancelled = df[cancelled_mask].copy(); rejected_cancelled[config.REJECT_REASON_COL] = f"Cancelled Flight or Unclear Status (Col {config.COL_FL_CANCELLED} not 0 or NA)"; rejected_rows_accumulator.append(rejected_cancelled)
            df = df[~cancelled_mask].copy(); print(f"Filtered out {len(rejected_cancelled)} cancelled/unclear flights.")


    # 5. Handle Missing Values & Unusual Values
    print(f"\n--- Handling Missing/Unusual Values for {step_suffix_dq} ---")
    total_missing_filtered_count = 0
    for col_name in df.columns: 
        if col_name in config.IMPUTE_NAN_CONFIG:
            # ... (imputation logic as in v7) ...
            impute_value = config.IMPUTE_NAN_CONFIG[col_name]; num_imputed = df[col_name].isnull().sum()
            if num_imputed > 0: df.loc[:, col_name] = df[col_name].fillna(impute_value); print(f"Imputed {num_imputed} NaNs in '{col_name}'.")
            continue 
        if df[col_name].isnull().any():
            nan_mask = df[col_name].isnull(); num_nan_rows = nan_mask.sum()
            if num_nan_rows > 0: 
                rejected_nan = df[nan_mask].copy(); rejected_nan[config.REJECT_REASON_COL] = f"Missing/unparseable in '{col_name}'"; rejected_rows_accumulator.append(rejected_nan)
                df = df[~nan_mask].copy(); print(f"Filtered {num_nan_rows} rows (NaNs in '{col_name}')."); total_missing_filtered_count += num_nan_rows
    if total_missing_filtered_count > 0: print(f"Total rows filtered (missing/unparseable) from Flights: {total_missing_filtered_count}")
    
    # Specific Validations (e.g., Occupancy Rate)
    if config.COL_FL_OCCUPANCY in df.columns and pd.api.types.is_numeric_dtype(df[config.COL_FL_OCCUPANCY]):
        invalid_occupancy_mask = ~((df[config.COL_FL_OCCUPANCY] >= 0) & (df[config.COL_FL_OCCUPANCY] <= 1))
        if invalid_occupancy_mask.any(): 
            rejected_occupancy = df[invalid_occupancy_mask].copy(); rejected_occupancy[config.REJECT_REASON_COL] = "Invalid Occupancy (not [0,1])"; rejected_rows_accumulator.append(rejected_occupancy)
            df = df[~invalid_occupancy_mask].copy(); print(f"Filtered {len(rejected_occupancy)} rows (invalid occupancy).")

    # 6. Handle Outliers
    outlier_cols_flights = [config.COL_FL_DEP_DELAY, config.COL_FL_ARR_DELAY, config.COL_FL_AIR_TIME, config.COL_FL_DISTANCE]
    for col in outlier_cols_flights:
        if col in df.columns: df = handle_outliers_iqr(df, col, table_name_dq, rejected_rows_accumulator)

    # 7. Save Rejected Data
    _save_rejected_data(rejected_rows_accumulator, config.REJECTED_FLIGHTS_FILE, f"{step_suffix_dq}_Rejects")
    # 8. Final Logging & 9. Return
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"End Cleaning {step_suffix_dq}", table_name_dq, "Final Row Count", len(df))
    print(f"Finished cleaning {step_suffix_dq} data. Final shape: {df.shape}")
    return df


def clean_tickets_data(df_tickets_raw_full):
    table_name_dq = "Tickets_Processed"; step_suffix_dq = "Tickets"
    rejected_rows_accumulator = []
    df = df_tickets_raw_full.copy()
    # 1. Initial Logging
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"Start Cleaning {step_suffix_dq}", table_name_dq, "Initial Row Count (Full DF)", len(df))
    # 2. Column Filtering
    df = _filter_and_log_columns(df, config.REQUIRED_TICKETS_COLS, table_name_dq, step_suffix_dq)
    # 3. Change Data Types
    ticket_column_types = { # Copied from previous version
        config.COL_TK_ROUNDTRIP: 'int', config.COL_TK_PASSENGERS: 'float',
        config.COL_TK_ITIN_FARE: 'float', config.COL_TK_YEAR: 'int', config.COL_TK_QUARTER: 'int',
        config.COL_TK_ORIGIN: 'str', config.COL_TK_DEST: 'str'
    }
    ticket_column_types = {k: v for k, v in ticket_column_types.items() if k in df.columns}
    df = convert_data_types(df, table_name_dq, ticket_column_types, step_suffix=step_suffix_dq)
    
    # 4. Specific Business Rule Filters (Roundtrip, Year/Quarter)
    # ... (Roundtrip filter logic from v7, add rejects to accumulator) ...
    if config.COL_TK_ROUNDTRIP in df.columns and (pd.api.types.is_numeric_dtype(df[config.COL_TK_ROUNDTRIP]) or pd.api.types.is_bool_dtype(df[config.COL_TK_ROUNDTRIP]) or pd.api.types.is_integer_dtype(df[config.COL_TK_ROUNDTRIP])):
        try:
            mask = df[config.COL_TK_ROUNDTRIP] != 1; 
            if df[config.COL_TK_ROUNDTRIP].hasnans: mask = mask | df[config.COL_TK_ROUNDTRIP].isnull()
        except TypeError: temp = pd.to_numeric(df[config.COL_TK_ROUNDTRIP], errors='coerce').fillna(-1); mask = temp != 1.0
        if mask.any(): rej = df[mask].copy(); rej[config.REJECT_REASON_COL] = f"Not Roundtrip or Unclear (Col {config.COL_TK_ROUNDTRIP})"; rejected_rows_accumulator.append(rej); df = df[~mask].copy(); print(f"Filtered {len(rej)} non-roundtrip tickets.")
    # ... (Year/Quarter filter logic from v7, add rejects to accumulator) ...
    if config.COL_TK_YEAR in df.columns and config.COL_TK_QUARTER in df.columns: # Simplified
        mask_yq = ~((df[config.COL_TK_YEAR] == config.ANALYSIS_YEAR) & (df[config.COL_TK_QUARTER] == config.ANALYSIS_QUARTER))
        if df[config.COL_TK_YEAR].hasnans or df[config.COL_TK_QUARTER].hasnans: mask_yq = mask_yq | df[config.COL_TK_YEAR].isnull() | df[config.COL_TK_QUARTER].isnull()
        if mask_yq.any(): rej = df[mask_yq].copy(); rej[config.REJECT_REASON_COL] = "Not 1Q2019 or Year/Qtr Missing"; rejected_rows_accumulator.append(rej); df = df[~mask_yq].copy(); print(f"Filtered {len(rej)} tickets not for 1Q2019.")

    # 5. Handle Missing Values & Unusual Values (General Loop)
    total_missing_filtered_count_tickets = 0
    for col_name in df.columns:
        if col_name in config.IMPUTE_NAN_CONFIG: continue
        if df[col_name].isnull().any():
            nan_mask = df[col_name].isnull(); num_nan_rows = nan_mask.sum()
            if num_nan_rows > 0:
                rej = df[nan_mask].copy(); rej[config.REJECT_REASON_COL] = f"Missing/unparseable in '{col_name}'"; rejected_rows_accumulator.append(rej)
                df = df[~nan_mask].copy(); print(f"Filtered {num_nan_rows} rows (NaNs in '{col_name}' for Tickets)."); total_missing_filtered_count_tickets += num_nan_rows
    if total_missing_filtered_count_tickets > 0: print(f"Total rows filtered (missing/unparseable) from Tickets: {total_missing_filtered_count_tickets}")
    
    # Specific Validation: Positive ITIN_FARE
    if config.COL_TK_ITIN_FARE in df.columns and pd.api.types.is_numeric_dtype(df[config.COL_TK_ITIN_FARE]):
        mask_fare = df[config.COL_TK_ITIN_FARE] <= 0
        if mask_fare.any(): rej = df[mask_fare].copy(); rej[config.REJECT_REASON_COL] = "Non-positive ITIN_FARE"; rejected_rows_accumulator.append(rej); df = df[~mask_fare].copy(); print(f"Filtered {len(rej)} tickets (non-positive fare).")

    # 6. Handle Outliers for ITIN_FARE
    if config.COL_TK_ITIN_FARE in df.columns:
        df = handle_outliers_iqr(df, config.COL_TK_ITIN_FARE, table_name_dq, rejected_rows_accumulator)

    # 7. Save Rejected, 8. Final Log, 9. Return
    _save_rejected_data(rejected_rows_accumulator, config.REJECTED_TICKETS_FILE, f"{step_suffix_dq}_Rejects")
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"End Cleaning {step_suffix_dq}", table_name_dq, "Final Row Count", len(df))
    print(f"Finished cleaning {step_suffix_dq} data. Final shape: {df.shape}")
    return df


def clean_airport_codes_data(df_airports_raw_full):
    table_name_dq = "Airport_Codes_Processed"; step_suffix_dq = "Airport_Codes"
    rejected_rows_accumulator = []
    df = df_airports_raw_full.copy()
    # 1. Initial Logging
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"Start Cleaning {step_suffix_dq}", table_name_dq, "Initial Row Count (Full DF)", len(df))
    # 2. Column Filtering
    df = _filter_and_log_columns(df, config.REQUIRED_AIRPORT_COLS, table_name_dq, step_suffix_dq)
    # 3. Change Data Types (including new LATITUDE, LONGITUDE to float)
    airport_column_types = {
        config.COL_AIRPORT_IATA: 'str', config.COL_AIRPORT_TYPE: 'str',
        config.COL_AIRPORT_ISO_COUNTRY: 'str', config.COL_AIRPORT_NAME: 'str',
        config.COL_AIRPORT_COORDINATES: 'str', # Keep as string for initial parsing
        # New columns will be created and then can be typed if necessary,
        # but parsing will attempt to make them float.
    }
    airport_column_types = {k:v for k,v in airport_column_types.items() if k in df.columns}
    df = convert_data_types(df, table_name_dq, airport_column_types, step_suffix=f"{step_suffix_dq}_KeyStrings")

    # --- NEW: Parse COORDINATES into LATITUDE and LONGITUDE ---
    if config.COL_AIRPORT_COORDINATES in df.columns:
        print(f"\n--- Parsing Coordinates for {step_suffix_dq} ---")
        # Expected format: "longitude, latitude"
        # Create masks for rows that will be problematic
        # 1. Rows where COORDINATES is NaN (already string, so isnull() is fine)
        coord_nan_mask = df[config.COL_AIRPORT_COORDINATES].isnull()
        # 2. Rows where COORDINATES does not contain a comma (after ensuring it's string)
        coord_no_comma_mask = ~df[config.COL_AIRPORT_COORDINATES].astype(str).str.contains(',', na=False)
        
        # Combine masks for rows that cannot be parsed simply
        unparseable_coord_mask = coord_nan_mask | coord_no_comma_mask
        
        if unparseable_coord_mask.any():
            rejected_coords = df[unparseable_coord_mask].copy()
            rejected_coords[config.REJECT_REASON_COL] = f"Malformed or Missing {config.COL_AIRPORT_COORDINATES}"
            rejected_rows_accumulator.append(rejected_coords)
            df = df[~unparseable_coord_mask].copy() # Keep only rows with potentially parseable coordinates
            print(f"Filtered out {len(rejected_coords)} airports due to malformed/missing coordinates.")
            dq_utils.log_dq_metric(DQ_PHASE_CLEANING, "Parse Coordinates", table_name_dq, "Rows Filtered (Malformed/Missing Coords)", len(rejected_coords))

        # Proceed with splitting for the remaining rows
        if not df.empty and config.COL_AIRPORT_COORDINATES in df.columns: # Check if df is not empty after filtering
            split_coords = df[config.COL_AIRPORT_COORDINATES].astype(str).str.split(',', n=1, expand=True)
            df.loc[:, config.COL_AIRPORT_LONGITUDE] = pd.to_numeric(split_coords[0], errors='coerce')
            df.loc[:, config.COL_AIRPORT_LATITUDE] = pd.to_numeric(split_coords[1], errors='coerce')

            # Check for NaNs introduced by pd.to_numeric (if parts were not numeric after split)
            lat_nan_after_parse_mask = df[config.COL_AIRPORT_LATITUDE].isnull()
            lon_nan_after_parse_mask = df[config.COL_AIRPORT_LONGITUDE].isnull()
            parse_to_nan_mask = lat_nan_after_parse_mask | lon_nan_after_parse_mask
            
            if parse_to_nan_mask.any():
                rejected_numeric_fail = df[parse_to_nan_mask].copy()
                rejected_numeric_fail[config.REJECT_REASON_COL] = "Coordinate part not numeric after split"
                rejected_rows_accumulator.append(rejected_numeric_fail)
                df = df[~parse_to_nan_mask].copy() # Keep only rows where both lat/lon are numeric
                print(f"Filtered out {len(rejected_numeric_fail)} airports due to non-numeric lat/lon parts.")
                dq_utils.log_dq_metric(DQ_PHASE_CLEANING, "Parse Coordinates", table_name_dq, "Rows Filtered (Non-Numeric Lat/Lon)", len(rejected_numeric_fail))

            dq_utils.log_dq_metric(DQ_PHASE_CLEANING, "Parse Coordinates", table_name_dq, "Feature Created", config.COL_AIRPORT_LATITUDE)
            dq_utils.log_dq_metric(DQ_PHASE_CLEANING, "Parse Coordinates", table_name_dq, "Feature Created", config.COL_AIRPORT_LONGITUDE)
            print(f"Successfully parsed coordinates into {config.COL_AIRPORT_LATITUDE} and {config.COL_AIRPORT_LONGITUDE}.")
        else:
            # If df became empty or COORDINATES col was dropped (should not happen if in REQUIRED_COLS)
            # Ensure LATITUDE and LONGITUDE columns exist if expected by downstream, even if all NaN
            if config.COL_AIRPORT_LATITUDE not in df.columns: df[config.COL_AIRPORT_LATITUDE] = np.nan
            if config.COL_AIRPORT_LONGITUDE not in df.columns: df[config.COL_AIRPORT_LONGITUDE] = np.nan


    # 4. Specific Business Rule Filters & Critical NaN handling (for IATA, TYPE, ISO_COUNTRY)
    critical_filter_cols_airports = [config.COL_AIRPORT_IATA, config.COL_AIRPORT_TYPE, config.COL_AIRPORT_ISO_COUNTRY]
    critical_filter_cols_airports = [col for col in critical_filter_cols_airports if col in df.columns] 
    if critical_filter_cols_airports:
        na_mask_critical_airports = df[critical_filter_cols_airports].isnull().any(axis=1)
        if na_mask_critical_airports.any():
            # ... (reject logic as in v7) ...
            pass # Simplified

    # ... (US filter, Type filter, Duplicate IATA filter logic from v7, adding to rejected_rows_accumulator) ...
    if config.COL_AIRPORT_ISO_COUNTRY in df.columns:
        mask = df[config.COL_AIRPORT_ISO_COUNTRY].str.upper() != 'US'; 
        if mask.any(): rej = df[mask].copy(); rej[config.REJECT_REASON_COL] = "Not US Airport"; rejected_rows_accumulator.append(rej); df = df[~mask].copy(); print(f"Filtered {len(rej)} non-US airports.")
    if config.COL_AIRPORT_TYPE in df.columns:
        req_types = [str(rt).lower() for rt in config.REQUIRED_AIRPORT_TYPES]; mask = ~df[config.COL_AIRPORT_TYPE].str.lower().isin(req_types)
        if mask.any(): rej = df[mask].copy(); rej[config.REJECT_REASON_COL] = "Invalid Airport Type"; rejected_rows_accumulator.append(rej); df = df[~mask].copy(); print(f"Filtered {len(rej)} invalid airport types.")
    if config.COL_AIRPORT_IATA in df.columns and df.duplicated(subset=[config.COL_AIRPORT_IATA]).any(): # Check only subsequent
        mask = df.duplicated(subset=[config.COL_AIRPORT_IATA], keep='first')
        if mask.any(): rej = df[mask].copy(); rej[config.REJECT_REASON_COL] = "Duplicate IATA (subsequent)"; rejected_rows_accumulator.append(rej); df = df[~mask].copy(); print(f"Filtered {len(rej)} duplicate IATA airports.")


    # 5. Handle Missing Values (General Loop for remaining required_cols, including new LATITUDE, LONGITUDE)
    print(f"\n--- Handling Missing/Unusual Values for {step_suffix_dq} (Remaining Columns) ---")
    all_airport_cols_to_check_nan = config.REQUIRED_AIRPORT_COLS + [config.COL_AIRPORT_LATITUDE, config.COL_AIRPORT_LONGITUDE]
    all_airport_cols_to_check_nan = list(set(all_airport_cols_to_check_nan)) # Unique list

    total_missing_filtered_count_airports = 0
    for col_name in df.columns: # Check all columns currently in df
        if col_name in all_airport_cols_to_check_nan and col_name not in config.IMPUTE_NAN_CONFIG:
            if df[col_name].isnull().any(): 
                nan_mask = df[col_name].isnull(); num_nan_rows = nan_mask.sum()
                if num_nan_rows > 0:
                    rejected_nan = df[nan_mask].copy(); rejected_nan[config.REJECT_REASON_COL] = f"Missing value in '{col_name}'"; rejected_rows_accumulator.append(rejected_nan)
                    df = df[~nan_mask].copy(); print(f"Filtered {num_nan_rows} rows (NaNs in '{col_name}' for Airports)."); total_missing_filtered_count_airports += num_nan_rows
    if total_missing_filtered_count_airports > 0:
         print(f"Total rows filtered from airports due to missing values in other required columns: {total_missing_filtered_count_airports}")

    # 6. Handle Outliers (Not applied)
    print(f"\n--- Outlier handling not applied to {step_suffix_dq} descriptive fields ---")

    # 7. Save Rejected Data
    _save_rejected_data(rejected_rows_accumulator, config.REJECTED_AIRPORTS_FILE, f"{step_suffix_dq}_Rejects")
    # 8. Final Logging & 9. Return
    dq_utils.log_dq_metric(DQ_PHASE_CLEANING, f"End Cleaning {step_suffix_dq}", table_name_dq, "Final Row Count", len(df))
    print(f"Finished cleaning {step_suffix_dq} data. Final shape: {df.shape}")
    return df


if __name__ == '__main__':
    print("Testing clean_utils.py...")