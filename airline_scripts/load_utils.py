# airline_scripts/load_utils.py
import pandas as pd
from . import config
from . import dq_utils

def load_csv_data(file_path, table_name_for_dq, phase="Data Loading"):
    """
    Loads all data from a CSV file and logs initial DQ metrics.
    Column filtering will now happen in the cleaning phase.

    Args:
        file_path (str): The path to the CSV file.
        table_name_for_dq (str): The name of the table for DQ logging purposes.
        phase (str): The current data processing phase for DQ logging.

    Returns:
        pandas.DataFrame: The loaded data, or None if loading fails.
    """
    step = f"Load {table_name_for_dq} (All Columns)"
    try:
        df = pd.read_csv(file_path)
        
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "File Path", file_path)
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "Initial Row Count", len(df))
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "Loaded Column Count (All)", len(df.columns))
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "Loaded Column Names (All)", list(df.columns))
        print(f"Successfully loaded all columns for {table_name_for_dq} from {file_path}. Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "Error", error_msg)
        print(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error loading {file_path}: {e}"
        dq_utils.log_dq_metric(phase, step, table_name_for_dq, "Error", error_msg)
        print(error_msg)
        return None

def load_all_data():
    """
    Loads all necessary datasets (Flights, Tickets, Airport Codes),
    loading all columns initially.
   
    Returns:
        tuple: (flights_df, tickets_df, airports_df)
    """
    print("--- Starting Data Loading Phase (Loading ALL columns for profiling) ---")
    # The required_cols parameter is removed from these calls, so load_csv_data loads all columns.
    flights_df = load_csv_data(config.FLIGHTS_FILE, "Flights_Raw_Full")
    tickets_df = load_csv_data(config.TICKETS_FILE, "Tickets_Raw_Full")
    airports_df = load_csv_data(config.AIRPORT_CODES_FILE, "Airport_Codes_Raw_Full")
    print("--- Data Loading Phase Complete (All columns loaded) ---")
    return flights_df, tickets_df, airports_df

if __name__ == '__main__':
    print("Testing load_utils.py...")
    
    # Create dummy config for standalone testing
    class MockConfig:
        FLIGHTS_FILE = "dummy_flights.csv"
        TICKETS_FILE = "dummy_tickets.csv"
        AIRPORT_CODES_FILE = "dummy_airports.csv"
        
        # These are NOT used by load_utils anymore for selection, but clean_utils will use them.
        REQUIRED_FLIGHTS_COLS = ['FL_DATE', 'ORIGIN', 'DESTINATION', 'DISTANCE']
        REQUIRED_TICKETS_COLS = ['ORIGIN', 'ITIN_FARE']
        REQUIRED_AIRPORT_COLS = ['IATA_CODE', 'TYPE']
        
        DQ_COLUMNS = ['Timestamp', 'RunID', 'Phase', 'Step', 'TableName', 'Metric', 'Value', 'Description']
        DQ_LOG_FILE_PATH = "dummy_dq_log.csv"


    class MockDQUtils:
        dq_metrics_df = None
        current_run_id = "test_run_load_all"
        def initialize_dq_log(self): MockDQUtils.dq_metrics_df = pd.DataFrame(columns=MockConfig.DQ_COLUMNS)
        def log_dq_metric(self, phase, step, table_name, metric, value, description=""):
            if MockDQUtils.dq_metrics_df is None: self.initialize_dq_log()
            entry = pd.DataFrame([{'Timestamp': pd.Timestamp.now(), 'RunID': MockDQUtils.current_run_id, 
                                   'Phase': phase, 'Step': step, 'TableName': table_name, 
                                   'Metric': metric, 'Value': value, 'Description': description}])
            MockDQUtils.dq_metrics_df = pd.concat([MockDQUtils.dq_metrics_df, entry], ignore_index=True)
        def save_dq_log_to_csv(self, path=None):
            if MockDQUtils.dq_metrics_df is not None and not MockDQUtils.dq_metrics_df.empty:
                print(f"Mock saving DQ log to {path or MockConfig.DQ_LOG_FILE_PATH}")
        def display_dq_summary(self):
            if MockDQUtils.dq_metrics_df is not None: print(MockDQUtils.dq_metrics_df.tail())

    # Replace actual config and dq_utils with mocks for this test
    # This setup is a bit more involved because of the relative imports in a standalone test.
    # In a real run from the notebook, this mocking isn't needed.
    import sys
    # Temporarily mock the airline_scripts package structure if not running from project root
    if 'airline_scripts.config' not in sys.modules:
        sys.modules['airline_scripts.config'] = MockConfig()
        sys.modules['airline_scripts.dq_utils'] = MockDQUtils()
        # Re-import to use mocks if they were imported before this block
        from . import config as mock_cfg_ref # to ensure the 'config' name in load_all_data uses the mock
        from . import dq_utils as mock_dq_ref
        global config, dq_utils # Make sure the global names are updated for the functions in this file
        config = mock_cfg_ref
        dq_utils = mock_dq_ref
        dq_utils.initialize_dq_log()


    # Create dummy CSV files for testing
    pd.DataFrame({
        'FL_DATE': ['2023-01-01'], 'ORIGIN': ['JFK'], 'DESTINATION': ['LAX'], 
        'DISTANCE': [2000], 'UNUSED_FL_COL': ['A'], 'ANOTHER_UNUSED_FL': [1]
    }).to_csv(MockConfig.FLIGHTS_FILE, index=False)
    pd.DataFrame({
        'ORIGIN': ['JFK'], 'ITIN_FARE': [300.50], 'UNUSED_TK_COL': ['B'], 'PASSENGERS': [100]
    }).to_csv(MockConfig.TICKETS_FILE, index=False)
    pd.DataFrame({
        'IATA_CODE': ['JFK'], 'TYPE': ['large_airport'], 'UNUSED_AP_COL': ['C'], 'ELEVATION_FT': [13]
    }).to_csv(MockConfig.AIRPORT_CODES_FILE, index=False)

    print("\n--- Running load_all_data with mocks (expecting all columns) ---")
    flights, tickets, airports = load_all_data() # This now calls the global load_csv_data
    
    if flights is not None: 
        print(f"\nLoaded Flights Sample (all columns):\n{flights.head()}")
        print(f"Flights columns: {flights.columns.tolist()}")
    if tickets is not None: 
        print(f"\nLoaded Tickets Sample (all columns):\n{tickets.head()}")
        print(f"Tickets columns: {tickets.columns.tolist()}")
    if airports is not None: 
        print(f"\nLoaded Airports Sample (all columns):\n{airports.head()}")
        print(f"Airports columns: {airports.columns.tolist()}")
    
    if 'airline_scripts.dq_utils' in sys.modules and hasattr(sys.modules['airline_scripts.dq_utils'], 'display_dq_summary'):
        sys.modules['airline_scripts.dq_utils'].display_dq_summary()

