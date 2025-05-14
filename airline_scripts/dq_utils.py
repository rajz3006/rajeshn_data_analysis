# airline_scripts/dq_utils.py
import pandas as pd
import datetime
import os
import uuid # For unique RunID

from . import config # Import config from the same package

# Global variable to hold the DataFrame for DQ metrics for the current run
dq_metrics_df = None
current_run_id = None

def initialize_dq_log(new_run=True):
    """
    Initializes the data quality logging system for a new run.
    As per the update, "dq_utils is to create a new CSV file for each run
    and update the same CSV with logging details."
    This interprets as: for each new run (notebook execution), a new in-memory log is started.
    The save function will then write this new log, potentially overwriting an old `dq_metrics.csv`
    or creating a timestamped one if that's preferred (current implementation overwrites `dq_metrics.csv`).

    To ensure a *truly new file per run that doesn't overwrite*, the save function or this
    initialization would need to generate unique filenames (e.g., with timestamps).
    However, "update the *same* CSV" is contradictory. Let's assume `dq_metrics.csv` is for the *latest* run.

    If `new_run` is True, it generates a new RunID and resets the in-memory DataFrame.
    """
    global dq_metrics_df, current_run_id
    if new_run:
        current_run_id = str(uuid.uuid4()) # Generate a unique ID for this run
        print(f"Initializing DQ Log for Run ID: {current_run_id}")
    # Always (re)initialize the DataFrame for the current run
    dq_metrics_df = pd.DataFrame(columns=config.DQ_COLUMNS)
    print(f"DQ Log initialized. Ready to log metrics for Run ID: {current_run_id}.")


def log_dq_metric(phase, step, table_name, metric, value, description=""):
    """
    Logs a data quality metric to the in-memory DataFrame.

    Args:
        phase (str): The phase of the data pipeline (e.g., 'Data Loading', 'Cleaning').
        step (str): A specific step within the phase (e.g., 'Load Flights.csv', 'Handle Missing Values').
        table_name (str): The name of the table/DataFrame the metric pertains to.
        metric (str): The name of the metric (e.g., 'Row Count', 'Null Percentage').
        value (any): The value of the metric.
        description (str, optional): Additional context or description for the metric.
    """
    global dq_metrics_df, current_run_id
    if dq_metrics_df is None or current_run_id is None:
        print("DQ Log not initialized. Call initialize_dq_log() first.")
        # Optionally, initialize it here if not done:
        # initialize_dq_log()
        # print("DQ Log was auto-initialized.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_log_entry = pd.DataFrame([{
        'Timestamp': timestamp,
        'RunID': current_run_id,
        'Phase': phase,
        'Step': step,
        'TableName': table_name,
        'Metric': metric,
        'Value': value,
        'Description': description
    }])
    dq_metrics_df = pd.concat([dq_metrics_df, new_log_entry], ignore_index=True)
    # print(f"Logged DQ Metric: {phase} - {step} - {metric} = {value}") # Optional: for verbose logging

def get_dq_log_df():
    """Returns the current in-memory DQ log DataFrame."""
    global dq_metrics_df
    if dq_metrics_df is None:
        initialize_dq_log() # Ensure it's initialized if accessed before explicit init
    return dq_metrics_df

def save_dq_log_to_csv(file_path=None):
    """
    Saves the accumulated DQ metrics from the current run to a CSV file.
    This will overwrite the file if it already exists, effectively creating a new file for the current run's logs.
    """
    global dq_metrics_df
    if dq_metrics_df is None or dq_metrics_df.empty:
        print("No DQ metrics to save.")
        return

    if file_path is None:
        file_path = config.DQ_LOG_FILE_PATH

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dq_metrics_df.to_csv(file_path, index=False)
        print(f"DQ log for Run ID {current_run_id} saved to {file_path}")
    except Exception as e:
        print(f"Error saving DQ log: {e}")

def display_dq_summary():
    """Prints a summary of the logged DQ metrics."""
    global dq_metrics_df
    if dq_metrics_df is None or dq_metrics_df.empty:
        print("No DQ metrics logged yet.")
        return
    print("\n--- Data Quality Log Summary ---")
    print(f"Run ID: {current_run_id}")
    print(f"Total DQ metrics logged: {len(dq_metrics_df)}")
    if not dq_metrics_df.empty:
        print("Last 5 entries:")
        print(dq_metrics_df.tail())
    print("--- End of DQ Log Summary ---\n")


# Example of use (typically done in the notebook):
if __name__ == '__main__':
    # This block is for testing the module directly
    initialize_dq_log()
    log_dq_metric(phase="Test Phase", step="Initial Row Count", table_name="TestTable", metric="Row Count", value=100, description="Initial load")
    log_dq_metric(phase="Test Phase", step="After Cleaning", table_name="TestTable", metric="Row Count", value=90, description="After removing duplicates")
    display_dq_summary()
    save_dq_log_to_csv(os.path.join(config.LOGS_DIR, "test_dq_metrics.csv")) # Save to a test file