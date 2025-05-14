# airline_scripts/config.py
import os

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
STEP_OUTPUT_DIR = os.path.join(BASE_DIR, "output_steps")


# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STEP_OUTPUT_DIR, exist_ok=True)


# --- File Names/Paths ---
FLIGHTS_FILE = os.path.join(DATA_DIR, "Flights.csv")
TICKETS_FILE = os.path.join(DATA_DIR, "Tickets.csv")
AIRPORT_CODES_FILE = os.path.join(DATA_DIR, "Airport_Codes.csv")

DQ_LOG_FILE_PATH = os.path.join(LOGS_DIR, "dq_metrics.csv")
REJECTED_FLIGHTS_FILE = os.path.join(LOGS_DIR, "rejected_flights_data.csv")
REJECTED_TICKETS_FILE = os.path.join(LOGS_DIR, "rejected_tickets_data.csv")
REJECTED_AIRPORTS_FILE = os.path.join(LOGS_DIR, "rejected_airports_data.csv")

# --- Column Names (Constants for used columns) ---
# Airport Codes Columns
COL_AIRPORT_IATA = "IATA_CODE"
COL_AIRPORT_TYPE = "TYPE"
COL_AIRPORT_NAME = "NAME"
COL_AIRPORT_ISO_COUNTRY = "ISO_COUNTRY"
COL_AIRPORT_COORDINATES = "COORDINATES"
COL_AIRPORT_LATITUDE = "LATITUDE"       # New 
COL_AIRPORT_LONGITUDE = "LONGITUDE"     # New 

# Flights Columns
COL_FL_DATE = "FL_DATE"; COL_FL_ORIGIN = "ORIGIN"; COL_FL_DEST = "DESTINATION"; COL_FL_CANCELLED = "CANCELLED"
COL_FL_DEP_DELAY = "DEP_DELAY"; COL_FL_ARR_DELAY = "ARR_DELAY"; COL_FL_DISTANCE = "DISTANCE"
COL_FL_AIR_TIME = "AIR_TIME"; COL_FL_OCCUPANCY = "OCCUPANCY_RATE"; COL_FL_OP_CARRIER_FL_NUM = "OP_CARRIER_FL_NUM"

# Tickets Columns
COL_TK_ORIGIN = "ORIGIN"; COL_TK_DEST = "DESTINATION"; COL_TK_ROUNDTRIP = "ROUNDTRIP"; COL_TK_PASSENGERS = "PASSENGERS"
COL_TK_ITIN_FARE = "ITIN_FARE"; COL_TK_YEAR = "YEAR"; COL_TK_QUARTER = "QUARTER"

# --- Required Columns for Loading (used by clean_utils to filter down) ---
REQUIRED_AIRPORT_COLS = [
    COL_AIRPORT_IATA,
    COL_AIRPORT_TYPE,
    COL_AIRPORT_NAME,
    COL_AIRPORT_ISO_COUNTRY,
    COL_AIRPORT_COORDINATES # Added COORDINATES
]
REQUIRED_FLIGHTS_COLS = [
    COL_FL_DATE, COL_FL_OP_CARRIER_FL_NUM, COL_FL_ORIGIN, COL_FL_DEST, COL_FL_DEP_DELAY,
    COL_FL_ARR_DELAY, COL_FL_CANCELLED, COL_FL_AIR_TIME, COL_FL_DISTANCE, COL_FL_OCCUPANCY
]
REQUIRED_TICKETS_COLS = [
    COL_TK_YEAR, COL_TK_QUARTER, COL_TK_ORIGIN, COL_TK_ROUNDTRIP,
    COL_TK_PASSENGERS, COL_TK_ITIN_FARE, COL_TK_DEST
]

# --- Business Constants & Assumptions ---
FUEL_OIL_MAINTENANCE_CREW_COST_PER_MILE = 8
DEPRECIATION_INSURANCE_OTHER_COST_PER_MILE = 1.18
MEDIUM_AIRPORT_OPERATIONAL_COST = 5000
LARGE_AIRPORT_OPERATIONAL_COST = 10000
DELAY_COST_PER_MINUTE = 75
FREE_DELAY_MINUTES = 15
MAX_PASSENGERS_PER_PLANE = 200
BAGGAGE_FEE_PER_BAG_PER_FLIGHT = 35
PASSENGER_CHECKED_BAG_RATIO = 0.50
AVERAGE_BAGS_PER_CHECKING_PASSENGER = 1
AIRPLANE_UPFRONT_COST = 90_000_000
REQUIRED_AIRPORT_TYPES = ["medium_airport", "large_airport"]
ANALYSIS_YEAR = 2019
ANALYSIS_QUARTER = 1

# --- Data Quality ---
DQ_COLUMNS = ['Timestamp', 'RunID', 'Phase', 'Step', 'TableName', 'Metric', 'Value', 'Description']
REJECT_REASON_COL = 'FILTER_REASON'
BATCH_ID_COLUMN = 'BATCH_ID' 

# Outlier Handling Configuration
OUTLIER_HANDLING_METHOD = 'impute_mean'  # Options: 'impute_mean', 'filter', 'none'
OUTLIER_IQR_MULTIPLIER = 1.5

# Configuration for columns where NaNs should be imputed
IMPUTE_NAN_CONFIG = {
    COL_FL_DEP_DELAY: 0,
    COL_FL_ARR_DELAY: 0
}

# --- Other ---
def get_canonical_route_pair(airport1, airport2):
    return "-".join(sorted([str(airport1), str(airport2)]))