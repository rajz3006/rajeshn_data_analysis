# Airline Data Analysis Submission

## Project Overview

This project's primary goal is to analyze the provided 1Q2019 airline data to identify potential U.S. domestic routes for a new airline venture. The analysis focuses on identifying busy and profitable routes, recommending five specific round-trip routes for investment, calculating their breakeven points, and proposing Key Performance Indicators (KPIs) for future tracking.

The approach emphasizes a "product mindset," aiming to deliver a structured and reproducible analysis. This includes robust data cleaning, feature engineering, and a clear presentation of findings, primarily using Python for data processing and Tableau for visualization. A key component of this project is the systematic tracking of data quality metrics throughout the pipeline.

## Project Structure

The project is organized as follows:

airline_data_challenge_submission/

      ├── airline_scripts/                  # Reusable Python modules to keep the notebook clean and logic organized.
      │   ├── __init__.py                   # Makes 'airline_scripts' a Python package.
      │   ├── config.py                     # Centralized configuration: file paths, column names, business constants.
      │   ├── load_utils.py                 # Functions for data loading.
      │   ├── dq_utils.py                   # Functions for data quality metric logging.
      │   ├── clean_utils.py                # Functions for data cleaning and validation.
      │   ├── feature_engineering_utils.py  # Functions for creating new features, calculating costs/revenues, etc.
      │   └── analysis_utils.py             # Functions for core analytical tasks (profitability, busiest routes, breakeven).
      │
      ├── data/                             # Contains the raw data CSVs (Flights.csv, Tickets.csv, Airport_Codes.csv)
      │                                     # Not for submission as per instructions, used locally.
      ├── images/                           # Snapshots of the Tableau Dashboards
      │
      ├── logs/                             # Contains the data quality metrics log (dq_metrics.csv).
      │
      ├── notebooks/
      │   └── Airline_Route_Analysis.ipynb  # Main Jupyter Notebook detailing the entire analysis workflow,
      │                                     # from data loading to final recommendations.
      │
      ├── output_steps/                     # Stores generated temporary data frames as CSV files for tableau visualisation and manual analysis.
      │
      ├── reports/                          # Stores generated reports, like data profiling HTML files from ydata-profiling.
      │
      ├── Assumptions.md                    # Clear Documentation describing all the Logical assumptions made for this analysis.
      │
      ├── Airline_Challenge_Submission.twbx # Packaged Tableau workbook with dashboards.
      │
      └── README.md                         # This file: project overview, setup, and execution instructions.


## Setup Instructions

To set up and run this project locally:

1.  **Prerequisites:**
    * Python (version 3.9+ recommended).
    * Access to a terminal or command prompt.
    * Tableau Desktop (free version is sufficient for viewing).

2.  **Clone the Repository (if applicable) or Download Files:**
    Ensure all project files (excluding the `data/` directory for submission) are in the structure outlined above.

3.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    cd airline_data_challenge_submission
    python -m venv .venv
    ```
    Activate the environment:
    * Windows (Command Prompt): `.venv\Scripts\activate.bat`
    * Windows (PowerShell): `.venv\Scripts\Activate.ps1`
    * macOS/Linux: `source .venv/bin/activate`

4.  **Install Dependencies:**
    With the virtual environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Data Files:**
    For local execution, place the provided `Flights.csv`, `Tickets.csv`, and `Airport_Codes.csv` files into the `airline_data_challenge/data/` directory. The `config.py` script is set up to look for them here. (Note: The `data/` folder itself is not part of the submission package).

## How to Run the Analysis

1.  **Jupyter Notebook:**
    * Ensure your virtual environment is activated.
    * Navigate to the `airline_data_challenge/notebooks/` directory in your terminal.
    * Launch Jupyter Notebook or JupyterLab:
        ```bash
        jupyter notebook Airline_Route_Analysis.ipynb
        # OR
        jupyter lab Airline_Route_Analysis.ipynb
        ```
    * Open the `Airline_Route_Analysis.ipynb` notebook.
    * Run the cells sequentially to execute the entire data analysis pipeline. The notebook is structured to follow the phases outlined in the project plan.

2.  **Tableau Dashboards:**
    * The final analytical dataset and the data quality log (`dq_metrics.csv`) are generated by the Jupyter Notebook.
    * Open the provided Tableau packaged workbook (e.g., `Airline_Challenge_Submission.twbx`) with Tableau Desktop to view the interactive dashboards. These dashboards visualize the key findings, route recommendations, and data quality metrics.

## Key Features & Modules

* **Modular Python Scripts (`airline_scripts/`):** Core logic for data loading, cleaning, feature engineering, DQ logging, and analysis is encapsulated in reusable Python functions. This keeps the main notebook focused on the narrative and high-level execution.
* **Configuration Driven (`config.py`):** File paths, business rules, and key parameters are managed centrally.
* **Data Quality Monitoring (`dq_utils.py`, `logs/dq_metrics.csv`):** Metrics are logged at various stages of data processing (loading, cleaning, transformation) to track data integrity and the impact of each step. These are visualized in a dedicated Tableau dashboard.
* **Comprehensive Analysis:** Addresses all questions from the challenge statement, including identifying top routes, recommending investments, and calculating breakeven points.

## A Note on this Challenge Submission

This project represents my approach to the Airline Data Analysis Challenge. I've focused on demonstrating a builder mindset, systematic data management, and clear business intelligence as requested in the instructions. All assumptions made during the analysis are documented within the `Airline_Route_Analysis.ipynb` notebook.

Thank you for the opportunity!
