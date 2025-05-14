## Key Assumptions Made in the Analysis

### Data & Scope:

 * **Data Representativeness:** The provided 1Q2019 data for flights, tickets (sample), and airport codes is assumed to be representative for initial strategic decision-making.

 * **No Seasonality:** Seasonal variations in demand, pricing, and operations are not considered in this analysis, as per the project guidelines.

 * **Market Focus:** The analysis is confined to U.S. domestic routes operating between designated medium and large U.S. airports.

 * **Aircraft Dedication:** Each of the 5 new aircraft will be dedicated to a single, distinct round-trip route.

### Operational & Financial Calculations:

 * **Fixed Cost Parameters:** Per-mile operational costs (fuel, maintenance, crew, depreciation, insurance) are constant as specified.

 * **Airport Fees:** Airport operational costs are fixed per landing based on airport size (medium/large).

 * **Delay Costs:** Delays incur a fixed cost per minute only after an initial 15-minute grace period for both departures and arrivals.

 * **Aircraft Capacity:** Each aircraft has a maximum capacity of 200 passengers.

 * **Occupancy Data:** Passenger occupancy is determined exclusively by the OCCUPANCY_RATE in the Flights dataset.

 * **Baggage Revenue:** Based on a fixed percentage of passengers checking an average number of bags, with a set fee per bag per flight leg.

 * **Ticket Revenue Allocation:** The average round-trip itinerary fare (derived from sample ticket data) is assumed to be split evenly (50/50) between the two legs of a round trip for per-leg revenue calculations.

 * **Initial Profitability Scope:** The "most profitable routes" analysis (Q2) considers operational profit and does not include the $90 million upfront airplane investment cost.

### Data Cleaning & Handling:

 * **Missing Delays:** Missing departure and arrival delay values are imputed with 0 (zero), assuming no delay if not recorded.

 * **Unparseable/Invalid Data:** Values in numeric or date columns that cannot be parsed are converted to 'missing' (NaN/NaT). Rows with missing values in critical analytical columns are generally filtered out and logged. Specific invalid entries (e.g., occupancy outside 0-1, non-positive fares) are also filtered.

 * **Outlier Treatment:** The impact of statistical outliers on key numeric fields (like delays, fares) is handled based on a configurable method (e.g., 'impute with mean', 'filter', or 'none'). The specific method active during the run affects the final dataset.

 * **Cancelled Flights:** All flights marked as cancelled are excluded from the main analysis of route busyness and profitability.

 * **Ticket Data Usage:** Only round-trip tickets from 1Q2019 with positive fares are used for calculating average route fares.

 * **Airport Data Usage:** Only U.S. airports classified as 'medium' or 'large' are included. Duplicate airport IATA codes are resolved by keeping the first instance.

 * **Coordinate Parsing:** Latitude and longitude are derived by splitting the COORDINATES string, assuming a "longitude, latitude" format; malformed entries result in missing coordinate data for that airport.

### Analysis & Recommendation:

 * **Round-Trip Definition:** A "round-trip route" is defined by its canonical pair (e.g., JFK-ORD and ORD-JFK are treated as the same route entity for aggregation). The number of round-trip flights is derived by dividing total flight legs by two.

 * **Breakeven Calculation:** Assumes that the operational profit per round-trip flight, as calculated from the 1Q2019 data, is a stable indicator for projecting the number of flights needed to recoup the $90 million airplane investment.

 * **Route Recommendation Model:** The advanced scoring model for recommending the top 5 routes relies on the defined metrics, their normalization, and the assigned strategic weights to reflect business priorities.


These assumptions are important context for interpreting the results and recommendations of this analysis.