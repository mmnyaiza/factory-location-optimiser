import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

# Transportation cost per kilometre per unit
TRANSPORT_COST_PER_KM = 2.50


# ============================================================
# HAVERSINE DISTANCE FUNCTION
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on Earth using the Haversine formula.

    Returns distance in kilometres.
    """

    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's approximate radius in km
    earth_radius = 6371

    return earth_radius * c


# ============================================================
# LOAD DATA
# ============================================================

customers = pd.read_csv(DATA_DIR / "customers.csv")
facilities = pd.read_csv(DATA_DIR / "facilities.csv")


print("=" * 60)
print("TRANSPORTATION DISTANCE CALCULATION")
print("=" * 60)

print(f"\nCustomers loaded:  {len(customers)}")
print(f"Facilities loaded: {len(facilities)}")


# ============================================================
# CREATE CUSTOMER-FACILITY COMBINATIONS
# ============================================================

# Create every possible customer-facility combination
customers["key"] = 1
facilities["key"] = 1

distance_matrix = customers.merge(
    facilities,
    on="key",
    suffixes=("_customer", "_facility")
)

# Remove temporary key
distance_matrix.drop(columns=["key"], inplace=True)


# ============================================================
# CALCULATE DISTANCES
# ============================================================

distance_matrix["distance_km"] = haversine_distance(
    distance_matrix["latitude_customer"],
    distance_matrix["longitude_customer"],
    distance_matrix["latitude_facility"],
    distance_matrix["longitude_facility"],
)


# ============================================================
# CALCULATE TRANSPORTATION COST
# ============================================================

distance_matrix["transport_cost_per_unit"] = (
    distance_matrix["distance_km"]
    * TRANSPORT_COST_PER_KM
)

distance_matrix["total_transport_cost"] = (
    distance_matrix["transport_cost_per_unit"]
    * distance_matrix["annual_demand"]
)


# ============================================================
# SELECT RELEVANT COLUMNS
# ============================================================

distance_matrix = distance_matrix[
    [
        "customer_id",
        "facility_id",
        "annual_demand",
        "distance_km",
        "transport_cost_per_unit",
        "total_transport_cost",
    ]
]


# ============================================================
# ROUND VALUES
# ============================================================

distance_matrix["distance_km"] = (
    distance_matrix["distance_km"].round(2)
)

distance_matrix["transport_cost_per_unit"] = (
    distance_matrix["transport_cost_per_unit"].round(2)
)

distance_matrix["total_transport_cost"] = (
    distance_matrix["total_transport_cost"].round(2)
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = DATA_DIR / "customer_facility_distances.csv"

distance_matrix.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nDISTANCE MATRIX")
print("-" * 40)

print(f"Customer-facility combinations: {len(distance_matrix):,}")

print(
    f"Average distance: "
    f"{distance_matrix['distance_km'].mean():.2f} km"
)

print(
    f"Minimum distance: "
    f"{distance_matrix['distance_km'].min():.2f} km"
)

print(
    f"Maximum distance: "
    f"{distance_matrix['distance_km'].max():.2f} km"
)

print(
    f"\nTotal potential transportation cost: "
    f"R{distance_matrix['total_transport_cost'].sum():,.2f}"
)

print("\nDATA SAVED TO:")
print(output_file)

print("=" * 60)