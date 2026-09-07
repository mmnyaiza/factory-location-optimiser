import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Random seed makes the dataset reproducible.
# Change this number to generate a different dataset.
np.random.seed(42)

N_CUSTOMERS = 500
N_FACILITIES = 20
N_SUPPLIERS = 10

# Buffers that guarantee capacity > demand and supply > capacity,
# regardless of what the random draws produce.
FACILITY_BUFFER = 1.25   # total facility capacity = demand * 1.25
SUPPLIER_BUFFER = 1.15   # total supplier capacity = facility capacity * 1.15

# Output directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SOUTH AFRICAN ECONOMIC CENTRES
# ============================================================

# latitude, longitude, relative demand weight
ECONOMIC_CENTRES = {
    "Johannesburg": {
        "lat": -26.2041,
        "lon": 28.0473,
        "weight": 0.25
    },

    "Pretoria": {
        "lat": -25.7479,
        "lon": 28.2293,
        "weight": 0.12
    },

    "Durban": {
        "lat": -29.8587,
        "lon": 31.0218,
        "weight": 0.15
    },

    "Cape Town": {
        "lat": -33.9249,
        "lon": 18.4241,
        "weight": 0.18
    },

    "Gqeberha": {
        "lat": -33.9608,
        "lon": 25.6022,
        "weight": 0.06
    },

    "East London": {
        "lat": -33.0153,
        "lon": 27.9116,
        "weight": 0.04
    },

    "Bloemfontein": {
        "lat": -29.0852,
        "lon": 26.2041,
        "weight": 0.05
    },

    "Polokwane": {
        "lat": -23.9045,
        "lon": 29.4689,
        "weight": 0.05
    },

    "Mbombela": {
        "lat": -25.4753,
        "lon": 30.6918,
        "weight": 0.03
    },

    "Kimberley": {
        "lat": -28.7282,
        "lon": 24.7499,
        "weight": 0.02
    }
}


# ============================================================
# PREPARE ECONOMIC CENTRES
# ============================================================

centre_names = list(ECONOMIC_CENTRES.keys())

centre_probabilities = np.array([
    ECONOMIC_CENTRES[centre]["weight"]
    for centre in centre_names
])

# Make sure probabilities sum to 1
centre_probabilities = (
    centre_probabilities /
    centre_probabilities.sum()
)


# ============================================================
# GENERATE CUSTOMERS
# ============================================================

customers = []

for i in range(1, N_CUSTOMERS + 1):

    # Select an economic centre
    centre = np.random.choice(
        centre_names,
        p=centre_probabilities
    )

    centre_data = ECONOMIC_CENTRES[centre]

    # Generate customer location around the economic centre.
    #
    # Standard deviation controls how far customers
    # can be from the centre.
    latitude = (
        centre_data["lat"] +
        np.random.normal(0, 0.35)
    )

    longitude = (
        centre_data["lon"] +
        np.random.normal(0, 0.35)
    )

    # Generate demand.
    #
    # Lognormal gives us realistic behaviour where:
    # - most customers have relatively moderate demand
    # - a small number have very high demand
    demand = int(
        np.random.lognormal(
            mean=6.5,
            sigma=0.65
        )
    )

    # Keep demand within reasonable limits
    demand = np.clip(
        demand,
        100,
        5000
    )

    customers.append({
        "customer_id": f"C{i:04d}",
        "economic_centre": centre,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "annual_demand": int(demand)
    })


customers = pd.DataFrame(customers)


# ============================================================
# GENERATE POTENTIAL FACILITIES
# ============================================================

facilities = []

for i in range(1, N_FACILITIES + 1):

    # Facilities can also be located near economic centres
    centre = np.random.choice(
        centre_names,
        p=centre_probabilities
    )

    centre_data = ECONOMIC_CENTRES[centre]

    latitude = (
        centre_data["lat"] +
        np.random.normal(0, 0.25)
    )

    longitude = (
        centre_data["lon"] +
        np.random.normal(0, 0.25)
    )

    # Facility capacity
    capacity = np.random.randint(
        8000,
        30000
    )

    # Fixed annual cost
    fixed_cost = np.random.randint(
        5_000_000,
        20_000_000
    )

    # Operating cost per unit
    operating_cost = round(
        np.random.uniform(5, 15),
        2
    )

    facilities.append({
        "facility_id": f"F{i:03d}",
        "near_centre": centre,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "capacity": capacity,
        "fixed_cost": fixed_cost,
        "operating_cost_per_unit": operating_cost
    })


facilities = pd.DataFrame(facilities)


# ------------------------------------------------------------
# SCALE FACILITY CAPACITY SO IT EXCEEDS TOTAL DEMAND
# ------------------------------------------------------------
# Random draws don't guarantee capacity > demand on their own,
# so we scale the whole capacity column up (preserving relative
# spread between facilities) until the total clears demand by
# FACILITY_BUFFER.
total_demand = customers["annual_demand"].sum()
target_facility_capacity = total_demand * FACILITY_BUFFER
total_facility_capacity = facilities["capacity"].sum()

if total_facility_capacity < target_facility_capacity:
    scale_factor = target_facility_capacity / total_facility_capacity
    facilities["capacity"] = (
        facilities["capacity"] * scale_factor
    ).round().astype(int)


# ============================================================
# GENERATE SUPPLIERS
# ============================================================

suppliers = []

for i in range(1, N_SUPPLIERS + 1):

    centre = np.random.choice(
        centre_names,
        p=centre_probabilities
    )

    centre_data = ECONOMIC_CENTRES[centre]

    latitude = (
        centre_data["lat"] +
        np.random.normal(0, 0.4)
    )

    longitude = (
        centre_data["lon"] +
        np.random.normal(0, 0.4)
    )

    supply_capacity = np.random.randint(
        15000,
        50000
    )

    unit_cost = round(
        np.random.uniform(30, 60),
        2
    )

    reliability = round(
        np.random.uniform(0.85, 0.99),
        3
    )

    suppliers.append({
        "supplier_id": f"S{i:03d}",
        "near_centre": centre,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "supply_capacity": supply_capacity,
        "unit_cost": unit_cost,
        "reliability": reliability
    })


suppliers = pd.DataFrame(suppliers)


# ------------------------------------------------------------
# SCALE SUPPLIER CAPACITY SO IT EXCEEDS TOTAL FACILITY CAPACITY
# ------------------------------------------------------------
# Suppliers feed facilities, not customers directly, so target
# the (already-scaled) facility total rather than raw demand.
total_facility_capacity = facilities["capacity"].sum()
target_supplier_capacity = total_facility_capacity * SUPPLIER_BUFFER
total_supplier_capacity = suppliers["supply_capacity"].sum()

if total_supplier_capacity < target_supplier_capacity:
    scale_factor = target_supplier_capacity / total_supplier_capacity
    suppliers["supply_capacity"] = (
        suppliers["supply_capacity"] * scale_factor
    ).round().astype(int)


# ============================================================
# SAVE DATASETS
# ============================================================

customers.to_csv(
    DATA_DIR / "customers.csv",
    index=False
)

facilities.to_csv(
    DATA_DIR / "facilities.csv",
    index=False
)

suppliers.to_csv(
    DATA_DIR / "suppliers.csv",
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("=" * 60)
print("SUPPLY CHAIN DATASET GENERATED")
print("=" * 60)

print()

print(f"Customers:       {len(customers):,}")
print(f"Facilities:      {len(facilities):,}")
print(f"Suppliers:       {len(suppliers):,}")

print()

print(
    f"Total demand:    "
    f"{customers['annual_demand'].sum():,} units"
)

print(
    f"Total capacity:  "
    f"{facilities['capacity'].sum():,} units"
)

print(
    f"Supplier supply: "
    f"{suppliers['supply_capacity'].sum():,} units"
)

print()

print("CUSTOMERS BY ECONOMIC CENTRE")
print("-" * 40)

print(
    customers["economic_centre"]
    .value_counts()
)

print()

print("DATA SAVED TO:")
print(DATA_DIR.resolve())

print("=" * 60)