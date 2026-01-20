# generator.py
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ------------------------
# CONFIG
# ------------------------
N_WAREHOUSES = 3
N_STORES = 10
N_DAYS = 180
OUTPUT_DIR = Path("data")

OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------
# STORE → WAREHOUSE MAPPING
# ------------------------
stores = [f"S{i}" for i in range(N_STORES)]
warehouses = [f"W{i}" for i in range(N_WAREHOUSES)]

store_warehouse_map = {
    store: np.random.choice(warehouses)
    for store in stores
}

# ------------------------
# DEMAND GENERATION
# ------------------------
rows = []

for store in stores:
    base_demand = np.random.randint(20, 50)
    trend = np.random.uniform(-0.02, 0.02)

    for day in range(N_DAYS):
        seasonality = 5 * np.sin(2 * np.pi * day / 30)
        promo = np.random.binomial(1, 0.1)
        promo_lift = promo * np.random.uniform(5, 15)

        noise = np.random.normal(0, 3)

        demand = (
            base_demand
            + trend * day
            + seasonality
            + promo_lift
            + noise
        )

        rows.append({
            "day": day,
            "store_id": store,
            "warehouse_id": store_warehouse_map[store],
            "promo_flag": promo,
            "demand": max(0, int(demand))
        })

demand_df = pd.DataFrame(rows)
demand_df.to_csv(OUTPUT_DIR / "demand_history.csv", index=False)

# ------------------------
# NETWORK DATA
# ------------------------
network_rows = []

for wh in warehouses:
    network_rows.append({
        "warehouse_id": wh,
        "capacity": np.random.randint(300, 600),
        "holding_cost": np.random.uniform(0.5, 2.0)
    })

network_df = pd.DataFrame(network_rows)

transport_rows = []

for store in stores:
    transport_rows.append({
        "store_id": store,
        "warehouse_id": store_warehouse_map[store],
        "transport_cost": np.random.uniform(1.0, 5.0)
    })

transport_df = pd.DataFrame(transport_rows)

network_df.to_csv(OUTPUT_DIR / "warehouse_data.csv", index=False)
transport_df.to_csv(OUTPUT_DIR / "transport_costs.csv", index=False)

print("Dummy data generated:")
print("- data/demand_history.csv")
print("- data/warehouse_data.csv")
print("- data/transport_costs.csv")
