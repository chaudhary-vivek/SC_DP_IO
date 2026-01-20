# optimization_model.py
import pandas as pd
import pulp


def solve_distribution_lp(
    demand_df: pd.DataFrame,
    warehouse_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    stockout_penalty: float = 20.0,
):
    """
    Bare-bones distribution optimization (LP).

    Inputs:
      demand_df: columns = ["store_id", "demand"]
      warehouse_df: columns = ["warehouse_id", "capacity"]
      transport_df: columns = ["store_id", "warehouse_id", "transport_cost"]

    Assumption (simple): Each store is served by exactly one warehouse
    (as produced by generator.py transport_costs.csv).
    """
    # ---- basic validation ----
    for col in ["store_id", "demand"]:
        if col not in demand_df.columns:
            raise ValueError(f"demand_df missing column: {col}")

    for col in ["warehouse_id", "capacity"]:
        if col not in warehouse_df.columns:
            raise ValueError(f"warehouse_df missing column: {col}")

    for col in ["store_id", "warehouse_id", "transport_cost"]:
        if col not in transport_df.columns:
            raise ValueError(f"transport_df missing column: {col}")

    demand = dict(zip(demand_df["store_id"], demand_df["demand"]))
    capacity = dict(zip(warehouse_df["warehouse_id"], warehouse_df["capacity"]))

    # store -> (warehouse, transport_cost)
    mapping = {}
    for _, r in transport_df.iterrows():
        mapping[r["store_id"]] = (r["warehouse_id"], float(r["transport_cost"]))

    stores = list(demand.keys())
    warehouses = list(capacity.keys())

    # ---- model ----
    prob = pulp.LpProblem("DistributionPlanning", pulp.LpMinimize)

    ship = pulp.LpVariable.dicts("ship", stores, lowBound=0, cat="Continuous")
    stockout = pulp.LpVariable.dicts("stockout", stores, lowBound=0, cat="Continuous")

    # Objective: transport + stockout penalty
    prob += pulp.lpSum(
        ship[s] * mapping[s][1] + stockout[s] * stockout_penalty
        for s in stores
    )

    # Store demand balance
    for s in stores:
        prob += ship[s] + stockout[s] == demand[s], f"demand_balance_{s}"

    # Warehouse capacity constraints
    for w in warehouses:
        served_stores = [s for s in stores if mapping[s][0] == w]
        prob += pulp.lpSum(ship[s] for s in served_stores) <= capacity[w], f"cap_{w}"

    # ---- solve ----
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]

    # ---- outputs ----
    shipments = []
    for s in stores:
        w, c = mapping[s]
        shipments.append(
            {
                "store_id": s,
                "warehouse_id": w,
                "demand": float(demand[s]),
                "ship_qty": float(ship[s].value()),
                "stockout_qty": float(stockout[s].value()),
                "transport_cost": float(c),
            }
        )

    shipments_df = pd.DataFrame(shipments)

    total_transport_cost = float((shipments_df["ship_qty"] * shipments_df["transport_cost"]).sum())
    total_stockout_cost = float((shipments_df["stockout_qty"] * stockout_penalty).sum())
    total_cost = total_transport_cost + total_stockout_cost

    summary = {
        "status": status,
        "total_cost": total_cost,
        "total_transport_cost": total_transport_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_shipped": float(shipments_df["ship_qty"].sum()),
        "total_stockout": float(shipments_df["stockout_qty"].sum()),
    }

    return shipments_df, summary


# ------------------------
# BASIC USAGE
# ------------------------
if __name__ == "__main__":
    demand_hist = pd.read_csv("data/demand_history.csv")
    warehouse_df = pd.read_csv("data/warehouse_data.csv")
    transport_df = pd.read_csv("data/transport_costs.csv")

    # Example: solve for a single day using TRUE demand
    day = 170
    demand_df = (
        demand_hist[demand_hist["day"] == day][["store_id", "demand"]]
        .copy()
    )

    shipments_df, summary = solve_distribution_lp(
        demand_df=demand_df,
        warehouse_df=warehouse_df,
        transport_df=transport_df,
        stockout_penalty=20.0,
    )

    print("Summary:", summary)
    print(shipments_df.sort_values(["warehouse_id", "store_id"]).head(10))
