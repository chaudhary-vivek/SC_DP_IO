import pandas as pd
from pathlib import Path

from ml.demand_model import DemandModel
from ml.uncertainty_estimator import UncertaintyEstimator
from opr.optimization_model import solve_distribution_lp


DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def run_pipeline(
        target_day: int = 170,
        train_end_day: int = 150,
        stockout_penalty: float = 20.0
):
    
    # ------------------------
    # 1) Load data
    # ------------------------
    demand_hist = pd.read_csv(DATA_DIR / "demand_history.csv")
    warehouse_df = pd.read_csv(DATA_DIR / "warehouse_data.csv")
    transport_df = pd.read_csv(DATA_DIR / "transport_costs.csv") 

    # ------------------------
    # 2) Train ML model
    # ------------------------
    train_df = demand_hist[demand_hist["day"] < train_end_day].copy()
    predict_df = demand_hist[demand_hist["day"] == target_day][
        ["day", "store_id", "warehouse_id", "promo_flag"]
    ].copy()

    dm = DemandModel()
    dm.fit(train_df)

    preds_df = dm.predict(predict_df)

    # ------------------------
    # 3) Fit uncertainty estimator (using train residuals)
    # ------------------------
    # Prediction on train for residuals
    train_for_pred = train_df[['day', "store_id", "warehouse_id", "promo_flag", "demand"]].copy()
    train_preds = dm.predict(train_for_pred[["day", "store_id", "warehouse_id", "promo_flag"]])
    train_preds = train_preds.merge(
        train_for_pred[["day", "store_id", "demand"]],
        on = ["day", "store_id"],
        how = "left"
    )

    ue = UncertaintyEstimator()
    ue.fit(train_preds)

    preds_with_u = ue.attach_uncertainty(preds_df)

    # ------------------------
    # 4) Build demand input for OR (use predicted demand)
    # ------------------------
    demand_for_or = preds_with_u[["store_id", "predicted_demand"]].rename(
        columns = {"predicted_demand" : "demand"}
    )    

    # ------------------------
    # 5) Solve OR optimization
    # ------------------------
    shipments_df, summary = solve_distribution_lp(
        demand_df=demand_for_or,
        warehouse_df=warehouse_df,
        transport_df=transport_df,
        stockout_penalty=stockout_penalty,
    )

    # ------------------------
    # 6) Save outputs
    # ------------------------
    preds_with_u.to_csv(OUT_DIR / f"predictions_day_{target_day}.csv", index=False)
    shipments_df.to_csv(OUT_DIR / f"shipments_day_{target_day}.csv", index=False)

    # ------------------------
    # 7) Print quick summary
    # ------------------------
    print("\n=== PIPELINE RUN COMPLETE ===")
    print(f"Target day: {target_day}")
    print(f"Train end day: {train_end_day}")
    print("Optimization summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nSample predictions with uncertainty:")
    print(preds_with_u[["store_id", "predicted_demand", "demand_sigma", "p10", "p50", "p90"]].head())

    print("\nSample shipments:")
    print(shipments_df.sort_values(["warehouse_id", "store_id"]).head())

    return preds_with_u, shipments_df, summary


if __name__ == "__main__":
    # You can change these knobs
    run_pipeline(
        target_day=170,
        train_end_day=150,
        stockout_penalty=20.0,
    )