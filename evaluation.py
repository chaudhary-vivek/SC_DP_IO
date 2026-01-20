# evaluation.py
import pandas as pd
from pathlib import Path

from ml.demand_model import DemandModel
from opr.optimization_model import solve_distribution_lp


DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def evaluate_over_days(
    train_end_day: int = 150,
    test_days=None,
    stockout_penalty: float = 20.0,
    save_csv: bool = True,
):
    """
    Bare-bones evaluation:
      1) Train DemandModel on days < train_end_day
      2) For each test day:
         - Solve OR using predicted demand
         - Solve OR using true demand (oracle benchmark)
         - Compare costs and stockouts
    """
    if test_days is None:
        test_days = list(range(train_end_day, train_end_day + 20))

    demand_hist = pd.read_csv(DATA_DIR / "demand_history.csv")
    warehouse_df = pd.read_csv(DATA_DIR / "warehouse_data.csv")
    transport_df = pd.read_csv(DATA_DIR / "transport_costs.csv")

    # ------------------------
    # 1) Train ML model
    # ------------------------
    train_df = demand_hist[demand_hist["day"] < train_end_day].copy()

    dm = DemandModel()
    dm.fit(train_df)

    results = []

    # ------------------------
    # 2) Evaluate on each test day
    # ------------------------
    for day in test_days:
        day_df = demand_hist[demand_hist["day"] == day].copy()

        # -- predicted demand --
        predict_in = day_df[["day", "store_id", "warehouse_id", "promo_flag"]].copy()
        preds = dm.predict(predict_in)  # adds predicted_demand

        demand_pred = preds[["store_id", "predicted_demand"]].rename(
            columns={"predicted_demand": "demand"}
        )

        ship_pred, summary_pred = solve_distribution_lp(
            demand_df=demand_pred,
            warehouse_df=warehouse_df,
            transport_df=transport_df,
            stockout_penalty=stockout_penalty,
        )

        # -- true demand (oracle) --
        demand_true = day_df[["store_id", "demand"]].copy()

        ship_true, summary_true = solve_distribution_lp(
            demand_df=demand_true,
            warehouse_df=warehouse_df,
            transport_df=transport_df,
            stockout_penalty=stockout_penalty,
        )

        # ------------------------
        # 3) Compare
        # ------------------------
        regret = summary_pred["total_cost"] - summary_true["total_cost"]
        stockout_gap = summary_pred["total_stockout"] - summary_true["total_stockout"]

        results.append(
            {
                "day": day,
                "pred_total_cost": summary_pred["total_cost"],
                "true_total_cost": summary_true["total_cost"],
                "regret_cost": regret,
                "pred_transport_cost": summary_pred["total_transport_cost"],
                "true_transport_cost": summary_true["total_transport_cost"],
                "pred_stockout_cost": summary_pred["total_stockout_cost"],
                "true_stockout_cost": summary_true["total_stockout_cost"],
                "pred_total_stockout": summary_pred["total_stockout"],
                "true_total_stockout": summary_true["total_stockout"],
                "stockout_gap": stockout_gap,
                "pred_status": summary_pred["status"],
                "true_status": summary_true["status"],
            }
        )

    results_df = pd.DataFrame(results)

    # ------------------------
    # 4) Print quick summary
    # ------------------------
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Train end day: {train_end_day}")
    print(f"Test days: {min(test_days)} to {max(test_days)}")

    print("\nAverages over test days:")
    print(
        results_df[
            ["pred_total_cost", "true_total_cost", "regret_cost", "pred_total_stockout", "true_total_stockout"]
        ].mean()
    )

    print("\nWorst 5 days by regret:")
    print(results_df.sort_values("regret_cost", ascending=False).head(5)[["day", "regret_cost", "pred_total_cost", "true_total_cost"]])

    # ------------------------
    # 5) Save report
    # ------------------------
    if save_csv:
        out_path = OUT_DIR / "evaluation_report.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return results_df


if __name__ == "__main__":
    evaluate_over_days(
        train_end_day=150,
        test_days=list(range(150, 180)),
        stockout_penalty=20.0,
        save_csv=True,
    )
