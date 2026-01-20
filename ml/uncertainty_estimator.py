import numpy as np
import pandas as pd

class UncertaintyEstimator:
    """
    Estimates per-store sigma from residuals (actual - predicted)
    and provides simple normal-approx intervals and scenario sampling
    """

    def __init__(self):
        self.store_sigma = {}

    def fit(self, df_with_preds : pd.DataFrame):
        """
        Expects columns: store_id, demand, predicted_demand
        Learns sigma per store from residuals
        """

        required = {"store_id", "demand", "predicted_demand"}
        missing = required - set(df_with_preds.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df_with_preds.copy()
        df["residual"] = df["demand"] - df["predicted_demand"]

        sigmas = (
            df.groupby("store_id")["residual"]
            .std(ddof = 1)
            .fillna(1.0) # if not enough data
        )

        # avoid sigma = 0 for tiny datasets
        sigmas = sigmas.clip(lower = 1.0)

        self.store_sigma = sigmas.to_dict()

    def attach_uncertainty(self, df_with_preds: pd.DataFrame) -> pd.DataFrame:
        """
        Adds demand_sigma, p10, p50, p90 using a normal approximation
        Expects: store_id, predicted_demand
        """

        if not self.store_sigma:
            raise ValueError("Estimator not fitted. Call fit() first")
        
        required = {"store_id", "predicted_demand"}
        missing = required - set(df_with_preds.columns)
        if missing:
            raise ValueError(f"Missing Required columns: {missing}")
        
        out = df_with_preds.copy()
        out["demand_sigma"] = out["store_id"].map(self.store_sigma).fillna(5.0)

        mu = out["predicted_demand"].astype(float)
        sigma = out["demand_sigma"].astype(float)

        # Normal approx quantiles
        z10, z90 = -1.2816, 1.2816

        out["p10"] = np.maximum(0, np.round(mu + z10*sigma)).astype(int)
        out["p50"] = np.maximum(0, np.round(mu)).astype(int)
        out["p90"] = np.maximum(0, np.round(mu + z90*sigma)).astype(int)

        return out
    

    def sample_scenarios(
            self,
            df_with_preds : pd.DataFrame,
            n_scenarios: int = 50,
            seed: int = 42
        ) -> pd.DataFrame:
        """
        Generates demand scenarios per row normal using Normal (mu, Sigma).
        Returns a long dataframe with a scenario_id column.

        Expects: store_id, predicted_demand
        Output columns: scenario_id + original columns + scenario_demand
        """
        df_u = self.attach_uncertainty(df_with_preds)

        rng = np.random.default_rng(seed)

        mu = df_u["predicted_demand"].astype(float).to_numpy()
        sigma = df_u["demand_sigma"].astype(float).to_numpy()

        # shape: (n_rows, n_scenarios)
        draws = rng.normal(loc = mu[:, None], scale = sigma[:, None], size = (len(df_u), n_scenarios))
        draws = np.maximum(0, np.round(draws)).astype(int)

        # build long format
        scenario_ids = np.arange(n_scenarios)
        long_df = pd.DataFrame({
            "row_id" : np.repeat(np.arange(len(df_u)), n_scenarios),
            "scenario_id": np.tile(scenario_ids, len(df_u)),
            "scenario_demand": draws.reshape(-1)
        })

        base = df_u.reset_index(drop = True).reset_index().rename(columns = {"index": "row_id"})
        out = long_df.merge(base, on = "row_id", how = "left").drop(columns = ["row_id"])

        return out

# ------------------------
# BASIC USAGE
# ------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/demand_history.csv")

    # Fake baseline prediction for standalone testing
    df["predicted_demand"] = df["demand"].rolling(7, min_periods = 1).mean().round().astype(int)

    train_df = df[df["day"] < 150]
    test_df = df[df["day"] >= 150][["day", "store_id", "warehouse_id", "promo_flag", "predicted_demand"]]

    ue = UncertaintyEstimator()
    ue.fit(train_df)

    test_with_u = ue.attach_uncertainty(test_df)
    print(test_with_u[["day", "store_id", "predicted_demand", "demand_sigma", "p10", "p50", "p90"]])

    scen = ue.sample_scenarios(test_df.head(3), n_scenarios = 5)
    print("\Scenario sample:")
    print(scen[["scenario_id", "day", "store_id", "predicted_demand", "demand_sigma", "scenario_demand"]])