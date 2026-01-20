# demand_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class DemandModel:
    """
    Bare-bones demand forecasting model.
    Trains a separate linear model per store.
    """

    def __init__(self):
        self.models = {}

    def fit(self, df: pd.DataFrame):
        """
        Train one model per store.
        """
        for store_id, store_df in df.groupby("store_id"):
            X = store_df[["day", "promo_flag"]]
            y = store_df["demand"]

            model = LinearRegression()
            model.fit(X, y)

            self.models[store_id] = model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict demand for each store.
        """
        predictions = []

        for store_id, store_df in df.groupby("store_id"):
            model = self.models.get(store_id)
            if model is None:
                raise ValueError(f"No model found for store {store_id}")

            X = store_df[["day", "promo_flag"]]
            y_pred = model.predict(X)

            tmp = store_df.copy()
            tmp["predicted_demand"] = np.maximum(0, y_pred.round().astype(int))
            predictions.append(tmp)

        return pd.concat(predictions, ignore_index=True)


# ------------------------
# BASIC USAGE
# ------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/demand_history.csv")

    # Train / test split by time
    train_df = df[df["day"] < 150]
    test_df = df[df["day"] >= 150]

    model = DemandModel()
    model.fit(train_df)

    preds = model.predict(test_df)

    print(preds[["day", "store_id", "demand", "predicted_demand"]].head())
