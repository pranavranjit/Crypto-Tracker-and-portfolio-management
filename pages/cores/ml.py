import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.linear_model import LinearRegression, ElasticNet


def prepare_features_targets(df: pd.DataFrame, feature_cols):
    """
    Shift target column up by lag to predict future returns
    """
    X = df[feature_cols]
    y = df["return_target"]
    return X, y


def train_forecaster(df, feature_cols, model):
    """
    Trains forecasting model to predict
    """
    X, y = prepare_features_targets(df=df, feature_cols=feature_cols)
    model.fit(X, y)
    return model


def predict_model(df, feature_cols, model):
    features, _ = prepare_features_targets(df=df, feature_cols=feature_cols)
    model_preds = model.predict(features)
    return model_preds


def rolling_forecast_pipeline(
    df, feature_cols, model, train_size=60, pred_horizon=7, max_train_window=None
):
    """
    Rolling retrain pipeline:
    1. Train on `train_size` points
    2. Predict `pred_horizon` points
    3. Retrain on the expanded window (train + predicted period)

    The feature/target columns are pulled into contiguous NumPy arrays once and
    sliced per step, so each refit skips pandas indexing and sklearn DataFrame
    validation -- the same math, but several times faster over a long history.

    Features are standardised at every refit with the mean/std of that refit's
    training window only (no future statistics leak in). RSI lives on 0-100
    while returns are ~0.01, so without this the penalised models were really
    only penalising the small-scale features.

    Predictions cover every row from ``train_size`` to the end, including a
    final partial block, so ``preds[i]`` belongs to ``df.iloc[train_size + i]``.

    ``max_train_window`` optionally caps how far back each refit looks (a
    rolling instead of fully-expanding window), which keeps per-step cost flat
    on very long series. ``None`` keeps the original expanding-window behaviour.
    """
    n = len(df)
    # Contiguous float arrays -> fast slicing and fitting.
    X_all = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=float))
    y_all = np.ascontiguousarray(df["return_target"].to_numpy(dtype=float))

    preds = []
    end = train_size
    while end < n:
        start = 0 if max_train_window is None else max(0, end - max_train_window)
        X_train = X_all[start:end]
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd == 0] = 1.0  # constant feature (e.g. volume for a mutual fund)
        model.fit((X_train - mu) / sd, y_all[start:end])
        block = slice(end, min(end + pred_horizon, n))
        preds.append(model.predict((X_all[block] - mu) / sd))
        end += pred_horizon

    if not preds:
        return np.empty(0, dtype=float)
    return np.concatenate(preds)


def get_model_preds(df, model_name, features, train_size=60, pred_horizon=7):
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "ols":
        model = LinearRegression()
    elif model_name == "elasticnet":
        # Daily returns are ~1e-2, so alpha=0.1 zeroed every coefficient and the
        # "model" was just each fund's historical mean. 1e-4 keeps it sparse
        # without flattening it.
        model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000)
    else:
        raise ValueError("Unknown model name")

    preds = rolling_forecast_pipeline(
        df=df,
        feature_cols=features,
        model=model,
        train_size=train_size,
        pred_horizon=pred_horizon,
    )
    return preds


def getModelToPreds(df, models: list[str], features, train_size=60, pred_horizon=7):
    model_to_preds = {}
    for model_name in models:
        preds = get_model_preds(
            df=df,
            model_name=model_name,
            train_size=train_size,
            pred_horizon=pred_horizon,
            features=features,
        )
        model_to_preds[model_name] = preds
    return model_to_preds
