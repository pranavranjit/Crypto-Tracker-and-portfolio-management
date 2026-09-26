import numpy as np
import pandas as pd
from .ml import getModelToPreds

CASH = "Cash"


def multi_symbol_rotation_by_model(
    symbol_to_df: dict[str, pd.DataFrame],
    symbols: list[str],
    model_name_list: list[str],
    features: list[str],
    train_size=60,
    pred_horizon=3,
    threshold=0.0,
    initial_capital=10000,
    cost_bps=0.0,
):
    """
    Rotational strategy for multiple models.

    At the close of each trading day every model forecasts each symbol's
    next-day return; the portfolio then holds the symbol with the highest
    forecast for the following day, or cash when no forecast exceeds
    ``threshold`` (a daily return, e.g. 0.0005 = 5 bps). Features on day t
    only use prices up to t and the return booked for that decision is day
    t+1's, so the backtest never trades on information it could not have had.

    ``cost_bps`` is charged each time the holding changes (including moves
    into or out of cash).

    Returns a dict of portfolio curves per model (columns: date,
    portfolio_value, holding) and a dict of annualised Sharpe ratios. Each
    curve row is the value at that day's close and the position chosen then.
    """
    model_symbol_preds = {model: {} for model in model_name_list}
    next_day_returns = {}

    # Step 1: Get predictions for each symbol and model
    for sym in symbols:
        df = symbol_to_df.get(sym)
        if df is None or df.empty:
            continue
        # Make sure the columns the rotation logic relies on exist. A 'date'
        # column may instead be sitting in the index (e.g. after set_index);
        # recover it rather than blowing up with a raw KeyError.
        df = df.copy()
        if "date" not in df.columns:
            if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "date"})
            else:
                continue
        if "return_target" not in df.columns:
            continue
        if len(df) <= train_size:
            continue

        dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        next_day_returns[sym] = pd.Series(
            df["return_target"].to_numpy(dtype=float), index=dates
        )

        preds_dict = getModelToPreds(
            df=df,
            models=model_name_list,
            train_size=train_size,
            pred_horizon=pred_horizon,
            features=features,
        )
        for model_name, preds in preds_dict.items():
            if preds is None or len(preds) == 0:
                continue
            # preds[i] belongs to row train_size + i (see rolling_forecast_pipeline)
            model_symbol_preds[model_name][sym] = pd.Series(
                preds, index=dates[train_size : train_size + len(preds)]
            )

    portfolio_curves = {}
    sharpe_ratios = {}
    if not next_day_returns:
        return portfolio_curves, sharpe_ratios

    fwd = pd.DataFrame(next_day_returns).sort_index()

    # Step 2: For each model, pick the best forecast per day and book the
    # following day's return of that pick.
    for model_name, sym_preds in model_symbol_preds.items():
        if not sym_preds:
            # No symbol produced usable predictions for this model.
            continue
        preds = pd.DataFrame(sym_preds).sort_index()
        # Start once every symbol has a forecast (funds with shorter histories
        # would otherwise leave the early years a one-horse race); after that a
        # symbol missing a day's quote simply sits that day out.
        complete = preds.dropna(how="any")
        if complete.empty:
            continue
        preds = preds.loc[complete.index[0] :].dropna(how="all")

        best_symbol = preds.idxmax(axis=1)
        best_pred = preds.max(axis=1)
        invested = best_pred > threshold
        holding = best_symbol.where(invested, CASH)

        fwd_aligned = fwd.reindex(index=preds.index, columns=preds.columns)
        col_idx = preds.columns.get_indexer(best_symbol)
        picked = fwd_aligned.to_numpy()[np.arange(len(preds)), col_idx]
        period_ret = pd.Series(
            np.where(invested.to_numpy(), picked, 0.0), index=preds.index
        ).fillna(0.0)

        # Transaction cost whenever the position changes.
        switched = holding.ne(holding.shift())
        switched.iloc[0] = holding.iloc[0] != CASH
        period_ret = period_ret - switched.astype(float) * (cost_bps / 1e4)

        # Value at each decision date's close, then one more point for the day
        # the final decision is realised on.
        growth = (1.0 + period_ret).cumprod().to_numpy()
        values = initial_capital * np.concatenate([[1.0], growth])
        curve_dates = preds.index.append(
            pd.DatetimeIndex([preds.index[-1] + pd.offsets.BDay(1)])
        )
        portfolio_curves[model_name] = pd.DataFrame(
            {
                "date": curve_dates,
                "portfolio_value": values,
                "holding": np.concatenate([holding.to_numpy(dtype=object), [None]]),
            }
        )

        # Annualised Sharpe on trading-day returns
        std_ret = period_ret.std()
        sharpe_ratios[model_name] = (
            period_ret.mean() / std_ret * np.sqrt(252) if std_ret > 0 else np.nan
        )

    return portfolio_curves, sharpe_ratios
