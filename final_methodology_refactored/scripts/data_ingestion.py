"""
Data Ingestion
==============
Functions for loading and preprocessing news and energy data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def run_ingestion_stage(
    news_path: str,
    energy_path: str,
    min_timestamp: str | None = None,
    news_sample: int | None = None,
    forecast_horizon: int = 24,
    spread_deadband: float = 5.0,
    random_state: int = 42,
):
    """
    Load and preprocess news and energy data.

    Args:
        news_path: Path to news CSV file
        energy_path: Path to energy data CSV file
        min_timestamp: Minimum timestamp for filtering (optional)
        news_sample: Number of news articles to sample (optional, for testing)
        forecast_horizon: Hours ahead to predict (default: 24)
        spread_deadband: EUR/MWh band for neutral class (default: 5.0)
        random_state: Random seed for sampling

    Returns:
        dict containing:
            - news_df: Preprocessed news dataframe
            - energy_df: Preprocessed energy dataframe
            - master_df: Master feature dataframe with target variable
    """
    # --- Load & normalize news feed ---
    news_df = pd.read_csv(news_path)
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df['publishedAt'] = news_df['publishedAt'].dt.tz_localize(None)
    news_df = news_df.set_index('publishedAt').sort_index()

    if min_timestamp:
        news_df = news_df[news_df.index >= min_timestamp]

    if news_sample is not None:
        news_df = news_df.sample(news_sample, random_state=random_state)

    # --- Load energy telemetry ---
    energy_df = pd.read_csv(energy_path)
    energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp']) - pd.Timedelta(hours=1)
    energy_df = energy_df.set_index('Timestamp').sort_index()

    # Drop unnecessary columns if they exist
    if 'Nuclear' in energy_df.columns:
        energy_df = energy_df.drop(columns=['Nuclear'])

    energy_df = energy_df.dropna()

    if min_timestamp:
        energy_df = energy_df[energy_df.index >= f"{min_timestamp} 00:00:00"]

    # --- Baseline feature computation ---
    energy_df['real_spread_abs'] = energy_df['Spot Price'] - energy_df['Day Ahead Auction']

    # Drop additional columns if they exist
    for col in ['Non-Renewable', 'Renewable']:
        if col in energy_df.columns:
            energy_df = energy_df.drop(columns=[col])

    # Create target variable based on spread
    spread_diff = energy_df['real_spread_abs']
    spread_target = np.sign(spread_diff).astype(int)
    neutral_mask = spread_diff.abs() <= spread_deadband
    spread_target[neutral_mask] = 0
    energy_df['spread_target'] = spread_target

    # --- Create master dataframe with lagged features ---
    master_df = energy_df.copy()
    master_df['real_spread_abs_shift_24'] = master_df['real_spread_abs'].shift(-forecast_horizon)
    master_df['spread_target_shift_24'] = master_df['spread_target'].shift(-forecast_horizon)
    master_df['price_lag_24'] = master_df['Spot Price'].shift(forecast_horizon)
    master_df['price_lag_168'] = master_df['Spot Price'].shift(168)  # 1 week
    master_df['load_lag_24'] = master_df['Load'].shift(forecast_horizon)
    master_df['load_lag_168'] = master_df['Load'].shift(168)  # 1 week

    # Add temporal features
    master_df['hour'] = master_df.index.hour
    master_df['week_of_year'] = master_df.index.isocalendar().week
    master_df['month'] = master_df.index.month
    master_df['day_of_week'] = master_df.index.dayofweek
    master_df['day_of_year'] = master_df.index.dayofyear

    master_df = master_df.dropna()

    print(f"News shape after filters: {news_df.shape}")
    print(f"Energy telemetry shape: {energy_df.shape}")
    print(f"Baseline feature frame: {master_df.shape}")

    return {
        "news_df": news_df,
        "energy_df": energy_df,
        "master_df": master_df,
    }
