import pandas as pd
import numpy as np
from utils import normalize_column

# Expected features by the model
EXPECTED_FEATURES = ['PRECTOTCORR', 'day_of_year', 'month', 'RH2M', 'RH2M_lag_3', 'T2M',
                     'RH2M_lag_7', 'RH2M_lag_4', 'RH2M_lag_6', 'RH2M_lag_1', 'T2M_lag_6',
                     'T2M_lag_7', 'RH2M_lag_5', 'T2M_lag_4', 'T2M_lag_1', 'T2M_lag_5', 
                     'RH2M_lag_2', 'T2M_lag_2', 'T2M_lag_3', 'season']

# ----------------------------------------
# MAIN PREPROCESSING FUNCTION
# ----------------------------------------
def preprocess_inputs(df):
    """
    Preprocessing pipeline to match model training features:
    - Handles PRECTOTCORR (precipitation), RH2M (relative humidity), T2M (temperature)
    - Creates temporal features (day_of_year, month, season)
    - Creates lag features
    - Normalizes features
    """
    
    # Make a copy to avoid modifying original
    df = df.copy()

    # Handle date/timestamp column for temporal features
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    elif "timestamp" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["timestamp"], unit="s", errors='coerce')
        except:
            df["Date"] = pd.to_datetime(df["timestamp"], errors='coerce')
    else:
        df["Date"] = pd.to_datetime(pd.Timestamp.now())

    # Sort by date
    df = df.sort_values("Date", na_position='last').reset_index(drop=True)

    # -------------------------------------------
    # EXTRACT TEMPORAL FEATURES
    # -------------------------------------------
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    
    # Define season
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    df['season'] = df['month'].apply(get_season)

    # -------------------------------------------
    # HANDLE WEATHER FEATURES
    # -------------------------------------------
    # Map common column names to expected names
    feature_mapping = {
        'PRECTOTCORR': ['PRECTOTCORR', 'precipitation', 'Precip', 'Rain'],
        'RH2M': ['RH2M', 'Humidity', 'RH', 'RelativeHumidity'],
        'T2M': ['T2M', 'Temp', 'Temperature', 'T'],
    }
    
    for target, aliases in feature_mapping.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[target] = df[alias]
                    break
    
    # Initialize missing features with defaults
    for col in ['PRECTOTCORR', 'RH2M', 'T2M']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            # Convert to numeric and fill NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0.0)
            # Normalize
            df[col] = normalize_column(df[col])

    # -------------------------------------------
    # CREATE LAG FEATURES
    # -------------------------------------------
    for lag in range(1, 8):
        for col in ['RH2M', 'T2M']:
            lag_col_name = f"{col}_lag_{lag}"
            df[lag_col_name] = df[col].shift(lag)
            # Fill lag NaN values
            df[lag_col_name] = df[lag_col_name].fillna(df[col].mean() if df[col].notna().any() else 0.0)

    # -------------------------------------------
    # BUILD FINAL FEATURE MATRIX IN CORRECT ORDER
    # -------------------------------------------
    final_df = pd.DataFrame()

    for feature in EXPECTED_FEATURES:
        if feature in df.columns:
            final_df[feature] = df[feature].astype(float)
        else:
            # If missing, fill with zeros (or normalized default)
            final_df[feature] = 0.0

    return final_df
