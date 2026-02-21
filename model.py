"""
Sri Lanka Rainfall Predictor — Model & Data
============================================
Pure data loading and model training. No Streamlit dependency.
Used by frontend.py for integration.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load CSV, aggregate by district and month, add derived columns and rainfall class.
    Returns a DataFrame ready for training.
    """
    df_raw = pd.read_csv(path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    col_map = {
        "city":                       "district",
        "time":                       "time",
        "temperature_2m_mean":        "temperature",
        "precipitation_sum":          "daily_precip_mm",
        "windspeed_10m_max":          "wind_speed",
        "shortwave_radiation_sum":    "radiation",
        "et0_fao_evapotranspiration": "evapotranspiration",
        "winddirection_10m_dominant": "wind_direction",
        "weathercode":                "weather_code",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    df_raw["time"]  = pd.to_datetime(df_raw["time"], errors="coerce")
    df_raw          = df_raw.dropna(subset=["time"])
    df_raw["month"] = df_raw["time"].dt.month
    df_raw["year"]  = df_raw["time"].dt.year

    df = (
        df_raw
        .groupby(["district", df_raw["time"].dt.to_period("M")])
        .agg(
            month               = ("month", "first"),
            year                = ("year", "first"),
            temperature         = ("temperature", "mean"),
            temp_max            = ("temperature_2m_max", "mean"),
            temp_min            = ("temperature_2m_min", "mean"),
            monthly_rainfall_mm = ("daily_precip_mm", "sum"),
            rain_days           = ("precipitation_hours", "sum"),
            wind_speed          = ("wind_speed", "mean"),
            wind_direction      = ("wind_direction", "mean"),
            radiation           = ("radiation", "mean"),
            evapotranspiration  = ("evapotranspiration", "mean"),
            latitude            = ("latitude", "first"),
            longitude           = ("longitude", "first"),
            elevation           = ("elevation", "first"),
        )
        .reset_index()
        .drop(columns=["time"])
    )

    p_max = df["monthly_rainfall_mm"].quantile(0.95)
    df["humidity"] = (55 + 40 * (df["monthly_rainfall_mm"] / (p_max + 1e-9))).clip(40, 100).round(1)

    df["rainfall_class"] = pd.cut(
        df["monthly_rainfall_mm"],
        bins=[-1, 60, 160, 9999],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return df


def train_model(path: str):
    """
    Load data, train Random Forest classifier, return model and artifacts.
    Returns:
        rf, le_district, le_label, features, df, acc, X_test, y_test, y_pred
    """
    df = load_and_prepare(path)

    le_district = LabelEncoder()
    le_label    = LabelEncoder()
    df["district_enc"]       = le_district.fit_transform(df["district"].astype(str))
    df["rainfall_class_enc"] = le_label.fit_transform(df["rainfall_class"].astype(str))

    features = [
        "district_enc", "month", "temperature", "temp_max", "temp_min",
        "wind_speed", "wind_direction",
        "rain_days", "radiation", "evapotranspiration",
        "latitude", "longitude", "elevation"
    ]

    df_clean = df[features + ["rainfall_class_enc"]].dropna()
    X = df_clean[features]
    y = df_clean["rainfall_class_enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_split=5, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    return rf, le_district, le_label, features, df, acc, X_test, y_test, y_pred
