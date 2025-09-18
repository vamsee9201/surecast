"""Shared column names and the feature list — the contract between modules.

Keep this tiny and stable; everyone imports from here so a rename can't drift
across the parallel-built modules.
"""

# Index
TS = "ts"  # hourly DatetimeIndex name

# Target
TARGET = "cnt"

# Raw exogenous columns kept from hour.csv (after load_hourly)
RAW_WEATHER = ["temp", "atemp", "hum", "windspeed", "weathersit"]
RAW_CALENDAR = ["workingday", "holiday", "season"]

# Forecast framing
HORIZON = 24  # hours ahead

# Engineered feature columns produced by build_features, in a stable order.
# Every one must be knowable at t - HORIZON (calendar/weather-forecast) or be a
# target lag of >= HORIZON hours. No lag shorter than HORIZON may appear here.
CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "workingday",
    "holiday",
    "hour_sin",
    "hour_cos",
]
WEATHER_FEATURES = ["temp", "hum", "windspeed", "weathersit"]
LAG_FEATURES = ["lag_24", "lag_25", "lag_48", "lag_168"]
ROLLING_FEATURES = ["roll_mean_24", "roll_std_24"]

FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES + ROLLING_FEATURES
