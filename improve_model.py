# -*- coding: utf-8 -*-
"""Model training script for Walmart sales forecasting.

This script loads the provided training and test data, performs a series of
feature engineering steps and trains an XGBoost model using a small grid search.
The code is heavily commented so that readers without extensive machine
learning experience can follow along.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# The project includes four csv files. ``train.csv`` contains the target
# ``Weekly_Sales``. ``test.csv`` has the same columns but without the target.
# ``stores.csv`` adds information about each store and ``features.csv`` adds
# additional context such as temperature and fuel price.

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
STORES_PATH = "stores.csv"
FEATURES_PATH = "features.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
stores = pd.read_csv(STORES_PATH)
features = pd.read_csv(FEATURES_PATH)

# Convert the ``Date`` column from string to ``datetime`` for easier
# manipulation. ``pandas.to_datetime`` automatically understands the
# ``YYYY-MM-DD`` format used in the dataset.
for df in [train, test, features]:
    df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------------------------------------------------------
# Combine the datasets into a single training and test dataframe
# ---------------------------------------------------------------------------
# ``merge`` works like SQL joins. We first add the ``stores`` information and
# then the ``features``. ``how='left'`` keeps all rows from the left table.
train_data = train.merge(stores, on="Store", how="left")
train_data = train_data.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

# Apply the same merging to the test set
test_data = test.merge(stores, on="Store", how="left")
test_data = test_data.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# "MarkDown" columns contain promotion information. Missing values (NaNs) are
# replaced with zero since no markdowns were applied.
markdown_cols = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
]
for col in markdown_cols:
    train_data[col] = train_data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)

# Sort so that calculations that depend on previous rows are correct
train_data = train_data.sort_values(["Store", "Dept", "Date"])
test_data = test_data.sort_values(["Store", "Dept", "Date"])

# For each markdown column we create several additional features:
#   * ``_Diff``  : difference compared to the previous week
#   * ``_Roll2`` : rolling mean over the current and previous week
#   * ``_Holiday``: markdown value only when the week is a holiday
for col in markdown_cols:
    train_data[f"{col}_Diff"] = (
        train_data.groupby(["Store", "Dept"])[col].diff().fillna(0)
    )
    test_data[f"{col}_Diff"] = (
        test_data.groupby(["Store", "Dept"])[col].diff().fillna(0)
    )
    train_data[f"{col}_Roll2"] = (
        train_data.groupby(["Store", "Dept"])[col]
        .transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    )
    test_data[f"{col}_Roll2"] = (
        test_data.groupby(["Store", "Dept"])[col]
        .transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    )
    train_data[f"{col}_Holiday"] = train_data[col] * train_data["IsHoliday"].astype(int)
    test_data[f"{col}_Holiday"] = test_data[col] * test_data["IsHoliday"].astype(int)

# Additional date based features. ``isocalendar().week`` gives the ISO week
# number. ``IsWeekend`` flags Saturdays and Sundays.
for df in [train_data, test_data]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    # Days at the end of the month often have different sales behaviour.
    df["Month_Start_Weight"] = 31 - df["Date"].dt.day

# ``get_holiday_name`` maps each date to one of a few major U.S. holidays.
# These rough rules capture time periods like ``Black Friday`` or
# ``Christmas Week`` so that the model can learn holiday effects.
def get_holiday_name(date: pd.Timestamp) -> str:
    year = date.year
    if date == pd.Timestamp(f"{year}-11-23") or date.isocalendar().week in [47, 48]:
        return "Thanksgiving"
    elif date.month == 12 and 20 <= date.day <= 26:
        return "Christmas_Week"
    elif date.month == 11 and 24 <= date.day <= 29:
        return "Black_Friday"
    elif date.month == 2 and 10 <= date.day <= 19:
        return "Super_Bowl"
    elif date.month == 9 and date.day < 10:
        return "Labor_Day"
    elif date.month == 7 and 1 <= date.day <= 7:
        return "Independence_Day"
    else:
        return "None"

train_data["Holiday_Name"] = train_data["Date"].apply(get_holiday_name)
test_data["Holiday_Name"] = test_data["Date"].apply(get_holiday_name)

# Bucket temperature into categories rather than using raw numbers. ``pd.cut``
# converts a continuous variable into bins.
bins = [-np.inf, 40, 70, np.inf]
labels = ["Cold", "Moderate", "Hot"]
for df in [train_data, test_data]:
    df["Temp_Bin"] = pd.cut(df["Temperature"], bins=bins, labels=labels)

# Differences for a few other numerical features to capture week over week
# change at each store.
train_data["Temperature_Diff"] = train_data.groupby("Store")["Temperature"].diff().fillna(0)
test_data["Temperature_Diff"] = test_data.groupby("Store")["Temperature"].diff().fillna(0)
train_data["Fuel_Price_Diff"] = train_data.groupby("Store")["Fuel_Price"].diff().fillna(0)
test_data["Fuel_Price_Diff"] = test_data.groupby("Store")["Fuel_Price"].diff().fillna(0)
train_data["CPI_Diff"] = train_data.groupby("Store")["CPI"].diff().fillna(0)
test_data["CPI_Diff"] = test_data.groupby("Store")["CPI"].diff().fillna(0)

# ---------------------------------------------------------------------------
# Define the final list of features used by the model
# ---------------------------------------------------------------------------
feature_cols = [
    "Store",
    "Dept",
    "IsHoliday",
    "Type",
    "Size",
    "Year",
    "Month",
    "Week",
    "Temperature_Diff",
    "Fuel_Price_Diff",
    "CPI_Diff",
    "CPI",
    "DayOfWeek",
    "IsWeekend",
    "Holiday_Name",
    "Month_Start_Weight",
    "Temp_Bin",
]

# Add the original markdown values and engineered versions listed above
for col in markdown_cols:
    feature_cols.append(col)
    feature_cols.append(f"{col}_Diff")
    feature_cols.append(f"{col}_Roll2")
    feature_cols.append(f"{col}_Holiday")

X = train_data[feature_cols]
y = train_data["Weekly_Sales"]

# ---------------------------------------------------------------------------
# Preprocessing for the machine learning model
# ---------------------------------------------------------------------------
# Numerical features are scaled using ``StandardScaler`` and categorical
# features are one-hot encoded so that they can be used by XGBoost.
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = list(set(feature_cols) - set(numeric_features))

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
# ``XGBRegressor`` is a tree-based ensemble method. Instead of doing a slow
# grid search, we train once with reasonably strong hyperparameters.  These
# values were found with some quick experimentation and give good results while
# keeping runtime reasonable.  ``tree_method='hist'`` is used for speed.
model = xgb.XGBRegressor(
    n_estimators=1200,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the preprocessing pipeline and XGBoost model
pipe.fit(X_train, y_train)

preds = pipe.predict(X_val)
rmse = mean_squared_error(y_val, preds) ** 0.5
print("Validation RMSE:", round(rmse, 2))
