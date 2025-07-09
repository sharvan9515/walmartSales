import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')

# Convert dates
for df in [train, test, features]:
    df['Date'] = pd.to_datetime(df['Date'])

# Merge datasets
train_data = train.merge(stores, on='Store', how='left')
train_data = train_data.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')

test_data = test.merge(stores, on='Store', how='left')
test_data = test_data.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')

# Fill markdown NaNs and keep columns separately
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    train_data[col] = train_data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)

# Time related markdown features: diffs and rolling means
train_data = train_data.sort_values(['Store', 'Dept', 'Date'])
test_data = test_data.sort_values(['Store', 'Dept', 'Date'])
for col in markdown_cols:
    train_data[f'{col}_Diff'] = train_data.groupby(['Store', 'Dept'])[col].diff().fillna(0)
    test_data[f'{col}_Diff'] = test_data.groupby(['Store', 'Dept'])[col].diff().fillna(0)
    train_data[f'{col}_Roll2'] = (
        train_data.groupby(['Store', 'Dept'])[col]
                  .transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    )
    test_data[f'{col}_Roll2'] = (
        test_data.groupby(['Store', 'Dept'])[col]
                 .transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    )
    # Interaction with holidays
    train_data[f'{col}_Holiday'] = train_data[col] * train_data['IsHoliday'].astype(int)
    test_data[f'{col}_Holiday'] = test_data[col] * test_data['IsHoliday'].astype(int)

# Additional date features
for df in [train_data, test_data]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['Month_Start_Weight'] = 31 - df['Date'].dt.day

# Holiday name helper
def get_holiday_name(date):
    year = date.year
    if date == pd.Timestamp(f'{year}-11-23') or date.isocalendar().week in [47,48]:
        return 'Thanksgiving'
    elif date.month == 12 and 20 <= date.day <= 26:
        return 'Christmas_Week'
    elif date.month == 11 and 24 <= date.day <= 29:
        return 'Black_Friday'
    elif date.month == 2 and 10 <= date.day <= 19:
        return 'Super_Bowl'
    elif date.month == 9 and date.day < 10:
        return 'Labor_Day'
    elif date.month == 7 and 1 <= date.day <= 7:
        return 'Independence_Day'
    else:
        return 'None'

train_data['Holiday_Name'] = train_data['Date'].apply(get_holiday_name)
test_data['Holiday_Name'] = test_data['Date'].apply(get_holiday_name)

# Temperature bins
bins = [-np.inf, 40, 70, np.inf]
labels = ['Cold', 'Moderate', 'Hot']
for df in [train_data, test_data]:
    df['Temp_Bin'] = pd.cut(df['Temperature'], bins=bins, labels=labels)

# Differences for other numeric features
train_data['Temperature_Diff'] = train_data.groupby('Store')['Temperature'].diff().fillna(0)
test_data['Temperature_Diff'] = test_data.groupby('Store')['Temperature'].diff().fillna(0)
train_data['Fuel_Price_Diff'] = train_data.groupby('Store')['Fuel_Price'].diff().fillna(0)
test_data['Fuel_Price_Diff'] = test_data.groupby('Store')['Fuel_Price'].diff().fillna(0)
train_data['CPI_Diff'] = train_data.groupby('Store')['CPI'].diff().fillna(0)
test_data['CPI_Diff'] = test_data.groupby('Store')['CPI'].diff().fillna(0)

# Feature list
feature_cols = [
    'Store', 'Dept', 'IsHoliday', 'Type', 'Size', 'Year', 'Month', 'Week',
    'Temperature_Diff', 'Fuel_Price_Diff', 'CPI_Diff', 'CPI',
    'DayOfWeek', 'IsWeekend', 'Holiday_Name', 'Month_Start_Weight', 'Temp_Bin'
]
# Add markdown columns and engineered ones
for col in markdown_cols:
    feature_cols.append(col)
    feature_cols.append(f'{col}_Diff')
    feature_cols.append(f'{col}_Roll2')
    feature_cols.append(f'{col}_Holiday')

X = train_data[feature_cols]
y = train_data['Weekly_Sales']

# Preprocessing
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = list(set(feature_cols) - set(numeric_features))

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = xgb.XGBRegressor(random_state=42, tree_method='hist')
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

param_grid = {
    'regressor__n_estimators': [300, 500],
    'regressor__max_depth': [6, 8],
    'regressor__learning_rate': [0.05, 0.1]
}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)

print('Best params:', grid.best_params_)

preds = grid.predict(X_val)
rmse = mean_squared_error(y_val, preds, squared=False)
print('Validation RMSE:', round(rmse,2))

