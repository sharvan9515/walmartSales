# Step-by-Step Explanation

This document walks through each part of `improve_model.py` and highlights
concepts that may be confusing if you are new to data science or Python.

## 1. Loading the Data
`pandas.read_csv` is used to read CSV files into dataframes.  The `Date`
column is converted to a real date object using `pd.to_datetime` so that we
can easily extract information like the year or week number.

## 2. Merging Datasets
The training and test files are merged with `stores.csv` and `features.csv`
using `pandas.merge`.  This is similar to a SQL join.  We use a **left join** so
that every row in the original dataset is kept even if there is no matching row
in the auxiliary tables.

## 3. Handling Missing Values
Promotion columns named `MarkDown1` ... `MarkDown5` often contain empty
values.  We replace these with zero because an empty cell means that no markdown
was offered that week.

## 4. Creating New Features
Many features are derived from existing columns:

- Differences (`*_Diff`) compare the current value with the previous week.
  This shows whether prices or markdowns are increasing or decreasing.
- Rolling averages (`*_Roll2`) smooth the markdown values across two weeks.
- Holiday interactions (`*_Holiday`) multiply the markdown by the `IsHoliday`
  flag so the model can learn the combined effect.
- Additional date features such as `Year`, `Month` and `IsWeekend` capture
  seasonality.
- The helper function `get_holiday_name` assigns well known holidays so the
  model can treat them individually.

## 5. Preparing the Data for Machine Learning
Machine learning models expect numerical input.  We scale numerical columns with
`StandardScaler` and convert categorical columns into one-hot encoded vectors
with `OneHotEncoder`.  `ColumnTransformer` applies these transformations
correctly to each column.

## 6. Training the Model
The code uses `XGBRegressor` from the XGBoost library, wrapped in a
`Pipeline` so that preprocessing and model training happen together.  A grid
search (`GridSearchCV`) tries a few combinations of hyperparameters to find the
best performing model.

Run the script with `python improve_model.py`.  The output will show the best
parameters found and the validation RMSE (root mean squared error).
