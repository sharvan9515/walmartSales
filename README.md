# Walmart Sales Forecasting

This project demonstrates how to predict weekly sales for Walmart stores using Python and machine learning. The dataset combines historical sales with store information and various economic indicators.

## Files
- `train.csv` and `test.csv` – base datasets provided by the competition
- `features.csv` – additional variables such as temperature and fuel price
- `stores.csv` – metadata about each store
- `improve_model.py` – Python script that trains the model
- `EXPLANATION.md` – step-by-step explanation of the workflow

## Quick Start
1. Install the required packages (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python improve_model.py
   ```
   The script will print the validation RMSE.

## About the Model
The script trains an XGBoost regressor with hand-tuned hyperparameters. Several engineered features capture holiday effects, promotions and temporal trends. See `EXPLANATION.md` for a beginner-friendly walkthrough.
