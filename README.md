# M5 Walmart Sales Forecasting Project

This project contains the complete pipeline for forecasting unit sales for 10 Walmart stores across California, Texas, and Wisconsin. The solution involves data preprocessing, extensive feature engineering, and store-specific model training using LightGBM and XGBoost, with hyperparameters optimized by Optuna.

## Project Objective

The goal is to accurately predict 28 days of future unit sales based on hierarchical sales data, calendar events, and selling prices. A separate, optimized model is trained for each of the 10 stores to capture unique local sales patterns.

## File Structure

The repository is structured as follows:

* `src/pipeline.py`: Contains all data preprocessing and feature engineering functions, including merging, lag/rolling feature creation, and mean encoding.
* `src/utils.py`: Utility functions for loading data and reducing memory usage.
* `notebooks/eda.ipynb`: Exploratory Data Analysis (EDA) of the sales, calendar, and price data.
* `notebooks/experiments.ipynb`: The main notebook for hyperparameter tuning. It uses **Optuna** to find the best parameters for both LightGBM and XGBoost for each of the 10 stores.
* `notebooks/lgb_model.ipynb`: Final training and prediction pipeline for the LightGBM model, using the optimized parameters found in the experiment phase to generate a submission file.
* `notebooks/xgb_model.ipynb`: Final training and prediction pipeline for the XGBoost model, using its optimized parameters to generate a submission file.
* `lgbmodels/`: (Output directory) Stores the optimized LightGBM model parameters (`.pkl` files) for each store.
* `xgbmodels/`: (Output directory) Stores the optimized XGBoost model parameters (`.pkl` files) for each store.

## Methodology

### 1. Data Preprocessing
* **Merge Data**: Combined the `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` files into a single master DataFrame.
* **Handle Missing Values**: `sell_price` nulls were filled using a backward-fill (`bfill`) grouped by item and store, assuming prices are stable backward in time.
* **Memory Reduction**: All columns were downcast to smaller, efficient data types (e.g., `float16`, `int8`) to handle the large dataset (50M+ rows).

### 2. Feature Engineering
An extensive set of time-series and categorical features was created:
* **Time Features**: `month`, `year`, `weekday`, `wday` from the calendar.
* **Lag Features**: `sold` values from previous days (lags: 1, 2, 3, 7, 15, 30 days).
* **Rolling/Expanding Means**: Rolling 7-day mean of `sold` and expanding mean of `sold`.
* **Mean Encoding**: Grouped averages of `sold` for various categorical combinations (e.g., `item_sold_avg`, `store_item_sold_avg`, `state_store_cat_sold_avg`).
* **Price Features**: `selling_trend` (price change over time).

### 3. Modeling and Optimization
* **Models**: Implemented both **LightGBM (LGBM)** and **XGBoost (XGB)**, two powerful gradient boosting frameworks.
* **Strategy**: Trained 10 separate models, one for each unique store (CA_1, CA_2, ..., WI_3).
* **Validation**: Used a local validation set (days 1885-1913) to evaluate models, with the final test set being days 1914-1941.
* **Hyperparameter Tuning**: Used the **Optuna** library to run 8 optimization trials per store for *each* model type, minimizing the **Root Mean Squared Error (RMSE)**.

## Evaluation and Results

The primary metric for optimization and validation was **Root Mean Squared Error (RMSE)**. The tuned XGBoost model for store `CA_4` achieved the best validation score.

### Best Validation RMSE Scores (from Optuna)

| Store | LightGBM RMSE | XGBoost RMSE |
| :--- | :---: | :---: |
| CA\_1 | 0.5726 | 0.5515 |
| CA\_2 | 0.5172 | 0.4446 |
| CA\_3 | 0.6950 | 0.6257 |
| **CA\_4** | 0.3299 | **0.2684** |
| TX\_1 | 0.5472 | 0.5152 |
| TX\_2 | 0.5461 | 0.4640 |
| TX\_3 | 0.4959 | 0.4893 |
| WI\_1 | 0.3675 | 0.3456 |
| WI\_2 | 1.1300 | 0.9524 |
| WI\_3 | 0.6283 | 0.5776 |

## Technologies Used

* **Data Analysis**: Python, Pandas, NumPy
* **Machine Learning**: Scikit-learn, LightGBM, XGBoost
* **Optimization**: Optuna
* **Utilities**: Joblib, os, gc
* **Visualization**: Matplotlib, Seaborn (used in `eda.ipynb`)