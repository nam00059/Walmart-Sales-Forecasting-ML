# 🛒 Walmart Sales Forecasting with Machine Learning

This project applies a comprehensive suite of machine learning models—including tree-based models and neural networks—to forecast item-level daily sales for Walmart. We experimented with various architectures, feature engineering strategies, and an ensemble model to optimize performance under the WRMSSE evaluation metric.

---

## 🎯 Objective

- Forecast 28-day daily sales across thousands of Walmart products using historical and calendar data
- Capture trends, promotions, seasonal cycles, and event effects
- Improve model accuracy through extensive feature engineering and ensemble learning

---

## 🛠 Tools & Libraries

- Python (Pandas, NumPy, Scikit-learn, Optuna)
- LightGBM, XGBoost, AdaBoost, Linear Regression
- LSTM / CNN-LSTM (TensorFlow / Keras)
- Jupyter Notebooks (.ipynb)
- Data: M5 Forecasting Dataset (item/store/state/calendar/pricing)

---

## 🧮 Feature Engineering

We engineered over 50 features in the following categories:

- **Categorical encoding**: `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id`, `event_type`
- **Calendar/time-based features**: `day_of_week`, `month`, `weekofyear`, `season`, `is_event`, `is_month_end`
- **Promotion features**: `snap_active`, `days_since_snap`, `days_until_next_snap`
- **Lag features**: `sales_lag_1`, `sales_lag_7`, `..._lag_98`
- **Rolling windows**: `rolling_mean_7`, `rolling_std_14`, `expanding_mean`
- **Group-level aggregations**: `rolling_mean_store_dept`, `cat_sales`, `dept_sales`
- **Shifted stats** to prevent data leakage

📂 See [`Feature_Engineering_1.ipynb`](analysis/Feature_Engineering_1.ipynb) and [`Feature_Engineering_2.ipynb`](analysis/Feature_Engineering_2.ipynb) for full pipeline.

---

## 🤖 Models Tested

| Model            | Key Characteristics                              |
|------------------|---------------------------------------------------|
| Linear Regression | Baseline |
| AdaBoost         | Shallow boosting with squared loss |
| XGBoost          | Tree boosting with regularization |
| LightGBM         | Optimized with Optuna; tweedie objective |
| LSTM (Encoder)   | Time-series neural net with categorical embeddings |
| CNN-LSTM         | Combined convolutional and sequential modeling |
| **Ensemble**     | Weighted average of LGBM, XGBoost, and LSTM |

📄 See [`Write Up - Predictive Project.pdf`](presentation/Write_Up_Predictive_Project.pdf) for full details.

---

## 🏆 Model Performance (WRMSSE)

| Model         | Score   |
|---------------|---------|
| Linear Reg    | 0.883   |
| AdaBoost      | 1.094   |
| XGBoost       | 2.120   |
| LightGBM      | **0.611** |
| LSTM (Enc)    | 0.980   |
| CNN-LSTM      | 1.285   |
| **Ensemble**  | **0.616** (best balance of accuracy & generalization)

---

