
# 🛒 Walmart Sales Forecasting

## 📌 Project Overview

This project focuses on **Walmart Sales Forecasting** using historical sales data and machine learning/time-series forecasting techniques.

The objective is to analyze historical Walmart sales patterns, identify important trends and seasonal effects, and build models capable of predicting future sales.

This project covers the complete workflow from **data preprocessing and exploratory data analysis (EDA) to feature engineering, model building, evaluation, and forecasting**.


- https://m5-demand-forecasting-dashboard-1.streamlit.app/
---

## 🎯 Objectives

- Analyze historical Walmart sales data.
- Identify trends, seasonality, and recurring sales patterns.
- Perform data cleaning and preprocessing.
- Handle missing values and outliers.
- Create meaningful time-series features.
- Compare different forecasting approaches.
- Evaluate model performance using appropriate metrics.
- Forecast future Walmart sales.

---

## 📊 Dataset

The dataset contains historical Walmart sales information, including features such as:

- Store
- Date
- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- CPI
- Unemployment

The **Weekly Sales** variable is used as the primary target for forecasting.

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

- Missing-value analysis
- Distribution analysis
- Store-wise sales comparison
- Weekly/monthly/yearly sales trends
- Holiday vs non-holiday sales analysis
- Correlation analysis
- Time-series visualization
- Identification of seasonal patterns and trends

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

1. Handling missing values.
2. Converting the `Date` column into datetime format.
3. Sorting observations chronologically.
4. Checking and handling duplicate records.
5. Detecting potential outliers.
6. Creating time-based features.
7. Preparing the data for time-series modeling.

Example time-based features:

```python
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week
df["Day"] = df["Date"].dt.day
```

---

## ⚙️ Feature Engineering

Important features can include:

- Year
- Month
- Week
- Quarter
- Holiday indicator
- Lag features
- Rolling mean
- Rolling standard deviation
- Store-level historical sales

Example:

```python
df["lag_1"] = df["Weekly_Sales"].shift(1)
df["rolling_mean"] = df["Weekly_Sales"].rolling(4).mean()
```

---

## 🤖 Models

Different approaches can be explored for forecasting:

### Statistical / Time-Series Models

- ARIMA
- SARIMA
- Auto ARIMA

### Machine Learning Models

- Linear Regression
- Random Forest
- XGBoost
- LightGBM

### Deep Learning

- RNN
- LSTM
- GRU

Models are compared based on their forecasting performance.

---

## 📏 Model Evaluation

The models can be evaluated using:

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
```

---

## 📈 Forecasting

After selecting the best-performing model, it is used to generate forecasts for future Walmart sales.

The forecasting process helps understand expected future demand and provides insights that can potentially support:

- Inventory planning
- Demand forecasting
- Store-level planning
- Resource allocation
- Business decision-making

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Pmdarima
- LightGBM
- Jupyter Notebook

---

## 📁 Project Structure

```text
Walmart-Sales-Forecasting/
│
├── data/
│   └── Walmart.csv
│
├── notebooks/
│   └── Walmart_Sales_Forecasting.ipynb
│
├── models/
│   └── trained_models/
│
├── visualizations/
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Walmart-Sales-Forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook
```

Open:

```text
Walmart_Sales_Forecasting.ipynb
```

---

## 💡 Key Insights

The analysis focuses on understanding:

- How sales change over time.
- Which stores generate higher sales.
- The impact of holidays on sales.
- Seasonal sales patterns.
- The relationship between economic factors and sales.
- Which forecasting model performs best.

---

## 🔮 Future Improvements

Potential improvements include:

- Hyperparameter tuning.
- Store-specific forecasting models.
- Advanced feature engineering.
- Ensemble forecasting.
- LSTM/GRU-based forecasting.
- Automated model selection.
- Deployment using Streamlit or Flask.
- Creating an interactive sales forecasting dashboard.

---

## 👨‍💻 Author

**DHARM PRAKASH**

If you found this project useful, consider ⭐ starring the repository.
