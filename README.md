# M5 Demand Forecasting + Inventory Optimization — Dashboard
- https://m5-demand-forecasting-dashboard-1.streamlit.app/

Interactive Streamlit dashboard for the M5 Walmart demand forecasting project.
Compares ARIMA/SARIMA, LightGBM, and LSTM forecasts, and converts predictions
into actionable inventory decisions (Safety Stock, Reorder Point, EOQ).

## Project Structure

```
your-repo/
├── app.py                 # Streamlit dashboard (this file)
├── requirements.txt        # Python dependencies
├── README.md
└── data/
    ├── lgb_test_predictions.parquet
    ├── inventory_recommendations.csv
    ├── model_comparison.csv
    └── arima_results.csv
```

## Step 1: Download outputs from Kaggle

From your 3 Kaggle notebooks, download these files (Output tab -> download):

| File | From notebook |
|---|---|
| `lgb_test_predictions.parquet` | Notebook 2 |
| `arima_results.csv` | Notebook 2 |
| `model_comparison.csv` | Notebook 3 |
| `inventory_recommendations.csv` | Notebook 3 |

Put all 4 files inside a folder named `data/` next to `app.py`.

## Step 2: Run locally (optional, to test before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Step 3: Deploy to Streamlit Community Cloud (free)

1. Create a new GitHub repository
2. Push `app.py`, `requirements.txt`, `README.md`, and the `data/` folder to it
   - Note: `lgb_test_predictions.parquet` can be large — if GitHub rejects it
     (>100MB), reduce it before uploading, e.g. keep only the last 28 days
     per item (which is what the dashboard actually uses)
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
4. Click "New app", select your repo, branch, and `app.py` as the main file
5. Click "Deploy" — you'll get a public URL like
   `https://your-app-name.streamlit.app`

## Dashboard Features

- **Item Forecast tab** — select any state/store/category/item, see actual
  vs LightGBM-forecasted sales
- **Model Comparison tab** — ARIMA vs LightGBM vs LSTM RMSE bar chart
- **Inventory Recommendation tab** — Safety Stock, Reorder Point, EOQ per
  item, plus an all-items overview
- **What-If Simulator tab** — interactively change service level / lead
  time and see how safety stock and reorder point change

## Notes

- If `lgb_test_predictions.parquet` is too large for GitHub, you can trim it
  in Kaggle before downloading:
  ```python
  test_data[['id','item_id','store_id','state_id','cat_id','date','sales','pred_lgb']].to_parquet(
      'lgb_test_predictions_small.parquet', index=False
  )
  ```
  (This is already what the notebook saves — should be well under 100MB
  since it's only the 28-day test window.)
