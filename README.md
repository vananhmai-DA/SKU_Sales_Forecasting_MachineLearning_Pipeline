# SKU Sales Forecasting with Machine Learning & SHAP

## 1.Overview

This project builds a machine learning model to forecast daily sales quantity for each SKU in an e-commerce business.

The business objective is to support inventory planning and purchasing decisions by reducing reliance on manual demand estimation.

```text
Input:  Date + SKU
Output: Predicted sales quantity
```

The project also uses SHAP to explain what drives the model predictions, making the forecast more understandable for business users.

---

## 2.Business Problem

Poor demand forecasting can lead to:

* Stockouts for fast-selling SKUs
* Overstock for slow-moving SKUs
* Inefficient purchasing decisions
* Higher inventory holding costs

This project helps estimate future SKU demand using historical sales patterns.

---

## 3.Dataset

| Item                |  Value |
| ------------------- | -----: |
| Raw records         | 48,363 |
| Original SKUs       |    676 |
| Sales channels      |      6 |
| Observed dates      |    184 |
| Final modeling SKUs |    180 |
| Final modeling rows | 29,004 |

The raw data was aggregated to the forecasting grain:

```text
shipped_date + sku
```

---

## 4.ML Process

```mermaid
flowchart LR
    A[Raw Sales Data] --> B[EDA & Cleaning]
    B --> C[Date-SKU Aggregation]
    C --> D[Feature Engineering]
    D --> E[Time-Based Split]
    E --> F[Model Training]
    F --> G[Model Improvement]
    G --> H[SHAP Explanation]
```

Key modeling decisions:

* Aggregated transaction data to Date-SKU level
* Removed negative quantity records as likely returns or corrections
* Did not fill missing calendar dates with zero because data was observed every 2 days
* Selected SKUs with at least 120 active selling days to ensure enough history
* Used time-based split instead of random split
* Used lag, rolling, and SKU historical features
* Excluded `revenue`, `cost_of_good_sold`, and `channel_count` to avoid data leakage

---

## 5.Feature Engineering

Main feature groups:

| Feature group    | Examples                                             | Purpose                     |
| ---------------- | ---------------------------------------------------- | --------------------------- |
| Date features    | `day_of_week`, `month`, `is_weekend`                 | Capture calendar patterns   |
| Lag features     | `qty_lag_1`, `qty_lag_7`, `qty_lag_14`               | Capture previous demand     |
| Rolling features | `qty_rolling_mean_7`, `qty_rolling_mean_14`          | Capture recent demand trend |
| SKU history      | `sku_expanding_mean_qty`, `sku_expanding_median_qty` | Capture SKU-level behavior  |

Rolling and expanding features were shifted before calculation to prevent leakage from the target period.

---

## 6.Models & Results

Models tested:

* Baseline Lag 1
* Random Forest
* XGBoost
* LightGBM

| Model                |       WAPE |
| -------------------- | ---------: |
| Baseline Lag 1       |     50.21% |
| Random Forest        |     45.64% |
| XGBoost              |     46.20% |
| LightGBM             |     45.54% |
| Final Tuned LightGBM | **45.30%** |

Final model:

```text
Final_LightGBM_V1_Tuned_1
```

The final tuned LightGBM improved WAPE from **50.21%** to **45.30%**, outperforming the simple historical baseline by **4.91 percentage points**.

---

## 7.Explainability with SHAP

SHAP was used to explain which features drive the final model's forecasts.

### Global Feature Importance

![SHAP Bar Plot](05_outputs/figures/shap_bar_plot.png)

Top drivers:

```text
qty_rolling_mean_7
qty_rolling_mean_14
day_of_week
qty_lag_1
qty_rolling_mean_3
moq_order
sku_expanding_median_qty
sku_expanding_mean_qty
```

The strongest driver is `qty_rolling_mean_7`, meaning recent average SKU demand is the most important signal for future demand.

### Feature Impact Direction

![SHAP Summary Plot](05_outputs/figures/shap_summary_plot.png)

Key interpretation:

* Higher recent rolling demand generally increases the forecast.
* Lower recent rolling demand usually reduces the forecast.
* Rolling averages are more stable than using only the previous sales period.
* `day_of_week` suggests that weekday patterns affect demand.
* SKU-level historical behavior helps distinguish high-demand and low-demand SKUs.

---

## 8.Business Insights

1. **Recent demand trend is the strongest signal.**
   The model relies most on rolling average sales, especially `qty_rolling_mean_7`.

2. **A single previous value is not enough.**
   The ML model outperforms the `qty_lag_1` baseline by combining multiple historical demand signals.

3. **Fast-moving SKUs should be monitored more closely.**
   SKUs with rising recent rolling demand may need earlier replenishment to reduce stockout risk.

4. **Forecasts should support, not replace, business decisions.**
   Purchasing decisions should still consider inventory level, supplier lead time, promotions, and business priorities.

---

## 9.Project Structure

```text
├── 01_data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── 02_notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_improvement.ipynb
│   └── 05_shap_explainability.ipynb
│
├── 04_models/
│   └── final_sales_forecasting_model.pkl
│
├── 05_outputs/
│   ├── model_results/
│   ├── forecasts/
│   └── figures/
│
├── README.md
└── requirements.txt
```

---

## 10.Key Outputs

| Output                              | Description                   |
| ----------------------------------- | ----------------------------- |
| `sales_features.csv`                | Final ML feature dataset      |
| `final_sales_forecasting_model.pkl` | Final trained LightGBM model  |
| `final_test_forecasts.csv`          | Forecast output on test set   |
| `final_model_comparison.csv`        | Final model performance       |
| `shap_bar_plot.png`                 | SHAP global importance        |
| `shap_summary_plot.png`             | SHAP feature impact direction |

---

## 11.Limitations

* Data is observed every 2 days, not daily.
* Only SKUs with enough history were modeled.
* Promotion, pricing, holiday, stock level, and marketing features were not available.
* SKU-level demand is noisy, so forecast output should be combined with inventory rules such as safety stock and lead time.

---

## 12.Conclusion

This project demonstrates an end-to-end SKU-level sales forecasting workflow with explainable machine learning.

The main value is not only the final model result, but the ML thinking behind the pipeline:

```text
right forecasting grain
time-based validation
leakage-safe features
baseline comparison
model improvement
SHAP explainability
```

The final tuned LightGBM achieved **45.30% WAPE**, improving over the **50.21% baseline**. SHAP analysis shows that recent rolling SKU demand is the strongest driver of the forecast.
