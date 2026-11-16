# AI-powered e-commerce business intelligence system

A decision-support system built on the Olist Brazilian e-commerce dataset. It cleans the
data, stores it in a relational database, runs four analytical agents (each a weighted
ensemble of five machine-learning models), and presents the results through a dashboard
and a chatbot.

## What it does

| Component | File | What it gives you |
|---|---|---|
| Revenue prediction | `models/revenue.py` | Daily revenue forecast up to 90 days ahead, trend and growth |
| Net revenue analysis | `models/netrevenue.py` | Exact gross, deductions and net revenue from SQL, plus a forecast |
| Product demand prediction | `models/demand.py` | Weekly units for the 20 largest categories, with stock advice |
| Customer churn prediction | `models/churn.py` | Churn probability and risk band for every customer |
| Central controller | `controller/main.py` | One entry point that routes requests to the agents |
| Dashboard | `ui/app.py` | Six pages: overview, revenue, customers, demand, net revenue, assistant |
| Chatbot | `chatbot/RAG.py` | Answers questions from the database, the agents and knowledge documents |

## Setup after cloning

The repository contains the code, the nine raw Kaggle CSV files, the trained models and the
reports. The database and the cleaned data are not included (the database is 246 MB, over
GitHub's file limit), so build them once. Run these from the project root, in order; each
step reads the output of the one before. Total time is about 10 minutes.

```bash
pip install -r requirements.txt

# 1. Clean the raw data -> dataset/clean dataset/ and the Excel reports (about 3 min)
cd olist-analysis
python -m src.profiling
python -m src.cleaning
python -m src.validation
python -m src.feature_engineering

# 2. Generate the synthetic repeat orders for the churn agent -> dataset/synthetic dataset/ (20 s)
python -m src.synthetic
cd ..

# 3. Build the database -> database/ecommerce.db (2 min)
python database/build_db.py

# 4. Train the four agents -> trained_models/*.pkl and the forecast tables in the database (4 min)
python controller/main.py train
```

Step 4 is needed even though the `.pkl` files are included: training also writes the
forecasts and churn scores into the new database, and the dashboard reads them from there.

## Running it

```bash
streamlit run ui/app.py                             # dashboard at http://localhost:8501
python controller/main.py full_analysis            # all four agents plus recommendations
python controller/main.py revenue --horizon_days 30
python chatbot/RAG.py "What was the total revenue last month?"
python chatbot/RAG.py                               # interactive chat
```

## Project layout

```
dataset/
  original dataset/     the nine raw Olist CSV files (never modified; checked by SHA-256)
  clean dataset/        cleaned tables used to build the database (generated, not in git)
  synthetic dataset/    generated repeat orders, churn agent only (generated, not in git)
olist-analysis/         cleaning pipeline (src/), notebooks and Excel quality reports
database/               db.sql (schema, load logic, views), build_db.py, ecommerce.db (generated, not in git)
models/                 ensemble.py (shared ensemble logic) and the four agents
trained_models/         churn.pkl, revenue.pkl, netrevenue.pkl, demand.pkl
controller/main.py      central controller
chatbot/                RAG.py, knowledge/ documents, tests
ui/                     Streamlit dashboard: app.py, sections/, components/, charts/
```

## How the weighted ensemble works

Every predictive agent follows the same five steps:

1. Train five algorithms on the training period: random forest, XGBoost, gradient boosting,
   decision tree, and linear (or logistic) regression.
2. Score each one on a validation period it has not seen.
3. Rank them and assign weights 0.8, 0.5, 0.3, 0.2, 0.1 by rank, normalised to sum to 1.
4. The final prediction is the weighted average of the five predictions.
5. Compare the ensemble with each single model on a later test period that played no part
   in choosing the weights.

Results on the test period:

| Agent | Metric | Ensemble | Best single model | Naive baseline |
|---|---|---|---|---|
| Revenue | RMSE (R$ per day) | 6,230 | 6,659 | 9,461 |
| Net revenue | RMSE (R$ per day) | 6,187 | 6,462 | 9,701 |
| Demand | RMSE (units per week) | 24.49 | 24.78 | 24.83 |
| Churn | ROC-AUC | 0.796 | 0.795 | not applicable |

The ensemble is ahead in all four, but only narrowly for demand and churn.

## Database

SQLite, in third normal form. Tables: `customers`, `orders`, `order_items`, `payments`,
`reviews`, `products`, `product_categories`, `sellers`, `geolocation`. Primary keys, foreign
keys and check constraints are enforced.

- `order_summary` is a derived table (one row per order with its totals), rebuilt with the
  database so that dashboard queries stay fast.
- Views `v_sales`, `v_monthly_revenue`, `v_net_revenue_monthly`, `v_category_weekly_demand`,
  `v_product_sales` and `v_customer_summary` use real orders only.
- The agents write to `revenue_forecast`, `demand_forecast`, `churn_predictions` and
  `model_evaluation`.

## Limitations you must know before using the results

- **Churn scores rest on synthetic data.** Only 3.1% of real customers (2,997 of 96,096) ever
  bought twice, too few to train on. 49,009 synthetic repeat orders were added for the churn
  agent, with a return pattern that was designed by hand. The churn scores show that the
  system works; they are not evidence about real customers. Synthetic orders are marked
  `is_synthetic = 1` and are excluded from every revenue, net revenue and demand figure.
- **No discounts, returns or refunds.** The dataset does not record them. Net revenue is
  gross billed minus cancelled and unavailable orders, so true net revenue is lower by an
  unknown amount. There is also no cost data, so profit cannot be calculated.
- **The data ends early.** Orders thin out after 19 August 2018 because the extract stops.
  The agents train on 9 January 2017 to 19 August 2018 and forecast from 20 August 2018.
  "Last month" in the chatbot means July 2018.
- **Demand is forecast by category, not by product.** Only 174 products sold 50 or more
  units. The demand ensemble is about 1% better than assuming next week equals last week.
- **Promotions cannot be forecast.** The history contains a single Black Friday.
- **Product and customer names do not exist.** Only anonymised ids and categories.

## Tests

```bash
python -X utf8 -m unittest ui/test_app.py        # dashboard: every page loads and shows correct totals
python -X utf8 chatbot/test_chatbot.py           # chatbot: 116 checks, including hostile input
```

The chatbot can optionally use Claude for questions its built-in router cannot match: install
`anthropic` and set `ANTHROPIC_API_KEY`. This mode has not been tested. Without a key the
chatbot runs fully offline.

## Data source

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
published on Kaggle under the CC BY-NC-SA 4.0 licence. The raw files in `dataset/original dataset/`
are redistributed unchanged under that licence, for non-commercial use.
