# Weighted ensemble learning

## How the weighted ensemble works
Every predictive agent (revenue, net revenue, demand, churn) is a weighted ensemble of five machine-learning models. The steps are:
1. train each of the five models on the training period;
2. score each one on a validation period it has not seen;
3. rank the models by their validation score and give rank-based weights 0.8, 0.5, 0.3, 0.2 and 0.1 (best to worst), normalised to sum to 1, which gives about 0.421, 0.263, 0.158, 0.105 and 0.053;
4. final prediction = weighted average of the five predictions (for churn, the weighted average of the five churn probabilities);
5. compare the ensemble with every single model on a later, untouched test period.
Weights come only from validation data, so the test comparison is unbiased. After testing, the models are refit on all data and keep the same weights.

## The five models
The revenue, net-revenue and demand agents use five regressors: random forest, XGBoost, gradient boosting, decision tree and linear regression (ridge). The churn agent uses five classifiers: random forest, XGBoost, gradient boosting, decision tree and logistic regression. Regression models are ranked by validation RMSE (lower is better); churn classifiers are ranked by validation ROC-AUC (higher is better). The weight each model received is stored in the `model_evaluation` table.

## Forecasting method
Revenue and net revenue are forecast per day from lag features (1, 2, 7, 14, 21 and 28 days earlier), trailing means and calendar features. Demand is forecast per category per week from weekly lags, trailing means, calendar features and the category. Forecasts are recursive: each predicted day or week is fed back as history for the next one, so errors can accumulate over long horizons.

## Training window
Models use only the stable window 2017-01-09 to 2018-08-19. The validation period is 2018-04-30 to 2018-06-24 and the test period starts on 2018-06-25 for the revenue, net-revenue and demand agents. Forecasts therefore start on 2018-08-20.
