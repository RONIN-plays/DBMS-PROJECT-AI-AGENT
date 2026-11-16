# What each agent predicts and how accurate it is

## Revenue agent
Predicts daily gross revenue: everything billed to customers (sum of payments) for real orders placed that day, for 1 to 90 days after 2018-08-19. On the test period (2018-06-25 to 2018-08-19, one day ahead) the ensemble had a MAPE of 16.4% and a WAPE of 12.9% (R2 0.687), better than the naive same-weekday-last-week forecast (MAPE 21.3%). Forecasting the whole 56-day test period without seeing it missed the total by +4.7%. History contains only one Black Friday (24 Nov 2017), so promotional spikes cannot be forecast.

## Net revenue agent
Historical net revenue is an exact SQL calculation. The forecast predicts daily net revenue (billed minus cancelled and unavailable orders). On the test period the ensemble had a MAPE of 16.9% and a WAPE of 13.5% (R2 0.674), against 22.7% MAPE for the naive same-weekday-last-week forecast. The 56-day recursive test missed the total by +5.1%.

## Demand agent
Predicts weekly units sold per product category for the 20 largest named categories (about 87% of all units), 1 to 12 weeks ahead. Individual products and smaller categories are too sparse to forecast. On the 8-week test period the ensemble had a WAPE of 23.3% and a MAPE of 28.5% (R2 0.817), slightly better than the naive last-week forecast (WAPE 24.2%). Forecasting the 8 test weeks recursively missed total units by -9.5%. A category is called increasing or decreasing when its forecast weekly average differs by more than 10% from the average of the last 4 weeks.

## Churn agent
Predicts the probability that a customer places no order in the next 180 days, scored at 2018-08-20. On the later test snapshot (cutoff 2018-02-20) the ensemble reached ROC-AUC 0.796, precision 0.843, recall 0.967, F1 0.901 and accuracy 0.826. The churn rate in that test snapshot was 0.818, so accuracy should be read against that base rate; ROC-AUC is the more informative measure. The model is trained on real plus synthetic repeat orders, so these figures describe performance on the designed synthetic pattern.

## Metric glossary
MAPE is the mean absolute percentage error per day or week. WAPE is the total absolute error divided by the total actual value. R2 is the share of variance explained (1 is perfect). RMSE is the root mean squared error. ROC-AUC measures how well churners are ranked above non-churners (0.5 is random, 1 is perfect). Precision is the share of predicted churners who did churn; recall is the share of actual churners who were flagged.
