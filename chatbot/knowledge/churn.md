# Customer churn

## What churn means here
A customer is churned if they place no order in the 180 days after a cutoff date. The churn agent predicts, for every customer, the probability of placing no order in the 180 days after the scoring date 2018-08-20. Features describe each customer using only orders placed before the cutoff, so nothing leaks from the future.

## Risk categories and threshold
Churn probability is turned into a risk category: High when the probability is at least 0.8, Medium when it is at least 0.5, and Low below 0.5. A customer is flagged as predicted to churn (`predicted_churn = 1`) when the probability is at least 0.5. Scores are stored in the table `churn_predictions`, one row per customer who ordered before the scoring date.

## Why synthetic repeat orders are used
Only about 3% of real customers ever came back for a second order, which makes a churn label almost always positive and too unbalanced to learn from. The project therefore adds synthetic follow-up orders (`orders.is_synthetic = 1`) for existing customers. A customer's chance of receiving one depends on their last real order (review score, late delivery, delivery time, order value, basket size, payment type, category) plus random noise; the synthetic order copies a real delivered order from the same state and shifts it to a later date. These coefficients were design choices, so the churn model learns a designed pattern: the scores demonstrate how the system works and are not evidence about real Olist customers. Only the churn agent uses synthetic orders (through `v_orders_all`); revenue, net revenue, demand and every historical answer use real orders only.

## How the churn model was evaluated
Evaluation is out-of-time. The training snapshot uses the cutoff 2017-08-23 (split 75/25 into training and validation data); the test snapshot uses the later cutoff 2018-02-20, whose 180-day label window ends 2018-08-19. The final model is refit on both snapshots and scores every customer at 2018-08-20. Five classifiers are combined by the weighted ensemble, ranked by validation ROC-AUC.

## Main churn drivers
Permutation importance shows that days since the last order is by far the strongest driver (more days means higher risk). Smaller effects come from the review score of the last order (higher score means lower risk), whether the last order was already delivered, the value of the last order and the time between first and last order.

## Recency in churn_predictions
The columns `recency_days`, `n_orders` and `total_paid` in `churn_predictions` describe the customer as the model saw them at 2018-08-20, including synthetic orders. For real purchase history use `v_customer_summary`.
