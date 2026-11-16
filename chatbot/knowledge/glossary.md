# Glossary of tables and views

## Core tables
`customers` has one row per person (`customer_unique_id`) with the city and state of their latest real order. `orders` has one row per order with status, timestamps, delivery state and the flag `is_synthetic`. `order_items` holds one row per item (product, seller, price, freight). `payments` holds one or more payments per order (type, instalments, value). `reviews` holds review scores (1 to 5). `products` and `product_categories` describe products by id and category only. `sellers` and `geolocation` hold seller and zip-code locations.

## Sales views (real orders only)
`v_sales` has one row per real order with order_status, purchase_date, purchase_month, customer_unique_id, delivery_state, n_items, items_value, freight_value, order_value, payment_total, voucher_value, max_installments and review_score. `v_monthly_revenue` gives orders, customers, items value, freight, payment_total (revenue) and average order value per month. `v_net_revenue_monthly` gives gross billed, cancelled, unavailable, pending, delivered and net revenue per month.

## Product and customer views
`v_category_weekly_demand` gives units and item value per category per week (weeks start on Monday; cancelled and unavailable orders excluded). `v_product_sales` gives units, item value, average price and first and last sale date per product. `v_customer_summary` gives each customer's real order count, first and last purchase, total paid, average review and whether they are a repeat customer.

## Synthetic data view
`v_orders_all` contains real and synthetic orders and is read only by the churn agent. It must not be used for historical facts.

## Prediction tables
These hold model outputs, not records: `revenue_forecast` (daily predicted gross or net revenue from 2018-08-20), `demand_forecast` (weekly predicted units for the 20 largest categories), `churn_predictions` (churn probability, risk category and predicted churn per customer at 2018-08-20) and `model_evaluation` (validation and test metrics and ensemble weights for every model).
