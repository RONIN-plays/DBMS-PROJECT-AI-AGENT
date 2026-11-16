# Data coverage and limitations

## Time coverage limitation
The data is the Olist Brazilian e-commerce extract. Orders run from 2016-09-04 to 2018-10-17, but the data is complete only from 2017-01-09 to 2018-08-19: 2016 has very few orders, and after 2018-08-19 the number of orders per day falls sharply because the extract was cut. Nothing happens "today". "Last month" means the last complete calendar month, July 2018; August 2018 is complete only up to the 19th. Forecasts ("next month", "next weeks") start on 2018-08-20.

## Missing information (dataset limitations)
The dataset has no discounts, no returns and no refunds, no product costs, so no profit or margin, no product names (only product ids and categories), no customer names or contact details, no stock or inventory levels, and no marketing or website data. Such questions are answered as "not available" rather than with zero or a guess.

## Revenue definitions
Revenue (gross billed) is the sum of payments (`payment_total`) for real orders, including freight, whatever their status. Items value is the sum of item prices without freight. Category and product sales use item prices of orders that were not cancelled or unavailable. Net revenue deducts cancelled and unavailable orders. All amounts are in Brazilian real (R$).

## Real and synthetic orders
The database also contains synthetic repeat orders (`is_synthetic = 1`) created only for the churn model. Historical facts (revenue, customers, returning customers, products) use real orders only; in reality 2,997 of 96,096 customers placed more than one order. Churn scores come from a model trained on real plus synthetic orders.

## Other caveats
Product categories have Portuguese and English names; one category is "unknown" for products without a category. A customer's city and state are those of their latest real order. Some orders have no payment record or no review. Promotional spikes such as Black Friday occurred only once in the history.
