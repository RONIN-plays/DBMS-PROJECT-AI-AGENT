# Net revenue

## How net revenue is calculated
Net revenue is calculated exactly with SQL; no model is involved. It is measured on the amount billed to customers (the sum of payments, `payment_total`) for real orders:

net revenue = gross billed - cancelled orders - unavailable orders

Gross billed is everything billed for orders placed in the period, whatever happened to them later. Orders with status `canceled` are deducted, and so are orders with status `unavailable` (the product was out of stock). The monthly figures are in the view `v_net_revenue_monthly`.

## Deductions that are not available
The dataset contains no discounts, no returns and no refunds. These deductions are reported as "not available in the dataset" instead of being estimated, so the true net revenue is lower than the reported figure by an unknown amount. Costs, profit and margins cannot be calculated either, because no cost data exists.

## Vouchers, freight and pending orders
Vouchers are not subtracted: a voucher is a way of paying, so voucher payments are part of the amount billed and are shown only as an information line. Freight is billed to the customer and is included in the billed amount. Orders that were still in progress at the end of the data (created, approved, invoiced, processing or shipped) are counted in net revenue but could still be cancelled; they are shown separately as pending value. Delivered value is the part of net revenue whose orders were delivered.

## Revenue leakage
Revenue leakage (the amount lost) is the value of cancelled plus unavailable orders, often expressed as a percentage of the amount billed. The net-revenue agent also breaks leakage down by delivery state and by payment pattern (paid partly by voucher, instalments, single payment).

## Net revenue forecast
Only the forecast of future net revenue uses machine learning: the same weighted-ensemble pipeline as the revenue agent, trained on the daily net revenue series (real orders excluding cancelled and unavailable ones). A forecast is a model estimate, not a recorded figure.
