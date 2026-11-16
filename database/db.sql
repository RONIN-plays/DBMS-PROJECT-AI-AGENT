-- =====================================================================
-- E-commerce Business Intelligence database (SQLite)
-- Built by database/build_db.py, which executes the sections below in order:
--   schema     -> create the normalised tables
--   (python)   -> bulk-load the cleaned + synthetic CSV files into stg_* staging tables
--   transform  -> INSERT ... SELECT from staging (de-duplication, normalisation) and UPDATE
--   views      -> reusable query layer for the four agents, the dashboard and the chatbot
--   validation -> integrity and reconciliation queries (results printed by the build script)
--
-- Design notes
--   * 3NF. The raw Olist "customers" file has one row per ORDER (customer_id is 1:1 with
--     order_id). It is split into customers (one row per person) and delivery columns on orders.
--   * Product categories are moved to their own table (name + English name stored once).
--   * No Returns / Refunds / Discounts tables: the dataset does not contain that information.
--   * orders.is_synthetic = 1 marks generated repeat orders. They exist ONLY for the churn
--     agent. Every revenue, net-revenue and demand view filters them out.
--   * Timestamps are ISO-8601 text ('YYYY-MM-DD HH:MM:SS'), which sorts and compares correctly.
-- =====================================================================

-- @@SECTION: schema
PRAGMA foreign_keys = ON;

CREATE TABLE geolocation (
    zip_code_prefix   TEXT PRIMARY KEY CHECK (length(zip_code_prefix) = 5),
    city              TEXT NOT NULL,
    state             TEXT NOT NULL CHECK (length(state) = 2),
    latitude          REAL,
    longitude         REAL,
    n_points          INTEGER NOT NULL,
    low_confidence    INTEGER NOT NULL CHECK (low_confidence IN (0, 1))
);

CREATE TABLE customers (
    customer_unique_id TEXT PRIMARY KEY,
    zip_code_prefix    TEXT NOT NULL CHECK (length(zip_code_prefix) = 5),   -- most recent delivery address
    city               TEXT NOT NULL,
    state              TEXT NOT NULL CHECK (length(state) = 2)
);

CREATE TABLE sellers (
    seller_id         TEXT PRIMARY KEY,
    zip_code_prefix   TEXT NOT NULL CHECK (length(zip_code_prefix) = 5),
    city              TEXT NOT NULL,
    state             TEXT NOT NULL CHECK (length(state) = 2)
);

CREATE TABLE product_categories (
    category_id            INTEGER PRIMARY KEY,
    category_name          TEXT NOT NULL UNIQUE,
    category_name_english  TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    product_id                  TEXT PRIMARY KEY,
    category_id                 INTEGER NOT NULL REFERENCES product_categories (category_id),
    product_name_length         INTEGER,
    product_description_length  INTEGER,
    product_photos_qty          INTEGER,
    product_weight_g            REAL NOT NULL CHECK (product_weight_g > 0),
    product_length_cm           REAL NOT NULL CHECK (product_length_cm > 0),
    product_height_cm           REAL NOT NULL CHECK (product_height_cm > 0),
    product_width_cm            REAL NOT NULL CHECK (product_width_cm > 0),
    physical_attrs_imputed      INTEGER NOT NULL DEFAULT 0 CHECK (physical_attrs_imputed IN (0, 1))
);

CREATE TABLE orders (
    order_id                       TEXT PRIMARY KEY,
    customer_id                    TEXT NOT NULL UNIQUE,          -- original per-order customer key
    customer_unique_id             TEXT NOT NULL REFERENCES customers (customer_unique_id),
    order_status                   TEXT NOT NULL CHECK (order_status IN
        ('created', 'approved', 'invoiced', 'processing', 'shipped', 'delivered', 'unavailable', 'canceled')),
    order_purchase_timestamp       TEXT NOT NULL,
    order_approved_at              TEXT,
    order_delivered_carrier_date   TEXT,
    order_delivered_customer_date  TEXT,
    order_estimated_delivery_date  TEXT NOT NULL,
    delivery_zip_code_prefix       TEXT NOT NULL CHECK (length(delivery_zip_code_prefix) = 5),
    delivery_state                 TEXT NOT NULL CHECK (length(delivery_state) = 2),
    flag_status_date_conflict      INTEGER NOT NULL DEFAULT 0 CHECK (flag_status_date_conflict IN (0, 1)),
    flag_date_sequence_anomaly     INTEGER NOT NULL DEFAULT 0 CHECK (flag_date_sequence_anomaly IN (0, 1)),
    is_synthetic                   INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    CHECK (order_estimated_delivery_date >= order_purchase_timestamp)
);

CREATE TABLE order_items (
    order_id             TEXT NOT NULL REFERENCES orders (order_id),
    order_item_id        INTEGER NOT NULL CHECK (order_item_id >= 1),
    product_id           TEXT NOT NULL REFERENCES products (product_id),
    seller_id            TEXT NOT NULL REFERENCES sellers (seller_id),
    shipping_limit_date  TEXT NOT NULL,
    price                REAL NOT NULL CHECK (price > 0),
    freight_value        REAL NOT NULL CHECK (freight_value >= 0),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE payments (
    order_id              TEXT NOT NULL REFERENCES orders (order_id),
    payment_sequential    INTEGER NOT NULL CHECK (payment_sequential >= 1),
    payment_type          TEXT NOT NULL CHECK (payment_type IN ('credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined')),
    payment_installments  INTEGER NOT NULL CHECK (payment_installments >= 1),
    payment_value         REAL NOT NULL CHECK (payment_value >= 0),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE reviews (
    review_id                TEXT NOT NULL,
    order_id                 TEXT NOT NULL REFERENCES orders (order_id),
    review_score             INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
    review_comment_title     TEXT,
    review_comment_message   TEXT,
    review_creation_date     TEXT NOT NULL,
    review_answer_timestamp  TEXT NOT NULL,
    is_latest_for_order      INTEGER NOT NULL CHECK (is_latest_for_order IN (0, 1)),
    PRIMARY KEY (review_id, order_id)
);

CREATE INDEX idx_orders_customer   ON orders (customer_unique_id);
CREATE INDEX idx_orders_purchase   ON orders (order_purchase_timestamp);
CREATE INDEX idx_orders_status     ON orders (order_status);
CREATE INDEX idx_items_product     ON order_items (product_id);
CREATE INDEX idx_items_seller      ON order_items (seller_id);
CREATE INDEX idx_products_category ON products (category_id);
CREATE INDEX idx_reviews_order     ON reviews (order_id);


-- @@SECTION: transform
-- Staging tables (stg_*) hold the CSV rows exactly as loaded. Everything below moves them
-- into the normalised tables. Parents are inserted before children so foreign keys hold.

INSERT INTO geolocation (zip_code_prefix, city, state, latitude, longitude, n_points, low_confidence)
SELECT zip_code_prefix, city, state, lat, lng, n_points, flag_low_confidence
FROM stg_geo_zip;

-- Normalisation: each category name is stored once (74 rows instead of 32,951 repeats).
INSERT INTO product_categories (category_name, category_name_english)
SELECT DISTINCT product_category_name, product_category_name_english
FROM stg_products
ORDER BY product_category_name;

INSERT INTO products
SELECT p.product_id, c.category_id, p.product_name_length, p.product_description_length, p.product_photos_qty,
       p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm, p.flag_physical_attrs_imputed
FROM stg_products p
JOIN product_categories c ON c.category_name = p.product_category_name;

INSERT INTO sellers (seller_id, zip_code_prefix, city, state)
SELECT seller_id, seller_zip_code_prefix, seller_city, seller_state
FROM stg_sellers;

-- Duplicate removal: the staging customer file has one row per order (148,450 rows).
-- GROUP BY collapses it to one row per person (96,096 rows).
INSERT INTO customers (customer_unique_id, zip_code_prefix, city, state)
SELECT customer_unique_id, MIN(customer_zip_code_prefix), MIN(customer_city), MIN(customer_state)
FROM stg_customers
GROUP BY customer_unique_id;

INSERT INTO orders
SELECT o.order_id, o.customer_id, c.customer_unique_id, o.order_status,
       o.order_purchase_timestamp, o.order_approved_at, o.order_delivered_carrier_date,
       o.order_delivered_customer_date, o.order_estimated_delivery_date,
       c.customer_zip_code_prefix, c.customer_state,
       o.flag_status_date_conflict, o.flag_date_sequence_anomaly, o.is_synthetic
FROM stg_orders o
JOIN stg_customers c ON c.customer_id = o.customer_id;

-- UPDATE: customers who ordered to more than one address get the address of their latest REAL order
-- (the GROUP BY above could only pick an arbitrary one).
UPDATE customers
SET (zip_code_prefix, city, state) = (
        SELECT s.customer_zip_code_prefix, s.customer_city, s.customer_state
        FROM stg_customers s
        JOIN stg_orders o ON o.customer_id = s.customer_id
        WHERE s.customer_unique_id = customers.customer_unique_id AND o.is_synthetic = 0
        ORDER BY o.order_purchase_timestamp DESC
        LIMIT 1)
WHERE customer_unique_id IN (
        SELECT customer_unique_id FROM stg_customers
        GROUP BY customer_unique_id
        HAVING COUNT(DISTINCT customer_zip_code_prefix) > 1);

INSERT INTO order_items
SELECT order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value
FROM stg_order_items;

INSERT INTO payments
SELECT order_id, payment_sequential, payment_type, payment_installments, payment_value
FROM stg_payments;

INSERT INTO reviews
SELECT review_id, order_id, review_score, review_comment_title, review_comment_message,
       review_creation_date, review_answer_timestamp, is_latest_review_for_order
FROM stg_reviews;


-- @@SECTION: views
-- Order-grain aggregates. Items and payments are both many-per-order, so they are summed
-- separately BEFORE being joined to orders; joining them to each other would inflate revenue.
CREATE VIEW v_order_items_agg AS
SELECT order_id,
       COUNT(*)                  AS n_items,
       COUNT(DISTINCT seller_id) AS n_sellers,
       SUM(price)                AS items_value,
       SUM(freight_value)        AS freight_value
FROM order_items
GROUP BY order_id;

CREATE VIEW v_order_payments_agg AS
SELECT order_id,
       SUM(payment_value)                                                      AS payment_total,
       SUM(CASE WHEN payment_type = 'voucher' THEN payment_value ELSE 0 END)   AS voucher_value,
       COUNT(*)                                                                AS n_payments,
       MAX(payment_installments)                                               AS max_installments
FROM payments
GROUP BY order_id;

-- One row per order, real and synthetic, with its item, payment and review totals.
-- Stored as a TABLE (not a view) because every dashboard query starts from it: computing the
-- aggregates once at build time makes those queries about 10x faster. It is derived data and
-- is rebuilt whenever the database is rebuilt.
CREATE TABLE order_summary AS
SELECT o.*,
       substr(o.order_purchase_timestamp, 1, 7)       AS purchase_month,
       date(o.order_purchase_timestamp)                AS purchase_date,
       COALESCE(i.n_items, 0)                          AS n_items,
       i.items_value, i.freight_value,
       i.items_value + i.freight_value                 AS order_value,
       p.payment_total, p.voucher_value, p.max_installments,
       r.review_score
FROM orders o
LEFT JOIN v_order_items_agg    i ON i.order_id = o.order_id
LEFT JOIN v_order_payments_agg p ON p.order_id = o.order_id
LEFT JOIN reviews              r ON r.order_id = o.order_id AND r.is_latest_for_order = 1;

CREATE UNIQUE INDEX idx_summary_order ON order_summary (order_id);
CREATE INDEX idx_summary_customer     ON order_summary (customer_unique_id);
CREATE INDEX idx_summary_date         ON order_summary (is_synthetic, purchase_date);

-- Real + synthetic orders (the churn agent reads this one).
CREATE VIEW v_orders_all AS
SELECT * FROM order_summary;

-- REAL orders only: the source for revenue, net revenue, demand, the dashboard and the chatbot.
CREATE VIEW v_sales AS
SELECT * FROM v_orders_all WHERE is_synthetic = 0;

CREATE VIEW v_monthly_revenue AS
SELECT purchase_month,
       COUNT(*)                                                  AS orders,
       SUM(order_status = 'delivered')                           AS delivered_orders,
       COUNT(DISTINCT customer_unique_id)                        AS customers,
       ROUND(SUM(items_value), 2)                                AS items_value,
       ROUND(SUM(freight_value), 2)                              AS freight_value,
       ROUND(SUM(payment_total), 2)                              AS payment_total,
       ROUND(SUM(payment_total) / COUNT(payment_total), 2)       AS avg_order_value
FROM v_sales
GROUP BY purchase_month;

-- Net revenue components per month, on the amount billed to customers (payment_total).
-- net_revenue = billed - cancelled - unavailable. Pending = orders still in flight.
CREATE VIEW v_net_revenue_monthly AS
SELECT purchase_month,
       ROUND(SUM(payment_total), 2)                                                                  AS gross_billed,
       ROUND(SUM(CASE WHEN order_status = 'canceled'    THEN payment_total ELSE 0 END), 2)           AS cancelled_value,
       ROUND(SUM(CASE WHEN order_status = 'unavailable' THEN payment_total ELSE 0 END), 2)           AS unavailable_value,
       ROUND(SUM(CASE WHEN order_status IN ('created', 'approved', 'invoiced', 'processing', 'shipped')
                      THEN payment_total ELSE 0 END), 2)                                             AS pending_value,
       ROUND(SUM(CASE WHEN order_status = 'delivered'   THEN payment_total ELSE 0 END), 2)           AS delivered_value,
       ROUND(SUM(CASE WHEN order_status NOT IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS net_revenue,
       ROUND(SUM(CASE WHEN order_status = 'delivered'   THEN freight_value ELSE 0 END), 2)           AS freight_delivered,
       ROUND(SUM(voucher_value), 2)                                                                  AS voucher_value
FROM v_sales
GROUP BY purchase_month;

-- Units and item value per category per week (week starts Monday). Cancelled/unavailable excluded.
CREATE VIEW v_category_weekly_demand AS
SELECT c.category_name_english                                        AS category,
       date(o.order_purchase_timestamp, 'weekday 0', '-6 days')       AS week_start,
       COUNT(*)                                                       AS units,
       ROUND(SUM(i.price), 2)                                         AS items_value
FROM order_items i
JOIN orders o             ON o.order_id = i.order_id
JOIN products p           ON p.product_id = i.product_id
JOIN product_categories c ON c.category_id = p.category_id
WHERE o.is_synthetic = 0 AND o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY c.category_name_english, week_start;

CREATE VIEW v_product_sales AS
SELECT p.product_id, c.category_name_english AS category,
       COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value, ROUND(AVG(i.price), 2) AS avg_price,
       MIN(date(o.order_purchase_timestamp)) AS first_sale, MAX(date(o.order_purchase_timestamp)) AS last_sale
FROM order_items i
JOIN orders o             ON o.order_id = i.order_id
JOIN products p           ON p.product_id = i.product_id
JOIN product_categories c ON c.category_id = p.category_id
WHERE o.is_synthetic = 0 AND o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY p.product_id;

-- Customer history from REAL orders (dashboard, chatbot).
CREATE VIEW v_customer_summary AS
SELECT c.customer_unique_id, c.city, c.state,
       COUNT(*)                              AS n_orders,
       MIN(s.order_purchase_timestamp)       AS first_purchase,
       MAX(s.order_purchase_timestamp)       AS last_purchase,
       ROUND(SUM(s.payment_total), 2)        AS total_paid,
       ROUND(AVG(s.review_score), 2)         AS avg_review_score,
       COUNT(*) > 1                          AS is_repeat_customer
FROM customers c
JOIN v_sales s ON s.customer_unique_id = c.customer_unique_id
GROUP BY c.customer_unique_id;


-- @@SECTION: validation
-- row counts
SELECT 'customers' AS table_name, COUNT(*) AS n FROM customers
UNION ALL SELECT 'orders (real)', COUNT(*) FROM orders WHERE is_synthetic = 0
UNION ALL SELECT 'orders (synthetic)', COUNT(*) FROM orders WHERE is_synthetic = 1
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'payments', COUNT(*) FROM payments
UNION ALL SELECT 'reviews', COUNT(*) FROM reviews
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'product_categories', COUNT(*) FROM product_categories
UNION ALL SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL SELECT 'geolocation', COUNT(*) FROM geolocation;

-- foreign-key violations (expected: no rows)
PRAGMA foreign_key_check;

-- orphans and duplicates (expected: all zero)
SELECT (SELECT COUNT(*) FROM order_items i LEFT JOIN orders o ON o.order_id = i.order_id WHERE o.order_id IS NULL)          AS items_without_order,
       (SELECT COUNT(*) FROM payments p LEFT JOIN orders o ON o.order_id = p.order_id WHERE o.order_id IS NULL)             AS payments_without_order,
       (SELECT COUNT(*) FROM reviews r LEFT JOIN orders o ON o.order_id = r.order_id WHERE o.order_id IS NULL)              AS reviews_without_order,
       (SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON c.customer_unique_id = o.customer_unique_id
         WHERE c.customer_unique_id IS NULL)                                                                               AS orders_without_customer,
       (SELECT COUNT(*) FROM (SELECT order_id FROM reviews WHERE is_latest_for_order = 1 GROUP BY order_id HAVING COUNT(*) > 1)) AS orders_with_two_latest_reviews,
       (SELECT COUNT(*) FROM v_orders_all) - (SELECT COUNT(*) FROM orders)                                                 AS rows_multiplied_by_view;

-- revenue reconciliation on REAL data: item grain vs payment grain vs the order-level view
SELECT ROUND((SELECT SUM(i.price) FROM order_items i JOIN orders o ON o.order_id = i.order_id WHERE o.is_synthetic = 0), 2)          AS items_value_item_grain,
       ROUND((SELECT SUM(items_value) FROM v_sales), 2)                                                                              AS items_value_view,
       ROUND((SELECT SUM(p.payment_value) FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE o.is_synthetic = 0), 2)     AS payments_payment_grain,
       ROUND((SELECT SUM(payment_total) FROM v_sales), 2)                                                                            AS payments_view;

-- order status mix (real orders)
SELECT order_status, COUNT(*) AS orders, ROUND(SUM(payment_total), 2) AS billed
FROM v_sales GROUP BY order_status ORDER BY orders DESC;

-- net revenue, whole period (real orders)
SELECT ROUND(SUM(gross_billed), 2) AS gross_billed, ROUND(SUM(cancelled_value), 2) AS cancelled,
       ROUND(SUM(unavailable_value), 2) AS unavailable, ROUND(SUM(net_revenue), 2) AS net_revenue,
       ROUND(SUM(delivered_value), 2) AS delivered, ROUND(SUM(voucher_value), 2) AS vouchers
FROM v_net_revenue_monthly;

-- top 5 categories by units (real orders)
SELECT category, SUM(units) AS units, ROUND(SUM(items_value), 2) AS items_value
FROM v_category_weekly_demand GROUP BY category ORDER BY units DESC LIMIT 5;

-- repeat customers: real data vs real + synthetic
SELECT (SELECT COUNT(*) FROM v_customer_summary WHERE is_repeat_customer)                                             AS repeat_customers_real,
       (SELECT COUNT(*) FROM (SELECT customer_unique_id FROM orders GROUP BY customer_unique_id HAVING COUNT(*) > 1)) AS repeat_customers_with_synthetic,
       (SELECT COUNT(*) FROM customers)                                                                               AS customers;


-- @@SECTION: predictions
-- Output tables written by the agents (models/*.py). Historical facts live in the tables above;
-- everything here is a MODEL OUTPUT and must be presented as a prediction, not a record.
CREATE TABLE IF NOT EXISTS revenue_forecast (
    target           TEXT NOT NULL CHECK (target IN ('gross', 'net')),
    forecast_date    TEXT NOT NULL,
    predicted_value  REAL NOT NULL,
    PRIMARY KEY (target, forecast_date)
);

CREATE TABLE IF NOT EXISTS demand_forecast (
    category         TEXT NOT NULL,
    week_start       TEXT NOT NULL,
    predicted_units  REAL NOT NULL,
    PRIMARY KEY (category, week_start)
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    customer_unique_id  TEXT PRIMARY KEY REFERENCES customers (customer_unique_id),
    snapshot_date       TEXT NOT NULL,
    churn_probability   REAL NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    predicted_churn     INTEGER NOT NULL CHECK (predicted_churn IN (0, 1)),
    risk_category       TEXT NOT NULL CHECK (risk_category IN ('Low', 'Medium', 'High')),
    -- customer history AS THE MODEL SAW IT (real + synthetic orders before the snapshot date)
    recency_days        INTEGER NOT NULL,
    n_orders            INTEGER NOT NULL,
    total_paid          REAL
);

-- One row per agent / model / data split / metric. model = 'ensemble' is the weighted ensemble.
CREATE TABLE IF NOT EXISTS model_evaluation (
    agent   TEXT NOT NULL,
    model   TEXT NOT NULL,
    weight  REAL,
    split   TEXT NOT NULL,
    metric  TEXT NOT NULL,
    value   REAL,
    PRIMARY KEY (agent, model, split, metric)
);
