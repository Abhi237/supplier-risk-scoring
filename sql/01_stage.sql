-- 01 Staging: typed, renamed copies of the raw Olist tables.
-- Inputs (views registered by supplier_risk.db): raw_orders, raw_items, raw_sellers, raw_customers,
-- raw_reviews, raw_products, raw_category_translation.  Customer data is reduced to state only.
CREATE OR REPLACE TABLE stg_orders AS
SELECT CAST(order_id AS VARCHAR)                              AS order_id,
       CAST(customer_id AS VARCHAR)                           AS customer_id,
       CAST(order_status AS VARCHAR)                          AS order_status,
       CAST(order_purchase_timestamp AS TIMESTAMP)            AS purchase_ts,
       CAST(order_approved_at AS TIMESTAMP)                   AS approved_ts,
       CAST(order_delivered_carrier_date AS TIMESTAMP)        AS carrier_ts,
       CAST(order_delivered_customer_date AS TIMESTAMP)       AS delivered_ts,
       CAST(order_estimated_delivery_date AS TIMESTAMP)       AS estimated_ts
FROM raw_orders;

CREATE OR REPLACE TABLE stg_items AS
SELECT CAST(order_id AS VARCHAR)                  AS order_id,
       CAST(order_item_id AS INTEGER)             AS order_item_id,
       CAST(product_id AS VARCHAR)                AS product_id,
       CAST(seller_id AS VARCHAR)                 AS seller_id,
       CAST(shipping_limit_date AS TIMESTAMP)     AS shipping_limit_ts,
       CAST(price AS DECIMAL(12, 2))              AS price,
       CAST(freight_value AS DECIMAL(12, 2))      AS freight_value
FROM raw_items;

CREATE OR REPLACE TABLE stg_sellers AS
SELECT CAST(seller_id AS VARCHAR)                         AS seller_id,
       LPAD(CAST(seller_zip_code_prefix AS VARCHAR), 5, '0') AS seller_zip_prefix,
       CAST(seller_city AS VARCHAR)                       AS seller_city,
       UPPER(CAST(seller_state AS VARCHAR))               AS seller_state
FROM raw_sellers;

-- customers: only the geography needed for lane analysis (no personal attributes kept downstream)
CREATE OR REPLACE TABLE stg_customers AS
SELECT CAST(customer_id AS VARCHAR) AS customer_id, UPPER(CAST(customer_state AS VARCHAR)) AS customer_state
FROM raw_customers;

CREATE OR REPLACE TABLE stg_products AS
SELECT CAST(p.product_id AS VARCHAR)                                        AS product_id,
       COALESCE(CAST(t.product_category_name_english AS VARCHAR),
                CAST(p.product_category_name AS VARCHAR), 'unknown')        AS category,
       CAST(p.product_weight_g AS DOUBLE)                                   AS weight_g,
       CAST(p.product_length_cm AS DOUBLE) * CAST(p.product_height_cm AS DOUBLE)
         * CAST(p.product_width_cm AS DOUBLE)                               AS volume_cm3
FROM raw_products p
LEFT JOIN raw_category_translation t ON p.product_category_name = t.product_category_name;

-- reviews: an order can have >1 review row; keep the latest per order (ROW_NUMBER window)
CREATE OR REPLACE TABLE stg_reviews AS
SELECT order_id, review_score, review_created_ts FROM (
  SELECT CAST(order_id AS VARCHAR) AS order_id,
         CAST(review_score AS INTEGER) AS review_score,
         CAST(review_creation_date AS TIMESTAMP) AS review_created_ts,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_creation_date DESC, review_id) AS rn
  FROM raw_reviews) r
WHERE rn = 1;
