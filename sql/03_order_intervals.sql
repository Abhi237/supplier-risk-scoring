-- 03 Item-grain fact with order-to-delivery intervals.  Grain: one row per (order_id, order_item_id).
-- Seller-controlled leg  = purchase -> carrier hand-off, judged against shipping_limit_date.
-- End-to-end leg         = purchase -> customer delivery, judged against estimated_delivery_date.
-- NB: Olist records ONE carrier/delivery timestamp per order; in multi-seller orders (~1%) every seller
-- inherits the order's timestamps (flagged as is_multi_seller_order).
CREATE OR REPLACE TABLE fct_items AS
WITH sellers_per_order AS (
  SELECT order_id, COUNT(DISTINCT seller_id) AS n_sellers FROM stg_items GROUP BY order_id
)
SELECT i.order_id, i.order_item_id, i.seller_id, i.product_id,
       s.seller_state, c.customer_state, p.category, p.weight_g,
       o.order_status, o.purchase_ts, o.approved_ts, o.carrier_ts, o.delivered_ts, o.estimated_ts,
       i.shipping_limit_ts,
       CAST(DATE_TRUNC('month', o.purchase_ts) AS DATE)                            AS purchase_month,
       i.price, i.freight_value,   -- DECIMAL(12,2): exact, order-independent sums downstream
       CAST(i.freight_value AS DOUBLE) / NULLIF(CAST(i.price AS DOUBLE), 0)         AS freight_to_price,
       (s.seller_state = c.customer_state)                                           AS same_state,
       (spo.n_sellers > 1)                                                           AS is_multi_seller_order,
       DATE_DIFF('second', o.purchase_ts, o.approved_ts) / 3600.0                    AS approval_hours,
       DATE_DIFF('second', o.purchase_ts, o.carrier_ts) / 86400.0                    AS handoff_days,
       DATE_DIFF('second', i.shipping_limit_ts, o.carrier_ts) / 86400.0              AS handoff_delay_days,
       DATE_DIFF('second', o.carrier_ts, o.delivered_ts) / 86400.0                   AS carrier_transit_days,
       DATE_DIFF('second', o.purchase_ts, o.delivered_ts) / 86400.0                  AS delivery_days,
       DATE_DIFF('day', CAST(o.estimated_ts AS DATE), CAST(o.delivered_ts AS DATE))  AS delay_days,
       (o.order_status IN ('canceled', 'unavailable'))                               AS is_canceled,
       (o.order_status = 'delivered' AND o.delivered_ts IS NOT NULL)                  AS is_delivered,
       CASE WHEN o.order_status = 'delivered' AND o.delivered_ts IS NOT NULL
            THEN CAST(o.delivered_ts AS DATE) > CAST(o.estimated_ts AS DATE) END     AS is_late,
       CASE WHEN o.carrier_ts IS NOT NULL AND o.carrier_ts >= o.purchase_ts
            THEN o.carrier_ts > i.shipping_limit_ts END                              AS is_handoff_late,
       r.review_score, r.review_created_ts
FROM stg_items i
JOIN stg_orders o           ON o.order_id = i.order_id
JOIN sellers_per_order spo  ON spo.order_id = i.order_id
LEFT JOIN stg_sellers s     ON s.seller_id = i.seller_id
LEFT JOIN stg_customers c   ON c.customer_id = o.customer_id
LEFT JOIN stg_products p    ON p.product_id = i.product_id
LEFT JOIN stg_reviews r     ON r.order_id = i.order_id;

-- order-grain (delivered orders) for lead-time distributions in R
CREATE OR REPLACE TABLE fct_orders_delivered AS
SELECT order_id, MIN(seller_id) AS seller_id, MIN(seller_state) AS seller_state, MIN(customer_state) AS customer_state,
       MIN(category) AS category, MIN(purchase_month) AS purchase_month, BOOL_OR(same_state) AS same_state,
       CAST(SUM(price) AS DOUBLE) AS order_gmv, CAST(SUM(freight_value) AS DOUBLE) AS order_freight, COUNT(*) AS n_items,
       CAST(SUM(freight_value) / NULLIF(SUM(price), 0) AS DOUBLE) AS freight_to_price,
       MIN(approval_hours) AS approval_hours, MIN(handoff_days) AS handoff_days, MAX(handoff_delay_days) AS handoff_delay_days,
       MIN(carrier_transit_days) AS carrier_transit_days, MIN(delivery_days) AS delivery_days,
       MIN(delay_days) AS delay_days, BOOL_OR(is_late) AS is_late, BOOL_OR(is_handoff_late) AS is_handoff_late,
       MIN(review_score) AS review_score, BOOL_OR(is_multi_seller_order) AS is_multi_seller_order
FROM fct_items
WHERE is_delivered
GROUP BY order_id;
