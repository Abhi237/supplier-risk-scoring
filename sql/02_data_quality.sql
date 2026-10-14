-- 02 Data-quality checks: one row per check (check_name, n, total, detail).
CREATE OR REPLACE TABLE dq_report AS
WITH o AS (SELECT * FROM stg_orders), i AS (SELECT * FROM stg_items)
SELECT 'orders_rows' AS check_name, COUNT(*) AS n, COUNT(*) AS total, 'raw orders' AS detail FROM o
UNION ALL SELECT 'order_items_rows', COUNT(*), COUNT(*), 'raw order items' FROM i
UNION ALL SELECT 'sellers_rows', COUNT(*), COUNT(*), 'raw sellers' FROM stg_sellers
UNION ALL SELECT 'duplicate_order_id', COUNT(*) - COUNT(DISTINCT order_id), COUNT(*), 'orders: duplicate keys' FROM o
UNION ALL SELECT 'duplicate_item_key', COUNT(*) - COUNT(DISTINCT order_id || '-' || order_item_id), COUNT(*),
                 'items: duplicate (order_id, order_item_id)' FROM i
UNION ALL SELECT 'items_orphan_order', COUNT(*), (SELECT COUNT(*) FROM i), 'items whose order is missing'
          FROM i WHERE order_id NOT IN (SELECT order_id FROM o)
UNION ALL SELECT 'items_orphan_seller', COUNT(*), (SELECT COUNT(*) FROM i), 'items whose seller is missing'
          FROM i WHERE seller_id NOT IN (SELECT seller_id FROM stg_sellers)
UNION ALL SELECT 'orders_without_items', COUNT(*), (SELECT COUNT(*) FROM o), 'orders with no item rows'
          FROM o WHERE order_id NOT IN (SELECT order_id FROM i)
UNION ALL SELECT 'delivered_missing_delivery_ts', COUNT(*), SUM(CASE WHEN order_status='delivered' THEN 1 END),
                 'status delivered but no customer delivery timestamp'
          FROM o WHERE order_status = 'delivered' AND delivered_ts IS NULL
UNION ALL SELECT 'missing_approved_ts', COUNT(*), (SELECT COUNT(*) FROM o), 'no approval timestamp' FROM o WHERE approved_ts IS NULL
UNION ALL SELECT 'missing_carrier_ts', COUNT(*), (SELECT COUNT(*) FROM o), 'no carrier hand-off timestamp' FROM o WHERE carrier_ts IS NULL
UNION ALL SELECT 'carrier_before_purchase', COUNT(*), (SELECT COUNT(*) FROM o), 'carrier hand-off precedes purchase (bad clock)'
          FROM o WHERE carrier_ts < purchase_ts
UNION ALL SELECT 'delivered_before_carrier', COUNT(*), (SELECT COUNT(*) FROM o), 'customer delivery precedes carrier hand-off'
          FROM o WHERE delivered_ts < carrier_ts
UNION ALL SELECT 'nonpositive_price', COUNT(*), (SELECT COUNT(*) FROM i), 'item price <= 0' FROM i WHERE price <= 0
UNION ALL SELECT 'zero_freight', COUNT(*), (SELECT COUNT(*) FROM i), 'free-freight items (valid, flagged)' FROM i WHERE freight_value = 0
UNION ALL SELECT 'orders_canceled_or_unavailable', COUNT(*), (SELECT COUNT(*) FROM o), 'status canceled / unavailable'
          FROM o WHERE order_status IN ('canceled', 'unavailable')
UNION ALL SELECT 'orders_without_review', COUNT(*), (SELECT COUNT(*) FROM o), 'no review record'
          FROM o WHERE order_id NOT IN (SELECT order_id FROM stg_reviews)
UNION ALL SELECT 'products_missing_category', COUNT(*), (SELECT COUNT(*) FROM stg_products), 'category unknown'
          FROM stg_products WHERE category = 'unknown';
