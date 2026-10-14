"""Shared fixtures: a deterministic synthetic Olist-shaped extract + handcrafted edge cases, run through SQL."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from supplier_risk.config import load_config
from supplier_risk.db import run_sql_model

FIX = Path(__file__).parent / "fixtures"
STATES = ["SP", "RJ", "MG", "PR", "BA", "CE", "AM"]


def _common_tables(order_ids, customer_ids, seller_ids, rng):
    customers = pd.DataFrame({"customer_id": customer_ids, "customer_unique_id": customer_ids,
                              "customer_zip_code_prefix": 1000, "customer_city": "x",
                              "customer_state": rng.choice(STATES, len(customer_ids))})
    sellers = pd.DataFrame({"seller_id": seller_ids, "seller_zip_code_prefix": 1001, "seller_city": "y",
                            "seller_state": rng.choice(STATES[:4], len(seller_ids))})
    products = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "product_category_name": ["cama", "beleza", None],
                             "product_name_lenght": 10, "product_description_lenght": 10, "product_photos_qty": 1,
                             "product_weight_g": [500, 300, 1000], "product_length_cm": 10, "product_height_cm": 10,
                             "product_width_cm": 10})
    cat = pd.DataFrame({"product_category_name": ["cama", "beleza"],
                        "product_category_name_english": ["bed_bath", "health_beauty"]})
    return customers, sellers, products, cat


def synthetic_sources(n_sellers: int = 120, n_orders: int = 6000, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Sellers with heterogeneous, persistent hand-off reliability over 2017-01 .. 2018-08."""
    rng = np.random.default_rng(seed)
    seller_ids = [f"s{i:04d}" for i in range(n_sellers)]
    p_ho = rng.beta(1, 8, n_sellers)                  # seller hand-off late propensity (persistent trait)
    weight = rng.pareto(1.2, n_sellers) + 0.05        # skewed volume
    start, span = pd.Timestamp("2017-01-01"), 600
    rows_o, rows_i, rows_r = [], [], []
    for j in range(n_orders):
        oid, cid = f"o{j:06d}", f"c{j:06d}"
        s = rng.choice(n_sellers, p=weight / weight.sum())
        t = start + pd.Timedelta(days=int(rng.integers(0, span)), hours=int(rng.integers(0, 24)))
        limit = t + pd.Timedelta(days=4)
        ho_late = rng.random() < p_ho[s]
        carrier = limit + pd.Timedelta(days=float(rng.uniform(0.2, 6))) if ho_late else \
            t + pd.Timedelta(days=float(rng.uniform(0.5, 3.8)))
        transit = float(rng.gamma(3, 2.5))
        delivered = carrier + pd.Timedelta(days=transit)
        est = t + pd.Timedelta(days=int(rng.integers(12, 25)))
        status = "canceled" if rng.random() < 0.01 else "delivered"
        if status == "canceled":
            carrier = delivered = pd.NaT
        rows_o.append((oid, cid, status, t, t + pd.Timedelta(hours=1), carrier, delivered, est))
        for k in range(1, 1 + (2 if rng.random() < 0.1 else 1)):
            rows_i.append((oid, k, str(rng.choice(["P1", "P2", "P3"])), seller_ids[s], limit,
                           round(float(rng.lognormal(4, 0.7)), 2), round(float(rng.uniform(5, 30)), 2)))
        if status == "delivered":
            late = delivered.normalize() > est.normalize()
            score = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])) if late else int(rng.choice([3, 4, 5],
                                                                                             p=[0.1, 0.3, 0.6]))
            rows_r.append((f"r{j}", oid, score, delivered + pd.Timedelta(days=1)))
    orders = pd.DataFrame(rows_o, columns=["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                                           "order_approved_at", "order_delivered_carrier_date",
                                           "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.DataFrame(rows_i, columns=["order_id", "order_item_id", "product_id", "seller_id",
                                          "shipping_limit_date", "price", "freight_value"])
    reviews = pd.DataFrame(rows_r, columns=["review_id", "order_id", "review_score", "review_creation_date"])
    customers, sellers, products, cat = _common_tables(orders.order_id, orders.customer_id, seller_ids, rng)
    return {"raw_orders": orders, "raw_items": items, "raw_sellers": sellers, "raw_customers": customers,
            "raw_reviews": reviews, "raw_products": products, "raw_category_translation": cat}


def edge_sources() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    orders = pd.read_csv(FIX / "edge_orders.csv").drop(columns="case")
    for c in orders.columns[3:]:
        orders[c] = pd.to_datetime(orders[c])
    items = pd.read_csv(FIX / "edge_items.csv", parse_dates=["shipping_limit_date"])
    customers, _, products, cat = _common_tables(orders.order_id, orders.customer_id, ["S1", "S2"], rng)
    customers["customer_state"] = "SP"
    sellers = pd.DataFrame({"seller_id": ["S1", "S2"], "seller_zip_code_prefix": [123, 45678],
                            "seller_city": ["a", "b"], "seller_state": ["sp", "RJ"]})
    # E1 has two reviews: the later one (score 2) must win
    reviews = pd.DataFrame({"review_id": ["a", "b", "c"], "order_id": ["E1", "E1", "E2"], "review_score": [5, 2, 1],
                            "review_creation_date": pd.to_datetime(["2017-03-09", "2017-03-15", "2017-03-26"])})
    return {"raw_orders": orders, "raw_items": items, "raw_sellers": sellers, "raw_customers": customers,
            "raw_reviews": reviews, "raw_products": products, "raw_category_translation": cat}


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sources():
    return synthetic_sources()


@pytest.fixture(scope="session")
def con(sources, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, sources=sources, cfg=cfg, verbose=False)
    yield c
    c.close()


@pytest.fixture(scope="session")
def edge_con(cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, sources=edge_sources(), cfg=cfg, verbose=False)
    yield c
    c.close()
