# src/tools/orders.py
import os
import sqlite3
from typing import Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "orders.db")

def _ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
      order_id TEXT PRIMARY KEY,
      status TEXT,
      eta TEXT,
      items TEXT
    )
    """)
    conn.commit()
    return conn

def seed_example_orders():
    conn = _ensure_db()
    cur = conn.cursor()
    sample = [
        ("12345", "shipped", "2 days", "Wireless Headphones"),
        ("98765", "processing", "4-6 days", "Smartphone Case")
    ]
    for o in sample:
        try:
            cur.execute("INSERT INTO orders (order_id, status, eta, items) VALUES (?, ?, ?, ?)", o)
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()

def track_order(order_id: str) -> str:
    conn = _ensure_db()
    cur = conn.cursor()
    cur.execute("SELECT status, eta, items FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return f"Order {order_id} not found. Please verify your order id."
    status, eta, items = row
    return f"Order {order_id}: status={status}. Expected delivery: {eta}. Items: {items}."
