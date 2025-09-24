from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timedelta
from core.evaluator import EvalPolicy, evaluate_bot

def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY,
      ts TEXT, run_id TEXT, bot TEXT, symbol TEXT, timeframe TEXT,
      event_type TEXT, side TEXT, qty REAL, price REAL, reason TEXT
    );
    CREATE TABLE IF NOT EXISTS equity (
      id INTEGER PRIMARY KEY,
      ts TEXT, run_id TEXT, bot TEXT, equity REAL, position_qty REAL, position_avg_price REAL
    );
    """)
    conn.commit()

def test_basic_metrics_tmp(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db.as_posix()) as conn:
        _init_db(conn)
        bot = "TestBot"
        start = datetime.utcnow() - timedelta(days=10)
        eq = 1000.0
        # 10 dagen equity, met lichte stijging en twee 'fills'
        for i in range(10):
            ts = (start + timedelta(days=i)).isoformat()
            eq *= 1.01 if i % 2 == 0 else 1.0
            conn.execute("INSERT INTO equity (ts, run_id, bot, equity, position_qty, position_avg_price) VALUES (?,?,?,?,?,?)",
                         (ts, "", bot, eq, 0.0, 0.0))
        # twee fills
        conn.execute("INSERT INTO events (ts, run_id, bot, symbol, timeframe, event_type, side, qty, price, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ((start+timedelta(days=2)).isoformat(),"",bot,"BTC/USDT","1h","fill","buy",0.01,25000.0,""))
        conn.execute("INSERT INTO events (ts, run_id, bot, symbol, timeframe, event_type, side, qty, price, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ((start+timedelta(days=5)).isoformat(),"",bot,"BTC/USDT","1h","fill","sell",0.01,26000.0,""))
        conn.commit()

        policy = EvalPolicy(lookback_days=14, min_trades=2, promote_return_pct=1.0)
        res = evaluate_bot(conn, bot, policy)
        assert res.bot == bot
        assert res.trades == 2
        assert res.ret_pct != 0.0
        assert res.recommendation in ("promote","hold","demote")