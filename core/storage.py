import sqlite3
from typing import Optional
from contextlib import contextmanager

class Storage:
    def __init__(self, db_path: str = "tradingbots.db"):
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create tables if not exist (idempotent)"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    run_id TEXT,
                    bot TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    event_type TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    run_id TEXT,
                    bot TEXT,
                    equity REAL,
                    position_qty REAL,
                    position_avg_price REAL
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def log_event(
        self,
        ts: str,
        run_id: str,
        bot: str,
        symbol: str,
        timeframe: str,
        event_type: str,
        side: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
        reason: Optional[str] = None
    ) -> None:
        """Log an event to the database"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO events (ts, run_id, bot, symbol, timeframe, event_type, side, qty, price, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, run_id, bot, symbol, timeframe, event_type, side, qty, price, reason))
            conn.commit()
    
    def log_equity(
        self,
        ts: str,
        run_id: str,
        bot: str,
        equity: float,
        position_qty: float,
        position_avg_price: float
    ) -> None:
        """Log equity snapshot to the database"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO equity (ts, run_id, bot, equity, position_qty, position_avg_price)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, run_id, bot, equity, position_qty, position_avg_price))
            conn.commit()
