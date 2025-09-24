#!/usr/bin/env python3
import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created: {path}")

print("Creating Trading Bot Platform files...")
print("=" * 60)

# Create requirements.txt first
create_file('requirements.txt', '''ccxt
pandas
python-dotenv
pyyaml
streamlit
tenacity
pytest
''')

# Create .env.example
create_file('.env.example', '''MEXC_API_KEY=
MEXC_API_SECRET=
DB_PATH=tradingbots.db
''')

# Create all core files
create_file('core/__init__.py', '')

create_file('core/strategy_base.py', '''from dataclasses import dataclass
from typing import List, Literal, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class Bar:
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Signal:
    side: Literal['buy', 'sell', 'close']
    qty: float
    reason: str

@dataclass
class Fill:
    ts: str
    side: Literal['buy', 'sell', 'close']
    qty: float
    price: float
    reason: str = ""

class Strategy(ABC):
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any]):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params
    
    @abstractmethod
    def prepare(self, bars: List[Bar]) -> None:
        """Prepare strategy with historical data"""
        pass
    
    @abstractmethod
    def on_bar(self, bar: Bar) -> List[Signal]:
        """Process new bar and return signals"""
        pass
''')

create_file('core/utils.py', '''import pandas as pd
from typing import List, Union
from datetime import datetime, timezone
from core.strategy_base import Bar

def to_iso(timestamp: Union[int, float, str, datetime, None] = None) -> str:
    """Convert timestamp to ISO-8601 format with Z suffix"""
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    elif isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.isoformat().replace('+00:00', 'Z')
    elif isinstance(timestamp, (int, float)):
        if timestamp > 1e10:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace('+00:00', 'Z')
    else:
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    mapping = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '12h': 43200,
        '1d': 86400,
        '1w': 604800,
    }
    return mapping.get(timeframe, 60)

def load_bars_csv(path: str) -> List[Bar]:
    """Load bars from CSV file"""
    df = pd.read_csv(path)
    bars = []
    for _, row in df.iterrows():
        ts = row['ts']
        if isinstance(ts, str):
            ts_str = ts if 'T' in ts else to_iso(float(ts))
        else:
            ts_str = to_iso(ts)
        
        bar = Bar(
            ts=ts_str,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        )
        bars.append(bar)
    
    return bars
''')

create_file('core/storage.py', '''import sqlite3
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
''')

create_file('core/broker_paper.py', '''from typing import Optional
from core.strategy_base import Fill

class PaperBroker:
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.pos_qty = 0.0
        self.pos_avg_price = 0.0
    
    def market(self, ts: str, side: str, qty: float, ref_price: float) -> Fill:
        """Execute a market order at the given price"""
        fill_price = ref_price
        
        if side == 'buy':
            total_value = (self.pos_qty * self.pos_avg_price) + (qty * fill_price)
            self.pos_qty += qty
            if self.pos_qty > 0:
                self.pos_avg_price = total_value / self.pos_qty
        elif side == 'sell':
            self.pos_qty -= qty
            if abs(self.pos_qty) < 1e-8:
                self.pos_qty = 0.0
                self.pos_avg_price = 0.0
        elif side == 'close':
            actual_qty = abs(self.pos_qty)
            self.pos_qty = 0.0
            self.pos_avg_price = 0.0
            return Fill(ts=ts, side=side, qty=actual_qty, price=fill_price)
        
        return Fill(ts=ts, side=side, qty=qty, price=fill_price)
    
    def mark_to_market(self, price: float) -> float:
        """Calculate current equity based on position and mark price"""
        return self.balance + (self.pos_qty * price)
''')

create_file('core/broker_mexc.py', '''import ccxt
from typing import Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.strategy_base import Fill
from core.utils import to_iso

class MexcBroker:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        exchange_type: Literal['spot', 'swap'] = 'spot',
        dry_run: bool = True,
        leverage: int = 1,
        slippage_bps: int = 10,
        max_retries: int = 3,
        exchange: Optional[ccxt.Exchange] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.exchange_type = exchange_type
        self.dry_run = dry_run
        self.leverage = leverage
        self.slippage_bps = slippage_bps
        self.max_retries = max_retries
        
        self.pos_qty = 0.0
        self.pos_avg_price = 0.0
        
        if exchange is None:
            self.exchange = ccxt.mexc({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if exchange_type == 'swap' else 'spot'
                }
            })
            self.exchange.load_markets()
            
            if exchange_type == 'swap' and not dry_run:
                try:
                    self.exchange.set_leverage(leverage, symbol)
                except Exception as e:
                    print(f"Warning: Could not set leverage: {e}")
        else:
            self.exchange = exchange
    
    def round_amount(self, symbol: str, qty: float) -> float:
        """Round amount to exchange precision"""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        market = self.exchange.markets[symbol]
        rounded = float(self.exchange.amount_to_precision(symbol, qty))
        
        if 'limits' in market and 'amount' in market['limits']:
            min_amount = market['limits']['amount'].get('min', 0)
            if min_amount and rounded < min_amount:
                raise ValueError(f"Amount {rounded} below minimum {min_amount}")
        
        return rounded
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision"""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        market = self.exchange.markets[symbol]
        rounded = float(self.exchange.price_to_precision(symbol, price))
        
        if 'limits' in market and 'price' in market['limits']:
            min_price = market['limits']['price'].get('min', 0)
            if min_price and rounded < min_price:
                raise ValueError(f"Price {rounded} below minimum {min_price}")
        
        return rounded
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.DDoSProtection))
    )
    def get_last_price(self, symbol: Optional[str] = None) -> float:
        """Get last price with retries"""
        ticker = self.exchange.fetch_ticker(symbol or self.symbol)
        return ticker['last']
    
    def market(self, ts: str, side: str, qty: float, ref_price: Optional[float] = None) -> Fill:
        """Execute market order"""
        try:
            qty = self.round_amount(self.symbol, qty)
            
            if self.dry_run:
                if ref_price is None:
                    ref_price = self.get_last_price(self.symbol)
                
                slippage = ref_price * self.slippage_bps / 10000
                if side == 'buy':
                    fill_price = ref_price + slippage
                else:
                    fill_price = ref_price - slippage
                
                if side == 'buy':
                    total_value = (self.pos_qty * self.pos_avg_price) + (qty * fill_price)
                    self.pos_qty += qty
                    if self.pos_qty > 0:
                        self.pos_avg_price = total_value / self.pos_qty
                elif side == 'sell':
                    self.pos_qty -= qty
                    if abs(self.pos_qty) < 1e-8:
                        self.pos_qty = 0.0
                        self.pos_avg_price = 0.0
                elif side == 'close':
                    actual_qty = abs(self.pos_qty)
                    self.pos_qty = 0.0
                    self.pos_avg_price = 0.0
                    return Fill(ts=ts, side=side, qty=actual_qty, price=fill_price)
                
                return Fill(ts=ts, side=side, qty=qty, price=fill_price)
            
            else:
                order_side = 'buy' if side == 'buy' else 'sell'
                if side == 'close':
                    qty = abs(self.pos_qty)
                    order_side = 'sell' if self.pos_qty > 0 else 'buy'
                
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=order_side,
                    amount=qty
                )
                
                fill_price = order.get('average', order.get('price'))
                if not fill_price:
                    fill_price = self.get_last_price(self.symbol)
                
                if side == 'buy' or (side == 'close' and order_side == 'buy'):
                    total_value = (self.pos_qty * self.pos_avg_price) + (qty * fill_price)
                    self.pos_qty += qty if side == 'buy' else -qty
                    if self.pos_qty > 0:
                        self.pos_avg_price = total_value / self.pos_qty
                else:
                    self.pos_qty -= qty if side == 'sell' else -qty
                    if abs(self.pos_qty) < 1e-8:
                        self.pos_qty = 0.0
                        self.pos_avg_price = 0.0
                
                return Fill(ts=ts, side=side, qty=qty, price=fill_price)
                
        except Exception as e:
            raise ValueError(f"Order failed: {e}")
    
    def mark_to_market(self, price: Optional[float] = None) -> float:
        """Calculate current equity"""
        if price is None:
            price = self.get_last_price(self.symbol)
        return 10000.0 + (self.pos_qty * (price - self.pos_avg_price) if self.pos_avg_price > 0 else 0)
''')

create_file('core/engine.py', '''import uuid
from typing import List, Optional
from dataclasses import dataclass
from core.strategy_base import Strategy, Bar, Signal, Fill
from core.storage import Storage
from core.broker_paper import PaperBroker
from core.broker_mexc import MexcBroker
from core.utils import to_iso

class Engine:
    def __init__(
        self,
        strategy: Strategy,
        broker,
        storage: Storage,
        bot_name: str,
        symbol: str,
        timeframe: str,
        run_id: str,
        is_live: bool = False
    ):
        self.strategy = strategy
        self.broker = broker
        self.storage = storage
        self.bot_name = bot_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.run_id = run_id
        self.is_live = is_live

    def backtest(self, bars: List[Bar]) -> None:
        """Run backtest on historical bars"""
        try:
            self.strategy.prepare(bars)
            
            for bar in bars:
                self._process_bar(bar, bar.close)
                
        except Exception as e:
            self.storage.log_event(
                ts=to_iso(),
                run_id=self.run_id,
                bot=self.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='error',
                reason=str(e)
            )
            raise

    def process_live_bar(self, bar: Bar) -> None:
        """Process a single live bar"""
        try:
            last_price = self.broker.get_last_price(self.symbol)
            self._process_bar(bar, last_price)
        except Exception as e:
            self.storage.log_event(
                ts=to_iso(),
                run_id=self.run_id,
                bot=self.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='error',
                reason=str(e)
            )

    def _process_bar(self, bar: Bar, mark_price: float) -> None:
        """Process a single bar and execute signals"""
        self.storage.log_event(
            ts=bar.ts,
            run_id=self.run_id,
            bot=self.bot_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_type='heartbeat',
            price=bar.close
        )
        
        signals = self.strategy.on_bar(bar)
        
        for signal in signals:
            self.storage.log_event(
                ts=bar.ts,
                run_id=self.run_id,
                bot=self.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='signal',
                side=signal.side,
                qty=signal.qty,
                reason=signal.reason
            )
            
            self.storage.log_event(
                ts=bar.ts,
                run_id=self.run_id,
                bot=self.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='order',
                side=signal.side,
                qty=signal.qty,
                price=bar.close,
                reason=signal.reason
            )
            
            fill = self.broker.market(
                ts=bar.ts,
                side=signal.side,
                qty=signal.qty,
                ref_price=bar.close
            )
            
            self.storage.log_event(
                ts=fill.ts,
                run_id=self.run_id,
                bot=self.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='fill',
                side=fill.side,
                qty=fill.qty,
                price=fill.price,
                reason=fill.reason
            )
        
        equity = self.broker.mark_to_market(mark_price)
        self.storage.log_equity(
            ts=bar.ts,
            run_id=self.run_id,
            bot=self.bot_name,
            equity=equity,
            position_qty=self.broker.pos_qty,
            position_avg_price=self.broker.pos_avg_price
        )
''')

create_file('core/live_runner.py', '''import time
from typing import Optional
from datetime import datetime, timezone
import ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.engine import Engine
from core.strategy_base import Bar
from core.broker_mexc import MexcBroker
from core.storage import Storage
from core.utils import to_iso, timeframe_to_seconds

class LiveRunner:
    def __init__(
        self,
        engine: Engine,
        broker: MexcBroker,
        storage: Storage,
        symbol: str,
        timeframe: str
    ):
        self.engine = engine
        self.broker = broker
        self.storage = storage
        self.symbol = symbol
        self.timeframe = timeframe
        self.last_bar_ts: Optional[int] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.DDoSProtection))
    )
    def fetch_latest_bars(self) -> Optional[Bar]:
        """Fetch latest closed bar"""
        try:
            ohlcv = self.broker.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=2
            )
            
            if len(ohlcv) >= 2:
                candle = ohlcv[-2]
                ts_ms = candle[0]
                
                if self.last_bar_ts is None or ts_ms > self.last_bar_ts:
                    self.last_bar_ts = ts_ms
                    
                    bar = Bar(
                        ts=to_iso(ts_ms),
                        open=candle[1],
                        high=candle[2],
                        low=candle[3],
                        close=candle[4],
                        volume=candle[5]
                    )
                    return bar
            
            return None
            
        except Exception as e:
            self.storage.log_event(
                ts=to_iso(),
                run_id=self.engine.run_id,
                bot=self.engine.bot_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type='error',
                reason=f"Failed to fetch bars: {str(e)}"
            )
            raise
    
    def run(self) -> None:
        """Main loop for live trading"""
        print(f"Starting live runner for {self.symbol} {self.timeframe}")
        
        interval_seconds = timeframe_to_seconds(self.timeframe)
        poll_interval = max(1, interval_seconds // 10)
        
        while True:
            try:
                bar = self.fetch_latest_bars()
                if bar:
                    print(f"Processing new bar: {bar.ts}")
                    self.engine.process_live_bar(bar)
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print("Stopping live runner...")
                break
            except Exception as e:
                print(f"Error in live loop: {e}")
                self.storage.log_event(
                    ts=to_iso(),
                    run_id=self.engine.run_id,
                    bot=self.engine.bot_name,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    event_type='error',
                    reason=str(e)
                )
                time.sleep(poll_interval * 3)
''')

# Create strategies
create_file('strategies/__init__.py', '')

create_file('strategies/template_strategy.py', '''from typing import List, Dict, Any
from core.strategy_base import Strategy, Bar, Signal

class TemplateStrategy(Strategy):
    """Minimal template strategy - no operations"""
    
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any]):
        super().__init__(symbol, timeframe, params)
    
    def prepare(self, bars: List[Bar]) -> None:
        """No preparation needed for template"""
        pass
    
    def on_bar(self, bar: Bar) -> List[Signal]:
        """Template returns no signals"""
        return []
''')

create_file('strategies/bollinger_revert.py', '''from typing import List, Dict, Any
from collections import deque
from core.strategy_base import Strategy, Bar, Signal

class BollingerRevert(Strategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any]):
        super().__init__(symbol, timeframe, params)
        self.window = params.get('window', 20)
        self.k = params.get('k', 2.0)
        self.qty = params.get('qty', 1.0)
        self.closes = deque(maxlen=self.window)
        self.position = 0.0
    
    def prepare(self, bars: List[Bar]) -> None:
        """Seed the buffer with historical closes"""
        for bar in bars[-self.window:]:
            self.closes.append(bar.close)
    
    def on_bar(self, bar: Bar) -> List[Signal]:
        """Generate signals based on Bollinger Bands"""
        self.closes.append(bar.close)
        
        if len(self.closes) < self.window:
            return []
        
        mean = sum(self.closes) / len(self.closes)
        variance = sum((x - mean) ** 2 for x in self.closes) / len(self.closes)
        stdev = variance ** 0.5
        
        upper = mean + self.k * stdev
        lower = mean - self.k * stdev
        
        signals = []
        
        if bar.close <= lower and self.position <= 0:
            signals.append(Signal(
                side='buy',
                qty=self.qty,
                reason=f"Price {bar.close:.2f} <= Lower BB {lower:.2f}"
            ))
            self.position = self.qty
            
        elif bar.close >= upper and self.position >= 0:
            signals.append(Signal(
                side='sell',
                qty=self.qty,
                reason=f"Price {bar.close:.2f} >= Upper BB {upper:.2f}"
            ))
            self.position = -self.qty
        
        return signals
''')

# Create configs
create_file('config/bb_revert_backtest.yml', '''mode: backtest
bot_name: bb_revert_backtest
strategy:
  module: bollinger_revert
  class: BollingerRevert
  params:
    window: 20
    k: 2.0
    qty: 1.0
symbol: BTC/USDT
timeframe: 1h
data_file: data/sample_bars.csv
''')

create_file('config/bb_revert_live_spot.yml', '''mode: live
bot_name: bb_revert_live
strategy:
  module: bollinger_revert
  class: BollingerRevert
  params:
    window: 20
    k: 2.0
    qty: 0.001
broker:
  exchange_type: spot
  dry_run: true
  slippage_bps: 10
  max_retries: 3
symbol: BTC/USDT
timeframe: 5m
''')

create_file('config/template.yml', '''mode: backtest
bot_name: template_test
strategy:
  module: template_strategy
  class: TemplateStrategy
  params: {}
symbol: BTC/USDT
timeframe: 1h
data_file: data/sample_bars.csv
''')

# Create sample data
create_file('data/sample_bars.csv', '''ts,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1000.0
2024-01-01T01:00:00Z,42300.0,42600.0,42100.0,42400.0,1100.0
2024-01-01T02:00:00Z,42400.0,42700.0,42200.0,42600.0,950.0
2024-01-01T03:00:00Z,42600.0,42800.0,42500.0,42750.0,1200.0
2024-01-01T04:00:00Z,42750.0,42900.0,42600.0,42850.0,1050.0
2024-01-01T05:00:00Z,42850.0,43000.0,42700.0,42950.0,1150.0
2024-01-01T06:00:00Z,42950.0,43100.0,42800.0,43000.0,1300.0
2024-01-01T07:00:00Z,43000.0,43200.0,42900.0,43150.0,1250.0
2024-01-01T08:00:00Z,43150.0,43300.0,43000.0,43250.0,1100.0
2024-01-01T09:00:00Z,43250.0,43400.0,43100.0,43350.0,1400.0
2024-01-01T10:00:00Z,43350.0,43500.0,43200.0,43450.0,1350.0
2024-01-01T11:00:00Z,43450.0,43600.0,43300.0,43550.0,1200.0
2024-01-01T12:00:00Z,43550.0,43700.0,43400.0,43500.0,1450.0
2024-01-01T13:00:00Z,43500.0,43650.0,43350.0,43400.0,1300.0
2024-01-01T14:00:00Z,43400.0,43550.0,43250.0,43300.0,1150.0
2024-01-01T15:00:00Z,43300.0,43450.0,43150.0,43200.0,1250.0
2024-01-01T16:00:00Z,43200.0,43350.0,43050.0,43100.0,1100.0
2024-01-01T17:00:00Z,43100.0,43250.0,42950.0,43000.0,1350.0
2024-01-01T18:00:00Z,43000.0,43150.0,42850.0,42900.0,1400.0
2024-01-01T19:00:00Z,42900.0,43050.0,42750.0,42800.0,1200.0
2024-01-01T20:00:00Z,42800.0,42950.0,42650.0,42700.0,1300.0
2024-01-01T21:00:00Z,42700.0,42850.0,42100.0,42200.0,1500.0
2024-01-01T22:00:00Z,42200.0,42400.0,41800.0,41900.0,1600.0
2024-01-01T23:00:00Z,41900.0,42100.0,41700.0,42000.0,1400.0
2024-01-02T00:00:00Z,42000.0,42200.0,41500.0,41600.0,1700.0
2024-01-02T01:00:00Z,41600.0,41800.0,41400.0,41500.0,1500.0
2024-01-02T02:00:00Z,41500.0,41700.0,41300.0,41400.0,1450.0
2024-01-02T03:00:00Z,41400.0,41600.0,41200.0,41300.0,1350.0
2024-01-02T04:00:00Z,41300.0,41500.0,41100.0,41200.0,1400.0
2024-01-02T05:00:00Z,41200.0,41400.0,41000.0,41100.0,1500.0
''')

# Create dashboard
create_file('dashboard/__init__.py', '')

create_file('dashboard/app.py', '''import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_db_path():
    return os.getenv('DB_PATH', 'tradingbots.db')

def load_data():
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        return None, None, None
    
    try:
        conn = sqlite3.connect(db_path)
        
        events_query = "SELECT * FROM events ORDER BY id DESC LIMIT 100"
        events_df = pd.read_sql_query(events_query, conn)
        
        equity_query = "SELECT * FROM equity ORDER BY id"
        equity_df = pd.read_sql_query(equity_query, conn)
        
        fills_query = "SELECT * FROM events WHERE event_type = 'fill' ORDER BY id DESC LIMIT 50"
        fills_df = pd.read_sql_query(fills_query, conn)
        
        conn.close()
        return events_df, equity_df, fills_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def main():
    st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
    st.title("Trading Bot Dashboard")
    
    events_df, equity_df, fills_df = load_data()
    
    if events_df is None or events_df.empty:
        st.info("No data available. Please run a bot first.")
        return
    
    bots = events_df['bot'].unique()
    selected_bot = st.sidebar.selectbox("Select Bot", bots)
    
    bot_events = events_df[events_df['bot'] == selected_bot]
    bot_equity = equity_df[equity_df['bot'] == selected_bot] if not equity_df.empty else pd.DataFrame()
    bot_fills = fills_df[fills_df['bot'] == selected_bot] if not fills_df.empty else pd.DataFrame()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        if not bot_equity.empty:
            bot_equity['ts'] = pd.to_datetime(bot_equity['ts'])
            st.line_chart(bot_equity.set_index('ts')['equity'])
        else:
            st.info("No equity data available")
    
    with col2:
        st.subheader("Position")
        if not bot_equity.empty:
            latest = bot_equity.iloc[-1]
            st.metric("Current Equity", f"${latest['equity']:.2f}")
            st.metric("Position Size", f"{latest['position_qty']:.4f}")
            if latest['position_avg_price'] > 0:
                st.metric("Avg Price", f"${latest['position_avg_price']:.4f}")
    
    st.subheader("Recent Events")
    if not bot_events.empty:
        display_cols = ['ts', 'event_type', 'side', 'qty', 'price', 'reason']
        available_cols = [col for col in display_cols if col in bot_events.columns]
        st.dataframe(bot_events[available_cols].head(20))
    
    st.subheader("Recent Fills")
    if not bot_fills.empty:
        display_cols = ['ts', 'side', 'qty', 'price', 'reason']
        available_cols = [col for col in display_cols if col in bot_fills.columns]
        st.dataframe(bot_fills[available_cols])
    else:
        st.info("No fills yet")

if __name__ == "__main__":
    main()
''')

# Create tests
create_file('tests/__init__.py', '')

create_file('tests/test_rounding.py', '''import pytest
from unittest.mock import Mock, MagicMock
import ccxt
from core.broker_mexc import MexcBroker

def test_rounding_and_limits():
    mock_exchange = Mock(spec=ccxt.Exchange)
    mock_exchange.markets = {
        'BTC/USDT': {
            'precision': {'amount': 0.001, 'price': 0.01},
            'limits': {
                'amount': {'min': 0.001, 'max': 1000},
                'price': {'min': 0.01, 'max': 100000}
            }
        }
    }
    
    mock_exchange.amount_to_precision = Mock(side_effect=lambda s, a: f"{a:.3f}")
    mock_exchange.price_to_precision = Mock(side_effect=lambda s, p: f"{p:.2f}")
    
    broker = MexcBroker(
        api_key='test',
        api_secret='test',
        symbol='BTC/USDT',
        dry_run=True,
        exchange=mock_exchange
    )
    
    assert broker.round_amount('BTC/USDT', 0.123456) == 0.123
    assert broker.round_price('BTC/USDT', 42000.123456) == 42000.12
    
    with pytest.raises(ValueError, match="below minimum"):
        broker.round_amount('BTC/USDT', 0.0001)
''')

create_file('tests/test_signals.py', '''import pytest
from strategies.bollinger_revert import BollingerRevert
from core.strategy_base import Bar

def test_bollinger_signals():
    strategy = BollingerRevert(
        symbol='BTC/USDT',
        timeframe='1h',
        params={'window': 5, 'k': 2.0, 'qty': 1.0}
    )
    
    bars = [
        Bar(ts='2024-01-01T00:00:00Z', open=100, high=101, low=99, close=100, volume=1000),
        Bar(ts='2024-01-01T01:00:00Z', open=100, high=101, low=99, close=100, volume=1000),
        Bar(ts='2024-01-01T02:00:00Z', open=100, high=101, low=99, close=100, volume=1000),
        Bar(ts='2024-01-01T03:00:00Z', open=100, high=101, low=99, close=100, volume=1000),
        Bar(ts='2024-01-01T04:00:00Z', open=100, high=101, low=99, close=100, volume=1000),
    ]
    
    strategy.prepare(bars)
    
    low_bar = Bar(ts='2024-01-01T05:00:00Z', open=90, high=91, low=89, close=90, volume=1000)
    signals = strategy.on_bar(low_bar)
    
    assert len(signals) == 1
    assert signals[0].side == 'buy'
''')

# Create run_bot.py
create_file('run_bot.py', '''import os
import sys
import uuid
import yaml
import importlib
from dotenv import load_dotenv
from core.storage import Storage
from core.engine import Engine
from core.broker_paper import PaperBroker
from core.broker_mexc import MexcBroker
from core.live_runner import LiveRunner
from core.utils import load_bars_csv

def main():
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python run_bot.py <config.yml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mode = config['mode']
    bot_name = config['bot_name']
    symbol = config['symbol']
    timeframe = config['timeframe']
    
    run_id = str(uuid.uuid4())
    
    print("=" * 60)
    print(f"Trading Bot Platform")
    print(f"Bot: {bot_name}")
    print(f"Mode: {mode}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Run ID: {run_id}")
    print("=" * 60)
    
    db_path = os.getenv('DB_PATH', 'tradingbots.db')
    storage = Storage(db_path)
    
    strategy_module = config['strategy']['module']
    strategy_class = config['strategy']['class']
    strategy_params = config['strategy'].get('params', {})
    
    module = importlib.import_module(f'strategies.{strategy_module}')
    StrategyClass = getattr(module, strategy_class)
    strategy = StrategyClass(symbol, timeframe, strategy_params)
    
    if mode == 'backtest':
        data_file = config.get('data_file', 'data/sample_bars.csv')
        print(f"Loading data from {data_file}")
        
        bars = load_bars_csv(data_file)
        print(f"Loaded {len(bars)} bars")
        
        broker = PaperBroker()
        
        engine = Engine(
            strategy=strategy,
            broker=broker,
            storage=storage,
            bot_name=bot_name,
            symbol=symbol,
            timeframe=timeframe,
            run_id=run_id,
            is_live=False
        )
        
        engine.backtest(bars)
        print("Backtest complete")
        
    elif mode == 'live':
        broker_config = config.get('broker', {})
        
        api_key = os.getenv('MEXC_API_KEY', '')
        api_secret = os.getenv('MEXC_API_SECRET', '')
        
        if not api_key or not api_secret:
            print("Error: MEXC_API_KEY and MEXC_API_SECRET must be set in .env")
            sys.exit(1)
        
        broker = MexcBroker(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            exchange_type=broker_config.get('exchange_type', 'spot'),
            dry_run=broker_config.get('dry_run', True),
            leverage=broker_config.get('leverage', 1),
            slippage_bps=broker_config.get('slippage_bps', 10),
            max_retries=broker_config.get('max_retries', 3)
        )
        
        engine = Engine(
            strategy=strategy,
            broker=broker,
            storage=storage,
            bot_name=bot_name,
            symbol=symbol,
            timeframe=timeframe,
            run_id=run_id,
            is_live=True
        )
        
        try:
            print("Fetching initial bars for strategy preparation...")
            ohlcv = broker.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            bars = []
            for candle in ohlcv:
                from core.utils import to_iso
                from core.strategy_base import Bar
                bar = Bar(
                    ts=to_iso(candle[0]),
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5]
                )
                bars.append(bar)
            strategy.prepare(bars)
            print(f"Strategy prepared with {len(bars)} bars")
        except Exception as e:
            print(f"Warning: Could not prepare strategy: {e}")
        
        runner = LiveRunner(
            engine=engine,
            broker=broker,
            storage=storage,
            symbol=symbol,
            timeframe=timeframe
        )
        
        runner.run()
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')

print("=" * 60)
print("✅ All files created successfully!")
print("\n📦 Next steps:")
print("1. Install dependencies:")
print("   pip install -r requirements.txt")
print("\n2. Set up environment (optional for backtest):")
print("   cp .env.example .env")
print("\n3. Run your first backtest:")
print("   python run_bot.py config/bb_revert_backtest.yml")
print("\n4. View the dashboard:")
print("   streamlit run dashboard/app.py")
print("\nHappy trading! 🚀")