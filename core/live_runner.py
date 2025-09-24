import time
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
