import uuid
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
