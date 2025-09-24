from typing import List, Dict, Any
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
