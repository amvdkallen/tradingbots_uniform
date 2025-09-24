from dataclasses import dataclass
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
