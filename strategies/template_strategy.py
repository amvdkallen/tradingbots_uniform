from typing import List, Dict, Any
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
