import pytest
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
