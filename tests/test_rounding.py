import pytest
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
