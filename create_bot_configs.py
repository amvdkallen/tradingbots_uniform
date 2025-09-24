import yaml
import os

configs = [
    {
        "name": "btc_bb_5m",
        "symbol": "BTC/USDT", 
        "timeframe": "5m",
        "qty": 0.001,
        "window": 20,
        "k": 2.0
    },
    {
        "name": "eth_bb_15m",
        "symbol": "ETH/USDT",
        "timeframe": "15m", 
        "qty": 0.01,
        "window": 20,
        "k": 2.5
    },
    {
        "name": "btc_bb_1h",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "qty": 0.001,
        "window": 24,
        "k": 2.0
    }
]

for cfg in configs:
    config = {
        "mode": "live",
        "bot_name": cfg["name"],
        "strategy": {
            "module": "bollinger_revert",
            "class": "BollingerRevert",
            "params": {
                "window": cfg["window"],
                "k": cfg["k"],
                "qty": cfg["qty"]
            }
        },
        "broker": {
            "exchange_type": "spot",
            "dry_run": True,  # Keep paper trading!
            "slippage_bps": 10,
            "max_retries": 3
        },
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"]
    }
    
    filename = f"config/{cfg['name']}.yml"
    with open(filename, 'w') as f:
        yaml.dump(config, f)
    print(f"Created: {filename}")
