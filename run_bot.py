import os
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
