import pandas as pd
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
