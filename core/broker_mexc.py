import ccxt
from typing import Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.strategy_base import Fill
from core.utils import to_iso

class MexcBroker:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        exchange_type: Literal['spot', 'swap'] = 'spot',
        dry_run: bool = True,
        leverage: int = 1,
        slippage_bps: int = 10,
        max_retries: int = 3,
        exchange: Optional[ccxt.Exchange] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.exchange_type = exchange_type
        self.dry_run = dry_run
        self.leverage = leverage
        self.slippage_bps = slippage_bps
        self.max_retries = max_retries
        
        self.pos_qty = 0.0
        self.pos_avg_price = 0.0
        
        if exchange is None:
            self.exchange = ccxt.mexc({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if exchange_type == 'swap' else 'spot'
                }
            })
            self.exchange.load_markets()
            
            if exchange_type == 'swap' and not dry_run:
                try:
                    self.exchange.set_leverage(leverage, symbol)
                except Exception as e:
                    print(f"Warning: Could not set leverage: {e}")
        else:
            self.exchange = exchange
    
    def round_amount(self, symbol: str, qty: float) -> float:
        """Round amount to exchange precision"""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        market = self.exchange.markets[symbol]
        rounded = float(self.exchange.amount_to_precision(symbol, qty))
        
        if 'limits' in market and 'amount' in market['limits']:
            min_amount = market['limits']['amount'].get('min', 0)
            if min_amount and rounded < min_amount:
                raise ValueError(f"Amount {rounded} below minimum {min_amount}")
        
        return rounded
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision"""
        if symbol not in self.exchange.markets:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        market = self.exchange.markets[symbol]
        rounded = float(self.exchange.price_to_precision(symbol, price))
        
        if 'limits' in market and 'price' in market['limits']:
            min_price = market['limits']['price'].get('min', 0)
            if min_price and rounded < min_price:
                raise ValueError(f"Price {rounded} below minimum {min_price}")
        
        return rounded
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.DDoSProtection))
    )
    def get_last_price(self, symbol: Optional[str] = None) -> float:
        """Get last price with retries"""
        ticker = self.exchange.fetch_ticker(symbol or self.symbol)
        return ticker['last']
    
    def market(self, ts: str, side: str, qty: float, ref_price: Optional[float] = None) -> Fill:
        """Execute market order"""
        try:
            qty = self.round_amount(self.symbol, qty)
            
            if self.dry_run:
                if ref_price is None:
                    ref_price = self.get_last_price(self.symbol)
                
                slippage = ref_price * self.slippage_bps / 10000
                if side == 'buy':
                    fill_price = ref_price + slippage
                else:
                    fill_price = ref_price - slippage
                
                if side == 'buy':
                    total_value = (self.pos_qty * self.pos_avg_price) + (qty * fill_price)
                    self.pos_qty += qty
                    if self.pos_qty > 0:
                        self.pos_avg_price = total_value / self.pos_qty
                elif side == 'sell':
                    self.pos_qty -= qty
                    if abs(self.pos_qty) < 1e-8:
                        self.pos_qty = 0.0
                        self.pos_avg_price = 0.0
                elif side == 'close':
                    actual_qty = abs(self.pos_qty)
                    self.pos_qty = 0.0
                    self.pos_avg_price = 0.0
                    return Fill(ts=ts, side=side, qty=actual_qty, price=fill_price)
                
                return Fill(ts=ts, side=side, qty=qty, price=fill_price)
            
            else:
                order_side = 'buy' if side == 'buy' else 'sell'
                if side == 'close':
                    qty = abs(self.pos_qty)
                    order_side = 'sell' if self.pos_qty > 0 else 'buy'
                
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=order_side,
                    amount=qty
                )
                
                fill_price = order.get('average', order.get('price'))
                if not fill_price:
                    fill_price = self.get_last_price(self.symbol)
                
                if side == 'buy' or (side == 'close' and order_side == 'buy'):
                    total_value = (self.pos_qty * self.pos_avg_price) + (qty * fill_price)
                    self.pos_qty += qty if side == 'buy' else -qty
                    if self.pos_qty > 0:
                        self.pos_avg_price = total_value / self.pos_qty
                else:
                    self.pos_qty -= qty if side == 'sell' else -qty
                    if abs(self.pos_qty) < 1e-8:
                        self.pos_qty = 0.0
                        self.pos_avg_price = 0.0
                
                return Fill(ts=ts, side=side, qty=qty, price=fill_price)
                
        except Exception as e:
            raise ValueError(f"Order failed: {e}")
    
    def mark_to_market(self, price: Optional[float] = None) -> float:
        """Calculate current equity"""
        if price is None:
            price = self.get_last_price(self.symbol)
        return 10000.0 + (self.pos_qty * (price - self.pos_avg_price) if self.pos_avg_price > 0 else 0)
