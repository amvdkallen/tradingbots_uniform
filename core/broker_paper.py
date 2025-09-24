from typing import Optional
from core.strategy_base import Fill

class PaperBroker:
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.pos_qty = 0.0
        self.pos_avg_price = 0.0
    
    def market(self, ts: str, side: str, qty: float, ref_price: float) -> Fill:
        """Execute a market order at the given price"""
        fill_price = ref_price
        
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
    
    def mark_to_market(self, price: float) -> float:
        """Calculate current equity based on position and mark price"""
        return self.balance + (self.pos_qty * price)
