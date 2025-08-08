from ..interface.itradereporter import ITradeReporter
from ..tradable.stock import Stock


class StockReporter(ITradeReporter):
    def __init__(self, stock: Stock):
        self.tradable = stock
