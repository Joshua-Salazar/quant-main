from ..interface.itradereporter import ITradeReporter
from ..tradable.cash import Cash


class CashReporter(ITradeReporter):
    def __init__(self, cash: Cash):
        self.tradable = cash
