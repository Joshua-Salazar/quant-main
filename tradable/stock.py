from ..interface.itradable import ITradable


class Stock(ITradable):
    def __init__(self, ticker, currency):
        self.ticker = ticker
        self.currency = currency

    def clone(self):
        return Stock(self.ticker, self.currency)

    def has_expiration(self):
        return False

    def name(self):
        return self.ticker + self.currency + 'Stock'
