from ..interface.itradable import ITradable


class FXSpot(ITradable):
    def __init__(self, ticker):
        self.ticker = ticker
        self.currency = ticker[3:]

    def clone(self):
        return FXSpot(self.ticker)

    def has_expiration(self):
        return False

    def name(self):
        return self.ticker
