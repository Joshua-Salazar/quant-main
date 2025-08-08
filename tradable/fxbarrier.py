from ..interface.itradable import ITradable


class FXBarrier(ITradable):
    """
    Wrap class to use in realised pnl calculation
    """
    def __init__(self, ticker):
        self.ticker = ticker

    def clone(self):
        return FXBarrier(self.ticker)

    def has_expiration(self):
        return False

    def name(self):
        return self.ticker
