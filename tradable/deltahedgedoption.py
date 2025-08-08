from ..interface.itradable import ITradable


class DeltaHedgedOption(ITradable):
    """
    Wrap class to use in realised pnl calculation
    """
    def __init__(self, ticker, underlying):
        self.ticker = ticker
        self.underlying = underlying

    def clone(self):
        return DeltaHedgedOption(self.ticker, self.underlying)

    def has_expiration(self):
        return True

    def name(self):
        return self.ticker

    def get_underlyings(self):
        return [self.underlying]
