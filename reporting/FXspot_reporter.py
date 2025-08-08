from ..interface.itradereporter import ITradeReporter
from ..tradable.fxspot import FXSpot


class FXSpotReporter(ITradeReporter):
    def __init__(self, fxspot: FXSpot):
        self.tradable = fxspot
