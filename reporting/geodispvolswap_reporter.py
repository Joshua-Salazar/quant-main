from ..interface.itradereporter import ITradeReporter
from ..tradable.geodispvolswap import GeoDispVolSwap


class GeoDispVolSwapReporter(ITradeReporter):
    def __init__(self, geodispvolswap: GeoDispVolSwap):
        self.tradable = geodispvolswap
