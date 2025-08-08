from ..interface.itradereporter import ITradeReporter
from ..tradable.geodispvarswap import GeoDispVarSwap


class GeoDispVarSwapReporter(ITradeReporter):
    def __init__(self, geodispvarswap: GeoDispVarSwap):
        self.tradable = geodispvarswap
