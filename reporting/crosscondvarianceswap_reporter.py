from ..interface.itradereporter import ITradeReporter
from ..tradable.crosscondvarianceswap import CrossCondVarianceSwap


class CrossCondVarianceSwapReporter(ITradeReporter):
    def __init__(self, crosscondvarianceswap: CrossCondVarianceSwap):
        self.tradable = crosscondvarianceswap
