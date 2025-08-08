from ..interface.itradereporter import ITradeReporter
from ..tradable.crosscondvolswap import CrossCondVolSwap


class CrossCondVolSwapReporter(ITradeReporter):
    def __init__(self, crosscondvolswap: CrossCondVolSwap):
        self.tradable = crosscondvolswap
