from ..interface.itradereporter import ITradeReporter
from ..tradable.forwardstartswap import ForwardStartSwap


class ForwardStartSwapReporter(ITradeReporter):
    def __init__(self, forwardstartswap: ForwardStartSwap):
        self.tradable = forwardstartswap
