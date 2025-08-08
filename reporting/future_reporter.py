from ..interface.itradereporter import ITradeReporter
from ..tradable.future import Future


class FutureReporter(ITradeReporter):
    def __init__(self, future: Future):
        self.tradable = future
