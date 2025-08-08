from ..interface.itradereporter import ITradeReporter
from ..tradable.FXforward import FXforward


class FXforwardReporter(ITradeReporter):
    def __init__(self, fxforward: FXforward):
        self.tradable = fxforward
