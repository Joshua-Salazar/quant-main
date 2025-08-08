from ..interface.itradereporter import ITradeReporter
from ..tradable.swaption import Swaption


class SwaptionReporter(ITradeReporter):
    def __init__(self, swaption: Swaption):
        self.tradable = swaption
