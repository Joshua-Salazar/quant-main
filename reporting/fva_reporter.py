from ..interface.itradereporter import ITradeReporter
from ..tradable.fva import FVA


class FVAReporter(ITradeReporter):
    def __init__(self, fva: FVA):
        self.tradable = fva
