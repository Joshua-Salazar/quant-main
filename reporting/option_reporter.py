from ..interface.itradereporter import ITradeReporter
from ..tradable.option import Option


class OptionReporter(ITradeReporter):
    def __init__(self, option: Option):
        self.tradable = option
