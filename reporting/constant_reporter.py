from ..interface.itradereporter import ITradeReporter
from ..tradable.constant import Constant


class ConstantReporter(ITradeReporter):
    def __init__(self, constant: Constant):
        self.tradable = constant
