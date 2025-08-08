from ..interface.itradereporter import ITradeReporter
from ..tradable.forwardstartoption import ForwardStartOption


class ForwardStartOptionReporter(ITradeReporter):
    def __init__(self, forwardstartoption: ForwardStartOption):
        self.tradable = forwardstartoption
