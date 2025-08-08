from ..interface.itradereporter import ITradeReporter
from ..tradable.vanillaswap import VanillaSwap


class VanillaSwapReporter(ITradeReporter):
    def __init__(self, vanillaswap: VanillaSwap):
        self.tradable = vanillaswap
