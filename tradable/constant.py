from ..interface.itradable import ITradable

class Constant(ITradable):
    """
    Special tradable class to handle the offset in future
    this is different from Cash as it is not real cash in portfolio, and doesn't accrue anything
    """
    def __init__(self, currency):
        self.currency = currency

    def clone(self):
        return Constant(self.currency)

    def has_expiration(self):
        return False

    def name(self):
        return self.currency + self.postfix()

    @staticmethod
    def postfix():
        return "Constant"
