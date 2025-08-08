from ..interface.itradable import ITradable


class Cash(ITradable):
    def __init__(self, currency):
        self.currency = currency

    def clone(self):
        return Cash(self.currency)

    def has_expiration(self):
        return False

    def name(self):
        return self.currency + self.postfix()

    @staticmethod
    def postfix():
        return "Cash"
