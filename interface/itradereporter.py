from ..interface.imarket import IMarket
from ..interface.ivaluer import IValuer


class ITradeReporter:
    def __init__(self):
        pass

    def get_cash_flows(self, market: IMarket):
        return {}

    def get_fixing_requirement(self, market: IMarket):
        return []

    def get_holiday_requirement(self, market: IMarket, valuer: IValuer=None):
        return []
