from abc import ABC, abstractmethod


class IScenario(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def instantaneous_scenario(self, instrument_asset_class, option_strategys, market, valuermap_overrides, base_date,
                               greek_types, currency, price_datastore=None):
        pass

    @abstractmethod
    def get_annualised_pnl(self, instrument_asset_class, option_strategy, market, valuermap_overrides, base_date, frequency, greek_types=[],
                           redistribute_hedge=True, currency=None, price_datastore=None):
        pass
