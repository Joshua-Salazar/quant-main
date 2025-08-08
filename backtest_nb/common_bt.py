from abc import abstractmethod
from ..backtest.backtester import LocalBacktester
from ..interface.istrategyinfo import StrategyInfo
from ..backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche


class CommonBT(StrategyInfo):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False, cache_market_data=False, data_cache_path=None, logging=False, log_file=""):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.parameters = parameters
        self.force_run = force_run
        self.cache_market_data = cache_market_data
        self.data_cache_path = data_cache_path
        self.logging = logging
        self.log_file = log_file
        self.strategy = None
        self.initial_state = None
        self.results = None
        self.pfo_df = None

    @abstractmethod
    def init(self):
        return

    @abstractmethod
    def get_pfo_df(self):
        return

    def get_nav(self):
        res = self.pfo_df.groupby("dt")[["price"]].sum()
        return res

    def get_strategy_daily_returns(self):
        res = self.get_nav().diff()
        return res

    def get_strategy_daily_details(self):
        raise Exception("Not Implemented")

    def run(self):
        if self.strategy is None or self.initial_state is None:
            raise Exception(f"strategy or initial state is not set")
        runner = LocalBacktester()
        self.results = runner.run(self.strategy, self.start_date, self.end_date, self.initial_state)
        self.pfo_df = self.get_pfo_df()
        return self

    @staticmethod
    def create_tranche(tranche):
        if not isinstance(tranche, str):
            return tranche
        if tranche.lower() == RollingAtExpiryTranche.__name__.lower():
            return RollingAtExpiryTranche()
        elif tranche.lower() == RollingAtExpiryDailyTranche.__name__.lower():
            return RollingAtExpiryDailyTranche()
        raise Exception(f"Unsupported tranche name: {tranche}")

