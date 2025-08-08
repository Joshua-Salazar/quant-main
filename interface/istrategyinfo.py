from abc import ABC, abstractmethod


class StrategyInfo(ABC):
    @abstractmethod
    def get_strategy_daily_returns(self, name):
        """
        Function that returns daily strategy/portfolio overall returns.

        Returns:
        Dataframe that has three columns with column names below:
            'date': NOT INDEX, with type datetime
            'return': float
            'level': float, compounded nav for easier resample
            'strategy_name': string, same value for all rows
        """
        pass

    # NOT YET FINALIZED
    @abstractmethod
    def get_strategy_daily_details(self):
        """
        Function that returns daily strategy details.

        Returns:
        Dataframe that has six columns with column names below:
            'date': NOT INDEX, with type datetime
            'symbol': string of underlier
            'leg': string of leg name
            'tranch': string of tranch name
            'trade_type': string of description of the trade, eg "SPY PUT 600 1/20/2025"
            'qty': float, size of the trade (NO MULTIPLIER)
            'spot': float, price of undelier
            'pnl': float
            'strategy': string of strategy name
        """
        pass
