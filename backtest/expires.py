from datetime import datetime

from ..interface.itradable import ITradable
from ..tradable.option import Option
from ..tradable.future import Future
from ..tradable.varianceswap import VarianceSwap
from ..tradable.FXforward import FXforward
from ..tradable.portfolio import Portfolio
from ..tradable.position import Position
from ..infrastructure.bmarket import BMarket


class ExpireContracts:
    def __init__(self):
        pass

    def expire(self, dt: datetime, portfolio: Portfolio, value_map_overide = None, position_path=(), cash_path=None, remove_zero=True):
        children = {k: v for k, v in portfolio.get_position(position_path).get_positions().items()}
        for k, v in children.items():
            if isinstance(v, Position):
                if v.tradable.has_expiration():
                    # TODO: expiry test to handle time
                    if dt.date() >= v.tradable.expiration.date():
                            # TODO: handle expiring logic properly
                        # this is just assuming expiring into cash at the market value
                        portfolio.unwind(position_path + (k,), self.expire_price(dt, v.tradable, value_map_overide), cash_path=cash_path, remove_zero=remove_zero)
            else:
                self.expire(dt, portfolio, value_map_overide, position_path=position_path + (k,), cash_path=cash_path, remove_zero=remove_zero)

    def expire_price(self, dt: datetime, contract: ITradable, value_map_overide ):
        pass


class ExpireContractsAtPrice(ExpireContracts):
    def __init__(self, backtest_market: BMarket, valuer_map={}):
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map

    def expire_price(self, dt: datetime, contract: ITradable, value_map_overide ):
        return contract.price(self.backtest_market.get_market(dt), self.valuer_map.get(type(contract), None), calc_types='price')

class ExpireVarswap(ExpireContracts):
    def __init__(self, backtest_market: BMarket, valuer_map={}):
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map

    def expire_price(self, dt: datetime, contract: ITradable, value_map_overide ):
        if isinstance( contract, VarianceSwap ):
            # replication pricer will use intrinsic value when prices on expiry are missing
            return contract.price(self.backtest_market.get_market(dt), self.valuer_map.get(type(contract), None), calc_types='price')
        else:
            # use intrinsic value, as some prices are missing
            s = self.backtest_market.get_market(dt).get_spot(contract.underlying)
            if contract.is_call:
                return max(0, s - contract.strike) * contract.contract_size
            else:
                return max(0, contract.strike - s) * contract.contract_size


class ExpireOptionAtIntrinsic(ExpireContracts):
    # support both option and future expiry
    def __init__(self, backtest_market: BMarket):
        self.backtest_market = backtest_market

    def expire_price(self, dt: datetime, contract: Option, value_map_overide ):
        #TODO: Update to use option intrinsic method
        if isinstance(contract.underlying, Future) or isinstance(contract, Future):
            future = contract.underlying if isinstance(contract.underlying, Future) else contract
            s = self.backtest_market.get_market(dt).get_future_price(None, None, future)
        else:
            s = self.backtest_market.get_market(dt).get_spot(contract.underlying)

        if isinstance(contract, Future):
            return s
        if contract.is_call:
            return max(0, s - contract.strike) * contract.contract_size
        else:
            return max(0, contract.strike - s) * contract.contract_size

        # return contract.intrinsic_value(self.backtest_market.get_market(dt)) * contract.contract_size


class ExpireFXOptionContracts(ExpireContracts):
    def __init__(self, backtest_market: BMarket, valuer_map={}):
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map

    def expire_price(self, dt: datetime, contract: Option, value_map_overide ):
        if len(contract.underlying) == 3:
            fx_pair_name = f'{contract.underlying}{contract.currency}'
        else:
            fx_pair_name = contract.underlying
        fx_vol_surface = self.backtest_market.get_market(dt).get_fx_vol_surface(fx_pair_name)
        if isinstance(contract, Option):
            if contract.expiration.date() < dt.date():
                spot = self.backtest_market.get_market(dt).get_fx_spot(fx_pair_name)
                return contract.intrinsic_value(spot)
            if value_map_overide is None:
                value_map = self.valuer_map.get(type(contract), None )
            else:
                value_map = value_map_overide.get(type(contract), None )
            return contract.price(market=self.backtest_market.get_market(dt), valuer=value_map, calc_types='price')
        elif isinstance(contract, FXforward):
            return fx_vol_surface.get_forward(contract.expiration)