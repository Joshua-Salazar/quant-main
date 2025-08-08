from datetime import datetime

from ..analytics.swaptions import atmf_yields_interpolate
from ..infrastructure.bmarket import BMarket
from ..infrastructure.market import Market
from ..tradable.portfolio import Portfolio
from ..tradable.swaption import Swaption
from ..tradable.option import Option
from ..tradable.future import Future
from ..tradable.fxspot import FXSpot
from ..tradable.stock import Stock
from ..interface.itradable import ITradable
from ..analytics.symbology import hedging_instrument_from_ticker
from ..backtest import hedge_utils
from ..tradable.forwardstartswap import ForwardStartSwap


class DeltaHedgeSingle:
    def __init__(self):
        self.hedge_instrument = None

    def determine_delta_hedge_instrument(self, dt: datetime, portfolio: Portfolio):
        pass

    def calc_price_and_delta_of_delta_hedge_instrument(self, dt: datetime, contract: ITradable):
        pass

    def calc_delta_of_portfolio(self, dt: datetime, portfolio: Portfolio):
        pass

    def delta_hedge(self, dt: datetime, portfolio: Portfolio, hedge_position_path=()):
        new_delta_hedge_instrument = self.determine_delta_hedge_instrument(dt, portfolio)
        # skip if not found hedge instrument
        if new_delta_hedge_instrument is None:
            return
        hedge_price, hedge_delta = self.calc_price_and_delta_of_delta_hedge_instrument(dt, new_delta_hedge_instrument)

        if (self.hedge_instrument is not None) and (new_delta_hedge_instrument.name != self.hedge_instrument.name):
            prev_hedge_price, prev_hedge_delta = self.calc_price_and_delta_of_delta_hedge_instrument(dt,
                                                                                                     self.hedge_instrument)
            portfolio.unwind(self.self.hedge_instrument, prev_hedge_price)

        portfolio_delta = self.calc_delta_of_portfolio(dt, portfolio)
        portfolio.trade(new_delta_hedge_instrument, -1.0 * portfolio_delta / hedge_delta, hedge_price, position_path=hedge_position_path)


# WIP
class SwaptionDailyDH(DeltaHedgeSingle):
    def __init__(self, backtest_market: BMarket, valuer_map={}, use_swaption_underlying=False):
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map
        self.use_swaption_underlying = use_swaption_underlying
        super().__init__()

    def delta_hedge(self, dt: datetime, portfolio: Portfolio, unwind_existing=False, hedge_position_path=(), residual_delta=0):
        if unwind_existing:
            position_names = list(portfolio.get_positions().keys())
            for pos_name in position_names:
                pos = portfolio.get_position(pos_name, hedge_position_path=hedge_position_path)
                if isinstance(pos.tradable, ForwardStartSwap):
                    unwind_price = pos.tradable.price(market=self.backtest_market.get_market(dt), valuer=self.valuer_map[type(pos.tradable)], calc_types='price')
                    portfolio.unwind(pos_name, unwind_price)

        # group the positions
        groups = {}
        pos_map = portfolio.filter_tradable_type(ForwardStartSwap).root
        pos_map.update(portfolio.filter_tradable_type(Swaption).root)
        for pos_name, pos in pos_map.items():
            trait = (pos.tradable.currency, pos.tradable.expiration, pos.tradable.tenor, pos.tradable.curve)
            if trait not in groups:
                groups[trait] = [(pos_name, pos)]
            else:
                groups[trait].append((pos_name, pos))
        # hedge each group
        for t, positions in list(groups.items()):
            group_port = Portfolio(positions)
            # hedge instrument
            market = self.backtest_market.get_market(dt)
            fwd_rate_curve = {dt: market.get_forward_rates(t[0], t[3])}
            spot_rate_curve = {dt: market.get_spot_rates(t[0], t[3]).data_dict}
            atmf = atmf_yields_interpolate(fwd_rate_curve, spot_rate_curve, dt, t[2], t[1])
            new_delta_hedge_instrument = ForwardStartSwap(t[0], t[1], t[2], atmf, "Payer", t[3])

            hedge_price, hedge_delta = self.calc_price_and_delta_of_delta_hedge_instrument(dt, new_delta_hedge_instrument)
            portfolio_delta = self.calc_delta_of_portfolio(dt, group_port)
            portfolio.trade(new_delta_hedge_instrument, -1.0 * (portfolio_delta - residual_delta) / hedge_delta, hedge_price, position_path=hedge_position_path)

    def determine_delta_hedge_instrument(self, dt: datetime, portfolio: Portfolio):
        underlyings = []
        for k, v in portfolio.net_positions().items():
            if isinstance(v.tradable, Swaption):
                underlyings.append(v.tradable.underlying)
        underlying_names = set([ul.name() for ul in underlyings])
        if len(underlying_names) == 0:
            return None
        elif len(underlying_names) > 1:
            raise RuntimeError('Only 1 Underlying Expected')
        else:
            swaption_underlying = underlyings.pop()
            if self.use_swaption_underlying:
                return swaption_underlying
            else:
                market = self.backtest_market.get_market(dt)
                fwd_rate_curve = {dt: market.get_forward_rates(swaption_underlying.currency, swaption_underlying.curve)}
                spot_rate_curve = {dt: market.get_spot_rates(swaption_underlying.currency, swaption_underlying.curve).data_dict}
                atmf = atmf_yields_interpolate(fwd_rate_curve, spot_rate_curve, dt, swaption_underlying.tenor, swaption_underlying.expiration)
                swaption_underlying.strike = atmf
                return swaption_underlying

    def calc_price_and_delta_of_delta_hedge_instrument(self, dt: datetime, contract: ITradable):
        try:
            price, delta = contract.price(market=self.backtest_market.get_market(dt), calc_types=['price', 'delta'],
                                          valuer=self.valuer_map[type(contract)])
        except:
            print("failed")
        return price, delta

    def calc_delta_of_portfolio(self, dt: datetime, portfolio: Portfolio):
        portfolio_delta = portfolio.aggregate('delta', default_value=0)
        return portfolio_delta


class FXOptionSpotDH(DeltaHedgeSingle):
    def __init__(self, backtest_market: BMarket, valuer_map={}):
        """
        value_function: a function taking a datetime and a tradable, and returns the price and delta as a dictionary {'price': xxx, 'delta': xxx}
        """
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map
        super().__init__()

    def determine_delta_hedge_instrument(self, dt: datetime, portfolio: Portfolio):
        underlyings = []
        for k, v in portfolio.net_positions().items():
            if isinstance(v.tradable, Option):
                underlyings.append(v.tradable.underlying)
        underlying_names = set(underlyings)
        if len(underlying_names) == 0:
            return None
        elif len(underlying_names) > 1:
            raise RuntimeError('Only 1 Underlying Expected')
        else:
            return FXSpot(underlyings[0])

    def calc_price_and_delta_of_delta_hedge_instrument(self, dt: datetime, contract: ITradable):
        price, delta = contract.price(market=self.backtest_market.get_market(dt), calc_types=['price', 'delta'],
                                      valuer=self.valuer_map.get(type(contract), None))
        return price, delta

    def calc_delta_of_portfolio(self, dt: datetime, portfolio: Portfolio):
        portfolio_delta = portfolio.aggregate('delta', default_value=0)
        return portfolio_delta


class EquityOptionDH(DeltaHedgeSingle):
    def __init__(self, backtest_market: BMarket, valuer_map={}, expiry_offset_for_front_future=0,
                 use_portfolio_expiry_future_for_index_underlying_future_hedge=False,
                 ):
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map
        self.expiry_offset_for_front_future = expiry_offset_for_front_future
        self.use_portfolio_expiry_future_for_index_underlying_future_hedge = use_portfolio_expiry_future_for_index_underlying_future_hedge
        super().__init__()

    def determine_delta_hedge_instrument(self, dt: datetime, portfolio: Portfolio):
        for k, v in portfolio.root.items():
            if isinstance(v.tradable, Option):
                hedge_type = hedging_instrument_from_ticker(v.tradable.root)
                if hedge_type == 'stock':
                    hedge_instrument = Stock(v.tradable.underlying, v.tradable.currency)
                elif hedge_type == 'future':
                    if isinstance(v.tradable.underlying, Future):
                        hedge_instrument = v.tradable.underlying
                    else:
                        if self.use_portfolio_expiry_future_for_index_underlying_future_hedge:
                            hedge_instrument = Future(v.tradable.underlying, v.tradable.currency, portfolio.get_expiration(),
                                                      exchange="", listed_ticker="", tz_name=v.tradable.tz_name)
                        else:
                            hedge_instrument = hedge_utils.find_front_future(
                                self.backtest_market.get_market(dt), v.tradable.root,
                                'last tradable date',
                                self.expiry_offset_for_front_future,
                            )
                else:
                    raise RuntimeError(f"unknown hedging instrument type {hedge_type}")

            break
        return hedge_instrument

    def calc_price_and_delta_of_delta_hedge_instrument(self, dt: datetime, contract: ITradable):
        price, delta = contract.price(market=self.backtest_market.get_market(dt), calc_types=['price', 'delta'],
                                      valuer=self.valuer_map.get(type(contract), None))
        return price, delta

    def calc_delta_of_portfolio(self, dt: datetime, portfolio: Portfolio):
        portfolio_delta = portfolio.aggregate('delta', default_value=0)
        return portfolio_delta
