from datetime import datetime

from ...analytics.future_finder import find_nth_future
from ...backtest.costs import FlatTradingCost
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.forwardstartswap import ForwardStartSwap
from ...tradable.future import Future
from ...valuation.forwardstartswap_valuer import ForwardStartSwapValuer, ForwardStartSwapValuerOld
from ...valuation.future_data_valuer import FutureDataValuer
from ...analytics.swaptions import atmf_yields_interpolate_old, rate_interpolate
from ...data.market import get_expiry_in_year
import re


class InvoiceSwapState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, current_future_expiry, current_fwd_swap_rate, reference_bond_yield):
        self.price = price
        self.cost = cost
        self.current_future_expiry = current_future_expiry
        self.current_fwd_swap_rate = current_fwd_swap_rate
        self.reference_bond_yield = reference_bond_yield
        super().__init__(time_stamp, portfolio)


class EODTradeEvent(Event):
    def get_forward_rate(self, market, expiry):
        spot_rate_curve = {
            self.time_stamp: market.get_spot_rates(self.strategy.currency, self.parameters['spot_rate_curve_type']).data_dict}
        if self.parameters['forward_rate_curve_type'] == 'SWAPTION':
            fwd_rate_curve = {self.time_stamp: market.get_forward_rates(self.strategy.currency)}
            fwd_swap_rate = atmf_yields_interpolate_old(fwd_rate_curve, spot_rate_curve, self.time_stamp,
                                                    self.parameters['tenor'], expiry)
        elif self.parameters['forward_rate_curve_type'] == 'SPOTRATE':
            # use spot rate as forward rate
            tenor_elements = re.match(r'(\d{1,2})(Y|M|W|D|y|m|w|d)', self.parameters['tenor']).groups()
            tenor_years = get_expiry_in_year(tenor_elements[0], tenor_elements[1])
            fwd_swap_rate = rate_interpolate(spot_rate_curve, self.time_stamp, tenor_years)
        else:
            raise RuntimeError(f"Unknown forward rate curve type {self.parameters['forward_rate_curve_type']}")
        return fwd_swap_rate

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        future = find_nth_future(market, self.parameters['root'], 1, self.parameters['expiry_reference'], self.parameters['expiry_offset'])
        roll_day = False
        if state.current_future_expiry is not None and state.current_future_expiry < future.expiration:
            roll_day = True
            for sub_portfolio in ['future', 'swap']:
                position_names = list(portfolio.get_position(sub_portfolio).get_positions().keys())
                for pos_name in position_names:
                    contract = portfolio.get_position((sub_portfolio, pos_name)).tradable
                    if isinstance(contract, Future) or isinstance(contract, ForwardStartSwap):
                        portfolio.unwind((sub_portfolio, pos_name), contract.price(market=market, valuer=self.strategy.valuer_map[type(contract)], calc_types='price'), contract.currency)

        current_future_expiry = state.current_future_expiry
        if state.current_future_expiry is None or roll_day:
            # find the strike of the forward starting swap
            fwd_swap_rate = self.get_forward_rate(market, future.expiration)
            swap = ForwardStartSwap(self.strategy.currency, future.expiration, self.parameters['tenor'], fwd_swap_rate, 'Payer', self.parameters['forward_rate_curve_type'])
            portfolio.trade(future, 1, future.price(market=market, valuer=self.strategy.valuer_map[Future], calc_types='price'), self.strategy.currency, position_path=('future',))
            portfolio.trade(swap, 100, swap.price(market=market, valuer=self.strategy.valuer_map[ForwardStartSwap], calc_types='price'), self.strategy.currency, position_path=('swap',))
            current_future_expiry = future.expiration

        # swap yield and reference bond yield
        current_fwd_swap_rate = self.get_forward_rate(market, current_future_expiry)
        reference_bond_yield = market.get_bond_yield(self.parameters['reference_bond_ticker'])

        # tc
        portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)

        # value portfolio
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        return InvoiceSwapState(self.time_stamp, portfolio, price, 0.0, current_future_expiry, current_fwd_swap_rate, reference_bond_yield)


class InvoiceSwap(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        self.valuer_map = {
            Future: FutureDataValuer('close'),
            ForwardStartSwap: ForwardStartSwapValuerOld(
                spot_rate_curve_type=self.parameters['spot_rate_curve_type'],
                discount_rate_curve_type=self.parameters['discount_rate_curve_type'],
                forward_rate_curve_type=self.parameters['forward_rate_curve_type']
            ),
        }

        self.parameters['trading_cost'] = FlatTradingCost({
            Future: self.parameters['future_tc'],
            ForwardStartSwap: self.parameters['swap_tc'],
        })

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
