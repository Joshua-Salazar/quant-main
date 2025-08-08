from datetime import datetime, time

from ...backtest.costs import FlatTradingCost, VariableVegaCost, PriceBasedTradingCostByTradableType

from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import count_business_days, add_business_days, tenor_to_years, add_tenor, bdc_adjustment
from ...backtest.expires import ExpireContractsAtPrice, ExpireOptionAtIntrinsic
from ...tradable.option import Option
from ...tradable.forwardstartcds import ForwardStartCDS

from ...valuation.croption_bs_valuer import CROptionBSValuer
from ...valuation.forwardstartcds_valuer import ForwardStartCDSSytheticValuer


class SimpleCROptionsState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost):
        self.price = price
        self.cost = cost
        super().__init__(time_stamp, portfolio)


class OptionEntry(Event):
    def find_next_expiry(self, tenor):
        next_expiry = add_tenor(self.time_stamp, tenor)
        next_expiry = bdc_adjustment(next_expiry, "following", self.strategy.holidays)
        return next_expiry

    def execute(self, state: StrategyState):
        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # trade
        if portfolio.is_empty():
            first_day = True
        else:
            first_day = False

        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                sub_portfolio = portfolio.get_position((leg_name, tranche_name))
                if sub_portfolio is None or len(sub_portfolio.find_children_of_tradable_type(Option)) == 0:
                    tenor = tranche['initial_tenor'] if first_day else leg['tenor']
                    TTM = tenor_to_years(tenor)

                    # backtest only handles certain (liquid) delta strikes
                    assert leg['strike'] in [x / 100 for x in range(5, 100, 5)]

                    vol_surface = market.get_cr_vol_surface(leg["underlying"])
                    abs_strike = vol_surface.abs_strikes[TTM][leg['strike']]
                    vol = vol_surface.get_vol_interpolation(TTM, abs_strike)

                    tranche_expiration_date = self.find_next_expiry(tenor)
                    option_to_trade = Option(
                        leg['underlying'], leg['underlying'], leg['currency'], tranche_expiration_date,
                        abs_strike, leg['type'] == 'C', False, 1.0, 'America/New_York', listed_ticker=None
                    )
                    assert len(leg['sizing']) == 1
                    size_by = list(leg['sizing'].keys())[0]
                    option_to_trade_risk = option_to_trade.price(
                        market=market,
                        valuer=self.strategy.valuer_map[Option],
                        return_struc=True,
                    )
                    option_to_trade_price = option_to_trade_risk['price']

                    option_to_trade_delta = round( option_to_trade_risk['forward_delta'], 3)
                    leg_sizing = leg['sizing'][size_by]
                    if size_by != 'quantity':
                        leg_sizing /= option_to_trade_risk[size_by]
                        assert abs(leg['sizing'][size_by] - leg_sizing * option_to_trade_risk[size_by]) < 0.001

                    if abs(round(option_to_trade_delta - leg['strike'], 3)) > 0.015:
                        print(f'entering position at {100*option_to_trade_delta} on {self.time_stamp}')

                    portfolio.trade(option_to_trade, leg_sizing, option_to_trade_price, leg['currency'], position_path=(leg_name, tranche_name))

        # add price manually
        price_pre_cost = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        if 'trading_cost' in self.parameters:
            self.parameters['trading_cost'].apply(portfolio, state.portfolio, self.time_stamp)
            price_post_cost = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            cost = price_pre_cost - price_post_cost
            price = price_post_cost
        else:
            cost = 0
            price = price_pre_cost

        return SimpleCROptionsState(self.time_stamp, portfolio, price, cost)


class ExpireOptions(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # expire options of each leg
        # cash left in each leg portfolio
        for leg_name, leg in self.parameters['legs'].items():
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                self.parameters['expires'].expire(self.time_stamp, leg_portfolio)

        return SimpleCROptionsState(self.time_stamp, portfolio, 0.0, 0.0)


class DeltaHedgeEOD(Event):
    def execute(self, state: StrategyState):
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        # copy the starting portfolio
        portfolio = state.portfolio.clone()
        for leg_name, leg in self.parameters['legs'].items():
            if leg['hedged']:
                for tranche_name, tranche in self.parameters['tranches'].items():
                    sub_portfolio = portfolio.get_position((leg_name, tranche_name))
                    delta = sub_portfolio.price_at_market(market, fields='delta', valuer_map_override=self.strategy.valuer_map)
                    # all the options
                    all_options = sub_portfolio.find_children_of_tradable_type(Option)
                    expirations = [x[1].tradable.expiration for x in all_options]
                    strikes = [x[1].tradable.strike for x in all_options]
                    expirations = list(set(expirations))
                    strikes = list(set(strikes))
                    assert len(expirations) == 1
                    assert len(strikes) == 1
                    if self.parameters.get('hedge_with_combo', False):
                        # combo
                        combo_call = Option(
                            leg['underlying'], leg['underlying'], leg['currency'], expirations[0],
                            strikes[0], leg['type'] == 'C', False, 1.0, 'America/New_York', listed_ticker=None
                        )
                        combo_call_risk = combo_call.price(market=market, valuer=self.strategy.valuer_map[Option], return_struc=True)
                        combo_put = Option(
                            leg['underlying'], leg['underlying'], leg['currency'], expirations[0],
                            strikes[0], leg['type'] == 'P', False, 1.0, 'America/New_York', listed_ticker=None
                        )
                        combo_put_risk = combo_put.price(market=market, valuer=self.strategy.valuer_map[Option], return_struc=True)
                        hedge_units = -delta / (combo_call_risk['delta'] - combo_put_risk['delta'])
                        # hedge put in each legs portfolio
                        portfolio.trade(combo_call, hedge_units, combo_call_risk['price'], leg['currency'], position_path=(leg_name, tranche_name, 'hedge'))
                        portfolio.trade(combo_put, -hedge_units, combo_put_risk['price'], leg['currency'], position_path=(leg_name, tranche_name, 'hedge'))
                    else:
                        fwdstartcds = ForwardStartCDS(leg['currency'], expirations[0], '5Y', strikes[0], 'Payer', leg['underlying'])
                        fwdstartcds_risk = fwdstartcds.price(market=market, valuer=self.strategy.valuer_map[ForwardStartCDS], return_struc=True)
                        hedge_units = -delta / fwdstartcds_risk['delta']
                        portfolio.trade(fwdstartcds, hedge_units, fwdstartcds_risk['price'], leg['currency'], position_path=(leg_name, tranche_name, 'hedge'))

        # add price manually
        price_pre_cost, delta_pre_cost = portfolio.price_at_market(market, fields=['price', 'delta'], valuer_map_override=self.strategy.valuer_map)

        if 'trading_cost' in self.parameters:
            self.parameters['trading_cost'].apply(portfolio, state.portfolio, self.time_stamp)
            price_post_cost, delta_post_cost = portfolio.price_at_market(market, fields=['price', 'delta'], valuer_map_override=self.strategy.valuer_map)
            cost = price_pre_cost - price_post_cost
            price = price_post_cost
        else:
            cost = 0
            price = price_pre_cost

        return SimpleCROptionsState(self.time_stamp, portfolio, price, state.cost + cost)


class SimpleCROptions(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.valuer_map = {
            Option: CROptionBSValuer('SWAP_LIBOR'),
            ForwardStartCDS: ForwardStartCDSSytheticValuer('SWAP_LIBOR')
        }

        self.parameters['expires'] = ExpireContractsAtPrice(self.backtest_market, self.valuer_map)

        # initialize the trading cost module
        if 'price_absolute_costs' in self.parameters.keys():
            cost_params = self.parameters['price_absolute_costs']
            self.parameters['trading_cost'] = PriceBasedTradingCostByTradableType(cost_params, absolute=True)

    def generate_events(self, dt: datetime):
        return [ExpireOptions(dt, self), OptionEntry(dt, self), DeltaHedgeEOD(dt, self)]


