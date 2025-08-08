from datetime import datetime
from ...backtest.costs import FlatTradingCost, VariableVegaCost
from ...backtest.expires import ExpireFXOptionContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import add_tenor, bdc_adjustment
from ...tradable.option import Option
from ...tradable.FXforward import FXforward
from ...valuation.fx_forward_fx_vol_surface_valuer import FXForwardDataValuer
from ...valuation.fxoption_sabr_bs_valuer import FXOptionSABRBSValuer
import pandas as pd


class FXCallWritingState(StrategyState):
    def __init__(self, time_stamp, portfolio, price):
        self.price = price
        super().__init__(time_stamp, portfolio)


class TradeEvent(Event):
    def find_next_expiry(self, tenor):
        next_expiry = add_tenor(self.time_stamp, tenor)
        next_expiry = bdc_adjustment(next_expiry, "following", self.strategy.holidays)
        return next_expiry

    def execute(self, state: StrategyState):

        # enter trade on month begin
        is_trade_date = len(self.strategy.states) == 0 or self.strategy.states[-1].time_stamp.month != self.time_stamp.month
        if not is_trade_date:
            return state

        ccy = self.parameters["position_ccy"]
        pct_pos = self.parameters["pos_data"].loc[self.time_stamp, ccy]
        if abs(pct_pos) <= self.parameters["pct_notional_threshold"]:
            return state

        market = self.strategy.backtest_market.get_market(self.time_stamp)
        portfolio = state.portfolio.clone(remove_zero=False)

        # trade
        is_call = pct_pos > 0
        pfo_name = f"PFO_{self.time_stamp.strftime('%Y-%m-%d')}"
        for leg_name, leg in self.parameters["legs"].items():
            assert leg["strike"] in [0.5, 0.25, 0.1]
            delta_strike = leg["strike"] if is_call else -leg["strike"]
            fx_pair = leg["base_currency"] + leg["term_currency"]
            vol_surface = market.get_fx_sabr_vol_surface(fx_pair)
            quotes = vol_surface.get_quote(leg["tenor"], delta_strike)
            strike = quotes["strike"]
            expiry = quotes["expiry"]
            option_to_trade = Option(leg["base_currency"], f"{leg['base_currency']}{leg['term_currency']}", leg["term_currency"], expiry, strike, is_call, False, 1.0, "America/New_York", listed_ticker=None)
            option_to_trade_price = option_to_trade.price(market=market, valuer=self.strategy.valuer_map[Option])

            ccy_conv = market.get_fx_spot(f"USD{ccy}")
            # For every position with initial position size of > 5% of NAV
            # Sell 25 delta 1m covered option for 50% notional of position size
            leg_sizing = -abs(pct_pos) * self.parameters["notional"] * ccy_conv * self.parameters["pct_notional_covered"]
            portfolio.trade(option_to_trade, leg_sizing, option_to_trade_price, leg["term_currency"], position_path=(pfo_name,))

        # add price manually
        post_state = FXCallWritingState(self.time_stamp, portfolio, state.price)
        return post_state


class ExpireEvent(Event):
    def execute(self, state: StrategyState):
        portfolio = state.portfolio.clone()
        self.parameters["expires"].expire(self.time_stamp, portfolio, remove_zero=False)
        return FXCallWritingState(self.time_stamp, portfolio, state.price)


class MTMEvent(Event):
    def add_attr(self, tradable, market):
        spot = 1
        strike = 1
        if isinstance(tradable, Option):
            spot = market.get_fx_spot(tradable.underlying)
            strike = tradable.strike
        return {"SpotRef": spot, "strike": strike}

    def execute(self, state: StrategyState):
        portfolio = state.portfolio.clone(remove_zero=False)
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        price = portfolio.price_at_market(market, fields=["price"], valuer_map_override=self.strategy.valuer_map)
        portfolio.value_positions(lambda tradable: self.add_attr(tradable, market))
        return FXCallWritingState(self.time_stamp, portfolio, price[0])


class FXCallWriting(DailyStrategy):
    def __init__(self, start_date, end_date, calendar, currency, parameters, data_requests):
        ccy = parameters["position_ccy"]
        pos_data = pd.read_excel(parameters["position_data_file"], sheet_name="CCS 2% Weights", parse_dates=["Date"])
        pos_data.columns = [x.upper() for x in pos_data.columns]
        if not pos_data["DATE"].is_monotonic_increasing:
            raise Exception("Found date in position data file is not monotonic increasing. Make sure that csv parse date is correct.")
        if pos_data["DATE"].shape[0] != pos_data["DATE"].unique().shape[0]:
            raise Exception("Found duplicate dates in position data file.")
        parameters["pos_data"] = pos_data[["DATE", ccy]].set_index("DATE")
        self.valuer_map = None
        super().__init__(start_date, end_date, calendar, currency, parameters, data_requests=data_requests)

    def preprocess(self):
        super().preprocess()
        self.valuer_map = {Option: FXOptionSABRBSValuer(), FXforward: FXForwardDataValuer()}

        # initialize the trading cost module
        if "variable_costs" in self.parameters.keys():
            cost_params = self.parameters["variable_costs"]
            self.parameters["trading_cost"] = VariableVegaCost(cost_params["cost_cap"], cost_params["cost_floor"],
                                                               cost_params["cost_scale"], cost_params["delta_cost"],
                                                               self.backtest_market, self.valuer_map)
        elif "flat_costs" in self.parameters.keys():
            cost_params = self.parameters["flat_costs"]
            self.parameters["trading_cost"] = FlatTradingCost(cost_params["tc_delta"], cost_params["tc_vega"])
        self.parameters["expires"] = ExpireFXOptionContracts(self.backtest_market, self.valuer_map)
        self.option_trade_log = {}

    def generate_events(self, dt: datetime):
        return [ExpireEvent(dt, self), TradeEvent(dt, self), MTMEvent(dt, self)]
