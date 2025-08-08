from datetime import datetime
from ...analytics.utils import float_equal
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import add_business_days
from ...constants.ccy import Ccy
from ...infrastructure.bmarket import BMarket
from ...reporting.trade_reporter import TradeReporter
from ...tradable.cash import Cash
from ...tradable.xccyswap import XCcySwap
from ...tools.timer import Timer


class XccySwapState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, details, prev_date):
        self.price = price
        self.cost = cost
        self.details = details
        self.prev_date = prev_date

        super().__init__(time_stamp, portfolio)


class AccrualCashFlowEvent(Event):
    def execute(self, state: StrategyState):
        # read accrual interest from previous date
        if self.time_stamp != self.strategy.start_date:
            prev_market = self.strategy.backtest_market.get_market(state.prev_date)
            portfolio = state.portfolio.clone()
            for position_name, position in state.portfolio.root.items():
                if isinstance(position.tradable, Cash):
                    ccy = position.tradable.currency
                    df = prev_market.get_df(Ccy(ccy), self.parameters["funding_curve_name"], self.time_stamp)
                    accrual_cash = position.quantity * (1. / df - 1)
                    portfolio.add_position(Cash(ccy), accrual_cash)
        else:
            portfolio = state.portfolio
        return XccySwapState(self.time_stamp, portfolio, state.price, 0., state.details, state.prev_date)


class ExpirationEvent(Event):
    def execute(self, state: StrategyState):
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        portfolio = state.portfolio.clone()
        for position_name, position in state.portfolio.root.items():
            if position.tradable.has_expiration():
                if self.time_stamp.date() == position.tradable.expiration.date():
                    if isinstance(position.tradable, XCcySwap):
                        # add cash flows in original currency then unwind xccy swap with zero price
                        cash_flows = TradeReporter(position.tradable).get_cash_flows(market)
                        for ccy, amount in cash_flows.items():
                            portfolio.add_position(Cash(ccy.value), amount)
                        portfolio.unwind(position_name, unwind_price=0, unwind_currency=self.strategy.currency)
                    else:
                        raise Exception(f"Found unexpected trade {position_name}. It should be only xccy swap to expire.")
        return XccySwapState(self.time_stamp, portfolio, state.price, 0., state.details, state.prev_date)


class EODTradeEvent(Event):
    def execute(self, state: StrategyState):
        timer = Timer(f"execute on date {self.time_stamp}", unit="sec")
        timer.start()
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # unwind existing portfolio using today market data if required
        par_rate_tminusone = None
        if self.parameters["unwind_daily"]:
            for position_name, position in state.portfolio.root.items():
                tradable_to_unwind = position.tradable
                # exclude cash and constant
                if tradable_to_unwind.has_expiration():
                    unwind_price = tradable_to_unwind.price(market=market, calc_types='diagnostic')
                    portfolio.unwind(position_name, unwind_price, unwind_currency=self.strategy.currency)
                    par_rate_tminusone = tradable_to_unwind.price(market, calc_types="par_rate")

        trade_date = market.base_datetime
        xccy_swap_with_zero_spread = self.parameters["xccy_swap_convention"].create_swap(
            trade_date, self.parameters["swap_term"], 0., self.parameters["notional"], self.parameters["pay_receive"],
            self.parameters["swap_start_tenor"])
        par_rate = xccy_swap_with_zero_spread.price(market, calc_types="par_rate")
        trade = True
        portfolio_copy = portfolio.clone()
        for position_name, position in portfolio_copy.root.items():
            if isinstance(position.tradable, XCcySwap):
                is_spot_starting = self.parameters["swap_start_tenor"] == "0D"
                if is_spot_starting:
                    trade = False
                else:
                    roll_days = self.parameters.get("roll_days", 30)
                    roll_date = add_business_days(position.tradable.start_date, -roll_days)
                    if roll_date >= position.tradable.start_date:
                        raise Exception(f"Fail to unwind trade")

                    if self.time_stamp >= roll_date:
                        try:
                            unwind_price = position.tradable.price(market=market, calc_types='diagnostic')
                        except Exception as e:
                            raise Exception(f"Fail unwind_price {str(e)}")

                        portfolio.unwind(position_name, unwind_price, unwind_currency=self.strategy.currency)
                    else:
                        trade = False
        if trade:
            xccy_swap = self.parameters["xccy_swap_convention"].create_swap(
                trade_date, self.parameters["swap_term"], par_rate/1e4, self.parameters["notional"],
                self.parameters["pay_receive"], self.parameters["swap_start_tenor"])
            pv = xccy_swap.price(market, calc_types="price")
            if not float_equal(pv, 0.):
                raise Exception(f"par rate {par_rate} not give zero pv {pv}")
            portfolio.trade(xccy_swap, 1, pv, execution_currency=self.strategy.currency)

        # catch all cash flows between prev date and current date, i.e. (prev_date, curr_date]
        # cash_flows = {}
        # dt = self.time_stamp if is_first_day else state.prev_date + timedelta(days=1)
        # while dt <= self.time_stamp:
        #     mkt = self.strategy.backtest_market.get_market(dt)
        #     cf = portfolio.get_cash_flows(mkt)
        #     for ccy, amount in cf.items():
        #         cash_flows[ccy] = cash_flows.get(ccy, 0.) + amount
        #     dt += timedelta(days=1)
        cash_flows = portfolio.get_cash_flows(market)
        for ccy, amount in cash_flows.items():
            portfolio.add_position(Cash(ccy.value), amount)

        # value function first
        portfolio.price_at_market(market, fields='price')
        # add fx attribute
        def fx(tradable):
            if "currency" not in tradable.__dict__:
                fx_pair_name = f'USD{self.strategy.currency}'
            else:
                fx_pair_name = f'{tradable.currency}{self.strategy.currency}'
            fx = market.get_fx_spot(fx_pair_name)
            return {'fx': fx}
        portfolio.value_positions(fx)
        # aggregate pv in USD
        price = portfolio.aggregate(lambda x: getattr(x, 'price') * getattr(x, 'fx'))

        timer.end()

        spread_leg = self.parameters["xccy_swap_convention"].spread_leg
        market_spread = market.get_xccy_basis(spread_leg.ccy, self.parameters["swap_term"])
        details = {"model_spread": par_rate, "market_spread": market_spread, "model_spread_tminusone": None}

        return XccySwapState(self.time_stamp, portfolio, price, 0., details, self.time_stamp)


class XccySwapDaily(DailyStrategy):
    def __init__(self, start_date, end_date, calendar, currency, parameters, data_requests, force_run):
        if "swap_start_tenor" not in parameters:
            parameters["swap_start_tenor"] = "0D"
        else:
            parameters["swap_start_tenor"] = parameters["swap_start_tenor"].upper()

        super().__init__(start_date, end_date, calendar, currency, parameters, data_requests, force_run=force_run,
                         logging=False, log_file=None)

    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market

        self.validate()

    def validate(self):
        pass

    def generate_events(self, dt: datetime):
        events = []
        if self.parameters["swap_start_tenor"] == "0D":
            events.append(AccrualCashFlowEvent(dt, self))
        events += [ExpirationEvent(dt, self), EODTradeEvent(dt, self)]
        return events
