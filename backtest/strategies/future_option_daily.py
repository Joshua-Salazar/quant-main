from datetime import datetime

from ...backtest.unwind import BaseUnwindEvent
from ...backtest.costs import FlatTradingCost, FlatVegaCostStock, BidOfferCost
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.portfolio import Portfolio
from ...tradable.future import Future, MONTH_TO_MONTH_CODE, FUTURE_MONTH_CODE
from ...tradable.constant import Constant
from ...tradable.option import Option
from ...tradable.position import ValuedPosition
from ...dates.utils import add_tenor, add_business_days, bdc_adjustment, minus_tenor, date_to_datetime, tenor_to_days
from ...backtest.expires import ExpireOptionAtIntrinsic
from ...backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ...analytics.utils import EPSILON
from ...data.market import find_nearest_listed_options
from ...constants.business_day_convention import BusinessDayConvention


MIN_SIZING_MEASURE = {
    'price': 0.05,
    'vega': 0.005,
}

MIN_ENTRY_PRICE = 0.05,


# Helpers
def short_dated_option(portfolio: Portfolio):
    option_list = get_option_list(portfolio)
    return min(option_list, key=lambda opt: opt.expiration)


def get_option_list(portfolio: Portfolio):
    option_list = []
    for name, position in portfolio.net_positions().items():
        if isinstance(position.tradable, Option):
            option_list.append(position.tradable)
    return option_list


class FutureOptionState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost):
        self.price = price
        self.cost = cost
        super().__init__(time_stamp, portfolio)


class ExpirationEvent(Event):
    def execute(self, state: StrategyState):
        portfolio = state.portfolio.clone()
        # expire options
        self.parameters['expires'].expire(self.time_stamp, portfolio, cash_path=())
        return FutureOptionState(self.time_stamp, portfolio, 0.0, 0.0)


class UnwindEvent(BaseUnwindEvent):
    def execute(self, state: StrategyState):
        if not self.need_unwind:
            return state

        portfolio = self.unwind_pfo(target_tradable_type=Option, state=state)
        return FutureOptionState(self.time_stamp, portfolio, 0, 0)


class FutureOptionEntryEvent(Event):
    @staticmethod
    def parse_option_type(type_string):
        return type_string.upper()[0]

    def find_underlying_future(self, market, root, tgt_future_tenor, min_future_tenor):
        if tgt_future_tenor is None:
            dummy_exp = add_tenor(self.time_stamp, "1M")
            dummy_future = Future(root, 'USD', dummy_exp, 'CME', listed_ticker='', ivol_futures_id=None, expiry_month=None, expiry_year=None)
            return dummy_future

        future_universe = market.get_future_universe(root, return_as_dict=False)
        future_universe = future_universe[future_universe.expiration_date > self.time_stamp]
        if self.parameters.get('tenor_by_calendar_month', False):
            # finds tenors by calendar month instead of futures
            assert tgt_future_tenor[-1] == 'M' and min_future_tenor is None
            zero_month_row = future_universe[future_universe.expiration_date == future_universe.expiration_date.min()].iloc[0]
            dummy_ts = datetime(zero_month_row.expiration_year, zero_month_row.expiration_month_id, 1)
            target_year = add_tenor(dummy_ts, tgt_future_tenor).year
            target_month = add_tenor(dummy_ts, tgt_future_tenor).month
            if target_month not in future_universe.expiration_month_id.unique():
                target_month = min( [ x for x in future_universe.expiration_month_id.unique() if x >= target_month ] )
            future_universe = future_universe[(future_universe.expiration_month_id == target_month) &
                                              (future_universe.expiration_year == target_year)]
            assert len(future_universe) == 1
            used_row = future_universe.iloc[0]
        else:
            future_universe = future_universe if min_future_tenor is None else \
                future_universe[future_universe.expiration_date >= add_tenor(self.time_stamp, min_future_tenor)]
            tgt_expiration = add_tenor(self.time_stamp, tgt_future_tenor)
            used_row = future_universe.iloc[(future_universe.expiration_date - tgt_expiration).abs().argsort().iloc[0]]
        exp_month = FUTURE_MONTH_CODE[used_row.contract_month[-1]]
        yr_2digit = int(used_row.contract_month[:-1])
        exp_yr = 1900 + yr_2digit if yr_2digit >= 50 else 2000 + yr_2digit
        used_future = Future(root, 'USD', used_row.expiration_date, 'CME',  # TODO: actually look up currency & exchange
                             listed_ticker='', ivol_futures_id=used_row.futures_id,
                             expiry_month=exp_month, expiry_year=exp_yr)
        return used_future

    def find_option_to_trade(self, market, root, tgt_future_tenor, min_future_tenor=None, tgt_option_tenor=None,
                             cp=None, tgt_pct_strike=None, tgt_delta=None,
                             target_opt_exp=None,
                             return_empty=False):
        used_future = self.find_underlying_future(market, root, tgt_future_tenor, min_future_tenor)
        # find the option to trade
        option_universe = market.get_future_option_universe(used_future)
        #screen out 0 price options
        option_universe = option_universe[option_universe['price'] > MIN_ENTRY_PRICE]
        option_type = FutureOptionEntryEvent.parse_option_type(cp)
        option_universe = option_universe[option_universe.call_put == option_type]

        if return_empty and len( option_universe ) < 1:
            return option_universe

        if tgt_option_tenor is None:
            tgt_option_expiration = add_tenor(self.time_stamp, '100Y')
        else:
            # special handle short days expiry, e.g. 2D so we target 2 bd tenor to avoid weekends.
            if "D" in tgt_option_tenor:
                days = tenor_to_days(tgt_option_tenor)
                tgt_option_expiration = add_business_days(self.time_stamp, days)
            else:
                tgt_option_expiration = add_tenor(self.time_stamp, tgt_option_tenor)
        if target_opt_exp is not None:
            tgt_option_expiration = target_opt_exp
        selected_expiration = option_universe.iloc[(option_universe.expiration_date - tgt_option_expiration)
                                                   .abs().argsort().iloc[0]].expiration_date
        option_universe = option_universe[option_universe.expiration_date == selected_expiration]

        if tgt_pct_strike is not None:
            if tgt_future_tenor is None:
                future_id = option_universe.futures_id.unique()
                if future_id.shape[0] == 1:
                    future_id = option_universe.futures_id.unique()[0]
                    used_future.ivol_futures_id = future_id
                else:
                    raise Exception(f"Found multiple future id")
            future_price = used_future.price(market, calc_types='price', valuer=self.strategy.valuer_map[Future])
            target_strike = future_price * (tgt_pct_strike + EPSILON)
            used_row = option_universe.iloc[(option_universe.strike - target_strike).abs().argsort().iloc[0]]
        elif tgt_delta is not None:
            used_row = option_universe.iloc[(option_universe.delta - tgt_delta).abs().argsort().iloc[0]]
        else:
            raise ValueError('Need to specify strike or delta')

        option_to_trade = Option(root, used_future, 'USD', used_row.expiration_date, used_row.strike, option_type == 'C'
                                 , True, 1, "", )
        if option_to_trade.expiration == self.time_stamp:
            raise Exception(f"selected option {option_to_trade.name()} is expired on {used_row.expiration_date.strftime('%Y-%m-%d')}")
        return option_to_trade

    def find_vix_option_to_trade(self, market, root, tgt_future_tenor, min_future_tenor=None, tgt_option_tenor=None,
                             cp=None, tgt_pct_strike=None, tgt_delta=None):

        used_future = self.find_underlying_future(market, root, tgt_future_tenor, min_future_tenor)
        future_price = used_future.price(market, calc_types='price', valuer=self.strategy.valuer_map[Future])

        opt_root = 'VIX'
        option_universe = market.get_option_universe(opt_root, return_as_dict=False)
        if tgt_pct_strike is not None:
            selection_value = future_price * (tgt_pct_strike + EPSILON)
            selection_by = 'strike'
        elif tgt_delta is not None:
            selection_value = tgt_delta
            selection_by = 'delta'
        else:
            raise ValueError('Need to specify strike or delta')

        option_to_trade = find_nearest_listed_options(used_future.expiration, selection_value, cp, option_universe,
                                                      return_as_tradables=True,
                                                      select_by=selection_by)
        assert len(option_to_trade) >= 1
        # TODO: more than one option expires on same day we have to choose
        option_to_trade = option_to_trade[0]
        option_to_trade.underlying = used_future
        return option_to_trade

    def trade_one_option(self, portfolio, market, valuer, root, tenor, min_future_tenor, cp, sizing_measure,
                         sizing_target, position_path, pct_strike = None, delta = None, opt_tenor=None, tranche=None):
        if root == 'VX':
            # option data source uses VIX as root
            option_to_trade = self.find_vix_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                            cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta)
        else:
            option_to_trade = self.find_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                        cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta,
                                                        tgt_option_tenor=opt_tenor)
        option_price, option_sizing_measure_value = option_to_trade.price(market, valuer,
                                                                          calc_types=['price', sizing_measure])

        # sizing
        if sizing_measure == 'units':
            sizing_measure_value = 1
        elif sizing_measure == 'notional':
            sizing_measure_value = option_to_trade.underlying.price(market, calc_types='price',
                                                                    valuer=self.strategy.valuer_map[Future])
        elif sizing_measure == 'strike_notional':
            sizing_measure_value = option_to_trade.strike
        else:
            sizing_measure_value = option_sizing_measure_value
        units_to_trade = sizing_target / sizing_measure_value
        if tranche is not None:
            if isinstance(tranche, RollingAtExpiryTranche):
                tranche.set_next_roll_date(option_to_trade.expiration, self.strategy.holidays)
            elif isinstance(tranche, RollingAtExpiryDailyTranche):
                expiry_dt = date_to_datetime(option_to_trade.expiration.date())
                tranche.update_tranche(self.time_stamp, expiry_dt, self.strategy.holidays)

            units_to_trade *= tranche.get_tranche_fraction(self.time_stamp)

        portfolio.trade(option_to_trade, units_to_trade, option_price, option_to_trade.currency,
                        position_path=position_path, cash_path=())

    def option_trading_logic(self, market, portfolio):
        for leg_name, leg in self.parameters['legs'].items():
            # unwind tranches
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                leg_position_names = list(leg_portfolio.get_positions().keys())
                for tranche_name in leg_position_names:
                    tranche_portfolio = leg_portfolio.get_position(tranche_name)
                    if isinstance(tranche_portfolio, Portfolio) and tranche_name != "delta_hedge":
                        if leg['tranche'].get_exit_datetime(tranche_name) == self.time_stamp:
                            tranche_position_names = list(tranche_portfolio.get_positions().keys())
                            for option_position_name in tranche_position_names:
                                option_position = tranche_portfolio.get_position(option_position_name)
                                if isinstance(option_position.tradable, Option):
                                    unwind_price = option_position.tradable.price(market,
                                                                                  self.strategy.valuer_map[Option],
                                                                                  calc_types='price')
                                    portfolio.unwind((leg_name, tranche_name, option_position_name), unwind_price,
                                                     option_position.tradable.currency, cash_path=())
            # trade new tranches
            if leg['tranche'].is_entry_datetime(self.time_stamp) or self.trade_first_day:
                self.trade_one_option(
                    portfolio, market, self.strategy.valuer_map[Option],
                    leg['root'], leg['tenor'], leg.get('min_tenor'), leg['type'],
                    leg['sizing_measure'],
                    leg['sizing_target'],
                    position_path=(leg_name, self.time_stamp), pct_strike=leg.get('strike'), delta=leg.get('delta'),
                    opt_tenor=leg.get('target_option_tenor'),
                    tranche=leg["tranche"]
                )
        return portfolio

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()
        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # allow trading on start_day
        start_date = bdc_adjustment(self.strategy.start_date, convention=BusinessDayConvention.FOLLOWING, holidays=self.strategy.holidays)
        self.trade_first_day = (self.time_stamp == start_date) and self.parameters['trade_first_day']

        # use stale data to forward fill missing price
        backfill_markets = []
        for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
            back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
            if back_date >= self.strategy.start_date:
                backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
        self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        # trade if needed here
        portfolio = self.option_trading_logic(market, portfolio)

        # value portfolio then apply costs then price portfolio
        portfolio.value_positions_at_market(market, fields=['price','delta', 'vega', 'bid', 'ask'],
                                            valuer_map_override=self.strategy.valuer_map)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp,
                                                  market=market, valuer_map=self.strategy.valuer_map)

        price, vega = portfolio.price_at_market(market, fields=['price', 'vega'],
                                                valuer_map_override=self.strategy.valuer_map)

        return FutureOptionState(self.time_stamp, portfolio, price, 0.0)


class ExpiryFutureOptionEntryEvent(FutureOptionEntryEvent):
    # options all be directly in leg portfolio, no tranching within a leg
    def option_trading_logic(self, market: BMarket, portfolio: Portfolio):
        for leg_name, leg in self.parameters['legs'].items():
            days_before_expiry = leg['days_before_expiry']
            # unwind tranches
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                net_positions = leg_portfolio.net_positions()
                for position_name, position in net_positions.items():
                    if isinstance(position.tradable, Option):
                        next_roll_time = add_business_days(position.tradable.expiration, -days_before_expiry,
                                                           self.strategy.holidays)
                        if next_roll_time == self.time_stamp:
                            unwind_price = position.tradable.price(market, self.strategy.valuer_map[Option],
                                                                   calc_types='price')
                            portfolio.unwind((leg_name, position_name), unwind_price,
                                             position.tradable.currency, cash_path=())
                            if leg['root'] == 'CO' and self.time_stamp > datetime( 2015, 1, 8 )\
                                    and self.time_stamp < datetime( 2015, 2, 2 ):
                                continue

                            if leg['root'] == 'CO' and self.time_stamp > datetime( 2014, 12, 9 ) \
                                    and self.time_stamp < datetime( 2015, 2, 2 ) and \
                                    leg['target_option_tenor'] == '3M':
                                continue

                            # trade new tranches
                            self.trade_one_option(
                                portfolio, market, self.strategy.valuer_map[Option],
                                leg['root'], leg['tenor'], leg.get('min_tenor'), leg['type'], leg['sizing_measure'],
                                leg['sizing_target'], position_path=(leg_name,), pct_strike=leg.get('strike'),
                                delta=leg.get('delta'), opt_tenor=leg.get('target_option_tenor'))

                    elif isinstance(position.tradable, Future ) and leg['root'] == 'CO' and \
                            self.time_stamp > datetime(2015, 1, 8) and self.time_stamp < datetime(2015, 2, 1 ) :
                        unwind_price = position.tradable.price(market, self.strategy.valuer_map[Future],
                                                               calc_types='price')
                        portfolio.unwind((leg_name, position_name), unwind_price,
                                         position.tradable.currency, cash_path=())

                    elif isinstance(position.tradable, Future ) and leg['root'] == 'CO' and \
                            self.time_stamp > datetime(2014, 12, 9) and self.time_stamp < datetime(2015, 2, 1 ) \
                            and leg['target_option_tenor'] == '3M':
                        unwind_price = position.tradable.price(market, self.strategy.valuer_map[Future],
                                                               calc_types='price')
                        portfolio.unwind((leg_name, position_name), unwind_price,
                                         position.tradable.currency, cash_path=())


            if self.trade_first_day or \
                ( leg['root'] == 'CO' and len( leg_portfolio.find_children_of_tradable_type_recursive( Option ) ) == 0
                and self.time_stamp > datetime( 2015, 2, 1 ) ):
                self.trade_one_option(
                    portfolio, market, self.strategy.valuer_map[Option],
                    leg['root'], leg.get('initial_tenor', leg['tenor']), leg.get('min_tenor'), leg['type'], leg['sizing_measure'],
                    leg['sizing_target'], position_path=(leg_name,), pct_strike=leg.get('strike'),
                    delta=leg.get('delta'), opt_tenor=leg.get('target_option_tenor'))
        return portfolio


class PairFutureOptionEntryEvent(FutureOptionEntryEvent):
    # unwind pairs trade portfolios at the minimum expiry of the two legs
    def option_trading_logic(self, market: BMarket, portfolio: Portfolio):

        options = portfolio.find_children_of_tradable_type_recursive( Option )
        if len(options) > 0:
            assert len(options) == len(self.parameters['legs'])
            min_exp = min([o[1].tradable.expiration for o in options])
        roots = list(set([x['root'] for x in list( self.parameters['legs'].values())]))
        assert len( roots ) == 1
        root = roots[0]
        if root != 'TY':
            for leg_name, leg in self.parameters['legs'].items():
                days_before_expiry = leg['days_before_expiry']
                # unwind tranches
                leg_portfolio = portfolio.get_position(leg_name)
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for position_name in leg_position_names:
                        position = leg_portfolio.get_position(position_name)
                        if isinstance(position.tradable, Option):
                            next_roll_time = add_business_days( min_exp, -days_before_expiry,
                                                               self.strategy.holidays)
                            if next_roll_time == self.time_stamp:
                                unwind_price = position.tradable.price(market, self.strategy.valuer_map[Option],
                                                                       calc_types='price')
                                portfolio.unwind((leg_name, position_name), unwind_price,
                                                 position.tradable.currency, cash_path=())
                                # trade new tranches
                                self.trade_one_option(
                                    portfolio, market, self.strategy.valuer_map[Option],
                                    leg['root'], leg['tenor'], leg.get('min_tenor'), leg['type'], leg['sizing_measure'],
                                    leg['sizing_target'], position_path=(leg_name,), pct_strike=leg.get('strike'),
                                    delta=leg.get('delta'), opt_tenor=leg.get('target_option_tenor'))
                if self.trade_first_day:
                    self.trade_one_option(
                        portfolio, market, self.strategy.valuer_map[Option],
                        leg['root'], leg['tenor'], leg.get('min_tenor'), leg['type'], leg['sizing_measure'],
                        leg['sizing_target'], position_path=(leg_name,), pct_strike=leg.get('strike'),
                        delta=leg.get('delta'), opt_tenor=leg.get('target_option_tenor'))
        else:

            opt_exp = [ x['target_option_tenor'] for x in list( self.parameters['legs'].values()) ]
            opt_exp = [ int(x[:-1]) for x in opt_exp ]
            max_ten = '%dM' %max( opt_exp )
            min_ten = '%dM' %min( opt_exp )

            tenor_to_name = { }
            for leg_name, leg in self.parameters['legs'].items():
                tenor_to_name[leg['target_option_tenor']] = { 'leg_name': leg_name,
                                                               'size' : leg['sizing_target'] }

            for leg_name, leg in self.parameters['legs'].items():
                days_before_expiry = leg['days_before_expiry']
                # unwind tranches
                leg_portfolio = portfolio.get_position(leg_name)
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for position_name in leg_position_names:
                        position = leg_portfolio.get_position(position_name)
                        if isinstance(position.tradable, Option) or isinstance(position.tradable, Future):
                            next_roll_time = add_business_days( min_exp, -days_before_expiry,
                                                               self.strategy.holidays)
                            if next_roll_time == self.time_stamp:
                                unwind_price = position.tradable.price(market,
                                                                       self.strategy.valuer_map[type(position.tradable)],
                                                                       calc_types='price')
                                portfolio.unwind((leg_name, position_name), unwind_price,
                                                 position.tradable.currency, cash_path=())

            # trade new positions
            if self.trade_first_day or \
                    len( portfolio.find_children_of_tradable_type_recursive( Option ) ) < 1:
                self.trade_one_pair(
                    portfolio, market, self.strategy.valuer_map[Option],
                    root, leg['tenor'], leg.get('min_tenor'), leg['type'], leg['sizing_measure'],
                    min_sizing_target = tenor_to_name[min_ten]['size'],
                    max_sizing_target = tenor_to_name[max_ten]['size'],
                    pct_strike=leg.get('strike'), delta=leg.get('delta'),
                    min_position_path=(tenor_to_name[min_ten]['leg_name'],),
                    max_position_path=(tenor_to_name[max_ten]['leg_name'],),
                    min_opt_tenor=min_ten,
                    max_opt_tenor=max_ten,
                    opts_to_ret = len(self.parameters['legs']) )

        return portfolio

    def trade_one_pair(self, portfolio, market, valuer,
                       root, tenor, min_future_tenor, cp, sizing_measure,
                       min_sizing_target,
                       max_sizing_target,
                       min_position_path,
                       max_position_path,
                       pct_strike=None, delta=None,
                       max_opt_tenor=None, min_opt_tenor=None,
                       opts_to_ret = 2 ):
        if root == 'VX':
            # option data source uses VIX as root
            option_to_trade = self.find_vix_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                            cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta,)
        else:
            option_to_trade = self.find_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                        cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta,
                                                        tgt_option_tenor=min_opt_tenor,
                                                        return_empty = True )
        if not isinstance( option_to_trade, Option ):
            return 1

        option_price, option_sizing_measure_value = option_to_trade.price(market, valuer,
                                                                          calc_types=['price', sizing_measure])

        # sizing
        if sizing_measure == 'units':
            sizing_measure_value = 1
        elif sizing_measure == 'notional':
            sizing_measure_value = option_to_trade.underlying.price(market, calc_types='price',
                                                                    valuer=self.strategy.valuer_map[Future])
        elif sizing_measure == 'strike_notional':
            sizing_measure_value = option_to_trade.strike
        else:
            sizing_measure_value = option_sizing_measure_value
        units_to_trade = min_sizing_target / sizing_measure_value

        if opts_to_ret == 1:
            portfolio.trade(option_to_trade, units_to_trade, option_price, option_to_trade.currency,
                            position_path=min_position_path, cash_path=())
        else:

            tgt_exp = minus_tenor( add_tenor( option_to_trade.expiration, max_opt_tenor ), min_opt_tenor )
            if root == 'VX':
                # option data source uses VIX as root
                option_to_trade2 = self.find_vix_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                                cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta)
            else:
                option_to_trade2 = self.find_option_to_trade(market, root, tenor, min_future_tenor=min_future_tenor,
                                                            cp=cp, tgt_pct_strike=pct_strike, tgt_delta=delta,
                                                            tgt_option_tenor=max_opt_tenor,
                                                            target_opt_exp = tgt_exp,
                                                            return_empty = True )

            if (option_to_trade.expiration >= option_to_trade2.expiration) or ( not isinstance( option_to_trade, Option ) ):
                return 1
            else:
                portfolio.trade(option_to_trade, units_to_trade, option_price, option_to_trade.currency,
                                position_path=min_position_path, cash_path=())

                option_price, option_sizing_measure_value = option_to_trade2.price(market, valuer,
                                                                                  calc_types=['price', sizing_measure])

                # sizing
                if sizing_measure == 'units':
                    sizing_measure_value = 1
                elif sizing_measure == 'notional':
                    sizing_measure_value = option_to_trade2.underlying.price(market, calc_types='price',
                                                                            valuer=self.strategy.valuer_map[Future])
                elif sizing_measure == 'strike_notional':
                    sizing_measure_value = option_to_trade2.strike
                else:
                    sizing_measure_value = option_sizing_measure_value
                units_to_trade = max_sizing_target / sizing_measure_value
                portfolio.trade( option_to_trade2, units_to_trade, option_price, option_to_trade2.currency,
                                position_path=max_position_path, cash_path=())





class FutureOptionDeltaHedgeEvent(Event):
    @staticmethod
    def find_front_future(market, portfolio):
        pass

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # use stale data to forward fill missing price
        backfill_markets = []
        for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
            back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
            if back_date >= self.strategy.start_date:
                backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
        self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        for leg_name in list(self.parameters['legs'].keys()):
            hedge_flag = self.parameters['legs'][leg_name].get('hedge')
            if hedge_flag is None or hedge_flag is False:
                continue

            # hedge future
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is None:
                continue

            held_options = []
            for k, v in leg_portfolio.net_positions().items():
                if isinstance(v.tradable, Option):
                    held_options.append(v.tradable)
            if not len(held_options):
                continue
            if held_options[0].underlying == 'VIX':
                pass
            else:
                hedge_instrument = min(held_options, key=lambda t: t.expiration).underlying

            hedge_instrument_price, hedge_instrument_delta = \
                hedge_instrument.price(market, self.strategy.valuer_map[Future], calc_types=['price', 'delta'])

            # first unwind any hedge that is not the selected hedge future
            if leg_portfolio is not None:
                delta_pfo = leg_portfolio.get_position("delta_hedge")
                if delta_pfo is not None:
                    net_positions = delta_pfo.net_positions()
                    for position_name, position in net_positions.items():
                        if isinstance(position.tradable, Future) and not position.tradable == hedge_instrument:
                            unwind_price = position.tradable.price(market, self.strategy.valuer_map[Future],
                                                                   calc_types='price')
                            delta_pfo.unwind(position_name, unwind_price, position.tradable.currency)

                # then hedge the portfolio delta with front future
                leg_portfolio.price_at_market(market, 'delta', self.strategy.valuer_map, default=0)
                portfolio_delta = leg_portfolio.aggregate('delta')

                if portfolio_delta != portfolio_delta:
                    print('did not hedge ' + leg_name + ' on ' + self.time_stamp.strftime('%Y-%m-%d'))
                    continue

                # all hedges are at the root
                leg_portfolio.trade(hedge_instrument, -portfolio_delta / hedge_instrument_delta, hedge_instrument_price, hedge_instrument.currency, position_path=("delta_hedge",))

        # value portfolio then apply costs then price portfolio
        portfolio.value_positions_at_market(market, fields=['price', 'vega'],
                                            valuer_map_override=self.strategy.valuer_map)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        price, vega = portfolio.price_at_market(market, fields=['price', 'vega'],
                                                valuer_map_override=self.strategy.valuer_map)

        return FutureOptionState(self.time_stamp, portfolio, price, 0.0)


class MTMEvent(Event):
    def execute(self, state: StrategyState):
        portfolio = state.portfolio.clone()
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        price = portfolio.price_at_market(market, fields=['price', 'delta', 'vega', 'theta'], valuer_map_override=self.strategy.valuer_map)
        return FutureOptionState(self.time_stamp, portfolio, price[0], 0.0)


class FutureOptionDaily(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        self.valuer_map = self.parameters['valuer_map']

        #self.force_run = True

        self.parameters['expires'] = ExpireOptionAtIntrinsic(self.backtest_market)
        # add trading costs
        if 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters['flat_costs']
            self.parameters['trading_cost'] = FlatTradingCost(cost_params['tc_delta'], cost_params['tc_vega'],
                                                              cost_params.get('per_unit', False),
                                                              cost_params.get('free_unwind', False))
        elif 'flat_vega_charge' in self.parameters.keys():
            cost_params = self.parameters['flat_vega_charge']
            self.parameters['trading_cost'] = FlatVegaCostStock( cost_params['tc_delta'], cost_params['tc_vega'] )
        elif 'bid_ask_charge' in self.parameters.keys():
            cost_params = self.parameters['bid_ask_charge']
            self.parameters['trading_cost'] = BidOfferCost( cost_params['tc_delta'], cost_params['tc_vega'],
                                                            cost_params.get('use_otm', False),
                                                            cost_params.get('free_unwind', False))

    def generate_events(self, dt: datetime):
        if self.parameters.get('roll_style') == 'expiry':
            exp_style = ExpiryFutureOptionEntryEvent(dt, self)
        elif self.parameters.get('roll_style') == 'pair':
            exp_style = PairFutureOptionEntryEvent(dt, self)
        else:
            exp_style = FutureOptionEntryEvent(dt, self)
        return [ExpirationEvent(dt, self),
                UnwindEvent(dt, self),
                exp_style,
                FutureOptionDeltaHedgeEvent(dt, self),
                MTMEvent(dt, self)]
