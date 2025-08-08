from datetime import datetime
import math
import pandas as pd
from ...analytics.symbology import currency_from_option_root, option_root_from_ticker
from ...analytics.utils import EPSILON
from ...analytics.utils import float_equal
from ...backtest import hedge_utils
from ...backtest.corpactions import CorpActionsProcessor
from ...backtest.costs import FlatTradingCost, FlatVegaCostStock, BidOfferCost
from ...backtest.expires import ExpireContractsAtPrice, ExpireOptionAtIntrinsic
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...backtest.tranche import Tranche, RollingTrancheExpiryOffset
from ...backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ...backtest.unwind import BaseUnwindEvent
from ...constants.business_day_convention import BusinessDayConvention
from ...data.market import find_nearest_listed_options
from ...dates.utils import add_tenor, add_business_days, bdc_adjustment, minus_tenor, date_to_datetime, tenor_to_days
from dateutil.relativedelta import relativedelta, FR
from ...infrastructure.bmarket import BMarket
from ...tradable.future import Future
from ...tradable.cash import Cash
from ...tradable.constant import Constant
from ...tradable.option import Option
from ...tradable.stock import Stock
from ...tradable.portfolio import Portfolio


MIN_SIZING_MEASURE = {
    'price': 0.05,
    'vega': 0.005,
    'theta': 0.0001
}


class StockOptionsDailyState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, portfolio_before_trades=None):
        self.price = price
        self.cost = cost
        self.portfolio_before_trades = portfolio_before_trades
        super().__init__(time_stamp, portfolio)


class CorpActionEvent(Event):
    def execute(self, state: StrategyState, daily_states = [ ]):
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # process corp actions
        self.parameters['corpaction_processor'].apply(self.time_stamp, portfolio, market, self.parameters.get('use_listed', True))

        return StockOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)


class ExpirationEvent(Event):
    def execute(self, state: StrategyState, daily_states = [ ]):
        portfolio = state.portfolio.clone()
        # expire options of each leg, cash in each leg portfolio
        for leg_name, leg in self.parameters['legs'].items():
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                self.parameters['expires'].expire(self.time_stamp, leg_portfolio)
        return StockOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)


class UnwindEvent(BaseUnwindEvent):
    def execute(self, state: StrategyState):
        if not self.need_unwind:
            return state

        portfolio = self.unwind_pfo(target_tradable_type=Option, state=state)
        return StockOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)


def make_tranche_key(entry_date, exit_date):
    if exit_date is None:
        exit_date = datetime.max
    return f"{entry_date.strftime('%Y-%m-%d')}_{exit_date.strftime('%Y-%m-%d')}"


def parse_tranche_key(tranche_name):
    elements = tranche_name.split('_')
    entry_str = elements[0]
    exit_str = elements[1]
    return datetime.strptime(entry_str, '%Y-%m-%d'), datetime.strptime(exit_str, '%Y-%m-%d')


class OptionEntryEvent(Event):
    def __init__(self, dt, strategy):
        super().__init__(dt, strategy)
        self.trade_first_day = None

    @staticmethod
    def parse_option_type(type_string):
        return type_string.upper()[0]

    @staticmethod
    def select_listed_option(market, root, sizing_measure, other_filters, target_expiration, selection_value, selection_by, option_type, expiration_search_method='absolute'):
        option_universe = market.get_option_universe(root, return_as_dict=False)
        if sizing_measure not in ['units', 'notional', 'strike_notional', None]:
            filters = other_filters + [lambda x: x[abs(x[sizing_measure]) >= MIN_SIZING_MEASURE.get(sizing_measure, EPSILON)]]
        else:
            filters = other_filters

        option_to_trade = find_nearest_listed_options(target_expiration, selection_value, option_type, option_universe,
                                                      other_filters=filters,
                                                      return_as_tradables=True,
                                                      select_by=selection_by,
                                                      expiration_search_method=expiration_search_method)
        assert len(option_to_trade) >= 1
        # TODO: more than one option expires on same day we have to choose
        option_to_trade = option_to_trade[0]
        return option_to_trade

    @staticmethod
    def option_selection_simple(
            # generic input
            strategy, strategy_state,
            # underlying
            underlying,
            # listed or otc
            use_listed,
            # type
            option_type,
            # strike selection
            selection_by, selection_value,
            # expiration selection
            tenor,
            # expiration selection overrides
            override_target_expiry=None, force_3rd_Fri=False,
            # additional filter on option selection
            other_filters=[],
            # sizing measure to filter selection
            sizing_measure_to_filter_selection=None,
            spread_strike = None
    ):
        time_stamp = strategy_state.time_stamp
        market = strategy.backtest_market.get_market(time_stamp)
        spot = market.get_spot(underlying)
        valuer = strategy.valuer_map[Option]

        root = option_root_from_ticker(underlying)

        # find the option to trade
        if tenor.endswith('+'):
            tenor_used = tenor[:-1]
            expiration_search_method = 'geq'
        elif tenor.endswith('-'):
            tenor_used = tenor[:-1]
            expiration_search_method = 'leq'
        else:
            tenor_used = tenor
            expiration_search_method = 'absolute'

        target_expiration = add_tenor(market.base_datetime, tenor_used)
        if force_3rd_Fri:
            fri3 = target_expiration.replace(day=1)
            fri3 = fri3 + relativedelta(weekday=FR(3))
            target_expiration = fri3

        option_type = OptionEntryEvent.parse_option_type(option_type)

        if override_target_expiry is not None:
            target_expiration = override_target_expiry
            #expiration_search_method = 'absolute'


        if selection_by == 'strike':
            target_strike = spot * selection_value
            if use_listed:
                option_to_trade = OptionEntryEvent.select_listed_option(market, root, sizing_measure_to_filter_selection, other_filters, target_expiration, target_strike, selection_by, option_type, expiration_search_method=expiration_search_method)
            else:
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration, target_strike, option_type == 'C', False, 1, '')
        elif selection_by == 'delta':
            if use_listed:
                option_to_trade = OptionEntryEvent.select_listed_option(market, root, sizing_measure_to_filter_selection, other_filters, target_expiration, selection_value, selection_by, option_type, expiration_search_method=expiration_search_method)
            else:
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration, spot, option_type == 'C', False, 1, '')
                strike = valuer.solve(option_to_trade, market, given='delta', solve_for='strike', value_given=selection_value)
                option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration, strike, option_type == 'C', False, 1, '')
        else:
            raise RuntimeError(f"Option selection by {selection_by} has not been implemented")

        if spread_strike is None:
            return [option_to_trade]
        else:
            if selection_by == 'strike':
                target_strike = spot * spread_strike
                if use_listed:
                    sprd_option_to_trade = OptionEntryEvent.select_listed_option(market, root,
                                                                            sizing_measure_to_filter_selection,
                                                                            other_filters, target_expiration,
                                                                            target_strike, selection_by, option_type,
                                                                            expiration_search_method=expiration_search_method)
                else:
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                             target_strike, option_type == 'C', False, 1, '')
            elif selection_by == 'delta':
                if use_listed:
                    sprd_option_to_trade = OptionEntryEvent.select_listed_option(market, root,
                                                                            sizing_measure_to_filter_selection,
                                                                            other_filters, target_expiration,
                                                                            spread_strike, selection_by, option_type,
                                                                            expiration_search_method=expiration_search_method)
                else:
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                             spot, option_type == 'C', False, 1, '')
                    strike = valuer.solve(sprd_option_to_trade, market, given='delta', solve_for='strike',
                                          value_given=spread_strike)
                    sprd_option_to_trade = Option(underlying, underlying, currency_from_option_root(root), target_expiration,
                                             strike, option_type == 'C', False, 1, '')
            else:
                raise RuntimeError(f"Option selection by {selection_by} has not been implemented")

            return [option_to_trade, sprd_option_to_trade ]


    @staticmethod
    def option_sizing_simple(
            # generic input
            strategy, strategy_state,
            # option selected
            options_to_trade,
            # sizing
            sizing_measure, sizing_target,
    ):
        assert len(options_to_trade) == 1
        option_to_trade = options_to_trade[0]

        time_stamp = strategy_state.time_stamp
        market = strategy.backtest_market.get_market(time_stamp)
        underlying = option_to_trade.underlying
        spot = market.get_spot(underlying)
        valuer = strategy.valuer_map[Option]

        if sizing_measure in ['units', 'notional', 'strike_notional']:
            bid, ask, option_price = option_to_trade.price(market, valuer, calc_types=['bid', 'ask', 'price'])
        else:
            bid, ask, option_price, option_sizing_measure_value = option_to_trade.price(market, valuer, calc_types=['bid', 'ask', 'price', sizing_measure])

        # sizing
        if sizing_measure == 'units':
            sizing_measure_value = 1
        elif sizing_measure == 'notional':
            sizing_measure_value = spot
        elif sizing_measure == 'strike_notional':
            sizing_measure_value = option_to_trade.strike
        else:
            sizing_measure_value = option_sizing_measure_value
        units_to_trade = sizing_target / sizing_measure_value

        return [(units_to_trade, option_price)]

    @staticmethod
    def trade_one_leg(tranche_name_has_to_be_new,
                      portfolio, tranche_fraction,
                      strategy, strategy_state,
                      option_selection_function,
                      option_sizing_function,
                      position_path,
                      scale_by_nav = None,
                      spread_strike = None):
        ts = strategy_state.time_stamp
        leg_name = position_path[0]
        tranche_name = position_path[1]

        if spread_strike is None:
            options_selected_list = option_selection_function(leg_name, tranche_name, strategy, strategy_state)
            option_sizing_list = option_sizing_function(leg_name, tranche_name, strategy, strategy_state, options_selected_list)
        else:
            # right now spread strike is not none only works when there is no customized option selection
            # TODO: we should deprecate the case of spread_strike but rather specify two legs
            options_selected_list = option_selection_function(leg_name, tranche_name, strategy, strategy_state, spread_strike)
            option_sizing_list = option_sizing_function(leg_name, tranche_name, strategy, strategy_state, [options_selected_list[0]])
            sprd_option_sizing_list = option_sizing_function(leg_name, tranche_name, strategy, strategy_state, [options_selected_list[1]])
            option_sizing_list = option_sizing_list + [(-option_sizing_list[0][0],sprd_option_sizing_list[0][1])]

        if scale_by_nav is not None:
            option_sizing_list = [ (x[0] * scale_by_nav, x[1]) for x in option_sizing_list ]

        time_stamp = strategy_state.time_stamp
        market = strategy.backtest_market.get_market(time_stamp)
        valuer = strategy.valuer_map[Option]

        def _build_options_to_trade_list(_x, _y):
            if isinstance(_y, tuple):
                assert len(_y) == 2
                _units = _y[0]
                _execution_price = _y[1]
            else:
                _units = _y
                _execution_price = _x.price(market, valuer, calc_types='price')
            return _x, _units, _execution_price

        option_to_trade_list = [_build_options_to_trade_list(x, y) for x, y in zip(options_selected_list, option_sizing_list)]

        # if isinstance(position_path[1], RollingTrancheExpiryOffset):
        #     min_exp = option_to_trade_list[0][0].expiration
        #     for o, u, e in option_to_trade_list:
        #         if o.expiration < min_exp:
        #             min_exp = o.expiration
        #     unwind_date = position_path[1].get_roll_date(expiry=min_exp)
        #     position_path = (position_path[0], make_tranche_key(ts, unwind_date))

        if tranche_name_has_to_be_new:
            # if tranching method is regular tranche, then test if the new trade tranche name already exists
            # this shouldn't happen usually but in edge cases where different tranches roll/enter at same day, this could happen
            original_key = position_path[-1]
            count = 0
            while portfolio.get_position(position_path) is not None:
                count += 1
                print(f"On {ts.strftime('%Y-%m-%d')} the new tranche {position_path} name already exists")
                position_path = position_path[:-1] + (f"{original_key}_{count}",)
                print(f"We are making it {position_path}")

        for o, u, p in option_to_trade_list:
            portfolio.trade(
                o, u * tranche_fraction, p, o.currency, position_path=position_path,
                # cash_path=(position_path[0],)
            )

    @staticmethod
    def unwind_tranche_portfolio(leg_name, tranche_name, portfolio, leg_portfolio, tranche_portfolio, market, strategy, parameters):
        tranche_position_names = list(tranche_portfolio.get_positions().keys())
        tranche_position_names = list(filter(lambda x: not isinstance(tranche_portfolio.get_position(x).tradable, Cash) and not isinstance(tranche_portfolio.get_position(x).tradable, Constant), tranche_position_names))

        for position_name in tranche_position_names:
            position = tranche_portfolio.get_position(position_name)
            if parameters.get('keep_hedges_in_tranche_portfolio', False):
                unwind_price = position.tradable.price(market, strategy.valuer_map[type(position.tradable)],
                                                       calc_types='price')
                portfolio.unwind((leg_name, tranche_name, position_name), unwind_price, position.tradable.currency,
                                 cash_path=(leg_name, tranche_name,))
            else:
                unwind_price = position.tradable.price(market, strategy.valuer_map[Option], calc_types='price')
                portfolio.unwind((leg_name, tranche_name, position_name), unwind_price, position.tradable.currency,
                                 cash_path=(leg_name,))

    @staticmethod
    def move_tranche_portfolio_with_cash_only_to_leg_level(tranche_name, leg_portfolio, tranche_portfolio):
        # move tranche portfolio with cash only to leg level
        pos_list = list(tranche_portfolio.get_positions().items())
        cash_only = all(
            [isinstance(x[1].tradable, Cash) or isinstance(x[1].tradable, Constant) for x in pos_list])
        if cash_only:
            for cash_pos in pos_list:
                if isinstance(cash_pos[1].tradable, Cash) or isinstance(cash_pos[1].tradable, Constant):
                    leg_portfolio.move(
                        cash_pos[1].tradable,
                        cash_pos[1].quantity,
                        (tranche_name,), ()
                    )
                else:
                    raise RuntimeError(f"found non cash in an unwound tranche portfolio")

    @staticmethod
    def trade_tranche(tranche_name_has_to_be_new, leg, leg_name, new_tranche_name, portfolio, tranche_fraction, selection_by, selection_value,
                      state, parameters, strategy, prev_nav, target_exp_override):
        if "option_selection_function" in leg:
            OptionEntryEvent.trade_one_leg(
                tranche_name_has_to_be_new,
                # portfolio obj to trade in, and fraction due to tranche
                portfolio,
                tranche_fraction,
                # the following three define the option selection and sizing
                strategy,
                state,
                leg['option_selection_function'],
                leg['option_sizing_function'],
                # other trading parameters
                position_path=(leg_name, new_tranche_name),
                scale_by_nav=prev_nav if ('scale_by_nav' in parameters.keys()
                                          and parameters['scale_by_nav']) else None,
                spread_strike=leg['spread_strike'] if 'spread_strike' in leg else None,
            )
        else:
            spread_stk = leg['spread_strike'] if 'spread_strike' in leg else None
            OptionEntryEvent.trade_one_leg(
                tranche_name_has_to_be_new,
                # portfolio obj to trade in, and fraction due to tranche
                portfolio,
                tranche_fraction,
                # the following four define the option selection and sizing
                strategy,
                state,
                lambda _leg_name, _tranche_name, _strategy, _state, spread_stk=None:
                OptionEntryEvent.option_selection_simple(
                    _strategy, _state,
                    leg['underlying'],
                    parameters.get('use_listed', True),
                    leg['type'],
                    selection_by, selection_value,
                    leg['tenor'],
                    target_exp_override,
                    'force_3rd_Fri' in leg,
                    leg.get('other_filters', []),
                    leg['sizing_measure'],
                    spread_strike=spread_stk
                ),
                lambda _leg_name, _tranche_name, _strategy, _state, _options:
                OptionEntryEvent.option_sizing_simple(
                    _strategy, _state, _options,
                    leg['sizing_measure'], leg['sizing_target'],
                ),
                # other trading parameters
                position_path=(leg_name, new_tranche_name),
                scale_by_nav=prev_nav if ('scale_by_nav' in parameters.keys()
                                          and parameters['scale_by_nav']) else None,
                spread_strike=leg['spread_strike'] if 'spread_strike' in leg else None,
            )

    @staticmethod
    def is_tranche_roll_date_from_expiry(_leg_name, _tranche_name, _strategy, _state):
        _tranche_portfolio = _state.portfolio.get_position(_leg_name).get_position(_tranche_name)
        tranche_position_names = list(_tranche_portfolio.get_positions().keys())
        min_exp = datetime.max
        for position_name in tranche_position_names:
            position = _tranche_portfolio.get_position(position_name)
            if isinstance(position.tradable, Option):
                if position.tradable.expiration < min_exp:
                    min_exp = position.tradable.expiration
        return _strategy.parameters['legs'][_leg_name]['tranche'].get_roll_date(expiry=min_exp).date() == _state.time_stamp.date()

    def trade_one_option(self, strategy, state, portfolio, underlying, use_listed, tenor, cp,
                         sizing_measure, sizing_target, selection_by, selection_value,
                         position_path, tranche=None):

        option_to_trade = \
            self.option_selection_simple(strategy, state, underlying, use_listed, cp, selection_by, selection_value,
                                         tenor, override_target_expiry=None, force_3rd_Fri=False, other_filters=[],
                                         sizing_measure_to_filter_selection=sizing_measure, spread_strike=None)[0]

        units_to_trade, option_price = \
            self.option_sizing_simple(strategy, state, [option_to_trade], sizing_measure, sizing_target)[0]

        if tranche is not None:
            if isinstance(tranche, RollingAtExpiryTranche):
                tranche.set_next_roll_date(option_to_trade.expiration, self.strategy.holidays)

            units_to_trade *= tranche.get_tranche_fraction(self.time_stamp)

        portfolio.trade(option_to_trade, units_to_trade, option_price, option_to_trade.currency,
                        position_path=position_path, cash_path=())

    def execute(self, state: StrategyState, daily_states=[]):
        #########################
        # Preparation
        #########################

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        start_date = bdc_adjustment(self.strategy.start_date, convention=BusinessDayConvention.FOLLOWING,
                                    holidays=self.strategy.holidays)
        self.trade_first_day = (self.time_stamp == start_date) and self.parameters['trade_first_day']

        if len(daily_states) > 0:
            prev_nav = daily_states[-1].price
        else:
            prev_nav = None

        # price before any trade
        portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        portfolio_before_trades = portfolio.clone()

        # use stale data to forward fill missing price
        if self.parameters.get('use_listed', True):
            backfill_markets = []
            for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
                back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
                if back_date >= self.strategy.start_date:
                    backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
            self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        #########################
        # Option Trading Per Leg
        #########################

        for leg_name, leg in self.parameters['legs'].items():
            # if there is no customize function we must have selection method as leg parameters
            if "option_selection_function" not in leg:
                if 'selection_by' in leg:
                    selection_by = leg['selection_by']
                elif 'strike' in leg:
                    selection_by = 'strike'
                elif 'delta' in leg:
                    selection_by = 'delta'
                else:
                    raise RuntimeError(
                        "Cannot find option selection method. In the leg config please specify 'selection_by' and the corresponding measure.")
                selection_value = leg[selection_by]
            else:
                selection_by = None
                selection_value = None

            # regular tranche
            if isinstance(leg['tranche'], Tranche):
                # unwind tranches and move cash into leg level to avoid keeping too many tranche portfolios
                unwinds_today = []
                leg_portfolio = portfolio.get_position(leg_name)
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        if isinstance(tranche_portfolio, Portfolio):
                            # unwind portfolio when it is the unwind date
                            if parse_tranche_key(tranche_name)[1].date() == self.time_stamp.date():
                                unwinds_today.append(tranche_name)
                                OptionEntryEvent.unwind_tranche_portfolio(leg_name, tranche_name, portfolio, leg_portfolio, tranche_portfolio, market, self.strategy, self.parameters)
                                OptionEntryEvent.move_tranche_portfolio_with_cash_only_to_leg_level(tranche_name, leg_portfolio, tranche_portfolio)

                # decide new tranche dates
                if leg['tranche'].is_entry_datetime(self.time_stamp):
                    # TODO: remove the case where the exit datetime can be None by making the tranche class logic take care of future exit dates
                    if isinstance(leg['tranche'].get_exit_datetime(self.time_stamp), datetime) or leg['tranche'].get_exit_datetime(self.time_stamp) is None:
                        exit_dates = [leg['tranche'].get_exit_datetime(self.time_stamp)]
                        fractions = [leg['tranche'].get_tranche_fraction(self.time_stamp)]
                    else:
                        # special case: this is when for one entry date, there are multiple exit dates
                        print("tranching with multiple exit dates on same entry date")
                        exit_dates = leg['tranche'].get_exit_datetime(self.time_stamp)
                        fractions = leg['tranche'].get_tranche_fraction(self.time_stamp)

                    # run through tranches
                    # Usually tranche should not provide target expiry, as this will override the target expiry specified in the strategy parameters
                    # The only case this override can happen is Custom Tranche
                    if leg['tranche'].get_target_expiry(self.time_stamp) is None:
                        target_exps = [None for x in exit_dates]
                    else:
                        if isinstance(leg['tranche'].get_target_expiry(self.time_stamp), datetime):
                            target_exps = [leg['tranche'].get_target_expiry(self.time_stamp)]
                        else:
                            target_exps = leg['tranche'].get_target_expiry(self.time_stamp)
                else:
                    exit_dates = fractions = target_exps = []

                # trade new tranche (multiple tranches in the special case of one entry multiple exit)
                for _tranche_exit, _tranche_fraction, _target_exp_from_tranche in zip(exit_dates, fractions, target_exps):
                    new_tranche_name = make_tranche_key(self.time_stamp, _tranche_exit)
                    OptionEntryEvent.trade_tranche(True, leg, leg_name, new_tranche_name, portfolio, _tranche_fraction, selection_by, selection_value,
                                                   state, self.parameters, self.strategy, prev_nav, _target_exp_from_tranche)

            # rolling tranche
            elif isinstance(leg['tranche'], RollingTrancheExpiryOffset):
                if self.time_stamp.date() in [x.date() for x in leg['tranche'].initial_entry_dates]:
                    new_tranche_name = f"rolling_starting_{self.time_stamp.strftime('%Y-%m-%d')}"
                    OptionEntryEvent.trade_tranche(True, leg, leg_name, new_tranche_name, portfolio, 1 / len(leg['tranche'].initial_entry_dates), selection_by, selection_value,
                                                   state, self.parameters, self.strategy, prev_nav, None)
                state.portfolio = portfolio

                # loop through tranches to decide if needs to roll
                leg_portfolio = portfolio.get_position(leg_name)
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        if isinstance(tranche_portfolio, Portfolio) and not tranche_portfolio.is_empty():
                            if 'is_tranche_roll_date_function' in leg:
                                def _is_tranche_roll_date_function(_leg_name, _tranche_name, _strategy, _state):
                                    return leg['is_tranche_roll_date_function'](_leg_name, _tranche_name, _strategy, _state)
                            else:
                                def _is_tranche_roll_date_function(_leg_name, _tranche_name, _strategy, _state):
                                    return OptionEntryEvent.is_tranche_roll_date_from_expiry(_leg_name, _tranche_name, _strategy, _state)

                            if _is_tranche_roll_date_function(leg_name, tranche_name, self.strategy, state):
                                
                                if 'keep_expiration' in leg.keys():
                                    if leg['keep_expiration'] and (len(portfolio.get_position(leg_name).get_position(tranche_name).find_children_of_tradable_type(Option)) != 0):
                                        target_expiration=portfolio.get_position(leg_name).get_position(tranche_name).find_children_of_tradable_type(Option)[0][1].tradable.expiration
                                    else:
                                        target_expiration=None
                                else:
                                    target_expiration=None

                                OptionEntryEvent.unwind_tranche_portfolio(leg_name, tranche_name, portfolio, leg_portfolio, tranche_portfolio, market, self.strategy, self.parameters)
                                OptionEntryEvent.trade_tranche(False, leg, leg_name, tranche_name, portfolio, 1 / len(leg['tranche'].initial_entry_dates), selection_by, selection_value,
                                                               state, self.parameters, self.strategy, prev_nav, target_expiration)

                                if len(portfolio.get_position(leg_name).get_position(tranche_name).find_children_of_tradable_type(Option)) == 0:
                                    raise RuntimeError(f"tranche portfolio is empty in rolling tranche")

                # if self.time_stamp.date() in [x.date() for x in leg['tranche'].initial_entry_dates] or len(unwinds_today):
                #     n_new_trades = len(unwinds_today)
                #     if self.time_stamp.date() in [x.date() for x in leg['tranche'].initial_entry_dates]:
                #         n_new_trades += 1
                #
                #     exit_dates = [leg['tranche']] * n_new_trades
                #     fractions = [1] * n_new_trades
                #     target_exps = [None] * n_new_trades
            elif isinstance(leg['tranche'], RollingAtExpiryTranche):
                # trade new tranches
                if leg['tranche'].is_entry_datetime(self.time_stamp) or self.trade_first_day:
                    self.trade_one_option(
                        self.strategy, state, portfolio,
                        leg['underlying'], self.parameters.get('use_listed'), leg['tenor'], leg['type'],
                        leg['sizing_measure'],
                        leg['sizing_target'],
                        selection_by,
                        selection_value,
                        position_path=(leg_name, self.time_stamp),
                        tranche=leg["tranche"]
                    )
            else:
                raise RuntimeError(f"Unknown type of tranche for leg {leg_name}")

        #########################
        # Finalising
        #########################

        # value portfolio
        if 'flat_vega_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'vega', 'theta', 'gamma'], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
            flds = ['price', 'vega', 'theta', 'gamma' ]
        elif 'bid_offer_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'bid', 'ask' ], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
            flds = ['price',  'bid', 'ask' ]
        else:
            price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price, 0.0)
            flds = ['price']

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            pre_trd_px = pre_trade_ptf.price_at_market(market, fields=flds, valuer_map_override=self.strategy.valuer_map)
            post_trade_ptf = post_state.portfolio
            post_trd_px = post_trade_ptf.price_at_market(market, fields=flds, valuer_map_override=self.strategy.valuer_map)
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        # value portfolio
        px = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        if math.isnan(px):
            raise RuntimeError(f"portfolio price after OptionEntryEvent is nan at {state.time_stamp}. Likely some price or greek (for sizing) data is missing")
        return StockOptionsDailyState(self.time_stamp, portfolio, px, 0.0, portfolio_before_trades)


class EODFuturesUnwind(Event):
    def execute(self, state: StrategyState, daily_states = [ ]):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # use stale data to forward fill missing price
        if self.parameters.get('use_listed', True):
            backfill_markets = []
            for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
                back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
                if back_date >= self.strategy.start_date:
                    backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
            self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        for leg_name in list(self.parameters['legs'].keys()):
            if 'hedging_instrument' not in self.parameters['legs'][leg_name]:
                continue

            # hedge future
            if self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future':
                hedge_instrument_type = Future
                hedge_instrument = hedge_utils.find_front_future(
                    market, self.parameters['legs'][leg_name]['hedging_instrument']['root'], 'last tradable date', self.parameters['legs'][leg_name]['hedging_instrument']['expiry_offset']
                )
                hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market, self.strategy.valuer_map[Future], calc_types=['price', 'delta'])
            elif self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'stock':
                hedge_instrument_type = Stock
                hedge_instrument = Stock(self.parameters['legs'][leg_name]['hedging_instrument']['ticker'], self.parameters['legs'][leg_name]['hedging_instrument']['currency'])
                hedge_instrument_price = market.get_spot(self.parameters['legs'][leg_name]['hedging_instrument']['ticker'])
                hedge_instrument_delta = 1.0
            else:
                raise RuntimeError('Unknown hedge instrument type')

            # unwind any hedge that is not the selected hedge future
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:
                net_positions = leg_portfolio.net_positions()
                for position_name, position in net_positions.items():
                    if isinstance(position.tradable, hedge_instrument_type):
                        unwind_price = position.tradable.price(market, self.strategy.valuer_map[hedge_instrument_type], calc_types='price')
                        leg_portfolio.trade(position.tradable, -position.quantity, unwind_price, position.tradable.currency)

        # value portfolio
        if 'flat_vega_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price','vega'], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
        elif 'bid_offer_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'bid', 'ask' ], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
        else:
            price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        # value portfolio
        [px, delta] = portfolio.price_at_market(market, fields=['price', 'delta'], valuer_map_override=self.strategy.valuer_map)
        return StockOptionsDailyState(self.time_stamp, portfolio, px, 0.0, state.portfolio_before_trades)


class OptionDeltaHedgeEvent(Event):
    def execute(self, state: StrategyState, daily_states = [ ]):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # use stale data to forward fill missing price
        if self.parameters.get('use_listed', True):
            backfill_markets = []
            for i in range(self.strategy.parameters.get('allow_fill_forward_missing_data', 0)):
                back_date = add_business_days(self.time_stamp, - i - 1, self.strategy.holidays)
                if back_date >= self.strategy.start_date:
                    backfill_markets.append(self.strategy.backtest_market.get_market(back_date))
            self.strategy.valuer_map[Option].set_backfill_markets(backfill_markets)

        for leg_name in list(self.parameters['legs'].keys()):
            if 'hedging_instrument' not in self.parameters['legs'][leg_name]:
                continue


            # hedge future
            if self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future':
                hedge_instrument_type = Future
                hedge_instrument = hedge_utils.find_front_future(
                    market, self.parameters['legs'][leg_name]['hedging_instrument']['root'], 'last tradable date', self.parameters['legs'][leg_name]['hedging_instrument']['expiry_offset']
                )
                hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market, self.strategy.valuer_map[Future], calc_types=['price', 'delta'])
            elif self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'stock':
                hedge_instrument_type = Stock
                hedge_instrument = Stock(self.parameters['legs'][leg_name]['hedging_instrument']['ticker'], self.parameters['legs'][leg_name]['hedging_instrument']['currency'])
                hedge_instrument_price = market.get_spot(self.parameters['legs'][leg_name]['hedging_instrument']['ticker'])
                hedge_instrument_delta = 1.0
            else:
                assert self.parameters.get('keep_hedges_in_tranche_portfolio', False)
                assert self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future_expiry_at_option_expiry', 'Unknown hedge instrument type'

            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is not None:

                if 'is_hedge_date_function' in self.parameters['legs'][leg_name]:

                    if not self.parameters['legs'][leg_name]['is_hedge_date_function'](leg_name, self.strategy, state):
                        continue

                # first unwind any hedge that is not the selected hedge future (the only reason this is the case is it is close to expiry)
                if self.parameters.get('keep_hedges_in_tranche_portfolio', False):
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        if isinstance(tranche_portfolio, Portfolio):

                            # find hedge instrument for this tranche
                            if self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future_expiry_at_option_expiry':
                                option_positions = tranche_portfolio.find_children_of_tradable_type(Option)
                                if len(option_positions):
                                    option_expiry = option_positions[0][1].tradable.expiration
                                    hedge_instrument_type = Future
                                    hedge_instrument = hedge_utils.find_future_with_expiration_date(market, self.parameters['legs'][leg_name]['hedging_instrument']['root'], 'last tradable date', option_expiry, 'nearest')
                                    hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market, self.strategy.valuer_map[Future], calc_types=['price', 'delta'])
                                    net_positions = tranche_portfolio.net_positions()
                                    for position_name, position in net_positions.items():
                                        if isinstance(position.tradable, hedge_instrument_type) and not position.tradable == hedge_instrument:
                                            unwind_price = position.tradable.price(market, self.strategy.valuer_map[hedge_instrument_type], calc_types='price')
                                            tranche_portfolio.trade(position.tradable, -position.quantity, unwind_price, position.tradable.currency)
                            else:
                                net_positions = tranche_portfolio.net_positions()
                                for position_name, position in net_positions.items():
                                    if isinstance(position.tradable, hedge_instrument_type) and not position.tradable == hedge_instrument:
                                        unwind_price = position.tradable.price(market, self.strategy.valuer_map[hedge_instrument_type], calc_types='price')
                                        tranche_portfolio.trade(position.tradable, -position.quantity, unwind_price, position.tradable.currency)
                else:
                    net_positions = leg_portfolio.net_positions()
                    for position_name, position in net_positions.items():
                        if isinstance(position.tradable, hedge_instrument_type) and not position.tradable == hedge_instrument:
                            unwind_price = position.tradable.price(market, self.strategy.valuer_map[hedge_instrument_type], calc_types='price')
                            leg_portfolio.trade(position.tradable, -position.quantity, unwind_price, position.tradable.currency)

                # then hedge the portfolio delta with front future
                leg_portfolio.price_at_market(market, 'delta', self.strategy.valuer_map, default=0)
                portfolio_delta = leg_portfolio.aggregate('delta')
                if self.parameters.get('keep_hedges_in_tranche_portfolio', False):
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        if isinstance(tranche_portfolio, Portfolio):

                            # find hedge instrument for this tranche
                            if self.parameters['legs'][leg_name]['hedging_instrument']['type'] == 'future_expiry_at_option_expiry':
                                option_positions = tranche_portfolio.find_children_of_tradable_type(Option)
                                if len(option_positions):
                                    option_expiry = option_positions[0][1].tradable.expiration
                                    hedge_instrument_type = Future
                                    hedge_instrument = hedge_utils.find_future_with_expiration_date(market, self.parameters['legs'][leg_name]['hedging_instrument']['root'], 'last tradable date', option_expiry, 'nearest')
                                    hedge_instrument_price, hedge_instrument_delta = hedge_instrument.price(market, self.strategy.valuer_map[Future], calc_types=['price', 'delta'])
                                    tranche_delta = tranche_portfolio.aggregate('delta')
                                    tranche_portfolio.trade(hedge_instrument, -tranche_delta / hedge_instrument_delta, hedge_instrument_price, hedge_instrument.currency)
                            else:
                                tranche_delta = tranche_portfolio.aggregate('delta')
                                tranche_portfolio.trade(hedge_instrument, -tranche_delta / hedge_instrument_delta, hedge_instrument_price, hedge_instrument.currency)
                else:
                    # all hedges are at the root
                    leg_portfolio.trade(hedge_instrument, -portfolio_delta / hedge_instrument_delta, hedge_instrument_price, hedge_instrument.currency)

        # value portfolio
        if 'flat_vega_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price','vega'], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
        elif 'bid_offer_charge' in self.parameters.keys():
            price = portfolio.price_at_market(market, fields=['price', 'bid', 'ask' ], valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)
        else:
            price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
            post_state = StockOptionsDailyState(self.time_stamp, portfolio, price, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        # value portfolio
        calc_types = ['price'] + self.parameters.get('greeks_to_include', ['delta'])
        pricing_results = portfolio.price_at_market(market, fields=calc_types, valuer_map_override=self.strategy.valuer_map)
        if math.isnan(pricing_results[0]):
            raise RuntimeError(f"portfolio price after OptionDeltaHedgeEvent is nan at {state.time_stamp}. Likely some price or delta (for hedging) data is missing")
        return StockOptionsDailyState(self.time_stamp, portfolio, pricing_results[0], 0.0, state.portfolio_before_trades)


class StockOptionsDaily(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        self.valuer_map = self.parameters['valuer_map']

        self.parameters['corpaction_processor'] = CorpActionsProcessor()

        if self.inc_trd_dts:
            opt_keys = [x for x in self.data_containers.keys() if ' option' in x]
            opt_dts = [ self.data_containers[ky].get_full_option_universe() for ky in opt_keys ]
            trd_dts = list(set.intersection(*map(set, opt_dts)))

            fut_keys = [x for x in self.data_containers.keys() if ' futur' in x]
            if len(fut_keys) > 0:
                fut_dts = [ self.data_containers[ky]._get_future_universe(None, True) for ky in fut_keys ]
                fut_dts = list(set.intersection(*map(set, fut_dts)))
                trd_dts = list( set(fut_dts) & set(trd_dts) )

            eq_keys = [x for x in self.data_containers.keys() if ' spot' in x]
            eq_dts = [ self.data_containers[ky]._get_data_dts() for ky in eq_keys]
            eq_dts = list(set.intersection(*map(set, eq_dts)))
            trd_dts = list(set(eq_dts) & set(trd_dts))

            trd_dts.sort()
            self.trading_days = trd_dts

        #self.force_run = True
        if 'scale_by_nav' in self.parameters.keys():
            self.inc_daily_states = True

        if not 'expires' in self.parameters:
            self.parameters['expires'] = ExpireOptionAtIntrinsic(self.backtest_market)
        elif self.parameters['expires'] == ExpireOptionAtIntrinsic:
            self.parameters['expires'] = ExpireOptionAtIntrinsic(self.backtest_market)
        elif self.parameters['expires'] == ExpireContractsAtPrice:
            self.parameters['expires'] = ExpireContractsAtPrice(self.backtest_market, self.valuer_map)

        # add trading costs
        if 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters[ 'flat_costs' ]
            self.parameters['trading_cost'] = FlatTradingCost( cost_params['tc_delta'], cost_params['tc_vega'] )
        elif 'flat_vega_charge' in self.parameters.keys():
            cost_params = self.parameters[ 'flat_vega_charge']
            self.parameters['trading_cost'] = FlatVegaCostStock( cost_params['tc_delta'], cost_params['tc_vega'] )
        elif 'bid_offer_charge' in self.parameters.keys():
            cost_params = self.parameters[ 'bid_offer_charge' ]
            self.parameters['trading_cost'] = BidOfferCost(cost_params['tc_delta'], cost_params['tc_vega'])

    def generate_events(self, dt: datetime):
        if self.parameters.get('process_dividends_and_corpactions', True):
            return [CorpActionEvent(dt, self),
                    ExpirationEvent(dt, self),
                    UnwindEvent(dt, self),
                    OptionEntryEvent(dt, self),
                    OptionDeltaHedgeEvent(dt, self)]
        elif self.parameters.get('naked', False):
            return [ExpirationEvent(dt, self),
                    UnwindEvent(dt, self),
                    OptionEntryEvent(dt, self)]
        else:
            return [ExpirationEvent(dt, self),
                    UnwindEvent(dt, self),
                    OptionEntryEvent(dt, self),
                    OptionDeltaHedgeEvent(dt, self)]

    @staticmethod
    def extract_positions_info(port, port_path, sub_port_name_map):
        positions = []
        pv = {}
        for k, v in port.root.items():
            if isinstance(v, Portfolio):
                sub_positions, sub_pv = StockOptionsDaily.extract_positions_info(v, port_path + (sub_port_name_map.get(k, k),), sub_port_name_map)
                positions = positions + sub_positions
                pv[k] = sub_pv
            elif isinstance(v.tradable, Option):
                positions.append((
                    port_path + ('Option', 'name'),
                    f"{v.tradable.underlying.split(' ')[0]} {v.tradable.expiration.strftime('%Y-%m-%d')} {v.tradable.strike} {'C' if v.tradable.is_call else 'P'}"
                ))
                positions.append((
                    port_path + ('Option', 'price'),
                    v.price
                ))
                positions.append((
                    port_path + ('Option', 'quantity'),
                    v.quantity
                ))
                pv[k] = v.price * v.quantity
            elif isinstance(v.tradable, Future):
                positions.append((
                    port_path + ('Future', 'name'),
                    f"{k.split(' ')[0]} {v.tradable.expiration.strftime('%Y-%m-%d')}"
                ))
                positions.append((
                    port_path + ('Future', 'price'),
                    v.price
                ))
                positions.append((
                    port_path + ('Future', 'quantity'),
                    v.quantity
                ))
                pv[k] = v.price * v.quantity
            else:
                tradable_type = v.tradable.__class__.__name__
                positions.append((
                    port_path + (tradable_type, 'name'),
                    k
                ))
                positions.append((
                    port_path + (tradable_type, 'price'),
                    v.price
                ))
                positions.append((
                    port_path + (tradable_type, 'quantity'),
                    v.quantity
                ))
                pv[k] = v.price * v.quantity
        return positions, pv

    @staticmethod
    def format_daily_portfolios_by_legs_and_tranches(state_series):
        def _aggregate(d):
            _total = 0
            for _k, _v in d.items():
                if isinstance(_v, dict):
                    _total += _aggregate(_v)
                else:
                    _total += _v
            return _total

        portfolio_expanded = []
        tranche_name_map = {}
        pnl_expanded = []
        pvs_map = {}
        for state in state_series:
            positions = []
            pnls = []
            # find new tranche from before trade portfolio
            for leg_name, leg_v in state.portfolio_before_trades.root.items():
                if not isinstance(leg_v, Portfolio) and isinstance(leg_v.tradable, Cash):
                    continue
                assert isinstance(leg_v, Portfolio)
                tranche_names = sorted([k for k, v in leg_v.root.items() if isinstance(v, Portfolio)])
                leg_tranche_name_map = tranche_name_map.get(leg_name, {})
                # tranches that disappeared
                for k in list(leg_tranche_name_map.keys()):
                    if k not in tranche_names:
                        del leg_tranche_name_map[k]
                # tranches that appeared
                for k in tranche_names:
                    if k not in list(leg_tranche_name_map.keys()):
                        current_tranche_numbers = list(leg_tranche_name_map.values())
                        max_current_tranche_numbers = max(current_tranche_numbers) if len(current_tranche_numbers) else 0
                        new_tranche_number = None
                        for n in range(1, max_current_tranche_numbers + 2):
                            if n not in current_tranche_numbers:
                                new_tranche_number = n
                                break
                        leg_tranche_name_map[k] = new_tranche_number
                tranche_name_map[leg_name] = leg_tranche_name_map
            # get before trade positions
            before_pvs_all_legs = {}
            for leg_name, leg_v in state.portfolio_before_trades.root.items():
                if not isinstance(leg_v, Portfolio) and isinstance(leg_v.tradable, Cash):
                    continue
                sub_port_name_map = {k: f".Tranche{v}" for k, v in tranche_name_map.get(leg_name, {}).items()}
                before_positions, before_pvs = StockOptionsDaily.extract_positions_info(leg_v, ('Before Trade', leg_name), sub_port_name_map)
                before_pvs_all_legs[leg_name] = before_pvs
                positions = positions + before_positions
                prev_cash_pv = 0
                for k, v in pvs_map[leg_name].items():
                    if not isinstance(v, dict):
                        prev_cash_pv += v
                cash_pv = 0
                for k, v in before_pvs.items():
                    if isinstance(v, dict):
                        tranche_pnl = _aggregate(v) - _aggregate(pvs_map[leg_name][k])
                        pnls.append(((leg_name, sub_port_name_map[k]), tranche_pnl))
                    else:
                        assert "Cash" in k or "Constant" in k
                        cash_pv += v
                assert float_equal(cash_pv, prev_cash_pv)

            # find new tranche from after trade portfolio
            for leg_name, leg_v in state.portfolio.root.items():
                if not isinstance(leg_v, Portfolio) and isinstance(leg_v.tradable, Cash):
                    continue
                assert isinstance(leg_v, Portfolio)
                tranche_names = sorted([k for k, v in leg_v.root.items() if isinstance(v, Portfolio)])
                leg_tranche_name_map = tranche_name_map.get(leg_name, {})
                # tranches that disappeared
                for k in list(leg_tranche_name_map.keys()):
                    if k not in tranche_names:
                        del leg_tranche_name_map[k]
                # tranches that appeared
                for k in tranche_names:
                    if k not in list(leg_tranche_name_map.keys()):
                        current_tranche_numbers = list(leg_tranche_name_map.values())
                        max_current_tranche_numbers = max(current_tranche_numbers) if len(current_tranche_numbers) else 0
                        new_tranche_number = None
                        for n in range(1, max_current_tranche_numbers + 2):
                            if n not in current_tranche_numbers:
                                new_tranche_number = n
                                break
                        leg_tranche_name_map[k] = new_tranche_number
                tranche_name_map[leg_name] = leg_tranche_name_map
            # get after trade positions
            for leg_name, leg_v in state.portfolio.root.items():
                if not isinstance(leg_v, Portfolio) and isinstance(leg_v.tradable, Cash):
                    continue
                sub_port_name_map = {k: f".Tranche{v}" for k, v in tranche_name_map.get(leg_name, {}).items()}
                after_positions, after_pvs = StockOptionsDaily.extract_positions_info(leg_v, ('After Trade', leg_name), sub_port_name_map)
                positions = positions + after_positions
                prev_cash_pv = 0
                for k, v in pvs_map.get(leg_name, {}).items():
                    if not isinstance(v, dict):
                        prev_cash_pv += v
                cash_pv = 0
                for k, v in after_pvs.items():
                    if isinstance(v, dict):
                        if k not in pvs_map.get(leg_name, {}):
                            # new tranche
                            assert float_equal(_aggregate(v), 0)
                        # else:
                            # existing tranche
                            # assert float_equal(_aggregate(v), _aggregate(before_pvs_all_legs[leg_name][k]))
                    else:
                        assert "Cash" in k or "Constant" in k
                        cash_pv += v
                total_disppearing_tranche_pnl = 0
                for k, v in before_pvs_all_legs.get(leg_name, {}).items():
                    if isinstance(v, dict):
                        if k not in after_pvs:
                            # disappering tranche
                            total_disppearing_tranche_pnl += _aggregate(v)
                cash_pnl = cash_pv - prev_cash_pv - total_disppearing_tranche_pnl
                pnls.append(((leg_name, 'cash'), cash_pnl))
                pvs_map[leg_name] = after_pvs

            if len(positions):
                # the key should be Before or After Trade, leg name, tranche name or positions at leg level, position type, column type
                max_key_levels = max([len(x[0]) for x in positions])
                assert max_key_levels <= 5
                max_key_levels = 5
                keys = []
                for p in positions:
                    if len(p[0]) < max_key_levels:
                        new_key = p[0][:-1] + ('',) + p[0][-1:]
                    else:
                        new_key = p[0]
                    # for the case of multiple options in a tranche
                    if new_key in keys:
                        new_key = new_key[:-2] + (new_key[-2] + '*',) + new_key[-1:]
                    keys.append(new_key)
                keys = [('',) * (max_key_levels - 1) + ('date',)] + keys
                values = [state.time_stamp] + [x[1] for x in positions]
                portfolio_expanded.append(dict(zip(keys, values)))
            if len(pnls):
                max_key_levels = max([len(x[0]) for x in pnls])
                keys = []
                for p in pnls:
                    if len(p[0]) < max_key_levels:
                        keys.append(p[0][:-1] + ('',) + p[0][-1:])
                    else:
                        keys.append(p[0])
                keys = [('',) * (max_key_levels - 1) + ('date',)] + keys + [('',) * (max_key_levels - 1) + ('total_pnl',)]
                values = [state.time_stamp] + [x[1] for x in pnls] + [sum([x[1] for x in pnls if not math.isnan(x[1])])]
                pnl_expanded.append(dict(zip(keys, values)))

        portfolio_expanded = pd.DataFrame.from_records(portfolio_expanded)
        ordered_columns = sorted(portfolio_expanded.columns)
        portfolio_expanded = portfolio_expanded[ordered_columns]
        portfolio_expanded.columns = pd.MultiIndex.from_tuples(portfolio_expanded.columns)

        pnl_expanded = pd.DataFrame.from_records(pnl_expanded)
        ordered_columns = sorted(pnl_expanded.columns)
        pnl_expanded = pnl_expanded[ordered_columns]
        pnl_expanded.columns = pd.MultiIndex.from_tuples(pnl_expanded.columns)
        return portfolio_expanded, pnl_expanded
