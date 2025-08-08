
from datetime import datetime, time

from ...backtest.costs import FlatTradingCost, VariableVegaCost

from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import count_business_days, add_business_days
from ...infrastructure.bmarket import BMarket
from ...tradable.option import Option

from ...valuation.fx_cr_option_bs_valuer import FXOptionBSValuer

global trade_date_info
trade_date_info = {}

global entry_unwind_schedule
entry_unwind_schedule = {}

class RollingCROptionsState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, entry_unwind_schedule = {}):
        self.price = price
        self.cost = cost
        self.entry_unwind_schedule = entry_unwind_schedule
        super().__init__(time_stamp, portfolio)

class OptionEntry(Event):
    def find_next_expiry(self, tenor ):
        assert tenor[-1] == 'M'
        next_expiry = add_business_days( self.time_stamp, 21 * int( tenor[:-1]), self.strategy.holidays)
        return next_expiry

    def execute(self, state: StrategyState):
        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # TODO: extend for more expiries
        TTM_conv = { '1M': 0.08333333333333333, '2M': 0.16666666666666666, '3M': 0.25 }

        # trade
        if portfolio.is_empty():
            first_day = True
        else:
            first_day = False
        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                sub_portfolio = portfolio.get_position((leg_name, tranche_name))
                if sub_portfolio is None or len(sub_portfolio.find_children_of_tradable_type(Option)) == 0:
                    # backtest only handles certain (liquid) delta strikes
                    assert leg['strike'] in [ x / 100 for x in range(5,100,5) ]
                    # TODO: support PUT OT
                    assert leg['OT'] == 'CALL'
                    vol_surface = market.storage[self.data_containers['vol'].get_market_key()][leg['underlier']]
                    tranche_expiration_date = self.find_next_expiry(tranche['initial_tenor'] if first_day else leg['tenor'] )
                    TTM = ( count_business_days( self.time_stamp, tranche_expiration_date, self.strategy.holidays )  - 1) / 252
                    vols_by_exp = vol_surface.vols[ TTM_conv[ tranche['initial_tenor'] if first_day else leg['tenor'] ] ]
                    vol = vols_by_exp[str(int(leg['strike']*100))]
                    delta_abs_conv = vol_surface.abs_strikes[ TTM_conv[ tranche['initial_tenor'] if first_day else leg['tenor'] ] ]
                    abs_strike = delta_abs_conv[leg['strike']]

                    option_to_trade = Option(
                        leg['underlier'], leg['underlier'], leg['currency'], tranche_expiration_date,
                         abs_strike, leg['OT'] == 'CALL', False, 1.0, 'America/New_York', listed_ticker=None )
                    assert len( leg[ 'sizing' ] ) == 1
                    size_by = list( leg['sizing'].keys() )[0]
                    option_to_trade_risk = option_to_trade.price( market=market,
                                                                  valuer=self.strategy.valuer_map[Option],
                                                                  return_struc=True,
                                                                  vol_override = vol )
                    option_to_trade_price = option_to_trade_risk[ 'price' ]
                    option_to_trade_delta = round( option_to_trade_risk[ 'forward_delta' ], 3 )
                    leg_sizing = leg['sizing'][size_by]
                    if size_by != 'quantity':
                        leg_sizing /= option_to_trade_risk[ size_by ]
                        assert abs(leg['sizing'][size_by] - leg_sizing * option_to_trade_risk[size_by]) < 0.001

                    if TTM in list( TTM_conv.values() ):
                        if abs( round( option_to_trade_delta - leg['strike'],3 ) ) > 0.015:
                            print( 'entering position at %.1fd on %s' %( 100*option_to_trade_delta, self.time_stamp) )
                    else:
                        assert abs( round( option_to_trade_risk[ 'forward_delta' ] - leg['strike'], 3 ) ) <= 0.015

                    tranche_unwind_date = self.find_next_expiry(leg['unwind'])
                    entry_unwind_schedule[tranche_expiration_date] = { 'unwind_dt': tranche_unwind_date,
                                                                       'unwind_path': (leg_name,
                                                                                       tranche_name,
                                                                                       option_to_trade.name_str ) }
                    portfolio.trade(option_to_trade, leg_sizing, option_to_trade_price, leg['currency'],
                                    position_path=(leg_name, tranche_name) )
                    # update <trade_date_info>
                    update_dict = { 'trade_date': self.time_stamp,
                                    'trade_date_vol': vol }
                    trade_date_info[ option_to_trade.name() ] = update_dict

                        # reset valuer as the valuer needs state dependent information
        # this uses a global variable so this line has to be called right before you use it
        # TODO: need to move away from using global variable
        self.strategy.valuer_map[Option] = FXOptionBSValuer('OIS',
                                                            trade_date_info )

        # add price manually
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        post_state = RollingCROptionsState(self.time_stamp, portfolio, 0.0, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        return post_state


class UnwindOptions(Event):
    def execute(self, state: StrategyState):

        # load entry_unwind_schedule where applicable from state
        # eg saved from previous backtest
        check_eus = 'entry_unwind_schedule' in globals()
        if len(globals()['entry_unwind_schedule']) == 0 and hasattr( state, 'entry_unwind_schedule'):
            globals()['entry_unwind_schedule'] = state.entry_unwind_schedule

        # Currently no option to hold to expiry: backtest is interpolating
        # between monthly tenors and below 1M is full extrapolation.

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # reset valuer as the valuer needs state dependent information
        # this uses a global variable so this line has to be called right before you use it
        # TODO: need to move away from using global variable
        self.strategy.valuer_map[Option] = FXOptionBSValuer('OIS', trade_date_info)

        # unwind options
        positions = {k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}
        for pos, vals in positions.items():
            exp = vals.tradable.expiration
            exp = datetime.combine(exp.date(), time(0, 0))
            if entry_unwind_schedule[ exp ][ 'unwind_dt' ] <= self.time_stamp:
                unwind_px = vals.tradable.price( market = market,
                                                  valuer=FXOptionBSValuer('OIS', trade_date_info),
                                                  calc_types='price' )
                portfolio.unwind( entry_unwind_schedule[ exp ][ 'unwind_path' ], unwind_px )

        return RollingCROptionsState(self.time_stamp, portfolio, 0.0, 0.0)


class DailyMtM(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # reset valuer as the valuer needs state dependent information
        # this uses a global variable so this line has to be called right before you use it
        # TODO: need to move away from using global variable
        self.strategy.valuer_map[Option] = FXOptionBSValuer('OIS', trade_date_info)

        # value portfolio
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        return RollingCROptionsState(self.time_stamp, portfolio, price, 0.0,
                                     entry_unwind_schedule = entry_unwind_schedule )

class RollingCROptions(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        rates_data = self.data_containers['df_curve'].get_spot_rate_curves()
        rates_data_dt = list(rates_data.keys())
        vol_data_dt = self.data_containers['vol'].get_trading_days()
        trade_days = list(set(vol_data_dt) & set(rates_data_dt))
        trade_days.sort()

        self.trading_days = trade_days

        trade_date_info['holidays'] = self.holidays
        self.valuer_map = {
            Option: FXOptionBSValuer( 'OIS', trade_date_info ),
        }

        self.force_run = True

        # initialize the trading cost module
        if 'variable_costs' in self.parameters.keys():
            cost_params = self.parameters[ 'variable_costs' ]
            self.parameters['trading_cost'] = VariableVegaCost( cost_params[ 'cost_cap' ], cost_params[ 'cost_floor' ], cost_params[ 'cost_scale' ], cost_params[ 'delta_cost' ], self.backtest_market, self.valuer_map )
        elif 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters[ 'flat_costs' ]
            self.parameters['trading_cost'] = FlatTradingCost( cost_params['tc_delta'], cost_params['tc_vega'] )

    def generate_events(self, dt: datetime):
        return [ UnwindOptions(dt, self), OptionEntry(dt, self), DailyMtM(dt, self) ]


