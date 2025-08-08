

from ...constants.asset_class import AssetClass
from ...dates.utils import bdc_adjustment
from ...tradable.varianceswap import VarianceSwap

from datetime import datetime

from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...dates.utils import add_tenor
from ...valuation.option_data_valuer import OptionDataValuer
from ...backtest.expires import ExpireVarswap
from ...backtest.costs import FlatTradingCost, VariableVegaCost, FlatVegaTradingCost
from ...tradable.option import Option
from ...analytics.symbology import OPTION_ROOT_FROM_TICKER


import numpy as np
from dateutil.relativedelta import relativedelta, FR



class RollingVarSwapState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost):
        self.price = price
        self.cost = cost
        super().__init__(time_stamp, portfolio)



class OptionEntry(Event):

    def find_next_expiry(self, tenor):
        next_expiry = add_tenor(self.time_stamp, tenor)
        next_expiry = bdc_adjustment(next_expiry, 'following', self.strategy.holidays)
        return next_expiry

    def execute(self, state: StrategyState, daily_states ):
        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get t-1 NAV for sizing
        prev_nav = daily_states[-1].price
        if isinstance(prev_nav, list):
            prev_nav = prev_nav[0]
        scale_by_nav = 'scale_by_nav' in self.parameters.keys() and self.parameters['scale_by_nav']

        # trade
        if len({k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}) < 1:
            first_day = True
            varswap_exp = []
        else:
            first_day = False
            varswap_pos = {k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}
            varswap_exp = [x.tradable.expiration for x in varswap_pos.values() if isinstance(x.tradable, VarianceSwap)]

        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                sub_portfolio = portfolio.get_position((leg_name, tranche_name))
                if sub_portfolio is None or len(sub_portfolio.find_children_of_tradable_type(VarianceSwap)) == 0:
                    tranche_exp = self.find_next_expiry( tranche['initial_tenor'] if first_day else leg['tenor'])

                    # trade 3rd Friday options for SPX as there is sparse data in 2007/08 and EOM options cause errors.
                    if leg['underlying'] == 'SPX Index' and self.time_stamp < datetime( 2012, 1, 4 ):
                        fri3 = tranche_exp.replace(day=1)
                        fri3 = fri3 + relativedelta(weekday=FR(3))
                        fri3 = bdc_adjustment(fri3, 'following', self.strategy.holidays)
                        tranche_exp = fri3

                    varswap = VarianceSwap( leg[ 'underlying' ], self.time_stamp, tranche_exp, 0, 1, "USD", asset_class=AssetClass.EQUITY)
                    fields = ["fair_strike", leg['sizing_measure'].replace('notional', 'spot'), "price"]
                    varswap_risk = dict(zip(fields, varswap.price(market=market, calc_types=fields,
                                                                  recalculate_replication_portfolio=True)))

                    varswap.strike_in_vol = varswap_risk[ 'fair_strike']
                    varswap.strike_in_var = varswap_risk['fair_strike'] * varswap_risk['fair_strike']
                    varswap.expiration = datetime.fromisoformat( varswap.rep_ptf.expiration.unique()[0] )
                    if varswap.expiration in varswap_exp:
                        continue
                    leg_sizing = leg['sizing_target']
                    if scale_by_nav:
                        leg_sizing *= prev_nav

                    if leg['sizing_measure'] != 'quantity':
                        leg_sizing /= varswap_risk[ leg['sizing_measure'].replace('notional', 'spot') ]
                        if scale_by_nav:
                            assert abs( prev_nav * leg['sizing_target'] -
                                        leg_sizing * varswap_risk[leg['sizing_measure'].replace('notional', 'spot')]) < 0.001
                        else:
                            assert abs( leg['sizing_target'] -
                                        leg_sizing * varswap_risk[ leg['sizing_measure'].replace('notional', 'spot') ]) < 0.001


                    portfolio.trade( varswap, leg_sizing, varswap_risk[ "price" ], self.strategy.currency, position_path=(leg_name, tranche_name) )

        # add price manually
        price = portfolio.price_at_market(market, fields=['price', 'vega'], valuer_map_override=self.strategy.valuer_map)
        post_state = RollingVarSwapState(self.time_stamp, portfolio, 0.0, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        return post_state

class RemoveExpiredOptions(Event):
    def execute(self, state: StrategyState, daily_states ):
        print("running backtest on %s" % self.time_stamp)
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # expire options
        self.parameters['expires'].expire(self.time_stamp, portfolio)
        return RollingVarSwapState( self.time_stamp, portfolio, 0.0, 0.0 )

class DailyMtM(Event):
    def execute(self, state: StrategyState, daily_states ):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # value portfolio
        price = portfolio.price_at_market(market, fields=['price', 'delta', 'gamma', 'vega', 'theta'], valuer_map_override=self.strategy.valuer_map)
        return RollingVarSwapState( self.time_stamp, portfolio, price, 0.0 )

class DeltaHedgeEOD(Event):

    def execute(self, state: StrategyState, daily_states ):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        positions = { k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k }
        all_und = list(set([x.tradable.underlying for x in positions.values()]))
        delta_by_und = {}
        # iterate over underliers
        for und in all_und:
            delta_by_exp = {}
            sub_ptf = [ x for x in list( positions.values() ) if x.tradable.underlying == und ]
            exp_by_und = [ x.tradable.expiration for x in sub_ptf ]
            exp_by_und = np.unique( np.array( exp_by_und ) )
            # iterate of expiries
            for exp in exp_by_und:
                to_hedge = [ x for x in sub_ptf if x.tradable.expiration == exp ]
                varswap = [ x for x in to_hedge if isinstance(x.tradable, VarianceSwap) ]
                assert len(varswap) == 1
                varswap = varswap[0]
                varswap_delta = varswap.tradable.price(market=market, calc_types="delta") * varswap.quantity
                fwd_pos = [ x for x in to_hedge if isinstance(x.tradable, Option ) ]

                synth_fwd = varswap.tradable.synth_fwd
                synth_fwd_call = synth_fwd[ synth_fwd.call_put == 'C' ]
                synth_fwd_put = synth_fwd[ synth_fwd.call_put == 'P' ]
                # unwind existing fwd
                if len( fwd_pos ) > 0:
                    assert len( fwd_pos ) == 2
                    call_pos = [ x for x in fwd_pos if x.tradable.is_call ]
                    assert len( call_pos ) == 1
                    call_qty = call_pos[0].quantity
                    call_pos = call_pos[0].tradable
                    call_px = call_pos.price( market=market, calc_types="price",
                                              valuer =  self.strategy.valuer_map.get( type( call_pos ) ) )

                    put_pos = [ x for x in fwd_pos if not x.tradable.is_call ]
                    assert len( put_pos ) == 1
                    put_qty = put_pos[0].quantity
                    put_pos = put_pos[0].tradable
                    put_px = put_pos.price( market=market, calc_types="price",
                                              valuer =  self.strategy.valuer_map.get( type( put_pos ) ) )

                    assert abs( call_qty + put_qty ) < 0.00001
                    # unwind fwd
                    portfolio.trade( call_pos, -call_qty, call_px, call_pos.currency )
                    portfolio.trade( put_pos, -put_qty, put_px, put_pos.currency )
                else:
                    call_qty = 0
                    put_qty = 0

                # enter new fwd pos
                call_pos = Option( OPTION_ROOT_FROM_TICKER[varswap.tradable.underlying],
                                   varswap.tradable.underlying,
                                   varswap.tradable.currency,
                                   datetime.fromisoformat(synth_fwd_call.expiration.values[0]),
                                   synth_fwd_call.strike.values[0],
                                   synth_fwd_call.call_put.values[0] == 'C',
                                   False,
                                   synth_fwd_call.contract_size.values[0],
                                   synth_fwd_call.tz_name.values[0],
                                   synth_fwd_call.ticker.values[0])
                call_px = call_pos.price(market=market, calc_types="price",
                                         valuer=self.strategy.valuer_map.get(type(call_pos)))
                call_delta = call_pos.price(market=market, calc_types="delta",
                                         valuer=self.strategy.valuer_map.get(type(call_pos)))

                put_pos = Option( OPTION_ROOT_FROM_TICKER[varswap.tradable.underlying],
                                  varswap.tradable.underlying,
                                  varswap.tradable.currency,
                                  datetime.fromisoformat(synth_fwd_put.expiration.values[0]),
                                  synth_fwd_put.strike.values[0],
                                  synth_fwd_put.call_put.values[0] == 'C',
                                  False,
                                  synth_fwd_put.contract_size.values[0],
                                  synth_fwd_put.tz_name.values[0],
                                  synth_fwd_put.ticker.values[0])
                put_px = put_pos.price(market=market, calc_types="price",
                                       valuer=self.strategy.valuer_map.get(type(put_pos)))
                put_delta = put_pos.price(market=market, calc_types="delta",
                                       valuer=self.strategy.valuer_map.get(type(put_pos)))

                delta_by_exp[ exp ] = varswap_delta
                hedge_delta = call_delta - put_delta
                net_delta = -delta_by_exp[ exp ] / hedge_delta
                # TODO: generalise DH thresh-hold
                position_thresh = 0
                delta_thresh = 0
                if abs(net_delta) < delta_thresh or abs( call_qty - net_delta) < position_thresh:
                    net_delta = 0

                # flatten delta
                if abs( net_delta ) > 0:
                    portfolio.trade( call_pos, net_delta, synth_fwd_call.price.values[0], call_pos.currency )
                    portfolio.trade( put_pos, -net_delta, synth_fwd_put.price.values[0], put_pos.currency )
                if abs( call_qty - net_delta ) > position_thresh:
                    check_delta = ( call_delta - put_delta ) * net_delta + varswap_delta
                    assert abs( check_delta ) < 0.00001
            delta_by_und[ und ] = delta_by_exp
        ptf_delta = portfolio.price_at_market(market, fields='delta', valuer_map_override=self.strategy.valuer_map)
        assert abs( ptf_delta ) < 0.00001

        # add price manually
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        post_state = RollingVarSwapState(self.time_stamp, portfolio, 0.0, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        return post_state

class RollingVarSwap(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item( data.get_market_key(), data )
        self.backtest_market = backtest_market
        self.valuer_map = {
            Option: OptionDataValuer() }

        # infer trade dates from data
        spot_dts = self.data_containers['spots'].get_data_dts()
        vol_dts = self.data_containers['options'].get_full_option_universe()
        trades_dts = list(set(spot_dts) & set(vol_dts))
        trades_dts.sort()
        self.trading_days = trades_dts

        #self.force_run = True
        self.inc_daily_states = True

        # initialize the trading cost module
        if 'variable_costs' in self.parameters.keys():
            cost_params = self.parameters[ 'variable_costs' ]
            self.parameters['trading_cost'] = VariableVegaCost( cost_params[ 'cost_cap' ], cost_params[ 'cost_floor' ], cost_params[ 'cost_scale' ], cost_params[ 'delta_cost' ], self.backtest_market, self.valuer_map )
        elif 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters[ 'flat_costs' ]
            self.parameters['trading_cost'] = FlatTradingCost( cost_params['tc_delta'], cost_params['tc_vega'] )
        elif 'flat_vega_charge' in self.parameters.keys():
            cost_params = self.parameters[ 'flat_vega_charge']
            self.parameters['trading_cost'] = FlatVegaTradingCost( cost_params['tc_delta'], cost_params['tc_vega'] )
        #self.parameters['expires'] = ExpireContractsAtPrice(self.backtest_market, self.valuer_map)
        self.parameters['expires'] = ExpireVarswap(self.backtest_market, self.valuer_map)


    def generate_events(self, dt: datetime):
        if 'no_delta_hedge' in self.parameters and self.parameters['no_delta_hedge']:
            return [RemoveExpiredOptions(dt, self), OptionEntry(dt, self), DailyMtM(dt, self)]
        else:
            return [RemoveExpiredOptions(dt, self), OptionEntry(dt, self), DeltaHedgeEOD(dt, self), DailyMtM(dt, self)]