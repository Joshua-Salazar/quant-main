from datetime import datetime

from ...backtest.costs import FlatTradingCost, VariableVegaCost
from ...backtest.expires import ExpireFXOptionContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import add_tenor, bdc_adjustment
from ...infrastructure.bmarket import BMarket
from ...tradable.option import Option
from ...tradable.FXforward import FXforward
from ...analytics.FX_vol_interpolation import delta_strike_to_relative

import numpy as np

from ...valuation.fx_forward_fx_vol_surface_valuer import FXForwardDataValuer
from ...valuation.fxoption_bs_valuer import FXOptionBSValuer


class FXOptionsDailyState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost):
        self.price = price
        self.cost = cost
        super().__init__(time_stamp, portfolio)


class TradeOptions(Event):
    def find_next_expiry(self, tenor):
        next_expiry = add_tenor(self.time_stamp, tenor)
        next_expiry = bdc_adjustment(next_expiry, 'following', self.strategy.holidays)
        return next_expiry

    def execute(self, state: StrategyState, daily_states):
        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        if 'scale_by_nav' in self.parameters and self.parameters['scale_by_nav']:
            prev_nav = daily_states[-1].price

        TTM_conv = {'1W': 0.019178082191780823, '2W': 0.038356164383561646, '1M': 0.08333333333333333,
                    '2M': 0.16666666666666666, '3M': 0.25, '6M': 0.5, '9M': 0.75}
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
                    if leg['style'] == 'CALL':
                        assert leg['strike'] == 0.5 or leg['strike'] == 0.25 or leg['strike'] == 0.1
                    if leg['style'] == 'PUT':
                        assert leg['strike'] == -0.5 or leg['strike'] == -0.25 or leg['strike'] == -0.1

                    fx_pair_name = leg['base_currency'] + leg['term_currency']
                    fx_spot = market.get_fx_spot(fx_pair_name)
                    vol_surface = market.get_fx_vol_surface(fx_pair_name)

                    tranche_expiration_date = self.find_next_expiry(
                        tranche['initial_tenor'] if first_day else leg['tenor'])
                    if tranche_expiration_date not in trade_date_info['trading_dates'] and \
                            tranche_expiration_date < max(trade_date_info['trading_dates']):
                        tranche_expiration_date = min(np.array(trade_date_info['trading_dates'])[
                                                          [x >= tranche_expiration_date for x in
                                                           trade_date_info['trading_dates']]])

                    TTM = (tranche_expiration_date - self.time_stamp).days / 365
                    vols_by_exp = vol_surface.vols[TTM_conv[tranche['initial_tenor'] if first_day else leg['tenor']]]
                    if leg['style'] == 'PUT' and leg['strike'] == -0.5:
                        vol = vols_by_exp[vol_surface.delta_strikes.index(-leg['strike'])]
                    else:
                        vol = vols_by_exp[vol_surface.delta_strikes.index(leg['strike'])]

                    fwd = vol_surface.get_forward(tranche_expiration_date)
                    disc = fx_spot / fwd
                    F_K = delta_strike_to_relative(disc, leg['strike'], vol, TTM,
                                                   True if leg['style'] == 'CALL' else False)
                    abs_strike = (fwd / F_K)
                    option_to_trade = Option(
                        leg['base_currency'], leg['base_currency'], leg['term_currency'], tranche_expiration_date,
                        abs_strike, leg['style'] == 'CALL', False, 1.0, 'America/New_York', listed_ticker=None)
                    assert len(leg['sizing']) == 1
                    size_by = list(leg['sizing'].keys())[0]
                    option_to_trade_risk = option_to_trade.price(market=market, valuer=self.strategy.valuer_map[Option],
                                                                 return_struc=True, vol_override=vol)
                    option_to_trade_price = option_to_trade_risk['price']
                    option_to_trade_delta = round(option_to_trade_risk['forward_delta'], 2)
                    leg_sizing = leg['sizing'][size_by]

                    if 'scale_by_nav' in self.parameters and self.parameters['scale_by_nav']:
                        leg_sizing *= prev_nav

                    if size_by != 'quantity':
                        leg_sizing /= option_to_trade_risk[size_by]
                        if 'scale_by_nav' in self.parameters and self.parameters['scale_by_nav']:
                            assert abs(
                                leg['sizing'][size_by] * prev_nav - leg_sizing * option_to_trade_risk[size_by]) < 0.001
                        else:
                            if not (abs(leg['sizing'][size_by] - leg_sizing * option_to_trade_risk[size_by]) < 0.001):
                                print("wrong", self.time_stamp)
                            assert abs(leg['sizing'][size_by] - leg_sizing * option_to_trade_risk[size_by]) < 0.001

                    if TTM in list(TTM_conv.keys()):
                        assert abs(option_to_trade_delta - leg['strike']) < 0.001
                    else:
                        assert abs(round(option_to_trade_risk['forward_delta'] - leg['strike'], 3)) <= 0.01

                    portfolio.trade(option_to_trade, leg_sizing, option_to_trade_price, leg['term_currency'],
                                    position_path=(leg_name, tranche_name))
                    # update <trade_date_info>
                    update_dict = {'trade_date': self.time_stamp, 'trade_date_vol': vol}
                    trade_date_info[option_to_trade.name()] = update_dict

                    # upate <option_trade_log>
                    if self.time_stamp in option_trade_log:
                        option_trade_log_dt = option_trade_log[self.time_stamp]
                    else:
                        option_trade_log_dt = {}
                    option_to_trade_risk['qty'] = leg_sizing
                    option_to_trade_risk['strike'] = option_to_trade.strike
                    option_to_trade_risk['expiry'] = option_to_trade.expiration
                    option_to_trade_risk['OT'] = 'Call' if option_to_trade.is_call else 'Put'
                    option_trade_log_dt[leg_name] = option_to_trade_risk
                    option_trade_log[self.time_stamp] = option_trade_log_dt

        # reset valuer as the valuer needs state dependent information
        # this uses a global variable so this line has to be called right before you use it
        # TODO: need to move away from using global variable
        self.strategy.valuer_map[Option] = FXOptionBSValuer('OIS', trade_date_info)

        # add price manually
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        post_state = FXOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        return post_state


class ExpireOptions(Event):
    def execute(self, state: StrategyState, daily_states):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()
        # expire options
        self.parameters['expires'].expire(self.time_stamp, portfolio)
        return FXOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)


class DailyMtM(Event):
    def execute(self, state: StrategyState, daily_states):
        print('running on %s' % self.time_stamp)

        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # reset valuer as the valuer needs state dependent information
        # this uses a global variable so this line has to be called right before you use it
        # TODO: need to move away from using global variable
        self.strategy.valuer_map[Option] = FXOptionBSValuer('OIS', trade_date_info)

        # value portfolio
        price = portfolio.price_at_market(market, fields=['price', 'forward_delta', 'vega', 'theta', 'gamma'],
                                          valuer_map_override=self.strategy.valuer_map)
        return FXOptionsDailyState(self.time_stamp, portfolio, price[0], 0.0)


class DeltaHedgeEOD(Event):
    def delta_to_hedge(self, to_hedge, market):
        fwd_delta = 0
        for instrument in to_hedge:
            #  handle when <to_hedge> contains tradable rather than PTF element
            if hasattr(instrument, 'quantity'):
                qty = instrument.quantity
                tradable = instrument.tradable
            else:
                qty = 1
                tradable = instrument
            _fx_pair_name = f'{tradable.underlying}{tradable.currency}'
            _fx_spot = market.get_fx_spot(_fx_pair_name)
            _vol_surface = market.get_fx_vol_surface(_fx_pair_name)

            if isinstance(tradable, Option):

                if tradable.name() in trade_date_info and self.time_stamp == trade_date_info[tradable.name()][
                    'trade_date']:
                    trade_info = trade_date_info[tradable.name()]
                    fwd_delta += qty * tradable.price(market=market,
                                                      valuer=self.strategy.valuer_map[Option],
                                                      calc_types='forward_delta',
                                                      vol_override=trade_info['trade_date_vol'])
                else:
                    fwd_delta += qty * tradable.price(market=market,
                                                      valuer=self.strategy.valuer_map[Option],
                                                      calc_types='forward_delta')
            elif isinstance(tradable, FXforward):
                fwd = _vol_surface.get_forward(tradable.expiration)
                fwd_delta += qty * _fx_spot / fwd
            else:
                raise RuntimeError('Unable to hedge tradable ' + tradable.name())
        return fwd_delta

    def fwd_price(self, tradable, market):
        if hasattr(tradable, 'tradable'):
            tradable = tradable.tradable
        _fx_pair_name = f'{tradable.underlying}{tradable.currency}'
        _fx_spot = market.get_fx_spot(_fx_pair_name)
        _vol_surface = market.get_fx_vol_surface(_fx_pair_name)

        if isinstance(tradable, FXforward):
            fwd = _vol_surface.get_forward(tradable.expiration)
            return fwd
        else:
            raise RuntimeError('Unable to hedge tradable ' + tradable.name())

    def execute(self, state: StrategyState, daily_states):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # compute delta by underlier and then by expiry
        positions = {k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}
        all_und = list(set([x.tradable.underlying for x in positions.values()]))
        delta_by_und = {}
        # iterate over underliers
        for und in all_und:
            delta_by_exp = {}
            sub_ptf = [x for x in list(positions.values()) if x.tradable.underlying == und]
            exp_by_und = [x.tradable.expiration for x in sub_ptf]
            exp_by_und = np.unique(np.array(exp_by_und))
            # iterate over expiries
            for exp in exp_by_und:
                to_hedge = [x for x in sub_ptf if x.tradable.expiration == exp]
                fwd_pos = [x for x in to_hedge if not hasattr(x.tradable, 'strike')]

                # find hedge instrument
                if len(fwd_pos) == 0:
                    # enter new fwd pos
                    opt = [x for x in to_hedge if hasattr(x.tradable, 'strike')][0].tradable
                    fwd_pos = FXforward(opt.currency, opt.underlying, opt.currency, exp, opt.tz_name)
                    fwd_qty = 0
                else:
                    assert len(fwd_pos) == 1
                    fwd_pos = fwd_pos[0]
                    fwd_qty = fwd_pos.quantity
                    fwd_pos = fwd_pos.tradable
                # unwind fwd, if applicable
                fwd_px = self.fwd_price(fwd_pos, market)
                portfolio.trade(fwd_pos, -fwd_qty, fwd_px, fwd_pos.currency)

                # compute fwd qty to hedge
                pos_upd = {k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}
                to_hedge = [x for x in [x for x in list(pos_upd.values()) if x.tradable.underlying == und] if
                            x.tradable.expiration == exp]
                delta_by_exp[exp] = self.delta_to_hedge(to_hedge, market)
                hedge_delta = self.delta_to_hedge([fwd_pos], market)
                net_delta = -delta_by_exp[exp] / hedge_delta

                # TODO: generalise DH thresh-hold
                position_thresh = 0
                delta_thresh = 0
                if abs(net_delta) < delta_thresh or abs(fwd_qty - net_delta) < position_thresh:
                    net_delta = 0

                # flatten delta
                if abs(net_delta) > 0:
                    portfolio.trade(fwd_pos, net_delta, fwd_px, fwd_pos.currency)

                # check updated sub portfolio (by und and expiry) has delta < thresh-hold
                new_pos = {k: v for k, v in portfolio.net_positions().items() if not 'Cash' in k}
                sub_ptf_upd = [x for x in list(new_pos.values()) if x.tradable.underlying == und]
                to_hedge_upd = [x for x in sub_ptf_upd if x.tradable.expiration == exp]
                delta_by_exp_upd = self.delta_to_hedge(to_hedge_upd, market)
                if abs(fwd_qty - net_delta) > position_thresh:
                    assert abs(delta_by_exp_upd - delta_thresh) < 1e-8

        delta_by_und[und] = delta_by_exp

        # add price manually
        price = portfolio.price_at_market(market, fields='price', valuer_map_override=self.strategy.valuer_map)
        post_state = FXOptionsDailyState(self.time_stamp, portfolio, 0.0, 0.0)

        if 'trading_cost' in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = post_state.portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp)

        return post_state


class FXOptionsDaily(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.valuer_map = {
            Option: FXOptionBSValuer('OIS'),
            FXforward: FXForwardDataValuer(),
        }

        # self.force_run = True

        all_data = self.data_containers['vol'].get_fx_surface()
        trade_dates = []
        for dt, v in all_data.items():
            assert len(v) == 1
            if len(v[list(v.keys())[0]].forwards) > 0:
                trade_dates.append(dt)
        rates_data = self.data_containers['df_curve'].get_spot_rate_curves()
        rates_data_dt = list(rates_data.keys())
        trade_dates = list(set(trade_dates) & set(rates_data_dt))
        trade_dates.sort()

        self.trading_days = trade_dates
        trade_date_info['trading_dates'] = trade_dates

        self.inc_daily_states = True

        # initialize the trading cost module
        if 'variable_costs' in self.parameters.keys():
            cost_params = self.parameters['variable_costs']
            self.parameters['trading_cost'] = VariableVegaCost(cost_params['cost_cap'], cost_params['cost_floor'],
                                                               cost_params['cost_scale'], cost_params['delta_cost'],
                                                               self.backtest_market, self.valuer_map)
        elif 'flat_costs' in self.parameters.keys():
            cost_params = self.parameters['flat_costs']
            self.parameters['trading_cost'] = FlatTradingCost(cost_params['tc_delta'], cost_params['tc_vega'])
        self.parameters['expires'] = ExpireFXOptionContracts(self.backtest_market, self.valuer_map)
        # TODO: only support USD as strategy currency
        # assert self.currency == 'USD'

    def generate_events(self, dt: datetime):
        if 'hedge_eod' not in self.parameters or ('hedge_eod' in self.parameters and self.parameters['hedge_eod']):
            return [ExpireOptions(dt, self), TradeOptions(dt, self), DeltaHedgeEOD(dt, self), DailyMtM(dt, self)]
        else:
            return [ExpireOptions(dt, self), TradeOptions(dt, self), DailyMtM(dt, self)]
