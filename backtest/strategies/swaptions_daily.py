import copy
from datetime import datetime, date
from ...analytics.swaptions import cube_interpolate, atmf_yields_interpolate, rate_interpolate
from ...backtest.unwind import BaseUnwindEvent
from ...backtest.costs import FlatTradingCost, RiskBasedTradingCostByTradableType
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...constants.business_day_convention import BusinessDayConvention
from ...dates.utils import add_tenor, add_business_days, bdc_adjustment, tenor_to_days, date_to_datetime, count_business_days
from ...dates.holidays import get_holidays
from ...tradable.portfolio import Portfolio
from ...tradable.swaption import Swaption
from ...tradable.forwardstartswap import ForwardStartSwap
from ...backtest.rolls import RollSwaptionContracts
from ...backtest.delta_hedges import SwaptionDailyDH
from ...backtest.tranche import Tranche, RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ...valuation.forwardstartswap_valuer import ForwardStartSwapValuer
from ...valuation.swaption_norm_valuer import SwaptionNormValuer
from ...tradable.cash import Cash
from ...tradable.constant import Constant
import math
import pandas as pd
import numpy as np


class SwaptionsDailyState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, legs_info, cost_info=None):
        self.price = price
        self.cost = cost
        self.legs_info = legs_info
        self.cost_info = cost_info
        super().__init__(time_stamp, portfolio)


def find_next_contract(dt, market, leg_params, strategy):
    atmf_yields = {dt: market.get_forward_rates(leg_params["currency"], leg_params["curve_type"])}
    spot_rates = {dt: market.get_spot_rates(leg_params["currency"], leg_params["curve_type"]).data_dict}
    target_expiration = bdc_adjustment(add_tenor(dt, leg_params["expiry"]), convention='following', holidays=strategy.holidays)
    if leg_params["strike_type"] == 'forward':
        target_strike = atmf_yields_interpolate(atmf_yields, spot_rates, dt, leg_params["tenor"], target_expiration) + leg_params["strike"] / 100.0
    elif leg_params["strike_type"] == 'spot':
        target_strike = rate_interpolate(spot_rates, dt, tenor_to_days(leg_params["tenor"])/360) + leg_params["strike"] / 100.0
    elif leg_params["strike_type"] == "delta":
        atmf = atmf_yields_interpolate(atmf_yields, spot_rates, dt, leg_params["tenor"], target_expiration)
        swaption = Swaption(leg_params["currency"], target_expiration, leg_params["tenor"], atmf, leg_params["style"], leg_params["curve_type"])
        target_strike = strategy.valuer_map[Swaption].get_strike_from_delta(swaption, leg_params["strike"], market)
    else:
        raise RuntimeError(f'Unknown strike type {leg_params["strike_type"]}')
    target_swaption = Swaption(leg_params["currency"], target_expiration, leg_params["tenor"], target_strike, leg_params["style"], leg_params["curve_type"])
    return target_swaption


class UnwindEvent(BaseUnwindEvent):
    def execute(self, state: StrategyState):
        if not self.need_unwind:
            return state

        portfolio = self.unwind_pfo(target_tradable_type=Swaption, state=state)
        return SwaptionsDailyState(self.time_stamp, portfolio, state.price, state.cost, state.legs_info, state.cost_info)


class EODTradeEvent(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()
        non_cash_pfo = {ky: v for ky, v in portfolio.net_positions().items() if not isinstance(v.tradable, Cash)}

        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # allow trading on start_day
        start_date = bdc_adjustment(self.strategy.start_date, convention=BusinessDayConvention.FOLLOWING, holidays=self.strategy.holidays)
        self.trade_first_day = (self.time_stamp == start_date) and self.parameters['trade_first_day']

        # trade
        legs_info = {}
        if "tranches" in self.parameters:
            # back compatibility
            has_swaption_trade = False
            if portfolio.is_empty() or len(non_cash_pfo) < 1:
                # this is the start day, we need to trade into each leg and tranche
                for leg_name, rolls_by_tranche in self.parameters['rolls_by_leg_tranche'].items():
                    leg_config = self.parameters['legs'][leg_name]
                    for tranche_name, roll in rolls_by_tranche.items():
                        initial_swaption = roll.find_initial_contract(self.time_stamp)
                        initial_swaption_price = initial_swaption.price(market=market, valuer=self.strategy.valuer_map[Swaption],
                                                                        calc_types='price')
                        portfolio.trade(initial_swaption, leg_config['sizing'], initial_swaption_price,
                                        leg_config['currency'], position_path=(leg_name, tranche_name))
                        legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['swaption_traded'] = True
                        has_swaption_trade = True
            else:
                # roll
                for leg_name, rolls_by_tranche in self.parameters['rolls_by_leg_tranche'].items():
                    for tranche_name, roll in rolls_by_tranche.items():
                        roll_geq = ( 'roll_geq' in self.parameters['legs'][leg_name] ) and ( self.parameters['legs'][leg_name]['roll_geq'] )

                        # is any swaption rolled
                        swaptions = portfolio.get_position((leg_name, tranche_name)).find_children_of_tradable_type_recursive(Swaption)
                        swaptions = [x[1].tradable for x in swaptions]
                        swaption_traded = False
                        for swaption in swaptions:
                            if roll.is_roll_time(self.time_stamp, roll.roll_time(self.time_stamp, swaption, roll_geq=roll_geq), roll_geq=roll_geq):
                                swaption_traded = True
                        legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['swaption_traded'] = swaption_traded
                        if swaption_traded:
                            has_swaption_trade = True
                        # roll
                        roll.roll(self.time_stamp, portfolio.get_position((leg_name, tranche_name)), roll_type='fixed_units', roll_geq=roll_geq)
        else:
            # use regular tranching
            # unwind tranches and move cash into leg level to avoid keeping too many tranche portfolios
            has_swaption_trade = False
            for leg_name, leg in self.parameters['legs'].items():
                legs_info.setdefault(leg_name, {})['tranche_name_map'] = copy.deepcopy(state.legs_info[leg_name]['tranche_name_map']) if state.legs_info is not None else {}
                leg_portfolio = portfolio.get_position(leg_name)
                tranche = leg["tranche"]
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        if isinstance(tranche_portfolio, Portfolio):
                            # unwind portfolio when it is the unwind date
                            tranche_name_mapped = legs_info[leg_name]['tranche_name_map'][tranche_name]
                            if isinstance(tranche, RollingAtExpiryDailyTranche) and tranche_name_mapped is None:
                                continue
                            if tranche_name_mapped is not None and Tranche.parse_tranche_key(tranche_name_mapped)[1].date() == self.time_stamp.date():
                                Tranche.unwind_tranche_portfolio(leg_name, tranche_name, portfolio, leg_portfolio, tranche_portfolio, market, self.strategy)
                                # leg["tranche"].move_tranche_portfolio_with_cash_only_to_leg_level(tranche_name, leg_portfolio, tranche_portfolio)
                                has_swaption_trade = True
                                legs_info[leg_name]['tranche_name_map'][tranche_name] = None
                # roll a tranche or initial position
                if tranche.is_entry_datetime(self.time_stamp) or self.trade_first_day:
                    usable_tranche_name = [k for k, v in legs_info[leg_name]['tranche_name_map'].items() if v is None]
                    if not isinstance(tranche, RollingAtExpiryDailyTranche):
                        assert len(usable_tranche_name) <= 1
                    largest_tranche_name = max(list(legs_info[leg_name]['tranche_name_map'].keys())) if len(list(legs_info[leg_name]['tranche_name_map'].keys())) else 0
                    new_tranche_name = usable_tranche_name[0] if len(usable_tranche_name) >= 1 else largest_tranche_name + 1
                    swaption_to_trade = find_next_contract(self.time_stamp, market, leg, self.strategy)
                    calc_types = ["price"]
                    sizing_measure = leg["sizing_measure"]
                    sizing_target = leg["sizing_target"]
                    calc_types.append(sizing_measure)
                    swaption_to_trade_calc_res = swaption_to_trade.price(market=market, valuer=self.strategy.valuer_map[Swaption], calc_types=calc_types)
                    swaption_to_trade_price = swaption_to_trade_calc_res[0]

                    # sizing
                    if sizing_measure == "units":
                        sizing_measure_value = 1
                    elif sizing_measure == "notional":
                        sizing_measure_value = swaption_to_trade.underlying.price(market, calc_types="price")
                    elif sizing_measure == 'strike_notional':
                        sizing_measure_value = swaption_to_trade.strike
                    else:
                        option_sizing_measure_value = swaption_to_trade_calc_res[1]
                        sizing_measure_value = option_sizing_measure_value
                    units_to_trade = sizing_target / sizing_measure_value
                    if isinstance(tranche, RollingAtExpiryTranche):
                        tranche.set_next_roll_date(swaption_to_trade.expiration, self.strategy.holidays)
                    elif isinstance(tranche, RollingAtExpiryDailyTranche):
                        expiry_dt = date_to_datetime(swaption_to_trade.expiration.date())
                        tranche.update_tranche(self.time_stamp, expiry_dt, self.strategy.holidays)
                    exit_date = tranche.get_exit_datetime(self.time_stamp)
                    new_tranche_name_mapped = Tranche.make_tranche_key(self.time_stamp, exit_date)
                    legs_info[leg_name]['tranche_name_map'][new_tranche_name] = new_tranche_name_mapped

                    units_to_trade *= tranche.get_tranche_fraction(self.time_stamp)
                    portfolio.trade(swaption_to_trade, units_to_trade, swaption_to_trade_price, leg['currency'], position_path=(leg_name, new_tranche_name))
                    has_swaption_trade = True

        # delta hedge
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
        hedge_position_path = ("delta_hedge", ) if self.parameters["use_delta_hedge_path"] else ()
        for leg_name, leg in self.parameters['legs'].items():
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is None:
                continue
            leg_position_names = list(leg_portfolio.get_positions().keys())
            for tranche_name in leg_position_names:
                tranche_portfolio = leg_portfolio.get_position(tranche_name)
                if not isinstance(tranche_portfolio, Portfolio):
                    continue
                hedged_today = False
                unwind_existing_hedge = legs_info.get(leg_name, {}).get(tranche_name, {}).get('swaption_traded', False)
                if self.parameters['legs'][leg_name]['delta_hedge'] == 'Daily':
                    self.parameters['delta_hedger'].delta_hedge(
                        self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge, hedge_position_path=hedge_position_path)
                    last_scheduled_hedge_date = self.time_stamp
                    hedged_today = True
                elif self.parameters['legs'][leg_name]['delta_hedge'] == 'Unhedged':
                    last_scheduled_hedge_date = None
                elif isinstance(self.parameters['legs'][leg_name]['delta_hedge'], dict) and 'rate_move_threshold' in self.parameters['legs'][leg_name]['delta_hedge']:
                    last_scheduled_hedge_date = None
                    threshold = self.parameters['legs'][leg_name]['delta_hedge']['rate_move_threshold']
                    swaptions = portfolio.get_position((leg_name, tranche_name)).find_children_of_tradable_type_recursive(Swaption)
                    assert len(swaptions) == 1
                    current_forward_rate = swaptions[0][1].forward_rate
                    last_hedge_date_forward_rate = None if state.legs_info is None else state.legs_info[leg_name][tranche_name]['last_hedge_date_forward_rate']
                    if last_hedge_date_forward_rate is not None:
                        if unwind_existing_hedge or abs(last_hedge_date_forward_rate - current_forward_rate) > threshold:
                            self.parameters['delta_hedger'].delta_hedge(self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                            hedged_today = True
                            last_hedge_date_forward_rate = current_forward_rate
                    else:
                        self.parameters['delta_hedger'].delta_hedge(self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                        hedged_today = True
                        last_hedge_date_forward_rate = current_forward_rate
                    legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['last_hedge_date_forward_rate'] = last_hedge_date_forward_rate
                elif isinstance(self.parameters['legs'][leg_name]['delta_hedge'], dict) and ('rate_move_threshold_in_vol' in self.parameters['legs'][leg_name]['delta_hedge'] or 'rate_move_threshold_in_vol_fixed_expiry' in self.parameters['legs'][leg_name]['delta_hedge']):
                    last_scheduled_hedge_date = None
                    if 'rate_move_threshold_in_vol' in self.parameters['legs'][leg_name]['delta_hedge']:
                        threshold_in_vol = self.parameters['legs'][leg_name]['delta_hedge']['rate_move_threshold_in_vol']
                        swaptions = portfolio.get_position((leg_name, tranche_name)).find_children_of_tradable_type_recursive(Swaption)
                        assert len(swaptions) == 1
                        current_forward_rate = swaptions[0][1].forward_rate
                        vol_cube = {self.time_stamp: market.get_swaption_vol_cube(swaptions[0][1].tradable.currency)}
                        fwd_rate_curve = {self.time_stamp: market.get_forward_rates(swaptions[0][1].tradable.currency, swaptions[0][1].tradable.curve)}
                        strike_vol = cube_interpolate(vol_cube, fwd_rate_curve, self.time_stamp, swaptions[0][1].tradable.tenor, swaptions[0][1].tradable.expiration, swaptions[0][1].tradable.strike, linear_in_vol=True)
                        threshold = threshold_in_vol * strike_vol / math.sqrt(252)
                    elif 'rate_move_threshold_in_vol_fixed_expiry' in self.parameters['legs'][leg_name]['delta_hedge']:
                        threshold_in_vol = self.parameters['legs'][leg_name]['delta_hedge']['rate_move_threshold_in_vol_fixed_expiry']
                        vol_cube = {self.time_stamp: market.get_swaption_vol_cube(leg['currency'])}
                        fwd_rate_curve = {self.time_stamp: market.get_forward_rates(leg['currency'], leg['curve_type'])}
                        spot_rate_curve = {self.time_stamp: market.get_spot_rates(leg['currency'], leg['curve_type']).data_dict}
                        atmf = atmf_yields_interpolate(fwd_rate_curve, spot_rate_curve, self.time_stamp, leg['tenor'], add_tenor(self.time_stamp, leg['expiry']))
                        vol = cube_interpolate(vol_cube, fwd_rate_curve, self.time_stamp, leg['tenor'], add_tenor(self.time_stamp, leg['expiry']), atmf, linear_in_vol=True)
                        current_forward_rate = atmf
                        threshold = threshold_in_vol * vol / math.sqrt(252)
                    last_hedge_date_forward_rate = None if state.legs_info is None else state.legs_info[leg_name][tranche_name]['last_hedge_date_forward_rate']
                    if last_hedge_date_forward_rate is not None:
                        if unwind_existing_hedge or abs(last_hedge_date_forward_rate - current_forward_rate) > threshold:
                            self.parameters['delta_hedger'].delta_hedge(self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                            hedged_today = True
                            last_hedge_date_forward_rate = current_forward_rate
                    else:
                        self.parameters['delta_hedger'].delta_hedge(self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                        hedged_today = True
                        last_hedge_date_forward_rate = current_forward_rate
                    legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['last_hedge_date_forward_rate'] = last_hedge_date_forward_rate
                elif isinstance(self.parameters['legs'][leg_name]['delta_hedge'], list):
                    last_scheduled_hedge_date = None if state.legs_info is None else state.legs_info[leg_name][tranche_name]['last_scheduled_hedge_date']
                    if self.time_stamp.date() in [x.date() for x in self.parameters['legs'][leg_name]['delta_hedge']]:
                        self.parameters['delta_hedger'].delta_hedge(
                            self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                        last_scheduled_hedge_date = self.time_stamp
                        hedged_today = True
                else:
                    last_scheduled_hedge_date = None if state.legs_info is None else state.legs_info[leg_name][tranche_name]['last_scheduled_hedge_date']
                    next_hedge_date = None
                    if last_scheduled_hedge_date is not None:
                        if self.parameters['legs'][leg_name]['delta_hedge'][-1].upper() == 'B':
                            next_hedge_date = add_business_days(last_scheduled_hedge_date, int(self.parameters['legs'][leg_name]['delta_hedge'][:-1]), self.strategy.holidays)
                        else:
                            next_hedge_date = add_tenor(last_scheduled_hedge_date, self.parameters['legs'][leg_name]['delta_hedge'])
                    if unwind_existing_hedge or self.time_stamp >= next_hedge_date:
                        self.parameters['delta_hedger'].delta_hedge(
                            self.time_stamp, portfolio.get_position((leg_name, tranche_name)), unwind_existing_hedge)
                        if next_hedge_date is None or self.time_stamp >= next_hedge_date:
                            last_scheduled_hedge_date = self.time_stamp # self.time_stamp if next_hedge_date is None else next_hedge_date
                        hedged_today = True
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['last_scheduled_hedge_date'] = last_scheduled_hedge_date
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['hedged_today'] = hedged_today

        # cost
        cost_info = {}
        if "trading_cost" in self.parameters:
            portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
            price_pre_cost = portfolio.aggregate('price')
            trading_costs, trading_costs_by_tradable_by_risk = self.parameters['trading_cost'].apply(portfolio, state.portfolio)
            cost_info_types = self.parameters.get("cost_info_types", ["cost_by_risk"])
            if "cost_by_risk" in cost_info_types:
                cost_by_risk = {}
                for tradable, trading_costs_by_risk in trading_costs_by_tradable_by_risk.items():
                    for risk_type, trading_cost in trading_costs_by_risk.items():
                        cost = cost_by_risk.setdefault(risk_type, 0)
                        cost_by_risk[risk_type] = cost + trading_cost
                cost_info["cost_by_risk"] = cost_by_risk
            if "cost_by_tranche_by_risk" in cost_info_types:
                cost_by_tranche_by_risk = {}
                for tradable, trading_costs_by_risk in trading_costs_by_tradable_by_risk.items():
                    tranche_names = []
                    for leg_name, leg in self.parameters['legs'].items():
                        for tranche_name, tranche_name_mapped in legs_info[leg_name]['tranche_name_map'].items():
                            tranche_portfolio = portfolio.get_position((leg_name, tranche_name))
                            for k, v in tranche_portfolio.net_positions().items():
                                if v.tradable.name() == tradable.name():
                                    tranche_names.append(tranche_name)
                        if state.legs_info is not None:
                            for tranche_name, tranche_name_mapped in state.legs_info[leg_name]['tranche_name_map'].items():
                                tranche_portfolio = state.portfolio.get_position((leg_name, tranche_name))
                                for k, v in tranche_portfolio.net_positions().items():
                                    if v.tradable.name() == tradable.name():
                                        tranche_names.append(tranche_name)
                    tranche_names = list(set(tranche_names))
                    assert len(tranche_names) == 1, f"found {len(tranche_names)} tranche(s) to map the cost of trading {tradable.name()}"
                    for risk_type, trading_cost in trading_costs_by_risk.items():
                        cost = cost_by_tranche_by_risk.setdefault(tranche_names[0], {}).setdefault(risk_type, 0)
                        cost_by_tranche_by_risk[tranche_names[0]][risk_type] = cost + trading_cost
                cost_info["cost_by_tranche_by_risk"] = cost_by_tranche_by_risk

        # value portfolio
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
        price = portfolio.aggregate('price')
        cost = price - price_pre_cost if "trading_cost" in self.parameters else 0
        total_delta = portfolio.aggregate('delta')
        total_vega = portfolio.aggregate('vega')

        for leg_name, leg in self.parameters['legs'].items():
            leg_portfolio = portfolio.get_position(leg_name)
            if leg_portfolio is None:
                continue
            leg_position_names = list(leg_portfolio.get_positions().keys())
            for tranche_name in leg_position_names:
                tranche_portfolio = leg_portfolio.get_position(tranche_name)
                if not isinstance(tranche_portfolio, Portfolio):
                    continue
                this_swaption = portfolio.get_position((leg_name, tranche_name)).find_children_of_tradable_type(Swaption)
                # allow early unwind so no swaption in pfo
                if len(this_swaption) == 0:
                    continue
                assert len(this_swaption) == 1
                this_swaption = this_swaption[0][1]

                vol_cube = {self.time_stamp: market.get_swaption_vol_cube(this_swaption.tradable.currency)}
                fwd_rate_curve = {self.time_stamp: market.get_forward_rates(this_swaption.tradable.currency, this_swaption.tradable.curve)}
                strike_vol = cube_interpolate(vol_cube, fwd_rate_curve, self.time_stamp, this_swaption.tradable.tenor, this_swaption.tradable.expiration, this_swaption.tradable.strike, linear_in_vol=True)

                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['forward_rate'] = this_swaption.forward_rate
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['strike_vol'] = strike_vol
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['gross_price'] = tranche_portfolio.aggregate('price')
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['delta'] = tranche_portfolio.aggregate('delta')
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['vega'] = tranche_portfolio.aggregate('vega')

        legs_info['total_delta'] = total_delta
        legs_info['total_vega'] = total_vega
        legs_info['has_swaption_trade'] = has_swaption_trade

        return SwaptionsDailyState(self.time_stamp, portfolio, price, cost, legs_info, cost_info)


class MTMEvent(Event):
    def execute(self, state: StrategyState):
        portfolio = state.portfolio.clone()
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        price = portfolio.price_at_market(market, fields=['price', 'delta', 'vega', 'theta'], valuer_map_override=self.strategy.valuer_map)
        return SwaptionsDailyState(self.time_stamp, portfolio, price[0], state.cost, state.legs_info, state.cost_info)


class SwaptionsDaily(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # init optional parameters
        self.parameters["mtm"] = self.parameters.get("mtm", False)
        self.parameters["trade_first_day"] = self.parameters.get("trade_first_day", False)
        self.parameters["use_delta_hedge_path"] = self.parameters.get("use_delta_hedge_path", False)

        self.valuer_map = {
            Swaption: SwaptionNormValuer(df_rates_type=self.parameters['df_rates_type']),
            ForwardStartSwap: ForwardStartSwapValuer(discount_rate_curve_type=self.parameters['df_rates_type']),
        }

        # initialize the trading cost module
        if "tc_rate" in self.parameters:
            self.parameters['trading_cost'] = RiskBasedTradingCostByTradableType(self.parameters['tc_rate'])

        # convert sizing into sizing_measure and sizing_target
        for leg_name, leg in self.parameters['legs'].items():
            if "sizing" in leg:
                if "sizing_measure" in leg or "sizing_target" in leg:
                    raise Exception(f"Cannot set both sizing and sizing_measure/sizing_target")
                leg["sizing_measure"] = "units"
                leg["sizing_target"] = leg["sizing"]
        # initialize the roll logic module by leg and tranche
        if "tranches" in self.parameters:
            rolls_by_leg_tranche = {}
            for leg_name, leg in self.parameters['legs'].items():
                for tranche_name, tranche in self.parameters['tranches'].items():
                    rolls_by_leg_tranche.setdefault(leg_name, {})[tranche_name] = RollSwaptionContracts(
                        leg['expiry'], leg['tenor'], leg['strike'], leg['strike_type'], leg['style'], leg['currency'], leg["curve_type"],
                        month_list=tranche['roll_months'], day=tranche['roll_day'], holidays=self.holidays,
                        backtest_market=self.backtest_market, valuer_map=self.valuer_map,
                    )
            self.parameters['rolls_by_leg_tranche'] = rolls_by_leg_tranche

        delta_hedger = (SwaptionDailyDH(backtest_market=self.backtest_market, valuer_map=self.valuer_map))
        self.parameters['delta_hedger'] = delta_hedger
        #self.hedging_holidays = get_holidays("NYC", self.start_date, self.end_date)

    def generate_events(self, dt: datetime):
        events = []
        events.append(UnwindEvent(dt, self))
        events.append(EODTradeEvent(dt, self))
        if self.parameters["mtm"]:
            events.append(MTMEvent(dt, self))
        return events

    @staticmethod
    def results_to_series(results, cost_info_types):
        # daily summary
        records = []
        cost_by_tranche_cum = {}
        for x in results:
            record = {
                'date': x.time_stamp,
                'pnl': x.price,
                'total_delta': x.legs_info['total_delta'],
                'total_vega': x.legs_info['total_vega'],
                'has_swaption_trade': x.legs_info['has_swaption_trade'],
            }
            if "cost_by_risk" in cost_info_types:
                record['vega_cost'] = x.cost_info["cost_by_risk"].get("vega", 0)
                record['delta_cost'] = x.cost_info["cost_by_risk"].get("delta", 0)
            # representiative info
            first_leg_first_tranche = list(x.legs_info.values())[0].get(1, None)
            if first_leg_first_tranche is None:
                record['forward_rate(1)'] = None
                record['strike_vol(1)'] = None
                record['hedged(1)'] = None
            else:
                record['forward_rate(1)'] = first_leg_first_tranche['forward_rate']
                record['strike_vol(1)'] = first_leg_first_tranche['strike_vol']
                record['hedged(1)'] = first_leg_first_tranche['hedged_today']

            # per leg and per tranche info
            pnl_by_tranche = {}
            for leg_name, leg_info in x.legs_info.items():
                if leg_name in ["has_swaption_trade", "total_delta", "total_vega"]:
                    continue
                for tranche_name, tranche_info in leg_info.items():
                    if tranche_name in ["tranche_name_map"]:
                        continue
                    # record[f"{leg_name} T{tranche_name} gross_pnl"] = tranche_info['gross_price']
                    # record[f"{leg_name} T{tranche_name} delta"] = tranche_info['delta']
                    # record[f"{leg_name} T{tranche_name} vega"] = tranche_info['vega']
                    # record[f"{leg_name} T{tranche_name} hedged"] = tranche_info['hedged_today']
                    # record[f"{leg_name} T{tranche_name} forward_rate"] = tranche_info['forward_rate']
                    # record[f"{leg_name} T{tranche_name} strike_vol"] = tranche_info['strike_vol']

                    tranche_gross_pnl = pnl_by_tranche.setdefault(tranche_name, {}).get("gross_pnl", 0)
                    pnl_by_tranche[tranche_name]["gross_pnl"] = tranche_gross_pnl + tranche_info['gross_price']

            # per tranche info
            for tranche_name, tranche_info in pnl_by_tranche.items():
                record[f"T{tranche_name} gross_pnl"] = tranche_info['gross_pnl']
                if "cost_by_tranche_by_risk" in cost_info_types:
                    tranche_vega_cost = pnl_by_tranche.setdefault(tranche_name, {}).get("vega_cost", 0)
                    pnl_by_tranche[tranche_name]["vega_cost"] = tranche_vega_cost + x.cost_info["cost_by_tranche_by_risk"][tranche_name].get('vega', 0)
                    tranche_delta_cost = pnl_by_tranche.setdefault(tranche_name, {}).get("delta_cost", 0)
                    pnl_by_tranche[tranche_name]["delta_cost"] = tranche_delta_cost + x.cost_info["cost_by_tranche_by_risk"][tranche_name].get('delta', 0)

                    tranche_vega_cost_cum = cost_by_tranche_cum.setdefault(tranche_name, {}).get("vega_cost", 0)
                    cost_by_tranche_cum[tranche_name]["vega_cost"] = tranche_vega_cost_cum + x.cost_info["cost_by_tranche_by_risk"][tranche_name].get('vega', 0)
                    tranche_delta_cost_cum = cost_by_tranche_cum.setdefault(tranche_name, {}).get("delta_cost", 0)
                    cost_by_tranche_cum[tranche_name]["delta_cost"] = tranche_delta_cost_cum + x.cost_info["cost_by_tranche_by_risk"][tranche_name].get('delta', 0)

                    record[f"T{tranche_name} vega_cost"] = tranche_info['vega_cost']
                    record[f"T{tranche_name} delta_cost"] = tranche_info['delta_cost']
                    record[f"T{tranche_name} pnl"] = tranche_info['gross_pnl'] - cost_by_tranche_cum[tranche_name]['vega_cost'] - cost_by_tranche_cum[tranche_name]['delta_cost']
            records.append(record)

        summary_series = pd.DataFrame.from_dict(records)
        # portfolios
        portfolio_expanded = []
        for state in results:
            for k, v in state.portfolio.net_positions().items():
                if not isinstance(v.tradable, Constant) and not isinstance(v.tradable, Cash):
                    portfolio_expanded.append({
                        'date': state.time_stamp,
                        'position': k,
                        'quantity': v.quantity,
                        'unit_price': v.price,
                        'price': v.price * v.quantity,
                        'delta': v.delta * v.quantity,
                        'vega': v.vega * v.quantity,
                    })
        portfolio_expanded = pd.DataFrame.from_records(portfolio_expanded)
        portfolio_expanded = pd.merge(portfolio_expanded, summary_series, on="date", how="left")
        return summary_series, portfolio_expanded

    @staticmethod
    def adjust_series_from_unit_cost_series(unit_cost_summary_series, delta_cost, vega_cost, adjust_tranche_pnl=False):
        summary_series = unit_cost_summary_series.copy()

        daily_delta_cost_adj = summary_series['delta_cost'].values * (delta_cost - 1)
        daily_vega_cost_adj = summary_series['vega_cost'].values * (vega_cost - 1)
        cum_cost_adj = np.cumsum(daily_delta_cost_adj + daily_vega_cost_adj)
        summary_series['pnl'] = summary_series['pnl'] - cum_cost_adj
        summary_series['vega_cost'] = summary_series['vega_cost'] + daily_vega_cost_adj
        summary_series['delta_cost'] = summary_series['delta_cost'] + daily_delta_cost_adj

        # tranches
        if adjust_tranche_pnl:
            tranche_tags = [x.split(" ")[0] for x in summary_series.columns if x.startswith("T")]
            tranche_tags = list(set(tranche_tags))
            for tranche_tag in tranche_tags:
                daily_delta_cost_adj = summary_series[f'{tranche_tag} delta_cost'].values * (delta_cost - 1)
                daily_vega_cost_adj = summary_series[f'{tranche_tag} vega_cost'].values * (vega_cost - 1)
                cum_cost_adj = np.cumsum(daily_delta_cost_adj + daily_vega_cost_adj)
                summary_series[f'{tranche_tag} pnl'] = summary_series[f'{tranche_tag} pnl'] - cum_cost_adj
                summary_series[f'{tranche_tag} vega_cost'] = summary_series[f'{tranche_tag} vega_cost'] + daily_vega_cost_adj
                summary_series[f'{tranche_tag} delta_cost'] = summary_series[f'{tranche_tag} delta_cost'] + daily_delta_cost_adj

        return summary_series
