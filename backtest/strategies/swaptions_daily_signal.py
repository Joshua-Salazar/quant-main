import copy
from datetime import datetime, date
from ...analytics.swaptions import cube_interpolate, atmf_yields_interpolate, rate_interpolate
from ...backtest.costs import FlatTradingCost, RiskBasedTradingCostByTradableType
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...dates.utils import add_tenor, add_business_days, bdc_adjustment, tenor_to_days
from ...dates.holidays import get_holidays
from ...tradable.portfolio import Portfolio
from ...tradable.swaption import Swaption
from ...tradable.forwardstartswap import ForwardStartSwap
from ...backtest.rolls import RollSwaptionContracts
from ...backtest.delta_hedges import SwaptionDailyDH
from ...valuation.forwardstartswap_valuer import ForwardStartSwapValuer
from ...valuation.swaption_norm_valuer import SwaptionNormValuer
from ...tradable.cash import Cash
from ...tradable.constant import Constant
import math
import pandas as pd
import numpy as np


class SwaptionsDailySignalState(StrategyState):
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
    else:
        raise RuntimeError(f'Unknown strike type {leg_params["strike_type"]}')
    target_swaption = Swaption(leg_params["currency"], target_expiration, leg_params["tenor"], target_strike, leg_params["style"], leg_params["curve_type"])
    return target_swaption


class EODTradeEvent(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        market = self.strategy.backtest_market.get_market(self.time_stamp)

        legs_info = copy.deepcopy(state.legs_info)
        for leg_name, leg in self.parameters['legs'].items():
            leg_info = legs_info.setdefault("legs", {}).setdefault(leg_name, {})
            leg["trade_func"](portfolio, leg_name, leg_info, self.time_stamp, self.strategy, state)
            #leg["exit_func"](portfolio, leg_name, leg_info, self.time_stamp, self.strategy, state)
            leg["roll_func"](portfolio, leg_name, leg_info, self.time_stamp, self.strategy, state)
            legs_info[leg_name] = leg_info

        # delta hedge
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
        for leg_name, leg in self.parameters['legs'].items():
            leg_info = legs_info["legs"][leg_name]
            leg["hedge_func"](portfolio, leg_name, leg_info, self.time_stamp, self.strategy, state)
            legs_info[leg_name] = leg_info

        # cost
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
        price_pre_cost = portfolio.aggregate('price')
        trading_costs, trading_costs_by_tradable_by_risk = self.parameters['trading_cost'].apply(portfolio, state.portfolio)
        cost_info_types = self.parameters.get("cost_info_types", ["cost_by_risk"])
        cost_info = {}
        if "cost_by_risk" in cost_info_types:
            cost_by_risk = {}
            for tradable, trading_costs_by_risk in trading_costs_by_tradable_by_risk.items():
                for risk_type, trading_cost in trading_costs_by_risk.items():
                    cost = cost_by_risk.setdefault(risk_type, 0)
                    cost_by_risk[risk_type] = cost + trading_cost
            cost_info["cost_by_risk"] = cost_by_risk

        # value portfolio
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta', 'vega'], valuer_map_override=self.strategy.valuer_map)
        price = portfolio.aggregate('price')
        total_delta = portfolio.aggregate('delta')
        total_vega = portfolio.aggregate('vega')
        legs_info["total_delta"] = total_delta
        legs_info["total_vega"] = total_vega
        return SwaptionsDailySignalState(self.time_stamp, portfolio, price, price - price_pre_cost, legs_info, cost_info)


class SwaptionsDailySignal(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.valuer_map = {
            Swaption: SwaptionNormValuer(df_rates_type=self.parameters['df_rates_type']),
            ForwardStartSwap: ForwardStartSwapValuer(discount_rate_curve_type=self.parameters['df_rates_type']),
        }

        # initialize the trading cost module
        self.parameters['trading_cost'] = RiskBasedTradingCostByTradableType(self.parameters['tc_rate'])
        delta_hedger = SwaptionDailyDH(backtest_market=self.backtest_market, valuer_map=self.valuer_map)
        self.parameters['delta_hedger'] = delta_hedger

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]

    @staticmethod
    def results_to_series(results, cost_info_types):
        # daily summary
        records = []
        for x in results:
            record = {
                'date': x.time_stamp,
                'pnl': x.price,
                'total_delta': x.legs_info['total_delta'],
                'total_vega': x.legs_info['total_vega'],
            }
            if "cost_by_risk" in cost_info_types:
                record['vega_cost'] = x.cost_info["cost_by_risk"].get("vega", 0)
                record['delta_cost'] = x.cost_info["cost_by_risk"].get("delta", 0)

            # per leg info
            for leg_name, leg_info in x.legs_info["legs"].items():
                leg_notional = sum([v.get("notional", 0) for k, v in leg_info.get("tranches", {}).items()])
                record[f"{leg_name}_notional"] = leg_notional
                record[f"{leg_name}_entry"] = leg_info.get("entry", 0)
                record[f"{leg_name}_exit"] = leg_info.get("exit", 0)
                record[f"{leg_name}_target_notional"] = leg_info.get("target_notional", 0)

            records.append(record)

        summary_series = pd.DataFrame.from_dict(records)

        # portfolios
        portfolio_expanded = []
        for state in results:
            for k, v in state.portfolio.net_positions().items():
                if not isinstance(v.tradable, Constant):
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
        #portfolio_expanded = pd.merge(portfolio_expanded, summary_series, on="date", how="left")
        return summary_series, portfolio_expanded