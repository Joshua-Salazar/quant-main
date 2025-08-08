from ..dates.utils import date_range, datetime_to_tenor
from ..reporting.trade_reporter import TradeReporter
from ..tradable.portfolio import Portfolio
from ..infrastructure.fixing_table import FixingTable
import numpy as np
import pandas as pd


def generate_most_likely_fixing_table(market, bumped_market, portfolio_or_strategys):
    today = market.get_base_datetime()
    shifted_today = bumped_market.get_base_datetime()
    sim_spots = []
    if shifted_today > today:
        fixing_requirements = []
        for portfolio_or_strategy in portfolio_or_strategys:
            if isinstance(portfolio_or_strategy, Portfolio):
                for name, pos in portfolio_or_strategy.root.items():
                    fixing_requirements += TradeReporter(pos.tradable).get_fixing_requirement(bumped_market)
            else:
                for name, pos in portfolio_or_strategy.portfolio.root.items():
                    fixing_requirements += TradeReporter(pos.tradable).get_fixing_requirement(bumped_market)
        if len(fixing_requirements) > 0:
            und_fixings = {}
            for fixing_requirement in fixing_requirements:
                if fixing_requirement.underlying in und_fixings:
                    und_fixings[fixing_requirement.underlying][0] = min(fixing_requirement.start_date,
                                                                        und_fixings[fixing_requirement.underlying][0])
                    und_fixings[fixing_requirement.underlying][1] = max(fixing_requirement.end_date,
                                                                        und_fixings[fixing_requirement.underlying][1])
                else:
                    und_fixings[fixing_requirement.underlying] = [fixing_requirement.start_date,
                                                                  fixing_requirement.end_date]

            yrs = datetime_to_tenor(shifted_today, today)
            for und, [start_date, end_date] in und_fixings.items():
                spot_und = und.split()[0] if und.split()[-1] == "Curncy" else und
                spot = market.get_spot(spot_und)
                bumped_spot = bumped_market.get_spot(spot_und)
                spot_shock = (bumped_spot - spot) / spot
                sigma = market.get_vol(spot_und, shifted_today, bumped_spot)
                noise_side_at_dt = 1
                fraction_at_prev_dt = 0
                sim_spot = spot
                for dt in date_range(today, shifted_today):
                    if dt != today:
                        fraction_at_dt = datetime_to_tenor(dt, today) / yrs
                        f = 1 + spot_shock * fraction_at_dt
                        f_prev = 1 + spot_shock * fraction_at_prev_dt
                        if dt == shifted_today:
                            noise_at_dt = 0
                        else:
                            noise_at_dt = noise_side_at_dt * sigma / np.sqrt(252) / 2
                        spot_shift = np.maximum(f + noise_at_dt * f_prev, 0.01)
                        sim_spot = spot_shift * spot
                        noise_side_at_dt *= -1
                        fraction_at_prev_dt = fraction_at_dt
                    if start_date <= dt.date() <= end_date:
                        sim_spots.append([dt.date(), und, sim_spot])
    if len(sim_spots) > 0:
        fixing_table = FixingTable(pd.DataFrame(sim_spots, columns=["date", "underlying", "fixing"]))
        return fixing_table
    else:
        return None
