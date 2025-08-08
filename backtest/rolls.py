import math
from datetime import datetime, timedelta

from ..analytics.swaptions import atmf_yields_interpolate, rate_interpolate, atmf_yields_interpolate_old
from ..data.imm import find_next_imm_date
from ..dates.utils import add_business_days, coerce_timezone, MAX_DATETIME, bdc_adjustment, add_tenor, tenor_to_days
from ..interface.itradable import ITradable
from ..tradable.future import Future
from ..tradable.FXforward import FXforward
from ..tradable.swaption import Swaption
from ..tradable.position import Position
from ..tradable.portfolio import Portfolio
from ..infrastructure.bmarket import BMarket


class RollContracts:
    def __init__(self):
        pass

    def roll_time(self, dt: datetime, current_contract: ITradable):
        pass

    def is_roll_time(self, dt, roll_time, roll_geq = False ):
        pass

    def past_roll_time(self, dt, roll_time):
        pass

    def find_next_contract(self, dt: datetime, current_contract: ITradable):
        pass

    def roll_price(self, dt: datetime, contract: ITradable):
        pass

    def roll(self, dt: datetime, portfolio: Portfolio, roll_type='fixed_notional',
             roll_geq = False ):
        children = {k: v for k, v in portfolio.root.items()}
        for k, v in children.items():
            if isinstance(v, Position):
                current_contract = v.tradable
                if self.is_roll_time( dt,
                                      self.roll_time(dt, current_contract, roll_geq = roll_geq),
                                      roll_geq = roll_geq ):
                    next_contract = self.find_next_contract(dt, current_contract)
                    current_contract_price = self.roll_price(dt, current_contract)
                    next_contract_price = self.roll_price(dt, next_contract)
                    if math.isnan(current_contract_price):
                        print(f"{current_contract.name()} price is nan")
                    if math.isnan(next_contract_price):
                        print(f"{next_contract.name()} price is nan")
                    portfolio.unwind(k, current_contract_price)
                    if roll_type == 'fixed_notional':
                        units = v.quantity * current_contract_price / next_contract_price
                    elif roll_type == 'fixed_units':
                        units = v.quantity
                    else:
                        raise RuntimeError(f'Unknown roll type {roll_type}')
                    portfolio.trade(next_contract, units, next_contract_price)
            else:
                self.roll(dt, v, roll_geq = roll_geq )
    
    def roll_to_target(self, dt: datetime, portfolio: Portfolio, roll_type='fixed_notional',
             roll_geq = False , target_tenor = 0, skip_months = []):
        children = {k: v for k, v in portfolio.root.items()}
        for k, v in children.items():
            if isinstance(v, Position):
                current_contract = v.tradable
                if (target_tenor != 0) and isinstance(v.tradable, Future):
                    front_contract = self.find_target_contract( add_business_days(dt, -1, self.holidays), skip_months, target_tenor = 0, root = v.tradable.root )
                else:
                    front_contract = current_contract
                if self.is_roll_time( dt,
                                      self.roll_time(dt, front_contract, roll_geq = roll_geq),
                                      roll_geq = roll_geq ):
                    next_contract = self.find_target_contract( dt, skip_months, target_tenor = target_tenor, root = v.tradable.root )
                    current_contract_price = self.roll_price(dt, current_contract)
                    next_contract_price = self.roll_price(dt, next_contract)
                    portfolio.unwind(k, current_contract_price)
                    if roll_type == 'fixed_notional':
                        units = v.quantity * current_contract_price / next_contract_price
                    elif roll_type == 'fixed_units':
                        units = v.quantity
                    else:
                        raise RuntimeError(f'Unknown roll type {roll_type}')
                    portfolio.trade(next_contract, units, next_contract_price)
            else:
                self.roll(dt, v, roll_geq = roll_geq )

    def roll_target(self, dt: datetime, portfolio: Portfolio, long_target, short_target, min_fut, skip_months, roll_type='fixed_notional',
                    roll_geq = False ):
        all_pos = portfolio.net_positions()
        for k, v in all_pos.items():
            if isinstance(v, Position) and isinstance( v.tradable, Future ) and \
                    self.is_roll_time(dt, self.roll_time(dt, min_fut.tradable ), roll_geq = roll_geq ):
                if v.quantity > 0:
                    target_tenor = long_target
                else:
                    target_tenor = short_target
                next_contract = self.find_target_contract( dt, skip_months, target_tenor = target_tenor, root = v.tradable.root )
                current_contract = v.tradable
                current_contract_price = self.roll_price(dt, current_contract)
                next_contract_price = self.roll_price(dt, next_contract)

                if roll_type == 'fixed_notional':
                    units = v.quantity * current_contract_price / next_contract_price
                elif roll_type == 'fixed_units':
                    units = v.quantity
                else:
                    raise RuntimeError(f'Unknown roll type {roll_type}')

                if k in portfolio.root[ 'long' ].root.keys():
                    portfolio.unwind( ( 'long', k ), current_contract_price )
                    portfolio.trade(next_contract, units, next_contract_price, position_path=('long',))
                elif k in portfolio.root[ 'short' ].root.keys():
                    portfolio.unwind( ( 'short', k ) , current_contract_price )
                    portfolio.trade(next_contract, units, next_contract_price, position_path=('short',))
                else:
                    raise RuntimeError('unwind key error')


class RollFXForwardContracts(RollContracts):
    def __init__(self, backtest_market: BMarket, offset, holidays):
        self.backtest_market = backtest_market
        self.offset = offset
        self.holidays = holidays

    def roll_time(self, dt: datetime, current_contract: ITradable, roll_geq=False):
        if isinstance(current_contract, FXforward):
            return add_business_days(current_contract.expiration, -self.offset, self.holidays)
        else:
            return None

    def is_roll_time(self, dt, roll_time, roll_geq=False):
        if roll_time is None:
            return False
        elif roll_geq:
            return dt.date() >= roll_time.date()
        else:
            return dt.date() == roll_time.date()

    def past_roll_time(self, dt, roll_time):
        if roll_time is None:
            return False
        else:
            return dt.date() > roll_time.date()

    def find_next_contract(self, dt: datetime, current_contract: FXforward):
        as_of_date = dt
        while True:
            next_imm_date = find_next_imm_date(as_of_date)
            next_contract = FXforward(current_contract.underlying, current_contract.underlying, current_contract.currency, next_imm_date, 'America/New_York')
            if self.past_roll_time(dt, self.roll_time(dt, next_contract)) or self.is_roll_time(dt, self.roll_time(dt, next_contract)):
                as_of_date = next_imm_date + timedelta(days=1)
            else:
                break
        return next_contract

    def roll_price(self, dt: datetime, contract: FXforward):
        return contract.price(market=self.backtest_market.get_market(dt), calc_types='price')


class RollFutureContracts(RollContracts):
    def __init__(self, backtest_market: BMarket, valuer_map, price_name,
                 offset, offset_reference, holidays, roll_month_map=None):
        # TODO: add roll month map logic to handle commodity futures
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map
        self.price_name = price_name
        self.offset = offset
        self.offset_reference = offset_reference
        self.holidays = holidays

    def roll_time(self, dt: datetime, current_contract: ITradable, roll_geq = False ):
        if isinstance(current_contract, Future):
            return add_business_days(current_contract.expiration, -self.offset, self.holidays)
        else:
            return None

    def is_roll_time(self, dt, roll_time, roll_geq = False ):
        if roll_time is None:
            return False
        elif roll_geq:
            return dt.date() >= roll_time.date()
        else:
            return dt.date() == roll_time.date()

    def past_roll_time(self, dt, roll_time):
        if roll_time is None:
            return False
        else:
            return dt.date() > roll_time.date()

    def find_initial_contract(self, dt, root=None, roll_geq = False ):
        next_future_ref = None
        next_future_expiry = MAX_DATETIME
        for k, v in self.backtest_market.get_market(dt).get_future_universe(root).items():
            dt_v, this_expriy = coerce_timezone(dt, v[self.offset_reference])
            if root == v['root'] and dt_v < this_expriy < next_future_expiry:
                next_future_ref = v
                next_future_expiry = this_expriy

        next_future = Future(next_future_ref['root'], next_future_ref['currency'],
                             next_future_ref['last tradable date'], next_future_ref['exchange'],
                             next_future_ref['ticker'])

        roll_dt = self.roll_time(dt, next_future)
        if self.past_roll_time(dt, roll_dt) or self.is_roll_time(dt, roll_dt, roll_geq = roll_geq ):
            next_future = self.find_next_contract(dt, next_future)
        return next_future

    def find_target_contract(self, dt, skip_months, target_tenor = 1, root=None ):
        try:
            assert isinstance( target_tenor, int )
        except:
            raise RuntimeError('target_tenor must be an integer')

        initial_contract = self.find_initial_contract(dt, root)
        target_contract = initial_contract
        counter = 0
        while counter < target_tenor:
            target_contract = self.find_next_contract(dt, target_contract)
            counter += 1
        target_month_code = target_contract.listed_ticker.replace(' Index', '').replace(target_contract.root, '')[ 0 ]
        # used to skip illiquid months: pass list of illiquid month codes into self.parameters
        if len( skip_months ) > 0:
            while target_month_code in skip_months:
                target_contract = self.find_next_contract( dt, target_contract )
                target_month_code = target_contract.listed_ticker.replace(' Index', '').replace(target_contract.root, '')[0]
        return target_contract

    def find_next_contract(self, dt: datetime, current_contract: Future):
        current_future_expiry = current_contract.expiration
        next_future_ref = None
        next_future_expiry = MAX_DATETIME
        for k, v in self.backtest_market.get_market(dt).get_future_universe(current_contract.root).items():
            if current_contract.root == v['root'] and current_future_expiry < v[self.offset_reference] < next_future_expiry:
                next_future_ref = v
                next_future_expiry = v[self.offset_reference]

        next_future = Future(next_future_ref['root'], next_future_ref['currency'],
                             next_future_ref['last tradable date'], next_future_ref['exchange'],
                             next_future_ref['ticker'])
        return next_future

    def roll_price(self, dt: datetime, contract: Future):
        return self.backtest_market.get_market(dt).get_future_data(contract)[self.price_name]


class RollSwaptionContracts(RollContracts):
    def __init__(self, contract_expiry, contract_tenor, contract_strike, strike_type, contract_style, contract_currency, contract_curve,
                 month_list, day, holidays,
                 backtest_market: BMarket, valuer_map={}, old_data_format=False):
        self.contract_expiry = contract_expiry
        self.contract_tenor = contract_tenor
        self.contract_strike = contract_strike
        self.strike_type = strike_type
        self.contract_style = contract_style
        self.contract_currency = contract_currency
        self.contract_curve = contract_curve
        self.month_list = month_list
        self.day = day
        self.holidays = holidays
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map
        self.old_data_format = old_data_format

    def roll_time(self, dt: datetime, current_contract: ITradable, roll_geq = False ):
        #TODO: consider making an util
        def enforce_valid_date(year, month, day):
            day_count_for_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if year%4==0 and (year%100 != 0 or year%400==0):
                day_count_for_month[2] = 29
            valid_day = min(day_count_for_month[month], day)
            return datetime(year, month, valid_day)

        if isinstance(current_contract, Swaption):
            roll_dates = []
            for year in [dt.year - 1, dt.year, dt.year + 1]:
                for month in self.month_list:
                    roll_dates.append(enforce_valid_date(year, month, self.day))
            roll_dates = [bdc_adjustment(x, convention='following', holidays=self.holidays) for x in
                          roll_dates]
            next_roll_date = min([x for x in roll_dates if x >= dt])
            if roll_geq:
                next_roll_date = min( next_roll_date, current_contract.expiration )
            return next_roll_date
        else:
            return None

    def is_roll_time(self, dt, roll_time, roll_geq = False):
        if roll_time is None:
            return False
        elif roll_geq:
            return dt.date() >= roll_time.date()
        else:
            return dt.date() == roll_time.date()

    def past_roll_time(self, dt, roll_time):
        if roll_time is None:
            return False
        else:
            return dt.date() > roll_time.date()

    def find_initial_contract(self, dt):
        return self.find_next_contract(dt, None)

    def find_next_contract(self, dt: datetime, current_contract: Swaption):
        market = self.backtest_market.get_market(dt)
        atmf_yields = {dt: market.get_forward_rates(self.contract_currency, self.contract_curve)}
        spot_rates = {dt: market.get_spot_rates(self.contract_currency, "SWAP" if self.contract_curve is None else self.contract_curve).data_dict}

        target_expiration = bdc_adjustment(add_tenor(dt, self.contract_expiry), convention='following',
                                           holidays=self.holidays)
        if self.strike_type == 'forward':
            if self.old_data_format:
                target_strike = atmf_yields_interpolate_old(
                    atmf_yields, spot_rates, dt, self.contract_tenor,
                    target_expiration) + self.contract_strike / 100.0
            else:
                target_strike = atmf_yields_interpolate(
                    atmf_yields, spot_rates, dt, self.contract_tenor,
                    target_expiration) + self.contract_strike / 100.0
        elif self.strike_type == 'spot':
            target_strike = rate_interpolate(spot_rates, dt, tenor_to_days(self.contract_tenor)/360) + self.contract_strike / 100.0
        else:
            raise RuntimeError(f'Unknown strike type {self.strike_type}')
        target_swaption = Swaption(self.contract_currency, target_expiration, self.contract_tenor, target_strike,
                                   self.contract_style, self.contract_curve)
        return target_swaption

    def roll_price(self, dt: datetime, contract: Swaption):
        return contract.price(
            market=self.backtest_market.get_market(dt), calc_types='price', valuer=self.valuer_map[Swaption])
