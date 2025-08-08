from datetime import datetime
from ..dates.utils import coerce_timezone, is_business_day
from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
import json
import pickle


class AutoCallable(ITradable):
    def __init__(self, und_list: list, type: str, start_spots: list, start_date: datetime, expiration: datetime,
                 currency: str, notional: float, call_put: str, strike: float, lower_strike: float,
                 exercise_type: str, knock_in_barrier: float, knock_in_barrier_obs: str, knock_in_put_strike: float,
                 put_grearing: float, autocall_dates: list, autocall_barriers: list, coupon_barrier: float,
                 coupon_dates: list, coupon_down: float, coupon_up: float, coupon_is_memory: bool,
                 coupon_is_snowball: bool, float_dates: list, float_fixed_dates: list, float_start_date: datetime,
                 funding_spread: float, rate_index: str, rate_index_ccy: str, glider_event: bool, guaranteed_coupon: float,
                 inst_id=None,
                 autocall_barrier_shift_size=0,
                 autocall_barrier_shift_side=0,   # side -1: shift left side, 1: shift right side, 0: shift both sides
                 coupon_barrier_shift_size=0,
                 coupon_barrier_shift_side=-1,   # side -1: shift left side, 1: shift right side, 0: shift both sides
                 knock_in_barrier_shift_size=0,
                 knock_in_barrier_shift_side=-1,   # side -1: shift left side, 1: shift right side, 0: shift both sides
                 ):
        self.und_list = und_list
        self.type = type
        self.start_spots = start_spots
        self.start_date = start_date
        self.expiration = expiration
        self.currency = currency
        # sell put
        self.notional = notional
        self.call_put = call_put.upper()
        self.strike = strike    # 0.55
        self.lower_strike = lower_strike    # 0.35
        self.exercise_type = exercise_type.upper()
        self.knock_in_barrier = knock_in_barrier    # 0.55
        self.knock_in_barrier_obs = knock_in_barrier_obs
        self.knock_in_put_strike = knock_in_put_strike  # 0.55
        self.put_gearing = put_grearing # 0.2
        #  coupon
        self.autocall_dates = autocall_dates
        self.autocall_barriers = autocall_barriers
        self.coupon_barrier = coupon_barrier
        self.coupon_dates = coupon_dates
        self.coupon_down = coupon_down
        self.coupon_up = coupon_up
        self.coupon_is_memory = coupon_is_memory
        self.coupon_is_snowball = coupon_is_snowball
        # float rate
        self.float_dates = float_dates
        self.float_fixed_dates = float_fixed_dates  # float rate fixed dates
        self.float_start_date = float_start_date
        self.funding_spread = funding_spread
        self.rate_index = rate_index
        self.rate_index_ccy = rate_index_ccy
        # glider even
        self.glider_event = glider_event
        self.guaranteed_coupon = guaranteed_coupon  # 0.0103

        self.inst_id = inst_id
        self.autocall_barrier_shift_size = autocall_barrier_shift_size
        self.autocall_barrier_shift_side = autocall_barrier_shift_side   # side -1: shift left side, 1: shift right side, 0: shift both sides
        self.coupon_barrier_shift_size = coupon_barrier_shift_size
        self.coupon_barrier_shift_side = coupon_barrier_shift_side   # side -1: shift left side, 1: shift right side, 0: shift both sides
        self.knock_in_barrier_shift_size = knock_in_barrier_shift_size
        self.knock_in_barrier_shift_side = knock_in_barrier_shift_side   # side -1: shift left side, 1: shift right side, 0: shift both sides

        self.contract_size = 1
        self.cdr_code = "#A"
        self.index_cdr_code = "GT"

    def validate(self):
        if not is_business_day(self.start_date):
            raise Exception(f"Start date {self.start_date.strftime('%Y-%m-%d')} must be a business date")
        if not is_business_day(self.expiration):
            raise Exception(f"Expiration date {self.expiration.strftime('%Y-%m-%d')} must be a business date")

    def clone(self):
        return AutoCallable(
            self.und_list, self.type, self.start_spots, self.start_date, self.expiration, self.currency,
            self.notional, self.call_put, self.strike, self.lower_strike, self.exercise_type, self.knock_in_barrier,
            self.knock_in_barrier_obs, self.knock_in_put_strike, self.put_gearing, self.autocall_dates,
            self.autocall_barriers, self.coupon_barrier, self.coupon_dates, self.coupon_down, self.coupon_up,
            self.coupon_is_memory, self.coupon_is_snowball, self.float_dates, self.float_fixed_dates,
            self.float_start_date, self.funding_spread, self.rate_index, self.rate_index_ccy,
            self.glider_event, self.guaranteed_coupon, self.inst_id, self.autocall_barrier_shift_size,
            self.autocall_barrier_shift_side, self.coupon_barrier_shift_size, self.coupon_barrier_shift_side,
            self.knock_in_barrier_shift_size, self.knock_in_barrier_shift_side
        )

    def has_expiration(self):
        return True

    def is_expired(self, market: IMarket) -> bool:
        dt, expiry = coerce_timezone(market.get_base_datetime(), self.expiration)
        expired = dt >= expiry
        return expired

    def name(self):
        return "_".join(self.und_list) + self.type + self.start_date.strftime("%Y%m%d") + "-" + self.expiration.strftime("%Y%m%d")

    def get_underlyings(self):
        return self.und_list

    def get_rate_underlyings(self):
        return [(self.rate_index_ccy, self.rate_index)]

    def get_cap(self):
        return self.cap

    def override_coupon_up(self, coupon_up):
        return AutoCallable(
            self.und_list, self.type, self.start_spots, self.start_date, self.expiration, self.currency,
            self.notional, self.call_put, self.strike, self.lower_strike, self.exercise_type, self.knock_in_barrier,
            self.knock_in_barrier_obs, self.knock_in_put_strike, self.put_gearing, self.autocall_dates,
            self.autocall_barriers, self.coupon_barrier, self.coupon_dates, self.coupon_down, coupon_up,
            self.coupon_is_memory, self.coupon_is_snowball, self.float_dates, self.float_fixed_dates,
            self.float_start_date, self.funding_spread, self.rate_index, self.rate_index_ccy,
            self.glider_event, self.guaranteed_coupon, self.inst_id, self.autocall_barrier_shift_size,
            self.autocall_barrier_shift_side, self.coupon_barrier_shift_size, self.coupon_barrier_shift_side,
            self.knock_in_barrier_shift_size, self.knock_in_barrier_shift_side
        )

    def __eq__(self, other):
        if not isinstance(other, AutoCallable):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())

    def to_dict(self):
        ac_dict = dict(
            und_list=self.und_list, type=self.type, start_spots=self.start_spots, start_date=self.start_date, expiration=self.expiration, currency=self.currency,
            notional=self.notional, call_put=self.call_put, strike=self.strike, lower_strike=self.lower_strike, exercise_type=self.exercise_type, knock_in_barrier=self.knock_in_barrier,
            knock_in_barrier_obs=self.knock_in_barrier_obs, knock_in_put_strike=self.knock_in_put_strike, put_gearing=self.put_gearing, autocall_dates=self.autocall_dates,
            autocall_barriers=self.autocall_barriers, coupon_barrier=self.coupon_barrier, coupon_dates=self.coupon_dates, coupon_down=self.coupon_down, coupon_up=self.coupon_up,
            coupon_is_memory=self.coupon_is_memory, coupon_is_snowball=self.coupon_is_snowball, float_dates=self.float_dates, float_fixed_dates=self.float_fixed_dates,
            float_start_date=self.float_start_date, funding_spread=self.funding_spread, rate_index=self.rate_index, rate_index_ccy=self.rate_index_ccy,
            glider_event=self.glider_event, guaranteed_coupon=self.guaranteed_coupon,
            inst_id=self.inst_id, autocall_barrier_shift_size=self.autocall_barrier_shift_size, autocall_barrier_shift_side=self.autocall_barrier_shift_side,
            coupon_barrier_shift_size=self.coupon_barrier_shift_size, coupon_barrier_shift_side=self.coupon_barrier_shift_side,
            knock_in_barrier_shift_size=self.knock_in_barrier_shift_size, knock_in_barrier_shift_side=self.knock_in_barrier_shift_side,
        )
        return ac_dict

    @classmethod
    def from_dict(cls, ac_dict):
        return cls(
            ac_dict["und_list"], ac_dict["type"], ac_dict["start_spots"], ac_dict["start_date"], ac_dict["expiration"], ac_dict["currency"],
            ac_dict["notional"], ac_dict["call_put"], ac_dict["strike"], ac_dict["lower_strike"], ac_dict["exercise_type"], ac_dict["knock_in_barrier"],
            ac_dict["knock_in_barrier_obs"], ac_dict["knock_in_put_strike"], ac_dict["put_gearing"], ac_dict["autocall_dates"],
            ac_dict["autocall_barriers"], ac_dict["coupon_barrier"], ac_dict["coupon_dates"], ac_dict["coupon_down"], ac_dict["coupon_up"],
            ac_dict["coupon_is_memory"], ac_dict["coupon_is_snowball"], ac_dict["float_dates"], ac_dict["float_fixed_dates"],
            ac_dict["float_start_date"], ac_dict["funding_spread"], ac_dict["rate_index"], ac_dict["rate_index_ccy"],
            ac_dict["glider_event"], ac_dict["guaranteed_coupon"],
            ac_dict.get("inst_id"), ac_dict.get("autocall_barrier_shift_size", 0), ac_dict.get("autocall_barrier_shift_side", 0),
            ac_dict.get("coupon_barrier_shift_size", 0), ac_dict.get("coupon_barrier_shift_side",-1),
            ac_dict.get("knock_in_barrier_shift_size", 0), ac_dict.get("knock_in_barrier_shift_side", -1),
        )

    def to_pickle(self):
        ac_dict = self.to_dict()
        return pickle.dumps(ac_dict)

    @staticmethod
    def from_pickle(pkl):
        ac_dict = pickle.loads(pkl)
        return AutoCallable.from_dict(ac_dict)

    def to_json(self, path):
        ac_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(ac_dict, f, indent=4)

    @staticmethod
    def from_json(path):
        with open(path, "r") as f:
            ac_dict = json.load(f)
        for name in ["start_date", "expiration", "float_start_date"]:
            ac_dict[name] = datetime.fromisoformat(ac_dict[name])
        for name in ["autocall_dates", "coupon_dates", "float_dates", "float_fixed_dates"]:
            ac_dict[name] = [datetime.fromisoformat(x) for x in ac_dict[name]]
        return AutoCallable.from_dict(ac_dict)
