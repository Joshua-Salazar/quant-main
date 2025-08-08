from datetime import datetime
from ..dates.utils import EXCHANGE_TZ, set_timezone
from ..interface.itradable import ITradable

FUTURE_MONTH_CODE = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
MONTH_TO_MONTH_CODE = {v: k for k, v in FUTURE_MONTH_CODE.items()}


class Future(ITradable):
    def __init__(self, root: str, currency: str, expiration: datetime, exchange: str, listed_ticker: str,
                 tz_name: str = "",  contract_size: int = 1, **kwargs):
        self.root = root
        self.underlying = root
        self.currency = currency
        self.exchange = exchange
        self.tz_name = tz_name if tz_name else EXCHANGE_TZ[exchange]
        self.expiration = set_timezone(expiration, self.tz_name)
        self.listed_ticker = listed_ticker
        self.ivol_futures_id = kwargs.get('ivol_futures_id', None)
        self.expiry_month = kwargs.get('expiry_month', None)
        self.expiry_year = kwargs.get('expiry_year', None)
        self.contract_size = contract_size

    def clone(self):
        ivol_futures_id = self.ivol_futures_id if hasattr(self, 'ivol_futures_id') else None
        expiry_month = self.expiry_month if hasattr(self, 'expiry_month') else None
        expiry_year = self.expiry_year if hasattr(self, 'expiry_year') else None
        contract_size = self.contract_size if hasattr(self, 'contract_size') else 1
        return Future(self.root, self.currency, self.expiration, self.exchange, self.listed_ticker, self.tz_name,
                      ivol_futures_id=ivol_futures_id, expiry_month=expiry_month, expiry_year=expiry_year,
                      contract_size=contract_size)

    def has_expiration(self):
        return True

    def name(self):
        if self.listed_ticker:
            return self.listed_ticker
        else:
            return self.root + self.expiration.date().isoformat() + self.tz_name + self.currency

    def get_ivol_futures_id(self, dlc):
        if self.ivol_futures_id is None:
            self.ivol_futures_id = dlc.get_ivol_future_id(self)
        return self.ivol_futures_id

    @property
    def expiry_month_code(self):
        return MONTH_TO_MONTH_CODE[self.expiry_month]

    @property
    def kibot_ticker(self):
        mapping_dict = {'HO': 'IHO', 'CO': 'EB'}
        kibot_root = mapping_dict.get(self.root, self.root)
        return kibot_root + str(self.expiry_month_code) + str(self.expiry_year)[-2:]

    @property
    def bbg_ticker(self):
        mapping_dict = {'VX': ('UX', 'Index'),
                        'S': ('S ', 'Comdty'),
                        'ES': ('ES', 'Index'), 
                        'C': ('C ', 'Comdty'),
                        'W': ('W ', 'Comdty')
                        }
        bbg_tup = mapping_dict.get(self.root, (self.root, 'Comdty')) # Comdty works for some indices too
        bbg_root = bbg_tup[0]
        bbg_suffix = bbg_tup[1]
        return bbg_root + str(self.expiry_month_code) + str(self.expiry_year)[-2:] + ' ' + bbg_suffix

    def __eq__(self, other):
        if not isinstance(other, Future):
            return False
        return self.name() == other.name()

    def __hash__(self):
        return hash(self.name())

    def get_underlyings(self):
        return [self.underlying]
