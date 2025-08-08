from datetime import datetime
import numpy as np
from ..dates.utils import datetime_diff
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..interface.ivaluer import IValuer
from ..valuation.future_data_valuer import FutureDataIntradayValuer
from ..analytics.options import Black76
# TODO: refactor to pull out df from swaption calc
from ..valuation import valuer_utils
from ..dates.utils import count_business_days

class FutureOptionIntradayBlack76Valuer(IValuer):

    def __init__(self, underlying_valuer=None):
        self.underlying_valuer = FutureDataIntradayValuer('close') if underlying_valuer is None else underlying_valuer

    def price(self, option: Option, market: Market, calc_types='price',
              fwd_vol_override = None, biz_day_calc = False, holidays = [ ], **kwargs):
        date = market.get_base_datetime()
        if fwd_vol_override is None:
            fwd = self.underlying_valuer.price(option.underlying, market)
            vol = market.get_future_option_data_prev_eod(option)['iv']
        else:
            fwd = fwd_vol_override[ option.name_str]['fwd']
            vol = fwd_vol_override[ option.name_str ]['vol']

        if vol <= 0:
            # return None for all values if no iv
            raise RuntimeError('missing iv')
            #return valuer_utils.return_results_based_on_dictionary(calc_types, {})

        if option.is_expired(market):
            TTM = 0
            disc = 1
            if vol is None:
                vol = 0.001
        else:
            # can only price a concrete option
            assert isinstance(option.expiration, datetime)
            if biz_day_calc:
                DTM = count_business_days( date, option.expiration, holidays ) + 1
                TTM = DTM / 252
            else:
                DTM = datetime_diff(option.expiration, date).days + 1  # adding one since it is using todays date
                TTM = DTM / 365
            # TODO: use yesterdays interest rate


            rate = market.get_interest_rate(option.currency, DTM)/100
            disc = np.exp(-rate * TTM)

        results = Black76(option.strike, option.expiration, option.is_call, date, fwd, vol, disc, TTM=TTM)
        results = valuer_utils.return_results_based_on_dictionary(calc_types, results)

        return results

