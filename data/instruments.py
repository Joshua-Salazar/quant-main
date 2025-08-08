import datetime
from datetime import timedelta
from dateutil.parser import parse
import pytz
from collections import defaultdict

from ..data.vola import find_underlying
from ..analytics.symbology import ticker_from_option_root
from ..data.instrument_cache_service import InstrumentCacheService
from ..constants.asset_class import AssetClass
from ..constants.underlying_type import UnderlyingType
from ..tradable.autocallable import AutoCallable
from ..tradable.condvarianceswap import CondVarianceSwap
from ..tradable.fxcorrelationswap import FXCorrelationSwap
from ..tradable.crosscondvarianceswap import CrossCondVarianceSwap
from ..tradable.crosscondvolswap import CrossCondVolSwap
from ..tradable.deltahedgedoption import DeltaHedgedOption
from ..tradable.forwardstartoption import ForwardStartOption
from ..tradable.future import Future
from ..tradable.fva import FVA
from ..tradable.fxbarrier import FXBarrier
from ..tradable.genericotc import GenericOTC
from ..tradable.geodispvarswap import GeoDispVarSwap
from ..tradable.geodispvolswap import GeoDispVolSwap
from ..tradable.option import Option
from ..tradable.replicatingvarianceswap import ReplVarianceSwap
from ..tradable.stock import Stock
from ..tradable.varianceswap import VarianceSwap
from ..tradable.volswap import VolSwap
from ..tradable.voloption import VolOption


class IInstrument(object):
    """
    Instrument interface defines the attributes a instrument class should represent
    """
    # TODO: This is only a very basic implementation, more attributes to add
    def inst_id(self):
        raise NotImplementedError('{}.inst_id'.format(self.__class__.__name__))

    def inst_def(self):
        raise NotImplementedError('{}.inst_def'.format(self.__class__.__name__))

    def symbol(self):
        raise NotImplementedError('{}.symbol'.format(self.__class__.__name__))

    def currency(self):
        raise NotImplementedError('{}.currency'.format(self.__class__.__name__))

    def trading_cal(self):
        raise NotImplementedError('{}.trading_cal'.format(self.__class__.__name__))

    def trade_date(self):
        """
        The current trade date based on the pricing time
        """
        raise NotImplementedError('{}.trade_date'.format(self.__class__.__name__))

    def settlement_lag(self):
        raise NotImplementedError('{}.settlement_lag'.format(self.__class__.__name__))

    def settlement_date(self):
        raise NotImplementedError('{}.settlement_date'.format(self.__class__.__name__))

    def __str__(self):
        return '{}({:d})'.format(self.symbol(), self.inst_id)

    def __repr__(self):
        return '{}({:d})'.format(self.__class__.__name__, self.inst_id)

    '''
    
    def quote_price(self):
        raise NotImplementedError('{}.quote_price'.format(self.__class__.__name__))    

    def dollar_price(self):
        """
        The economic value of the instrument in USD
        """
        raise NotImplementedError('{}.dollar_price'.format(self.__class__.__name__))

    def local_price(self):
        """
        The economic value of the instrument in the currency of the asset
        """
        raise NotImplementedError('{}.local_price'.format(self.__class__.__name__))            
    '''


class IDerivative(IInstrument):
    """
    Derivative interface defines the attributes a derivative class should represent
    """
    def underlying_id(self):
        raise NotImplementedError('{}.underlying_id'.format(self.__class__.__name__))

    def underlying_pricing_id(self):
        raise NotImplementedError('{}.underlying_pricing_id'.format(self.__class__.__name__))

    def expiration_time(self):
        raise NotImplementedError('{}.expiration_time'.format(self.__class__.__name__))

    def settlement_type(self):
        raise NotImplementedError('{}.settlement_type'.format(self.__class__.__name__))

    def time_to_expiration(self):
        raise NotImplementedError('{}.time_to_expiration'.format(self.__class__.__name__))


class ITradableInstrument(IInstrument):
    """
    Derivative interface defines the attributes a derivative class should represent
    """
    def get_tradable(self):
        raise NotImplementedError('{}.get_tradabale'.format(self.__class__.__name__))

    def underlying_cals(self):
        return None

    def barrier_cal(self):
        return None


class Instrument(IInstrument):
    """
    The base class for instrument
    """

    def __init__(self, inst_id):
        super(Instrument, self).__init__()
        self._inst_id = inst_id
        self._instrument_cache = InstrumentCacheService()
        self._instruments = InstrumentService()

    @property
    def inst_id(self):
        return self._inst_id

    def inst_def(self):
        return self._instrument_cache[self.inst_id]

    def symbol(self):
        # TODO: should we map the field names?
        return self.inst_def()['Symbol']

    def currency(self):
        inst_def = self.inst_def()
        if 'CCY' not in inst_def:
            raise ValueError('No currency specified for {}'.format(inst_def))
        return inst_def['CCY']

    def trading_cal(self):
        inst_def = self.inst_def()
        if 'ExchCdrCode' not in inst_def:
            raise ValueError('No trading calendar code specified for {}'.format(inst_def))
        # TODO: get the calendar from the code
        return inst_def['ExchCdrCode']

    def time_zone(self):
        inst_def = self.inst_def()
        if inst_def["InstrumentId"] in [70273543, 70273544]:
            return "Australia/Sydney"
        if 'TimeZone' not in inst_def:
            # raise ValueError('No time zone specified for {}'.format(inst_def))
            # a few OTC products don't have time_zone info, their maturity dates are presented in New York time.
            return 'America/New_York'
        return inst_def['TimeZone']

    def trade_date(self):
        # TODO this should be implemented correctly considering the trading calendar of the asset
        return datetime.datetime.now(tz=pytz.timezone('US/Eastern')).date()

    def settlement_lag(self):
        # TODO: there is no settlement days in the instrument defnition, assume it is 2 for now
        return 2.

    def settlement_date(self):
        # TODO: Use actual calendar later
        return self.trade_date() + timedelta(self.settlement_lag())


class DerivativeInstrument(IDerivative, Instrument):
    def __init__(self, inst_id):
        Instrument.__init__(self, inst_id)

    def underlying_id(self):
        """
            Retun Underlying Id
        """
        if self.inst_def()['UnderlyerId'] == 48540800:
            return 1320462
        else:
            return self.inst_def()['UnderlyerId']

    def underlying_def(self):
        """
        Returns: Underlying definition
        """
        return self._instrument_cache[self.underlying_id()]

    def underlying_pricing_id(self):
        """
            Retun Underlying pricing Id
        """
        return self.inst_def()['UnderlyerPricerId']

    def underlying_pricing_def(self):
        """
        Returns: Pricing underlying definition
        """
        return self._instrument_cache[self.underlying_pricing_id()]

    def expiration_time(self):
        inst_def = self.inst_def()
        if 'MaturityDate' not in inst_def:
            raise ValueError('No expiration time specified for {}'.format(inst_def))
        if inst_def["InstType"] == "OTCOption" and "MaturityDateUTC" in inst_def:
            raw_expiration_time = datetime.datetime.strptime(inst_def['MaturityDateUTC'][:19], "%Y-%m-%dT%H:%M:%S")
            expiration_time = pytz.timezone("UTC").localize(raw_expiration_time).astimezone(pytz.timezone(self.time_zone()))
        else:
            # TODO: assume the format stays consistent
            raw_expiration_time = datetime.datetime.strptime(inst_def['MaturityDate'][:19], "%Y-%m-%dT%H:%M:%S")
            expiration_time = pytz.timezone(self.time_zone()).localize(raw_expiration_time)
        return expiration_time

    def local_expiration_time(self):
        """
        function to read maturity date with its time zone and then convert it to trade local time
        """
        inst_def = self.inst_def()
        if 'MaturityDate' not in inst_def:
            raise ValueError('No expiration time specified for {}'.format(inst_def))
        raw_expiration_time = parse(inst_def['MaturityDate'])
        expiration_time = raw_expiration_time.astimezone(pytz.timezone(self.time_zone()))
        return expiration_time

    def settlement_type(self):
        inst_def = self.inst_def()
        if not 'SettleStyle' in inst_def:
            raise ValueError('No settle style specified for '.format(inst_def))
        return inst_def['SettleStyle']


class StockInstrument(Instrument, ITradableInstrument):

    def get_tradable(self):
        inst_def = self.inst_def()
        root = inst_def["BBGTicker"]
        stock = Stock(root, self.currency())
        return stock

    def spot_price(self):
        raise NotImplementedError('{}.spot_price'.format(self.__class__.__name__))

    def dividend_cash_stream(self):
        raise NotImplementedError('{}.dividend_cash_stream'.format(self.__class__.__name__))


class FutureInstrument(DerivativeInstrument, ITradableInstrument):

    def get_tradable(self):
        inst_def = self.inst_def()
        # work around where ticker is not in symbology table, e.g. ESM4 Index
        root = inst_def["UnderlyingTicker"] if inst_def["UnderInstClass"] == "Commodity" else inst_def["UnderlyingPricerBBGTicker"]
        fut = Future(root, self.currency(), self.local_expiration_time(), inst_def['PrimExchange'], inst_def["BBGTicker"],
                     self.time_zone(), contract_size=self.contract_size())
        return fut

    def contract_size(self):
        inst_def = self.inst_def()
        if 'Multiplier' not in inst_def:
            raise ValueError('No multiplier specified for '.format(inst_def))
        return inst_def['Multiplier']


class VIXFutInstrument(FutureInstrument):
    pass


class IOptionInstrument(IDerivative):
    def option_type(self):
        raise NotImplementedError('{}.option_type'.format(self.__class__.__name__))

    def exercise_type(self):
        raise NotImplementedError('{}.exercise_style'.format(self.__class__.__name__))

    def contract_size(self):
        raise NotImplementedError('{}.contract'.format(self.__class__.__name__))

    def strike(self):
        raise NotImplementedError('{}.strike'.format(self.__class__.__name__))

    def implied_volatility(self):
        raise NotImplementedError('{}.implied_volatility'.format(self.__class__.__name__))

    def underlying_price(self):
        raise NotImplementedError('{}.underlying_price'.format(self.__class__.__name__))

    def market_quote(self):
        raise NotImplementedError('{}.market_quote'.format(self.__class__.__name__))


class OptionInstrument(IOptionInstrument, DerivativeInstrument, ITradableInstrument):
    def __init__(self, inst_id):
        DerivativeInstrument.__init__(self, inst_id)

    def get_tradable(self):
        expiry_time = self.local_expiration_time()
        time_zone = self.time_zone()
        currency = self.currency()
        root = self.underlying_def()["BBGTicker"]
        underlying_in_option = find_underlying(UnderlyingType.EQUITY, root, expiry_time, time_zone, currency)
        option = Option(root, underlying_in_option, currency, expiry_time,
                        self.strike(), self.option_type() == 'CALL', self.exercise_type() == 'American',
                        self.contract_size(), time_zone, self.symbol())
        return option

    def option_type(self):
        inst_def = self.inst_def()
        if 'PutCall' not in inst_def:
            raise ValueError('No option type specified for '.format(inst_def))
        return inst_def['PutCall']

    def exercise_type(self):
        inst_def = self.inst_def()
        if 'ExerciseStyle' not in inst_def:
            raise ValueError('No exercise type specified for '.format(inst_def))
        return inst_def['ExerciseStyle']

    def contract_size(self):
        inst_def = self.inst_def()
        if 'Multiplier' not in inst_def:
            raise ValueError('No multiplier specified for '.format(inst_def))
        return inst_def['Multiplier']

    def strike(self):
        inst_def = self.inst_def()
        if 'Strike' not in inst_def:
            raise ValueError('No strike specified for '.format(inst_def))
        return inst_def['Strike']


class OptionOnFutureInstrument(OptionInstrument, ITradableInstrument):

    def get_tradable(self):
        expiry_time = self.local_expiration_time()
        time_zone = self.time_zone()
        currency = self.currency()
        inst_def = self.inst_def()
        und_def = self.underlying_def()
        if "BBGTicker" in und_def:
            root = und_def["BBGTicker"]
        else:
            if inst_def["UnderInstClass"] == "Commodity":
                root = und_def["Ticker"]
            else:
                raise
        underlying_in_option = find_underlying(UnderlyingType.FUTURES, root, expiry_time, time_zone, currency)
        option = Option(root, underlying_in_option, currency, expiry_time,
                        self.strike(), self.option_type() == 'CALL', self.exercise_type() == 'American',
                        self.contract_size(), time_zone, self.symbol())
        return option


class VarianceSwapInstrument(DerivativeInstrument):
    def __init__(self, inst_id):
        super(VarianceSwapInstrument, self).__init__(inst_id)

    def multiplier(self):
        inst_def = self.inst_def()
        if 'Multiplier' not in inst_def:
            raise ValueError('No multiplier specified for '.format(inst_def))
        return inst_def['Multiplier']

    def obs_start_time(self):
        inst_def = self.inst_def()
        if 'EffectiveDate' not in inst_def:
            raise ValueError('No effective time specified for {}'.format(inst_def))
        # TODO: assume the format stays consistent
        return datetime.datetime.strptime(inst_def['EffectiveDate'][:19], "%Y-%m-%dT%H:%M:%S")

    def obs_start_date(self):
        obs_start_time = self.obs_start_time()
        return obs_start_time.date()

    def obs_end_time(self):
        raise NotImplementedError('{}.obs_end_time'.format(self.__class__.__name__))

    def strike(self):
        inst_def = self.inst_def()
        if 'Strike' not in inst_def:
            raise ValueError('No strike specified for '.format(inst_def))
        return inst_def['Strike']

    def cap(self):
        inst_def = self.inst_def()
        if 'Cap' not in inst_def:
            raise ValueError('No Cap specified for '.format(inst_def))
        return inst_def['Cap']

    def is_capped(self):
        cap = self.cap()
        return True if cap > 0 else False

    def fixings(self):
        raise NotImplementedError('{}.fixings'.format(self.__class__.__name__))

    def accrued_volatility(self):
        raise NotImplementedError('{}.accrued_volatility'.format(self.__class__.__name__))

    def implied_volatility(self):
        raise NotImplementedError('{}.implied_volatility'.format(self.__class__.__name__))


class TradableInstrument(ITradableInstrument, DerivativeInstrument):
    def __init__(self, inst_id):
        super(TradableInstrument, self).__init__(inst_id)
        inst_def = self.inst_def()
        inst_type = inst_def["InstrumentType"]
        if inst_type not in ["BarrierFXOption", "GenericOTC"]:
            self.validate()
            inception = parse(inst_def["EffectiveDate"]) if "EffectiveDate" in inst_def else \
                parse(inst_def["InitDate"])

        if inst_type == "FwdStartingOption":
            self.tradable = ForwardStartOption(
                underlying=inst_def["UnderlyingPricerBBGTicker"],
                currency=inst_def["UnderlyingCCY"],
                strike_date=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike=inst_def["Strike"],
                is_call=inst_def["PutCall"] == "CALL"
            )
        elif inst_type == "FVA":
            self.tradable = FVA(
                underlying=inst_def["UnderlyingPricerBBGTicker"],
                currency=inst_def["UnderlyingCCY"],
                strike_date=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike=inst_def["Strike"],
                is_call=inst_def["PutCall"] == "CALL"
            )
        elif inst_type in ["LaggedVarSwap", "VarianceSwap"]:
            lag = inst_def["Lag"] if inst_type == "LaggedVarSwap" else 1
            underlying = inst_def["UnderlyingPricerBBGTicker"]
            asset_class = AssetClass.FX if inst_def["InstClass"] == "FX" else AssetClass.EQUITY
            self.tradable = VarianceSwap(
                underlying=underlying.split()[0] if asset_class == AssetClass.FX else underlying,
                inception=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike_in_vol=inst_def["Strike"],
                notional=1,
                currency=inst_def["CCY"],
                lag=lag,
                cap=inst_def["Cap"],
                cdr_code=self.trading_cal(),
                fixing_src=underlying,
                asset_class=asset_class,
                inst_id=inst_id,
            )
        elif inst_type in ["GeoDispVarSwap", "GeoDispVolSwap"]:
            underlyings = []
            weights = []
            ref_data = inst_def["ReferenceData"]
            if len(ref_data) % 2 != 0:
                raise Exception(f"ReferenceData for {inst_type} {len(ref_data)} must be even.")
            num_underlyings = int(len(ref_data) / 2)
            for idx in range(num_underlyings):
                und_id_key = f"Under{idx+1}"
                if und_id_key not in ref_data:
                    raise Exception(f"Cannot find {und_id_key}")
                und_weight_key = f"Weight{idx+1}"
                if und_weight_key not in ref_data:
                    raise Exception(f"Cannot find {und_weight_key}")
                underlying = self._instrument_cache[ref_data[und_id_key]]["BBGTicker"]
                underlyings.append(underlying)
                weights.append(ref_data[und_weight_key])
            if inst_type == "GeoDispVarSwap":
                self.tradable = GeoDispVarSwap(underlyings=underlyings,
                                               weights=weights,
                                               inception=inception,
                                               expiration=parse(inst_def["MaturityDate"]),
                                               strike_in_vol=inst_def["Strike"],
                                               notional=1,
                                               currency=inst_def["UnderlyingCCY"],
                                               lag=inst_def["Lag"],
                                               cap=inst_def["Cap"],
                                               )
            else:
                self.tradable = GeoDispVolSwap(underlyings=underlyings,
                                               weights=weights,
                                               inception=inception,
                                               expiration=parse(inst_def["MaturityDate"]),
                                               strike_in_vol=inst_def["Strike"],
                                               notional=1,
                                               currency=inst_def["UnderlyingCCY"],
                                               lag=inst_def["Lag"],
                                               cap=inst_def["Cap"],
                                               )
        elif inst_type == "CondVarianceSwap":
            barrier_type_in = inst_def["ReferenceData"]["BarrierType"]
            if barrier_type_in == "btUp":
                barrier_type = "UpVar"
                down_var_barrier = None
                up_var_barrier = inst_def["ReferenceData"]["Barrier2"]
            elif barrier_type_in == "btDown":
                barrier_type = "DownVar"
                down_var_barrier = inst_def["ReferenceData"]["Barrier"]
                up_var_barrier = None
            elif barrier_type_in == "btCorridor":
                barrier_type = "Corridor"
                down_var_barrier = inst_def["ReferenceData"]["Barrier1"]
                up_var_barrier = inst_def["ReferenceData"]["Barrier2"]
            else:
                raise Exception(f"Unsupport barrier type {barrier_type_in}")

            underlying = inst_def["UnderlyingPricerBBGTicker"]
            asset_class = AssetClass.FX if inst_def["InstClass"] == "FX" else AssetClass.EQUITY
            self.tradable = CondVarianceSwap(
                underlying=underlying,
                inception=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike_in_vol=inst_def["Strike"],
                notional=1,
                currency=inst_def["UnderlyingCCY"],
                barrier_condition=inst_def["ReferenceData"]["BarrierCondition"],
                barrier_type=barrier_type,
                down_var_barrier=down_var_barrier,
                up_var_barrier=up_var_barrier,
                cap=inst_def["Cap"],
                cdr_code=self.trading_cal(),
                fixing_src=underlying,
                asset_class=asset_class,
                inst_id=inst_id,
            )
        elif inst_type == "ReplVarSwap":
            self.tradable = ReplVarianceSwap(
                underlying=inst_def["UnderlyingPricerBBGTicker"],
                inception=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike_in_vol=inst_def["Strike"],
                notional=1,
                currency=inst_def["UnderlyingCCY"],
                strike_star=float(inst_def["ReferenceData"]["Strikestar"]),
                strike_step=int(float(inst_def["ReferenceData"]["Strikestep"])),
                strike_min=float(inst_def["ReferenceData"]["Strikemin"]),
                strike_max=float(inst_def["ReferenceData"]["Strikemax"]),
                reference_vol_level=inst_def["Vol"]
            )
        elif inst_type in ["CrossCorrCondVarSwap", "CrossCorrCondVolSwap"]:
            barrier_type_in = inst_def["ReferenceData"]["BarrierType"]
            if barrier_type_in == "btUp":
                barrier_type = "UpVar"
                down_var_barrier = None
                up_var_barrier = inst_def["ReferenceData"]["Barrier2"]
            elif barrier_type_in == "btDown":
                barrier_type = "DownVar"
                down_var_barrier = inst_def["ReferenceData"]["Barrier"]
                up_var_barrier = None
            elif barrier_type_in == "btCorridor":
                barrier_type = "CorridorVar"
                down_var_barrier = inst_def["ReferenceData"]["Barrier1"]
                up_var_barrier = inst_def["ReferenceData"]["Barrier2"]
            else:
                raise Exception(f"Unsupport barrier type {barrier_type_in}")

            barrier_underlying_id = inst_def["ReferenceData"]["Under2"]
            if inst_type == "CrossCorrCondVarSwap":
                self.tradable = CrossCondVarianceSwap(
                    underlying=inst_def["UnderlyingPricerBBGTicker"],
                    barrier_underlying=self._instrument_cache[barrier_underlying_id]["BBGTicker"],
                    inception=inception,
                    expiration=parse(inst_def["MaturityDate"]),
                    strike_in_vol=inst_def["Strike"],
                    notional=1,
                    currency=inst_def["UnderlyingCCY"],
                    barrier_condition=inst_def["ReferenceData"]["BarrierCondition"],
                    barrier_type=barrier_type,
                    down_var_barrier=down_var_barrier,
                    up_var_barrier=up_var_barrier,
                    cap=inst_def["Cap"]
                )
            else:
                self.tradable = CrossCondVolSwap(
                    underlying=inst_def["UnderlyingPricerBBGTicker"],
                    barrier_underlying=self._instrument_cache[barrier_underlying_id]["BBGTicker"],
                    inception=inception,
                    expiration=parse(inst_def["MaturityDate"]),
                    strike_in_vol=inst_def["Strike"],
                    notional=1,
                    currency=inst_def["CCY"],
                    barrier_condition=inst_def["ReferenceData"]["BarrierCondition"],
                    barrier_type=barrier_type,
                    down_var_barrier=down_var_barrier,
                    up_var_barrier=up_var_barrier,
                    cap=inst_def["Cap"]
                )
        elif inst_type in ["LaggedVolSwap", "VolSwap"]:
            underlying = inst_def["UnderlyingPricerBBGTicker"]
            asset_class = AssetClass.FX if inst_def["InstClass"] == "FX" else AssetClass.EQUITY
            self.tradable = VolSwap(
                underlying=underlying.split()[0] if asset_class == AssetClass.FX else underlying,
                inception=inception,
                expiration=parse(inst_def["MaturityDate"]),
                strike_in_vol=inst_def["Strike"],
                notional=1,
                currency=inst_def["CCY"],
                lag=inst_def["Lag"] if "Lag" in inst_def else 1,
                cap=inst_def["Cap"] if "Cap" in inst_def else 0,
                cdr_code=self.trading_cal(),
                fixing_src=underlying,
                asset_class=asset_class
            )
        elif inst_type in ["VolatilityOption"]:
            underlying = inst_def["UnderlyingPricerBBGTicker"]
            asset_class = AssetClass.FX if inst_def["InstClass"] == "FX" else AssetClass.EQUITY
            self.tradable = VolOption(
                underlying=underlying.split()[0] if asset_class == AssetClass.FX else underlying,
                inception=inception,
                expiration=parse(inst_def["MaturityDate"]),
                vol_strike=inst_def["Strike"],
                notional=1,
                currency=inst_def["CCY"],
                lag=inst_def["Lag"],
                is_cap=inst_def["OptionType"] != "Floor",
                cap=inst_def["Cap"],
                cdr_code=self.trading_cal(),
                fixing_src=underlying,
                asset_class=asset_class
            )
        elif inst_type == "CorrelationSwap" and inst_def["InstClass"] == "FX":
            pair1, fixing_src1 = inst_def["Leg1"].split(".")
            pair2, fixing_src2 = inst_def["Leg2"].split(".")
            if fixing_src1 != fixing_src2:
                raise Exception(f"Found inconsistent fixing source {fixing_src1} vs {fixing_src2}")
            expiration = inst_def["Symbol"].split()[-1].split("-")[-1]
            expiration = pytz.timezone(self.time_zone()).localize(parse(expiration))
            self.tradable = FXCorrelationSwap(
                pair1=pair1,
                pair2=pair2,
                inception=inception,
                expiration=expiration,
                strike=inst_def["Strike"],
                notional=1,
                fixing_src=fixing_src1,
                cdr_code=inst_def['CountryCdrCode'],
            )
        elif inst_type == "BarrierFXOption":
            self.tradable = FXBarrier(ticker=inst_def["Symbol"])
        elif inst_type == "GenericOTC":
            if inst_def["InstrumentCategory"] == "AUTOCALL_SWAP":
                ref_data = inst_def["ReferenceData"]
                und_idx_list = sorted([int(x.replace("M_UNDERLYERS_", "")) for x in ref_data.keys() if "M_UNDERLYERS_" in x])

                und_list = [ticker_from_option_root(ref_data[f"M_UNDERLYERS_{idx}"]) for idx in und_idx_list]
                type = inst_def["InstrumentCategory"]
                start_spots = [float(ref_data[f"M_START_SPOT_{idx}"]) for idx in und_idx_list]
                start_date = parse(ref_data["M_STARTDATE"])
                expiration = parse(ref_data["MATURITY"])
                currency = inst_def["CCY"]
                notional = 1
                assert ref_data["M_CALLPUT"] == inst_def["PutCall"]
                call_put = ref_data["M_CALLPUT"]
                assert float(ref_data["STRIKE"]) == inst_def["Strike"]
                strike = inst_def["Strike"]
                lower_strike = float(ref_data.get("M_LOWERSTRIKE", "0"))
                exercise_type = ref_data["M_KNOCKINBARRIEROBS"].upper() # todo: double check
                knock_in_barrier = float(ref_data["M_KNOCKINBARRIER"])
                knock_in_barrier_obs = ref_data["M_KNOCKINBARRIEROBS"]
                knock_in_put_strike = float(ref_data["M_KNOCKINBARRIER"])
                put_grearing = float(ref_data["M_PUTGEARING"])
                autocall_idx_list = sorted([int(x.replace("M_AUTOCALLDATES_", "")) for x in ref_data.keys() if "M_AUTOCALLDATES_" in x])
                autocall_dates = [parse(ref_data[f"M_AUTOCALLDATES_{x}"]) for x in autocall_idx_list]
                autocall_barriers = [float(ref_data[f"M_AUTOCALLBARRIERS_{x}"]) for x in autocall_idx_list]
                coupon_barrier = float(ref_data["M_COUPONBARRIER"])
                coupon_idx_list = sorted([int(x.replace("M_COUPONDATES_", "")) for x in ref_data.keys() if "M_COUPONDATES_" in x])
                coupon_dates = [parse(ref_data[f"M_COUPONDATES_{x}"]) for x in coupon_idx_list]
                coupon_down = float(ref_data.get("M_COUPONDOWN", "0"))
                coupon_up = float(ref_data.get("M_COUPONUP", "0"))
                coupon_is_memory = ref_data["M_COUPONISMEMORY"] == "TRUE"
                coupon_is_snowball = ref_data["M_COUPONISSNOWBALL"] == "TRUE"
                float_idx_list = sorted([int(x.replace("M_FLOATDATES_", "")) for x in ref_data.keys() if "M_FLOATDATES_" in x])
                float_dates = [parse(ref_data[f"M_FLOATDATES_{x}"]) for x in float_idx_list]
                float_fixed_dates = [parse(ref_data[f"M_FLOATFIXDATE_{x}"]) for x in float_idx_list]
                float_start_date = parse(ref_data["M_FLOATSTARTDATE"])
                funding_spread = float(ref_data["M_FUNDINGSPREAD"])
                rate_index = ref_data["M_RATEINDEXID"]
                rate_index_ccy = "USD"  # todo: only support SOFR
                glider_event = ref_data["M_GLIDEREVENT"] == "TRUE"
                guaranteed_coupon = ref_data.get("M_GUARANTEEDCOUPON", "FALSE") == "TRUE"
                self.tradable = AutoCallable(
                    und_list=und_list, type=type, start_spots=start_spots, start_date=start_date, expiration=expiration,
                    currency=currency, notional=notional, call_put=call_put, strike=strike, lower_strike=lower_strike,
                    exercise_type=exercise_type, knock_in_barrier=knock_in_barrier, knock_in_barrier_obs=knock_in_barrier_obs,
                    knock_in_put_strike=knock_in_put_strike, put_grearing=put_grearing, autocall_dates=autocall_dates,
                    autocall_barriers=autocall_barriers, coupon_barrier=coupon_barrier, coupon_dates=coupon_dates,
                    coupon_down=coupon_down, coupon_up=coupon_up, coupon_is_memory=coupon_is_memory,
                    coupon_is_snowball=coupon_is_snowball, float_dates=float_dates, float_fixed_dates=float_fixed_dates,
                    float_start_date=float_start_date, funding_spread=funding_spread, rate_index=rate_index,
                    rate_index_ccy=rate_index_ccy, glider_event=glider_event, guaranteed_coupon=guaranteed_coupon,
                    inst_id=inst_id,
                )
            else:
                self.tradable = GenericOTC(ticker=inst_def["Symbol"], underlying=inst_def["Ticker"])
        elif inst_type == "DeltaHedgedOption":
            self.tradable = DeltaHedgedOption(ticker=inst_def["Symbol"], underlying=inst_def["Ticker"])
        else:
            raise Exception(f"Unsupported instrument type {inst_type}")

    def validate(self):
        inst_def = self.inst_def()
        if 'EffectiveDate' not in inst_def and 'InitDate' not in inst_def:
            raise ValueError('No effective date and Init Date specified for {}'.format(inst_def))
        # if 'MaturityDate' not in inst_def:
            # raise ValueError('No maturity date specified for {}'.format(inst_def))
        # if 'Strike' not in inst_def:
        #     raise ValueError('No Strike specified for {}'.format(inst_def))

    def get_tradable(self):
        return self.tradable

    def multiplier(self):
        inst_def = self.inst_def()
        if 'Multiplier' not in inst_def:
            raise ValueError('No multiplier specified for '.format(inst_def))
        return inst_def['Multiplier']

    def obs_start_time(self):
        inst_def = self.inst_def()
        inception = parse(inst_def["EffectiveDate"]) if "EffectiveDate" in inst_def else \
            parse(inst_def["InitDate"])
        return inception

    def obs_start_date(self):
        obs_start_time = self.obs_start_time()
        return obs_start_time.date()

    def obs_end_time(self):
        return self.tradable.expiration

    def strike(self):
        return self.tradable.strike_in_vol

    def underlying_cals(self):
        if isinstance(self.tradable, GeoDispVarSwap) or isinstance(self.tradable, GeoDispVolSwap):
            num_underlyings = len(self.tradable.underlyings)
            underlying_cdrs = set()
            for idx in range(num_underlyings):
                underlying_id = self.inst_def()["ReferenceData"][f"Under{idx+1}"]
                underlying_cdr = self._instrument_cache[underlying_id]["ExchCdrCode"]
                underlying_cdrs.add(underlying_cdr)
            return list(underlying_cdrs)
        else:
            return None

    def barrier_cal(self):
        if isinstance(self.tradable, CrossCondVarianceSwap) or isinstance(self.tradable, CrossCondVolSwap):
            barrier_underlying_id = self.inst_def()["ReferenceData"]["Under2"]
            barrier_underlying_cdr = self._instrument_cache[barrier_underlying_id]["ExchCdrCode"]
            return barrier_underlying_cdr
        else:
            return None

class InstrumentService(object):
    # TODO: metaclass later
    # TODO: logger later
    #__metaclass__ = Singleton

    # region InstrumentType Definition
    InstrumentType = {
      'itEquity': 'Equity',
      'itETF': 'ETF',
      'itFuture': 'Future',
      'itVIX': 'VIX',
      'itOption': 'Option',
      'itOTCOption': 'OTCOption',
      'itOptionOnFuture': 'OptionOnFuture',
      'itVarianceSwap': 'VarianceSwap',
      # instrument type for structure
      'FwdStartingOption': 'FwdStartingOption',
      'FVA': 'FVA',
      'LaggedVarSwap': 'LaggedVarSwap',
      'LaggedVolSwap': 'LaggedVolSwap',
      'CondVarianceSwap': 'CondVarianceSwap',
      'GeoDispVarSwap': 'GeoDispVarSwap',
      'GeoDispVolSwap': 'GeoDispVolSwap',
      'CrossCorrCondVarSwap': 'CrossCorrCondVarSwap',
      'CrossCorrCondVolSwap': 'CrossCorrCondVolSwap',
      'VarianceSwap': 'VarianceSwap',
      'VolSwap': 'VolSwap',
      'CorrelationSwap': 'CorrelationSwap',
      'ReplVarSwap': 'ReplVarSwap',
      'BarrierFXOption': 'BarrierFXOption',
      'GenericOTC': 'GenericOTC',
      'VolatilityOption': 'VolatilityOption',
      'DeltaHedgedOption': 'DeltaHedgedOption',
    }
    # endregion

    def __init__(self):
        # region Instrument map
        self._instrument_map = {
            # region Vanilla Instruments
            (self.InstrumentType['itEquity'],): StockInstrument,
            (self.InstrumentType['itETF'],): StockInstrument,
            (self.InstrumentType['itFuture'],): FutureInstrument,
            (self.InstrumentType['itFuture'], self.InstrumentType['itVIX']): VIXFutInstrument,
            (self.InstrumentType['itOption'],): OptionInstrument,
            (self.InstrumentType['itOTCOption'],): OptionInstrument,
            (self.InstrumentType['itOption'], self.InstrumentType['itFuture']): OptionOnFutureInstrument,
            (self.InstrumentType['itOptionOnFuture'], self.InstrumentType['itFuture']): OptionOnFutureInstrument,
            (self.InstrumentType['itOTCOption'], self.InstrumentType['itFuture']): OptionOnFutureInstrument,
            # endregion
            # region Volatility Instruments
            (self.InstrumentType['itVarianceSwap'],): VarianceSwapInstrument,
            (self.InstrumentType['FVA'],): TradableInstrument,
            (self.InstrumentType['FwdStartingOption'],): TradableInstrument,
            (self.InstrumentType['LaggedVarSwap'],): TradableInstrument,
            (self.InstrumentType['LaggedVolSwap'],): TradableInstrument,
            (self.InstrumentType['CondVarianceSwap'],): TradableInstrument,
            (self.InstrumentType['GeoDispVarSwap'],): TradableInstrument,
            (self.InstrumentType['GeoDispVolSwap'],): TradableInstrument,
            (self.InstrumentType['CrossCorrCondVarSwap'],): TradableInstrument,
            (self.InstrumentType['CrossCorrCondVolSwap'],): TradableInstrument,
            (self.InstrumentType['VarianceSwap'],): TradableInstrument,
            (self.InstrumentType['VolSwap'],): TradableInstrument,
            (self.InstrumentType['CorrelationSwap'],): TradableInstrument,
            (self.InstrumentType['ReplVarSwap'],): TradableInstrument,
            (self.InstrumentType['BarrierFXOption'],): TradableInstrument,
            (self.InstrumentType['GenericOTC'],): TradableInstrument,
            (self.InstrumentType['VolatilityOption'],): TradableInstrument,
            (self.InstrumentType['DeltaHedgedOption'],): TradableInstrument,
            # endregion
        }
        # endregion
        self._cache = dict()
        self._inst_service = InstrumentCacheService()

    @staticmethod
    def _has_ul_px_id(inst_def):
        return 'UnderlyerPricerId' in inst_def and inst_def['UnderlyerPricerId'] > 0

    @staticmethod
    def _has_ul_id(inst_def):
        return 'UnderlyerId' in inst_def and inst_def['UnderlyerId'] > 0

    def _get_key(self, inst_def):
        key = [inst_def['InstType']]
        if self._has_ul_px_id(inst_def):
            ul_def = self._inst_service[inst_def['UnderlyerPricerId']]
            key.append(ul_def['InstType'])
        if self._has_ul_id(inst_def):
            ul_def = self._inst_service[inst_def['UnderlyerId']]
            key.append(ul_def['InstType'])
        return tuple(key)

    def _get_instrument_class(self, key):
        clazz = None
        while clazz is None and key:
            clazz = self._instrument_map.get(key)
            key = key[:-1]
        return clazz

    def _load_instruments(self, inst_ids):
        self._inst_service.load_instruments(inst_ids)
        by_key = defaultdict(set)
        for inst_id in inst_ids:
            inst_def = self._inst_service[inst_id]
            key = self._get_key(inst_def)
            by_key[key].add(inst_id)

        for key, key_ids in by_key.items():
            clazz = self._get_instrument_class(key)
            if clazz is None:
                # TODO: logger later
                print('{} has no instrument class mapped, defaulting to Instrument'.format(key))
                clazz = Instrument

            for inst_id in key_ids:
                instrument = clazz(inst_id)
                self._cache[instrument.inst_id] = instrument

    def get_instruments(self, inst_ids):
        ids_to_load = set(inst_ids).difference(self._cache.keys())
        if ids_to_load:
            self._load_instruments(ids_to_load)
        return [self._cache.get(inst_id) for inst_id in inst_ids]


if __name__ == '__main__':
    svc = InstrumentService()
    inst_ids = [75051666]
    ac = svc.get_instruments(inst_ids=inst_ids)
    print(ac)