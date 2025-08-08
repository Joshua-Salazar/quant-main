from ..backtest_nb.common_bt_future_options import CommonBTFutureOption
from ..backtest_nb.common_bt_fx_options import CommonBTFXOptions
from ..backtest_nb.common_bt_swaptions import CommonBTSwaption
from ..backtest_nb.common_bt_stock_options_daily import CommonBTStockOptionsDaily
from ..backtest_nb.common_bt_cr_options import CommonBTCROptions
from ..backtest_nb.common_bt import CommonBT
from ..backtest_nb.common_bt_config import *


class CommonBTFactory:
    def __init__(self):
        pass

    # def create(self, leg_type, *args, **kwargs) -> CommonBT:
    #     # @replay
    #     def nested_create(leg_type, *args, **kwargs):
    #         if leg_type.lower() == "future_option":
    #             return CommonBTFutureOption(*args, **kwargs)
    #         elif leg_type.lower() == "fx_option":
    #             return CommonBTFXOptions(*args, **kwargs)
    #         elif leg_type.lower() == "swaption":
    #             return CommonBTSwaption(*args, **kwargs)
    #         elif leg_type.lower() == "stock_options_daily":
    #             # kwargs.pop('replay', None)
    #             # kwargs.pop('replay_file', None)
    #             return CommonBTStockOptionsDaily(*args, **kwargs)
    #         elif leg_type.lower() == "cr_options":
    #             return CommonBTCROptions(*args, **kwargs)
    #
    #         raise Exception(f"Unsupported leg type: {leg_type}. \
    #         Valid types are: future_option, fx_option, swaption, stock_options_daily, cr_options")
    #
    #     nested_create(leg_type, *args, **kwargs)

    def create(self, leg_type, *args, **kwargs) -> CommonBT:
        if leg_type.lower() == "future_option":
            return CommonBTFutureOption(*args, **kwargs)
        elif leg_type.lower() == "fx_option":
            return CommonBTFXOptions(*args, **kwargs)
        elif leg_type.lower() == "swaption":
            return CommonBTSwaption(*args, **kwargs)
        elif leg_type.lower() == "stock_options_daily":
            # kwargs.pop('replay', None)
            # kwargs.pop('replay_file', None)
            return CommonBTStockOptionsDaily(*args, **kwargs)
        elif leg_type.lower() == "cr_options":
            return CommonBTCROptions(*args, **kwargs)

        raise Exception(f"Unsupported leg type: {leg_type}. \
        Valid types are: future_option, fx_option, swaption, stock_options_daily, cr_options")


    def get_valid_values(self, param_name):
        if param_name.lower() == "leg_type":
            return LEG_TYPE
        elif param_name.lower() == "asset":
            assets = []
            [assets.extend(v) for k, v in ASSET_MAP.items()]
            return assets
        else:
            raise Exception(f"Unsupported parameter name: {param_name}. Valid are leg_type, asset")
