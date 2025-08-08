from dataclasses import dataclass, field
from datetime import datetime
from ..constants.market_item_type import MarketItemType
from ..data.instruments import InstrumentService
from ..dates.utils import get_ny_timezone
from ..infrastructure import shock
from ..infrastructure.market_builder import MarketBuilder
from ..risk import greeks
from ..tools.timer import Timer
import pandas as pd
import pickle


@dataclass
class RiskEngineConfig:
    base_dt: datetime = None
    inst_ids: list = None
    greeks_list: list = None
    one_side: bool = False
    valuer_map_override: dict = field(default_factory=lambda: {})
    load_shared_file_eq_vol: bool = False
    flat_corr: float = None     # use flat correlation matrix. support both scaler or matrix
    sticky_strike_in_vega: bool = False
    same_vol_grid_in_delta: bool = False
    bpipe_get_bbg_history: object = None
    # calculate spot/vol ladder
    calc_ladder: bool = False
    spot_ladder: list = None
    vol_ladder: list = None
    spot_ladder_method: str = "percentage"
    vol_ladder_method: str = "level"
    # timing
    verbose: bool = False
    # local or distribution
    run_mode: str = "local"
    num_processes: int = 20
    market_pkl: str = "mkt.pkl"
    dakelake_credentials: dict = field(default_factory=lambda: {})
    throw_if_failure: bool = False

    # market snapshot timestep
    market_snapshot_timestamp: datetime = None


class RiskEngine:
    def __init__(self, config):
        self.config = config
        self.timer = Timer("RiskEngine", verbose=self.config.verbose, timezone=get_ny_timezone())
        self.timer.start()
        self.base_dt = self.timer.get_now() if config.base_dt is None else config.base_dt
        self.is_live = self.base_dt.date() == self.timer.get_now().date()
        self.greeks_list = ["price"] if config.greeks_list is None else config.greeks_list
        self.analytic_greeks_list = None
        self.numerical_greeks_list = None
        self.svc = InstrumentService()
        self.validate()
        self.inst_ids = []
        self.tradable_list = []
        self.market = None
        self.keys = None
        self.res_error = None
        self.res_full = None

    def validate_core(self):
        self.numerical_greeks_list = [x for x in self.greeks_list if "numerical#" in x]
        self.analytic_greeks_list = [x for x in self.greeks_list if "numerical#" not in x]
        if "price" not in self.analytic_greeks_list:
            self.analytic_greeks_list = ["price"] + self.analytic_greeks_list
        if self.config.calc_ladder:
            if self.config.spot_ladder is None and self.config.vol_ladder is None:
                raise Exception(f"Need supply spot or vol ladder to calculate ladder.")
            # fill default ladder if not found
            if self.config.spot_ladder is None:
                self.config.spot_ladder = [0]
            if self.config.vol_ladder is None:
                self.config.vol_ladder = [0]

    def validate(self):
        if self.config.inst_ids is None:
            raise Exception(f"Missing inst_ids")
        if self.config.greeks_list is None:
            raise Exception(f"Missing greeks_list")
        self.validate_core()

    def load_trades(self):
        self.timer.print("start load trade")
        if self.config.inst_ids is not None:
            self.inst_ids = self.config.inst_ids if isinstance(self.config.inst_ids, list) else [self.config.inst_ids]
            inst_lists = self.svc.get_instruments(inst_ids=self.inst_ids)
            self.tradable_list = [inst.get_tradable() for inst in inst_lists]
            valid_inst_class = ["Equity", "Index", "FX"]
            if not all([inst.inst_def()["InstClass"] in valid_inst_class for inst in inst_lists]):
                raise Exception(f"Only support inst class: {','.join(valid_inst_class)}")
            self.keys = []
            for tradable in self.tradable_list:
                self.keys += tradable.ask_keys(valuer=self.config.valuer_map_override.get(type(tradable), None))
            self.keys = list(set(self.keys))
        self.timer.reset("completed load trade")

    def create_market(self):
        self.timer.print("start create market")
        if self.is_live:
            self.market = MarketBuilder.build_market(self.tradable_list, self.base_dt, bpipe_get_bbg_history=self.config.bpipe_get_bbg_history, valuer_map_override=self.config.valuer_map_override, load_shared_file_eq_vol=self.config.load_shared_file_eq_vol, flat_corr=self.config.flat_corr)
        else:
            self.market = MarketBuilder.build_market(
                self.tradable_list, self.base_dt, bpipe_get_bbg_history=None, valuer_map_override=self.config.valuer_map_override,
                load_shared_file_eq_vol=self.config.load_shared_file_eq_vol, flat_corr=self.config.flat_corr, datalake_credentials=self.config.dakelake_credentials)
        self.config.market_snapshot_timestamp = self.timer.get_now() if self.is_live else self.base_dt
        if self.config.run_mode == "distribution":
            with open(self.config.market_pkl, 'wb') as f:
                pickle.dump(self.market, f)
        self.timer.reset("completed create market")

    @staticmethod
    def price_tradable(tradable, market, valuer, calc_types="price", trade_suffix=None, throw_if_failure=False):
        try:
            price = tradable.price(market=market, valuer=valuer, calc_types=calc_types, trade_suffix=trade_suffix)
        except Exception as e:
            if throw_if_failure:
                raise
            else:
                price = str(e)
        return price

    def calculate_numerical_greeks(self, market_override=None, base_analytic_greeks_res_map={}):
        if self.config.run_mode == "local":
            res = []
            market = self.market if market_override is None else market_override
            for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
                # price once
                valuer = self.config.valuer_map_override.get(type(tradable), None)
                if market_override is None and inst_id in base_analytic_greeks_res_map:
                    analytic_greeks_res_dict = base_analytic_greeks_res_map[inst_id]
                else:
                    analytic_greeks_res = tradable.price(market, valuer=valuer, calc_types=self.analytic_greeks_list, trade_suffix="_base",)
                    analytic_greeks_res_dict = dict(zip(self.analytic_greeks_list, analytic_greeks_res))
                price = analytic_greeks_res_dict["price"]
                greeks_kwargs = dict(origin_price=price)
                numerical_greeks_trd = greeks.calculate_numerical_greeks(
                    tradable, market, self.numerical_greeks_list, valuer=valuer, one_side=self.config.one_side,
                    same_vol_grid_in_delta=self.config.same_vol_grid_in_delta, sticky_strike_in_vega=self.config.sticky_strike_in_vega,
                    **greeks_kwargs)
                res_trd = [inst_id]
                for greeks_type in self.greeks_list:
                    if greeks_type in self.analytic_greeks_list:
                        res_trd.append(analytic_greeks_res_dict[greeks_type])
                    elif greeks_type in self.numerical_greeks_list:
                        res_trd.append(numerical_greeks_trd[greeks_type])
                res_trd = greeks.flatten_numerical_greeks(res_trd, ["inst_id"] + self.greeks_list)
                res.append(res_trd.set_index("Name").drop_duplicates().transpose().set_index("inst_id"))
            res = pd.concat(res)
        else:
            assert self.config.run_mode == "distribution"
            risk_params = self.collect_risk_tasks()
            res_list = self.run_tasks(RiskEngine.calculate_risk_per_tradable, risk_params, self.config.num_processes)
            res = pd.concat(res_list)
        return res

    @staticmethod
    def calculate_risk_per_tradable(inst_id, tradable, valuer, mkt_or_mkt_pkl, config, analytic_greeks_list, numerical_greeks_list, greeks_list):
        timer = Timer(f"calc risk for {inst_id}", verbose=config.verbose)
        timer.start()
        if isinstance(mkt_or_mkt_pkl, str):
            with open(mkt_or_mkt_pkl, 'rb') as f:
                mkt = pickle.load(f)
        else:
            mkt = mkt_or_mkt_pkl
        # price once
        analytic_greeks_res = tradable.price(mkt, valuer=valuer, calc_types=analytic_greeks_list, trade_suffix="_base")
        analytic_greeks_res_dict = dict(zip(analytic_greeks_list, analytic_greeks_res))
        price = analytic_greeks_res_dict["price"]
        greeks_kwargs = dict(origin_price=price)
        numerical_greeks_trd = greeks.calculate_numerical_greeks(
            tradable, mkt, numerical_greeks_list, valuer=valuer, one_side=config.one_side,
            same_vol_grid_in_delta=config.same_vol_grid_in_delta,
            sticky_strike_in_vega=config.sticky_strike_in_vega,
            **greeks_kwargs)
        res_trd = [inst_id]
        for greeks_type in greeks_list:
            if greeks_type in analytic_greeks_list:
                res_trd.append(analytic_greeks_res_dict[greeks_type])
            elif greeks_type in numerical_greeks_list:
                res_trd.append(numerical_greeks_trd[greeks_type])
        res_trd = greeks.flatten_numerical_greeks(res_trd, ["inst_id"] + greeks_list)
        res = res_trd.drop_duplicates().set_index("Name").transpose().set_index("inst_id")
        timer.end()
        return res

    @staticmethod
    def run_tasks(func, params, num_processes):
        chunksize = 1
        timer = Timer(f"run multiprocess on {len(params)} tasks over {num_processes} processes and chunk size {chunksize}")
        timer.start()
        from multiprocessing import get_context, set_start_method
        set_start_method("spawn", force=True)
        with get_context("spawn").Pool(processes=num_processes) as pool:
            res = pool.starmap(func, params, chunksize=chunksize)
        return res

    @staticmethod
    def calculate_numerical_greeks_per_tradable(
            spot_shock, vol_shock, inst_id, tradable, valuer, mkt_or_mkt_pkl, config,
            analytic_greeks_list, numerical_greeks_list, greeks_list, base_analytic_greeks_res_map={}):
        timer = Timer(f"calc numerical greeks for spot shock {spot_shock} vol shock {vol_shock} for {inst_id}", verbose=config.verbose)
        timer.start()
        if isinstance(mkt_or_mkt_pkl, str):
            with open(mkt_or_mkt_pkl, 'rb') as f:
                mkt = pickle.load(f)
        else:
            mkt = mkt_or_mkt_pkl
        try:
            no_shock = spot_shock == 0 and vol_shock == 0
            if no_shock:
                shocked_mkt = mkt
                trade_suffix = "_base"
            else:
                keys = tradable.ask_keys(valuer=valuer)
                vol_keys = [key for key in keys if (MarketItemType.VOLATILITY.value in key) or (MarketItemType.FXVOLATILITY.value in key)]
                spot_shock_obj = shock.SpotShock(size=spot_shock, method=config.spot_ladder_method, sticky_strike=True)
                vol_shock_obj = shock.VolShock(method=config.vol_ladder_method, parameters=vol_shock, sticky_strike=True)
                # lower down min vol for fx spot ladder calculation
                fx_vol_shock_obj = shock.VolShock(method=config.vol_ladder_method, parameters=vol_shock, sticky_strike=True, min_vol0=0.01)
                shocks = {}
                for vol_key in vol_keys:
                    if MarketItemType.FXVOLATILITY.value in vol_key:
                        shocks[vol_key] = [spot_shock_obj, fx_vol_shock_obj]
                    else:
                        shocks[vol_key] = [spot_shock_obj, vol_shock_obj]
                # apply shock
                shocked_mkt = mkt.apply(shocks)
                trade_suffix = f"_sc_spot_{spot_shock}_vol_{vol_shock}"
            # price once
            if no_shock and inst_id in base_analytic_greeks_res_map:
                analytic_greeks_res_dict = base_analytic_greeks_res_map[inst_id]
            else:
                analytic_greeks_res = tradable.price(shocked_mkt, valuer=valuer, calc_types=analytic_greeks_list, trade_suffix=trade_suffix)
                analytic_greeks_res_dict = dict(zip(analytic_greeks_list, analytic_greeks_res))
            price = analytic_greeks_res_dict["price"]
            greeks_kwargs = dict(origin_price=price)
            numerical_greeks_trd = greeks.calculate_numerical_greeks(
                tradable, shocked_mkt, numerical_greeks_list, valuer=valuer, one_side=config.one_side,
                same_vol_grid_in_delta=config.same_vol_grid_in_delta, sticky_strike_in_vega=config.sticky_strike_in_vega,
                **greeks_kwargs)
            res_trd = [inst_id]
            for greeks_type in greeks_list:
                if greeks_type in analytic_greeks_list:
                    res_trd.append(analytic_greeks_res_dict[greeks_type])
                elif greeks_type in numerical_greeks_list:
                    res_trd.append(numerical_greeks_trd[greeks_type])
            res_trd = greeks.flatten_numerical_greeks(res_trd, ["inst_id"] + greeks_list)
            res = res_trd.set_index("Name").drop_duplicates().transpose().set_index("inst_id")
            res["spot_shock"] = spot_shock
            res["vol_shock"] = vol_shock
            res["error"] = None
        except Exception as e:
            if config.run_mode == "local" and config.throw_if_failure:
                raise
            res = pd.DataFrame([[inst_id, spot_shock, vol_shock, str(e)]], columns=["inst_id", "spot_shock", "vol_shock", "error"])
        timer.end()
        return res

    def calculate_price(self):
        self.timer.print("start calculate price")
        if self.config.run_mode == "local":
            self.timer.print("price trade:")
            spot_shock = 0
            vol_shock = 0
            numerical_greeks_list = []
            greeks_list = self.analytic_greeks_list
            base_analytic_greeks_res_map = {}
            base_analytic_greeks_res_map_error = {}
            for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
                valuer = self.config.valuer_map_override.get(type(tradable), None)
                tmp = self.calculate_numerical_greeks_per_tradable(
                    spot_shock, vol_shock, inst_id, tradable, valuer, self.market, self.config,
                    self.analytic_greeks_list, numerical_greeks_list, greeks_list, base_analytic_greeks_res_map)
                if tmp.loc[inst_id, "error"] is None:
                    base_analytic_greeks_res_map[inst_id] = tmp.loc[inst_id].to_dict()
                else:
                    base_analytic_greeks_res_map_error[inst_id] = tmp.loc[inst_id, "error"]
            self.timer.reset("completed calculate original price")
        else:
            assert self.config.run_mode == "distribution"
            price_params = self.collect_price_tasks()
            analytic_greeks_res_list = self.run_tasks(RiskEngine.calculate_numerical_greeks_per_tradable, price_params, self.config.num_processes)
            analytic_greeks_res_df = pd.concat(analytic_greeks_res_list)
            base_analytic_greeks_res_map = {}
            base_analytic_greeks_res_map_error = {}
            for inst_id, row in analytic_greeks_res_df.iterrows():
                if row.error is None:
                    base_analytic_greeks_res_map[inst_id] = row.to_dict()
                else:
                    base_analytic_greeks_res_map_error[inst_id] = row.error
        return base_analytic_greeks_res_map, base_analytic_greeks_res_map_error

    def calculate_ladder(self):
        self.timer.print("start calculate ladder")
        assert self.config.calc_ladder
        # 1. price once
        base_analytic_greeks_res_map, base_analytic_greeks_res_map_error = self.calculate_price()
        # 2. calculate ladder
        self.timer.print("calc ladder:")
        if self.config.run_mode == "local":
            greeks = []
            for spot_shock in self.config.spot_ladder:
                for vol_shock in self.config.vol_ladder:
                    for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
                        print(inst_id, spot_shock, vol_shock)
                        valuer = self.config.valuer_map_override.get(type(tradable), None)
                        tmp = self.calculate_numerical_greeks_per_tradable(
                            spot_shock, vol_shock, inst_id, tradable, valuer, self.market, self.config,
                            self.analytic_greeks_list, self.numerical_greeks_list, self.greeks_list, base_analytic_greeks_res_map)
                        greeks.append(tmp)
        else:
            assert self.config.run_mode == "distribution"
            params = self.collect_ladder_tasks(base_analytic_greeks_res_map)
            greeks = self.run_tasks(RiskEngine.calculate_numerical_greeks_per_tradable, params, self.config.num_processes)
        greeks_df = pd.concat(greeks)
        res_full = greeks_df[greeks_df.error.isna()]
        res_error = greeks_df[~greeks_df.error.isna()]
        res_final = self.post_process_ladder(res_full, base_analytic_greeks_res_map)
        if res_full[(res_full.spot_shock == 0) & (res_full.vol_shock == 0)].empty:
            base_price = pd.DataFrame.from_dict(base_analytic_greeks_res_map, orient="index")
            res_full = pd.concat([res_full, base_price])
        self.timer.reset("completed calculate scenario")
        return res_final, res_error, res_full

    def collect_risk_tasks(self):
        params = []
        for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
            valuer = self.config.valuer_map_override.get(type(tradable), None)
            params.append([
                inst_id, tradable, valuer, self.config.market_pkl, self.config,
                self.analytic_greeks_list, self.numerical_greeks_list, self.greeks_list])
        return params

    def collect_price_tasks(self):
        params = []
        spot_shock = 0
        vol_shock = 0
        numerical_greeks_list = []
        greeks_list = self.analytic_greeks_list
        base_analytic_greeks_res_map = {}
        for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
            valuer = self.config.valuer_map_override.get(type(tradable), None)
            params.append([
                spot_shock, vol_shock, inst_id, tradable, valuer, self.config.market_pkl, self.config,
                self.analytic_greeks_list, numerical_greeks_list, greeks_list, base_analytic_greeks_res_map])
        return params

    def collect_ladder_tasks(self, base_analytic_greeks_res_map):
        params = []
        for spot_shock in self.config.spot_ladder:
            for vol_shock in self.config.vol_ladder:
                for inst_id, tradable in zip(self.inst_ids, self.tradable_list):
                    valuer = self.config.valuer_map_override.get(type(tradable), None)
                    params.append([
                        spot_shock, vol_shock, inst_id, tradable, valuer, self.config.market_pkl, self.config,
                        self.analytic_greeks_list, self.numerical_greeks_list, self.greeks_list,
                        base_analytic_greeks_res_map])
        return params

    def post_process_ladder(self, res, base_analytic_greeks_res_map):
        tmp = res.pivot_table(index="spot_shock", columns="vol_shock", aggfunc="sum")
        if "price" in self.greeks_list:
            base_analytic_greeks_res_df = pd.DataFrame(base_analytic_greeks_res_map)
            base_price = base_analytic_greeks_res_df.loc["price"].sum()
            pnl = tmp.loc[:, "price"] - base_price
            pnl.columns = pd.MultiIndex.from_product([["pnl"], pnl.columns], names=tmp.columns.names)
            tmp = pd.concat([tmp, pnl], axis=1)
        return tmp

    def run(self):
        self.load_trades()
        self.create_market()
        if self.config.calc_ladder:
            res, self.res_error, self.res_full = self.calculate_ladder()
        else:
            res = self.calculate_numerical_greeks()
        self.timer.end()
        return res


if __name__ == "__main__":
    RiskEngine(config=RiskEngineConfig()).run()