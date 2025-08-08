from ..tradable.position import ValuedPosition
from ..tradable.portfolio import Portfolio
from datetime import datetime


class PnlAttribution:
    def __init__(self, unds: list, expiry: datetime, strike: float=None):
        self.unds = unds
        self.expiry = expiry
        self.strike = strike

    @staticmethod
    def add_spot_pnl_attribution(curr_pfo, prev_pfo, und, curr_spot, prev_spot, curr_market, prev_market):
        spot_pct_change = 0 if prev_spot is None else curr_spot / prev_spot - 1
        for k, v in curr_pfo.root.items():
            if isinstance(v, Portfolio):
                PnlAttribution.add_spot_pnl_attribution(curr_pfo.root[k], None if prev_pfo is None else prev_pfo.root.get(k, None), und, curr_spot, prev_spot, curr_market, prev_market)
            else:
                assert isinstance(v, ValuedPosition)
                if prev_pfo is None or prev_pfo.root.get(k, None) is None:
                    # new trade so zero position pnl and not support trading pnl yet
                    delta_pnl = 0
                    gamma_pnl = 0
                else:
                    fx_adj = 1 if v.tradable.currency == "USD" else curr_market.get_fx_spot(f"{v.tradable.currency}USD") / prev_market.get_fx_spot(f"{v.tradable.currency}USD")
                    prev_rm = prev_pfo.root[k].get_additional_attributes()
                    delta_pnl = prev_rm[f"DeltaUSD#{und}"] * spot_pct_change * fx_adj
                    gamma_pnl = 0.5 * prev_rm[f"GammaUSD#{und}"] * spot_pct_change ** 2 / 0.01 * fx_adj
                setattr(curr_pfo.root[k], f"DeltaPnl#{und}", delta_pnl)
                setattr(curr_pfo.root[k], f"GammaPnl#{und}", gamma_pnl)
                setattr(curr_pfo.root[k], f"SpotChange#{und}", spot_pct_change)

    @staticmethod
    def add_vol_pnl_attribution(curr_pfo, prev_pfo, und, curr_vol, prev_vol, curr_market, prev_market):
        for k, v in curr_pfo.root.items():
            if isinstance(v, Portfolio):
                PnlAttribution.add_vol_pnl_attribution(curr_pfo.root[k], None if prev_pfo is None else prev_pfo.root.get(k, None), und, curr_vol, prev_vol, curr_market, prev_market)
            else:
                if prev_pfo is None or prev_pfo.root.get(k, None) is None:
                    # new trade so zero position pnl and not support trading pnl yet
                    assert isinstance(v, ValuedPosition)
                    vol_pnl = 0
                    vol_change = 0
                else:
                    fx_adj = 1 if v.tradable.currency == "USD" else curr_market.get_fx_spot(f"{v.tradable.currency}USD") / prev_market.get_fx_spot(f"{v.tradable.currency}USD")
                    prev_rm = prev_pfo.root[k].get_additional_attributes()
                    curr_rm = curr_pfo.root[k].get_additional_attributes()
                    vol_change = (curr_rm[f"VolRef#{und}"] - prev_rm[f"VolRef#{und}"]) * 100 if f"VolRef#{und}" in prev_rm else 0
                    vol_pnl = prev_rm[f"VegaUSD#{und}"] * vol_change * fx_adj
                setattr(curr_pfo.root[k], f"VolPnl#{und}", vol_pnl)
                setattr(curr_pfo.root[k], f"VolChange#{und}", vol_change)

    @staticmethod
    def add_vanna_pnl_attribution(curr_pfo, prev_pfo, und, curr_spot, prev_spot, curr_vol, prev_vol, curr_market, prev_market):
        spot_pct_change = 0 if prev_spot is None else curr_spot / prev_spot - 1
        for k, v in curr_pfo.root.items():
            if isinstance(v, Portfolio):
                PnlAttribution.add_vanna_pnl_attribution(curr_pfo.root[k], None if prev_pfo is None else prev_pfo.root.get(k, None), und, curr_spot, prev_spot, curr_vol, prev_vol, curr_market, prev_market)
            else:
                if prev_pfo is None or prev_pfo.root.get(k, None) is None:
                    # new trade so zero position pnl and not support trading pnl yet
                    assert isinstance(v, ValuedPosition)
                    vanna_pnl = 0
                else:
                    fx_adj = 1 if v.tradable.currency == "USD" else curr_market.get_fx_spot(f"{v.tradable.currency}USD") / prev_market.get_fx_spot(f"{v.tradable.currency}USD")
                    prev_rm = prev_pfo.root[k].get_additional_attributes()
                    curr_rm = curr_pfo.root[k].get_additional_attributes()
                    vol_change = (curr_rm[f"VolRef#{und}"] - prev_rm[f"VolRef#{und}"]) * 100 if f"VolRef#{und}" in prev_rm else 0
                    vanna_pnl = prev_rm[f"VannaUSD#{und}"] * spot_pct_change * vol_change * fx_adj
                setattr(curr_pfo.root[k], f"VannaPnl#{und}", vanna_pnl)

    @staticmethod
    def add_total_pnl_attribution(curr_pfo, prev_pfo, unds, curr_market, prev_market):
        for k, v in curr_pfo.root.items():
            if isinstance(v, Portfolio):
                PnlAttribution.add_total_pnl_attribution(curr_pfo.root[k], None if prev_pfo is None else prev_pfo.root.get(k, None), unds, curr_market, prev_market)
            else:
                if prev_pfo is None or prev_pfo.root.get(k, None) is None:
                    # new trade so zero position pnl and not support trading pnl yet
                    assert isinstance(v, ValuedPosition)
                    total_pnl = 0
                    explained_pnl = 0
                else:
                    fx_adj = 1 if v.tradable.currency == "USD" else curr_market.get_fx_spot(f"{v.tradable.currency}USD") / prev_market.get_fx_spot(f"{v.tradable.currency}USD")
                    prev_rm = prev_pfo.root[k].get_additional_attributes()
                    curr_rm = curr_pfo.root[k].get_additional_attributes()
                    total_pnl = (curr_rm["price"] - prev_rm["price"]) * prev_pfo.root[k].quantity * fx_adj
                    explained_pnl = 0
                    for und in unds:
                        explained_pnl += curr_rm[f"DeltaPnl#{und}"] + curr_rm[f"GammaPnl#{und}"] + curr_rm[f"VolPnl#{und}"] + curr_rm[f"VannaPnl#{und}"]
                setattr(curr_pfo.root[k], f"TotalPnl", total_pnl)
                setattr(curr_pfo.root[k], f"ExplainedPnl", explained_pnl)
                setattr(curr_pfo.root[k], f"UnExplainedPnl", total_pnl-explained_pnl)

    def apply(self, curr_pfo, prev_pfo, curr_market, prev_market):

        for und in self.unds:
            # spot pnl attribution
            if prev_market is None:
                curr_spot = curr_market.get_spot(und)
                prev_spot = None
            else:
                curr_spot = curr_market.get_spot(und)
                prev_spot = prev_market.get_spot(und)
            self.add_spot_pnl_attribution(curr_pfo, prev_pfo, und, curr_spot, prev_spot, curr_market, prev_market)

            # ref vol strike at prev atm level
            if prev_market is None:
                curr_vol = None
                prev_vol = None
            else:
                strike = prev_market.get_vol_surface(und).get_forward(self.expiry) if self.strike is None else self.strike
                curr_vol = curr_market.get_vol(und, self.expiry, strike)
                prev_vol = prev_market.get_vol(und, self.expiry, strike)
            # vol pnl attribution
            self.add_vol_pnl_attribution(curr_pfo, prev_pfo, und, curr_vol, prev_vol, curr_market, prev_market)
            # vanna pnl attribution
            self.add_vanna_pnl_attribution(curr_pfo, prev_pfo, und, curr_spot, prev_spot, curr_vol, prev_vol, curr_market, prev_market)

        self.add_total_pnl_attribution(curr_pfo, prev_pfo, self.unds, curr_market, prev_market)