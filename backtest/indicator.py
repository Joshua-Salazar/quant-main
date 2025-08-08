import pandas as pd


class Indicator:

    def __init__(self, name, one_range):
        self.name = name
        self.one_range = one_range


class ExternalIndicator:

    def __init__(self, name, one_range, external_source_csv, get_indicator_func=None):
        self.name = name
        self.one_range = one_range
        self.external_source = pd.read_csv(external_source_csv, parse_dates=["date"], index_col="date")
        self.get_indicator_func = self.get_indicator_func_default if get_indicator_func is None else get_indicator_func

    @staticmethod
    def get_indicator_func_default(value, one_range):
        return 1 if one_range[0] <= value <= one_range[1] else 0

    def get_indicator(self, dt):
        value = self.external_source.loc[dt, self.name]
        indicator = self.get_indicator_func(value, self.one_range)
        return indicator
