from ..tradable.option import Option
import numpy as np

class OptionChain:
    def __init__(self, dt, root, underlying, stock_id, data):
        self.dt = dt
        self.root = root
        self.underlying = underlying
        self.stock_id = stock_id
        self.data = data

class SelectionTool:

    def __init__(self, option_chain, call_put, select_expiry_func, select_strike_func):
        self.option_chain = option_chain
        self.call_put = call_put
        self.select_expiry_func = select_expiry_func
        self.select_strike_func = select_strike_func

    def select(self):
        self.option_chain.data = self.option_chain.data[self.option_chain.data['call_put'] == self.call_put]
        self.option_chain = self.select_expiry_func(option_chain=self.option_chain)
        self.option_chain = self.select_strike_func(option_chain=self.option_chain)

        assert len(self.option_chain.data) != 0, 'Empty option selected'
        assert len(self.option_chain.data) == 1, 'Selected multiple options'
        option_chain_data = self.option_chain.data
        strike = option_chain_data['strike'].values[0]
        expiration = option_chain_data['expiration'].values[0]
        is_call = option_chain_data['call_put'].values[0]
        ticker = option_chain_data['option_symbol'].values[0]
        stock_id = self.option_chain.stock_id
        root = self.option_chain.root
        underlying = self.option_chain.underlying
        contract_size = np.nan
        is_american = False
        tz_name = None   #eidt later

        return Option(root=root, underlying=underlying, currency='USD', expiration=expiration,
                       strike=strike, is_call=is_call, is_american=is_american, contract_size=contract_size,
                       tz_name=tz_name, listed_ticker=ticker)

