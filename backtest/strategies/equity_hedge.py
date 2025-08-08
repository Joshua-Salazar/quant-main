from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ...backtest.costs import FlatTradingCost
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.stock import Stock
from ...dates.utils import bdc_adjustment,get_business_days
from ...constants.business_day_convention import BusinessDayConvention

class EquityState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, notional,flag_dict,leg_entries):
        self.price = price
        self.cost = cost
        self.notional = notional
        self.flag_dict=flag_dict
        self.leg_entries=leg_entries
        super().__init__(time_stamp, portfolio)


class EODTradeEvent(Event):

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        

        if 'flag_function' in self.parameters.keys():

            flag_dict=self.parameters['flag_function'](self.strategy,state)
        
        elif isinstance(self.parameters['flag_method'],tuple):

            if self.parameters['flag_method'][0]=='Pre_defined':
                
                if self.time_stamp in self.parameters['flag_method'][1].index:

                    flag_dict=self.parameters['flag_method'][1].loc[self.time_stamp].to_dict()

                else:
                    
                    flag_dict=state.flag_dict

        leg_entries=state.leg_entries

        for leg in self.parameters['legs'].keys():

            if leg in portfolio.get_positions().keys():

                sub_portfolio=portfolio.get_positions()[leg]

            else:
                    
                sub_portfolio=portfolio

            flag=flag_dict[leg]

            equity=Stock(self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['currency'])

            current_position=sub_portfolio.get_position(equity.name())


            if flag:

                if current_position is None:

                    hedge_dict=self.parameters['legs'][leg]['hedge_function'](self.strategy,state,leg)

                    equity_price = equity.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                    quantity = self.parameters['legs'][leg]['trade_notional'] / equity_price
                    portfolio.trade(equity, quantity, execution_price=equity_price, execution_currency=self.parameters['legs'][leg]['currency'],position_path=(leg,))

                    for hedge_instrument in hedge_dict.keys():

                        hedge_equity=Stock(hedge_instrument, self.parameters['legs'][leg]['currency'])

                        hedge_price = hedge_equity.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                        hedge_quantity = hedge_dict[hedge_instrument] / hedge_price
                        portfolio.trade(hedge_equity, hedge_quantity, execution_price=hedge_price, execution_currency=self.parameters['legs'][leg]['currency'],position_path=(leg,))


                    leg_entries[leg]=self.time_stamp
            
            else:

                if current_position is not None:

                    current_holdings=sub_portfolio.get_positions().copy()

                    for holding in current_holdings.keys():

                        if 'Cash' not in holding:
                            current_position=sub_portfolio.get_position(holding)

                            equity_price = current_position.tradable.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                            portfolio.trade(current_position.tradable, -current_position.quantity, execution_price=equity_price, execution_currency=self.parameters['legs'][leg]['currency'],position_path=(leg,))
                    
                    leg_entries[leg]=None

        price_pre_cost = portfolio.price_at_market(market, fields='price', currency=self.strategy.currency)
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)
        price_after_cost = portfolio.price_at_market(market, fields='price', currency=self.strategy.currency)


        return EquityState(self.time_stamp, portfolio, price_after_cost, price_after_cost - price_pre_cost, self.parameters['notional']+price_after_cost, flag_dict, leg_entries)
    


class EquityHedge(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]

if __name__ == '__main__':

    from ..infrastructure.equity_data_container import EquityDataRequest,DatalakeBBGEquityDataSource
    from ..backtest.backtester import LocalBacktester
    from ..tradable.portfolio import Portfolio

    def flag_function(strategy,state):

        result={}


        for leg in strategy.parameters['legs'].keys():
            
            if state.leg_entries[leg] is not None:
                
                if (state.time_stamp-state.leg_entries[leg]).days+1 >= 179:

                     result[leg]=False

            if leg not in result.keys():
                
                result[leg]=strategy.parameters['signals'][leg]


        return result

    def hedge_function(strategy,state,leg):
        
        hedge_dict={k:-0.2*100 for k in hedging_instrument}

        return hedge_dict
        

    start_date = datetime(2020, 1, 3)
    end_date = datetime(2024, 1, 4)

    currency='USD'

    calendar = ['XCBO']

    equities=['AAPL US Equity','MSFT US Equity','GOOGL US Equity']

    hedging_instrument=['NVDA US Equity','TSLA US Equity','META US Equity','AMZN US Equity','ORCL US Equity']

    singal_dict={'AAPL US Equity':True,'MSFT US Equity':True,'GOOGL US Equity':True}
    
    strategy = EquityHedge(
        start_date=start_date,
        end_date=end_date,
        calendar=calendar,
        currency=currency,
        parameters={
            'flag_function': flag_function,
            'signals': singal_dict,
            'tc_rate': 0.02,
            'notional':0,
            'legs':{
                k:{
                'underlying': k,
                'currency': currency,
                'trade_notional': 100,
                'hedge_function':hedge_function
                } for k in equities
            }
        },
        data_requests={
            k: (
                EquityDataRequest(start_date, end_date, calendar, k , currency),
                DatalakeBBGEquityDataSource()
            )
            for k in equities + hedging_instrument
        }
    )
    
    runner = LocalBacktester()
    results = runner.run(strategy, start_date, end_date, EquityState(start_date, Portfolio([]), 0.0, 0.0,100,{},{k:None for k in equities}))

    records = [{'date': x.time_stamp, 'pnl': x.price} for x in results]
    pnl_series = pd.DataFrame.from_dict(records)

    print(pnl_series)





