from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ...backtest.costs import FlatTradingCost
from ...backtest.rolls import RollFutureContracts, RollFXForwardContracts
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.FXforward import FXforward
from ...dates.schedules import MonthDaySchedule
from ...valuation.fx_forward_fx_vol_surface_valuer import FXForwardDataValuer
from ...dates.utils import bdc_adjustment,get_business_days
from ...constants.business_day_convention import BusinessDayConvention


class RollingFXForwardsBasketState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, notional, weights,price_hist):
        self.price = price
        self.cost = cost
        self.notional = notional
        self.weights =  weights
        self.price_hist = price_hist
        super().__init__(time_stamp, portfolio)



class EODTradeEvent(Event):

    def equal_weights(self):

        w = {}

        constituents = list(self.strategy.parameters['legs'].keys())

        n = len(constituents)

        for name in constituents:
            
            w[name] = 1 / n
        
        return w
    
    def CTA_momentum(self,state):

        price_df=pd.DataFrame(state.price_hist).T

        zs=pd.DataFrame()
        
        for pairs in self.strategy.parameters['weights_method'][1]:
            ll=price_df.ewm(pairs[1]).mean()
            sl=price_df.ewm(pairs[0]).mean()

            x=sl-ll

            y=x/price_df.rolling(63).std()

            z=(y/y.rolling(252).std())

            if len(z.dropna()) > 0:

                z=z.iloc[-1,:]
                
                rz=z.apply(lambda x: (x*np.exp((-x**2)/4)/0.89))

                zs[str(pairs)]=rz

        if len(zs) > 0:

            S_CTA=zs.mean(axis=1)

            longs=S_CTA[S_CTA>0].sort_values(ascending=False).iloc[:3]

            shorts=S_CTA[S_CTA<0].sort_values(ascending=True).iloc[:3]

            w = {}
        
            for index in longs.index:

                w[index] = 0.5/len(longs.index)

            for index in shorts.index:

                w[index] = -0.5/len(shorts.index)

            for leg in self.parameters['legs'].keys():

                if leg not in w.keys():
                    
                    w[leg] = 0

            return w
        else:       
            w = {}

            for leg in self.parameters['legs'].keys(): 

                w[leg] = 0
                
            return w
        
    
    def is_rebalance_date(self, dt):
        if isinstance(self.strategy.parameters['rebalance_schedule'], tuple):
            if self.strategy.parameters['rebalance_schedule'][0] == 'monthly':
                day_of_month = self.strategy.parameters['rebalance_schedule'][1]
                this_month_rebalance_day = datetime(dt.year, dt.month, day_of_month)
                this_month_rebalance_day = bdc_adjustment(this_month_rebalance_day, convention=BusinessDayConvention.FOLLOWING, holidays=self.strategy.holidays)
                if dt.date() == this_month_rebalance_day.date():
                    return True
                else:
                    return False
            else:
                RuntimeError(f"Cannot handle rebalance_schedule as tuple with first element being {self.strategy.parameters['rebalance_schedule'][0]}")
        
        elif self.strategy.parameters['rebalance_schedule'] == 'daily':
            
            return True

        else:
            raise RuntimeError(f"Unknown type of rebalance_schedule")

    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()

        # get market data for this day
        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # roll if need to
        self.parameters['roll'].roll(self.time_stamp, portfolio)

        price_dict={}

        
        if (self.is_rebalance_date(self.time_stamp) or self.time_stamp == self.strategy.start_date):
            
            # weight calculation
            if 'weight_function' in self.parameters.keys():

                weights_dict=self.parameters['weight_function'](self.strategy,state)

            elif self.parameters['weights_method'] == 'Equal':

                weights_dict=self.equal_weights()
            
            elif isinstance(self.parameters['weights_method'],tuple):

                if self.parameters['weights_method'][0]=='Momentum':

                    weights_dict=self.CTA_momentum(state)

                elif self.parameters['weights_method'][0]=='Pre_defined':
                    
                    if self.time_stamp in self.parameters['weights_method'][1].index:

                        weights_dict=self.parameters['weights_method'][1].loc[self.time_stamp].to_dict()

                    else:
                        
                        weights_dict=state.weights



            #rebalance trading

            
            for leg in self.parameters['legs'].keys():

                if self.parameters['notional_type'] == 'fixed':

                    notional=self.parameters['notional']
                
                else:

                    notional=state.notional

                if leg in portfolio.get_positions().keys():

                    sub_portfolio=portfolio.get_positions()[leg]

                    current_contract=self.parameters['roll'].find_next_contract(self.time_stamp, FXforward(self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['currency'], self.time_stamp, 'America/New_York'))


                    current_position=sub_portfolio.get_position(current_contract.name())

                    if current_position is None:

                        first_contract = self.parameters['roll'].find_next_contract(self.time_stamp, FXforward(self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['currency'], self.time_stamp, 'America/New_York'))
                        execution_price = first_contract.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                        quantity = (self.parameters['notional']*weights_dict[leg]) / execution_price
                        portfolio.trade(first_contract, quantity, execution_price=execution_price, execution_currency=self.parameters['legs'][leg]['currency'],position_path=(leg,))
                    
                    else:

                        tradable=current_position.tradable

                        # print('before')
                        # print(current_porition.quantity)

                        execution_price = tradable.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                        
                        # print(execution_price)

                        new_units = (notional*weights_dict[leg]) / execution_price

                        # print(new_units)

                        trade_units = new_units - (0 if current_position is None else current_position.quantity)

                        # print(trade_units)

                        portfolio.trade(tradable, trade_units, execution_price,position_path=(leg,))

                else:

                    first_contract = self.parameters['roll'].find_next_contract(self.time_stamp, FXforward(self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['currency'], self.time_stamp, 'America/New_York'))
                    execution_price = first_contract.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])
                    quantity = (self.parameters['notional']*weights_dict[leg]) / execution_price
                    portfolio.trade(first_contract, quantity, execution_price=execution_price, execution_currency=self.parameters['legs'][leg]['currency'],position_path=(leg,))


                price_dict[leg]=execution_price
                

                    # print('after')

                    # print(portfolio.get_positions()[leg].get_position(current_contract.name()).quantity)
        else:

            for leg in self.parameters['legs'].keys():

                current_contract=self.parameters['roll'].find_next_contract(self.time_stamp, FXforward(self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['underlying'], self.parameters['legs'][leg]['currency'], self.time_stamp, 'America/New_York'))
                
                forward_price = current_contract.price(market, calc_types='price', currency=self.parameters['legs'][leg]['currency'])

                price_dict[leg]=forward_price







        # cost
        price_pre_cost = portfolio.price_at_market(market, fields='price')
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)
        price_after_cost = portfolio.price_at_market(market, fields='price')

        state.price_hist.update({self.time_stamp: price_dict})

        return RollingFXForwardsBasketState(self.time_stamp, portfolio, price_after_cost, price_after_cost - price_pre_cost, self.parameters['notional']+price_after_cost, weights_dict if 'weights_dict' in locals() else state.weights, state.price_hist)


class RollingFXForwardsBasket(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        self.parameters['roll'] = RollFXForwardContracts(self.backtest_market, self.parameters['roll_offset'], self.holidays)
        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
