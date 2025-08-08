from ..infrastructure.market import Market
from ..interface.ivaluer import IValuer
from ..tradable.varianceswap import VarianceSwap
from ..valuation import valuer_utils
from datetime import datetime
from ..analytics.symbology import OPTION_ROOT_FROM_TICKER


import numpy as np
import pandas as pd

class VarianceSwapReplicationValuer(IValuer):
    def __init__(self, max_delta_strike=None, min_delta_strike=None, recalculate_replication_portfolio=None):
        self.max_delta_strike = 0.001 if max_delta_strike is None else max_delta_strike
        self.min_delta_strike = -0.001 if min_delta_strike is None else min_delta_strike
        self.recalculate_replication_portfolio = True if recalculate_replication_portfolio is None else \
            recalculate_replication_portfolio

        self.validate()

    def validate(self):
        if self.max_delta_strike <= 0:
            raise Exception(f"max delta strike ({self.max_delta_strike}) must be positive")
        if self.min_delta_strike >= 0:
            raise Exception(f"min delta strike ({self.min_delta_strike}) must be negative")

    def price(self, varswap: VarianceSwap, market: Market, calc_types='price', recalculate_replication_portfolio = False, **kwargs):

        if recalculate_replication_portfolio:
            ## compute rep_ptf
            rep_ptf = self.rep_ptf(varswap, market)
            rep_ptf_price = rep_ptf.ptf_px.unique()[0]
            fair_strike = rep_ptf.K_var.unique()[0]
            rep_ptf_delta = ( rep_ptf.wgt * rep_ptf.delta ).sum()
            rep_ptf_vega = ( rep_ptf.wgt * rep_ptf.vega ).sum()
            rep_ptf_gamma = ( rep_ptf.wgt * rep_ptf.gamma ).sum()
            rep_ptf_theta = ( rep_ptf.wgt * rep_ptf.theta ).sum()
            spot_dt = market.get_spot( varswap.underlying )
            # update synthetic fwd
            synt_fwd = self.synth_fwd(varswap, market)
        else:
            ## pull updated mkt data + update prices
            rep_ptf = varswap.rep_ptf
            rep_ptf = rep_ptf.drop_duplicates()

            # update synthetic fwd
            synt_fwd = self.synth_fwd(varswap, market)
            assert len( rep_ptf ) > 0

            option_universe = market.get_option_universe( OPTION_ROOT_FROM_TICKER[varswap.underlying],
                                                          return_as_dict=False)
            option_universe = option_universe[option_universe.expiration_date == rep_ptf.expiration_date.unique()[0]]
            option_universe = option_universe[[ x not in [32150, 25757, 32229]
                                                for x in option_universe.stock_id.values]]

            core_ptf = rep_ptf.copy()
            core_ptf = core_ptf[['strike', 'call_put', 'wgt', 'K_var', 'expiration_date']]
            if len( option_universe ) == 0 and datetime.fromisoformat( rep_ptf.expiration_date.unique()[0] ) == market.base_datetime:
                core_ptf['price'] = [ max( market.get_spot(varswap.underlying) - x, 0 ) * ( y == 'C' ) +
                                      max( x - market.get_spot(varswap.underlying), 0) * ( y == 'P' ) for x,y in zip( core_ptf.strike, core_ptf.call_put ) ]
                core_ptf['delta'] = [ 0 for x in core_ptf.strike ]
                core_ptf['gamma'] = [ 0 for x in core_ptf.strike ]
                core_ptf['theta'] = [ 0 for x in core_ptf.strike ]
                core_ptf['vega'] = [ 0 for x in core_ptf.strike ]
                new_rep_ptf = core_ptf
            else:
                new_rep_ptf = [ ]
                for ix in np.arange( len( core_ptf ) ):
                    row = core_ptf.iloc[ ix ]
                    upd_row = option_universe[ option_universe.strike == row.strike ]
                    upd_row = upd_row[ upd_row.call_put == row.call_put ]
                    upd_row = upd_row[ upd_row.expiration_date == row.expiration_date ]
                    if len( upd_row ) == 0:
                        continue

                    # add intrinsic value when prices on expiry are missing
                    if upd_row.expiration_date.values[0] == upd_row.date.values[0] and option_universe.price.max() < 0.000000001:
                        if upd_row.call_put.values[0] == 'C':
                           upd_row[ 'price' ] = max( market.get_spot(varswap.underlying) - upd_row['strike'].values[0], 0 )
                        else:
                            upd_row['price'] = max( upd_row['strike'].values[0] - market.get_spot(varswap.underlying), 0)

                    upd_row[ 'K_var'] = row.K_var
                    upd_row[ 'wgt' ] = row.wgt
                    #new_rep_ptf = pd.concat( [ new_rep_ptf, upd_row ] )
                    new_rep_ptf.append( upd_row )

                # fix for when early data does not load correctly on expiration day
                if option_universe.expiration_date.unique()[0] == option_universe.date.unique()[0] and len(new_rep_ptf) == 0:
                    spot_dt = market.get_spot(varswap.underlying)
                    new_rep_ptf = core_ptf.copy()
                    new_rep_ptf['price'] = [ max( spot_dt - new_rep_ptf.iloc[ix].strike, 0 ) if new_rep_ptf.iloc[ix].call_put == 'C'
                                             else max( new_rep_ptf.iloc[ix].strike - spot_dt, 0 ) for ix in range(len(new_rep_ptf)) ]
                    new_rep_ptf['delta'] = [ 0 for ix in range(len(new_rep_ptf)) ]
                    new_rep_ptf['gamma'] = [0 for ix in range(len(new_rep_ptf))]
                    new_rep_ptf['vega'] = [0 for ix in range(len(new_rep_ptf))]
                    new_rep_ptf['theta'] = [0 for ix in range(len(new_rep_ptf))]
                else:
                    new_rep_ptf = pd.concat( new_rep_ptf )

            # update ptf_px
            new_rep_ptf['ptf_px'] = [ ( new_rep_ptf.wgt * new_rep_ptf.price ).sum() for x in new_rep_ptf.call_put ]
            varswap.rep_ptf = new_rep_ptf
            rep_ptf_price = new_rep_ptf.ptf_px.unique()[ 0 ]
            fair_strike = new_rep_ptf.K_var.unique()[0]

            rep_ptf_delta = ( new_rep_ptf.wgt * new_rep_ptf.delta ).sum()
            rep_ptf_vega = ( new_rep_ptf.wgt * new_rep_ptf.vega ).sum()
            rep_ptf_gamma = ( new_rep_ptf.wgt * new_rep_ptf.gamma ).sum()
            rep_ptf_theta = ( new_rep_ptf.wgt * new_rep_ptf.theta ).sum()
            spot_dt = market.get_spot( varswap.underlying )

        return valuer_utils.return_results_based_on_dictionary(calc_types, {
            # TODO: to add for mid-term varswap, realized + implied
            'price_var': varswap.notional * (fair_strike * fair_strike - varswap.strike_in_var),
            'price': varswap.notional * rep_ptf_price,
            'delta': varswap.notional * rep_ptf_delta,
            'vega': varswap.notional * rep_ptf_vega,
            'gamma': varswap.notional * rep_ptf_gamma,
            'theta': varswap.notional * rep_ptf_theta,
            'fair_strike': fair_strike,
            'spot': spot_dt })

    def rep_ptf(self, varswap: VarianceSwap, market: Market):
        dt = market.get_base_datetime()

        option_universe = market.get_option_universe( OPTION_ROOT_FROM_TICKER[varswap.underlying],
                                                      return_as_dict = False )
        option_universe = option_universe[[x not in [32150, 25757, 32229]
                                           for x in option_universe.stock_id.values]]
        spot = market.get_spot(varswap.underlying)

        # get data expiration
        option_universe_expiration_date = list(map(lambda x: datetime.fromisoformat(x), option_universe.expiration_date))
        sub_opt_universe = option_universe[
            [x.date() == varswap.expiration.date() for x in option_universe_expiration_date]].copy()
        if len(sub_opt_universe) == 0:
            sub_opt_universe = option_universe[ [(x.date().month == varswap.expiration.date().month) and
                ( x.date().year == varswap.expiration.date().year ) for x in option_universe_expiration_date] ].copy()
            if len(sub_opt_universe.expiration.unique()) > 1:
                sub_opt_universe['abs_dte'] = [abs((datetime.fromisoformat(x).date() - varswap.expiration.date()).days)
                                               for x in sub_opt_universe.expiration]
                sub_opt_universe = sub_opt_universe[sub_opt_universe.abs_dte == sub_opt_universe.abs_dte.min()]
                sub_opt_universe.drop(columns=['abs_dte'], inplace=True)

        if len(sub_opt_universe) == 0:
            # find first expiration after <varswap.expiration>
            sub_opt_universe = option_universe[[x.date() > varswap.expiration.date() for x in option_universe_expiration_date]]
            sub_opt_universe_expiration_date = list( map( lambda x: datetime.fromisoformat(x), sub_opt_universe.expiration_date ) )
            sub_target_exp = min( sub_opt_universe_expiration_date )
            sub_opt_universe = sub_opt_universe[[datetime.fromisoformat(x).date() == sub_target_exp.date() for x in sub_opt_universe.expiration_date]]

        if len(sub_opt_universe) == 0:
            print('Missing data for target tenor')
            return pd.DataFrame()

        if len(sub_opt_universe.expiration_date.unique()) > 1:
            sub_opt_universe = sub_opt_universe[
                sub_opt_universe.expiration_date == sub_opt_universe.expiration_date.min()]

        assert len(sub_opt_universe.expiration_date.unique()) == 1

        # drop duplicate data
        sub_opt_universe = sub_opt_universe.drop_duplicates()

        # handle missing fwd data
        if pd.isna( sub_opt_universe.forward_price.unique() )[0]:
            sub_opt_universe.forward_price = [spot for x in sub_opt_universe.strike]

        sub_opt_universe[ 'varswap_wgt' ] = [ 1 / ( x * x ) for x in sub_opt_universe.strike ]
        # get calls
        calls = sub_opt_universe[sub_opt_universe.call_put == 'C'].copy()
        calls = calls[calls.delta > self.max_delta_strike]

        # get puts
        puts = sub_opt_universe[sub_opt_universe.call_put == 'P'].copy()
        puts = puts[puts.delta < self.min_delta_strike]

        # partition by fwd
        assert len(puts.forward_price.unique()) == 1 and len(calls.forward_price.unique()) == 1
        assert puts.iloc[0].forward_price == calls.iloc[0].forward_price
        fwd = puts.iloc[0].forward_price
        fwd_K = puts[puts.strike < fwd].strike.max()
        puts = puts[puts.strike <= fwd_K]
        calls = calls[calls.strike >= fwd_K]

        # compute put mesh width
        puts.sort_values(by=['strike'], ascending=True, inplace=True)
        put_midpoints = 0.5 * (puts.strike.shift(1) + puts.strike).values
        put_midpoints[0] = put_midpoints[1] - (put_midpoints[2] - put_midpoints[1])
        put_midpoints = np.append(put_midpoints, 0.5 * (fwd_K + put_midpoints[-1]))
        puts['mesh_width'] = put_midpoints[1:] - put_midpoints[:-1]
        assert puts.mesh_width.min() > 0

        # compute call mesh width
        calls.sort_values(by=['strike'], ascending=True, inplace=True)
        call_midpoints = 0.5 * (calls.strike.shift(1) + calls.strike).values
        call_midpoints[0] = 0.5 * (fwd_K + call_midpoints[1])
        call_midpoints = np.append(call_midpoints, call_midpoints[-1] + (call_midpoints[-1] - call_midpoints[-2]))
        calls['mesh_width'] = call_midpoints[1:] - call_midpoints[:-1]
        assert calls.mesh_width.min() > 0

        rep_ptf = pd.concat([puts, calls])
        rep_ptf['wgt'] = rep_ptf.mesh_width * rep_ptf.varswap_wgt

        ## compute fair strike
        TTM = (datetime.fromisoformat(rep_ptf.expiration_date.values[0]).date() - dt.date()).days / 365.2425
        DF = fwd / spot
        K0 = rep_ptf[rep_ptf.call_put == 'P'].strike.max()
        K2_var = (2 * DF / TTM) * (rep_ptf.wgt * rep_ptf.price).sum()
        K_var = np.sqrt(K2_var - (1 / TTM) * (fwd / K0 - 1) ** 2)

        rep_ptf['K_var'] = [K_var for x in rep_ptf.call_put]
        rep_ptf['ptf_px'] = [(rep_ptf.wgt * rep_ptf.price).sum() for x in rep_ptf.call_put]
        rep_ptf.drop(columns=['varswap_wgt', 'mesh_width'], inplace=True)
        varswap.rep_ptf = rep_ptf
        return rep_ptf

    def synth_fwd(self, varswap: VarianceSwap, market: Market ):

        # get rep_ptf
        rep_ptf = varswap.rep_ptf
        assert len(rep_ptf) > 0

        if market.base_datetime == varswap.expiration:
            return varswap.synth_fwd
        else:

            # update prices
            option_universe = market.get_option_universe( OPTION_ROOT_FROM_TICKER[varswap.underlying],
                                                          return_as_dict=False )
            option_universe = option_universe[option_universe.expiration_date == rep_ptf.expiration_date.unique()[0]]
            option_universe = option_universe[[ x not in [32150, 25757, 32229]
                                                for x in option_universe.stock_id.values]]

            # get closest to money Put and Call to compute synthetic fwd
            puts = option_universe[option_universe.call_put == 'P']
            put_ATM = puts[puts.delta >= -0.5]
            put_ATM = put_ATM[put_ATM.delta == put_ATM.delta.min()]

            # handle delta = 0 case that occurs close to expiry
            if abs( put_ATM.delta.min() ) < 0.00000001:
                put_ATM['rel_strike'] =  [ abs( x / market.get_spot( varswap.underlying ) - 1 ) for x in put_ATM.strike ]
                put_ATM = put_ATM[put_ATM.rel_strike == put_ATM.rel_strike.min()]

            if len( put_ATM ) < 1:
                print('missing data on %s' %market.base_datetime )
            put_ATM = put_ATM.drop_duplicates()

            if len(put_ATM.stock_id.unique()) == 1 and \
                    len(put_ATM.strike.unique()) == 1 and \
                    len(put_ATM.call_put.unique()) == 1 and \
                    len(put_ATM.expiration.unique()) == 1  and \
                    put_ATM.forward_price.min() > 0:
                put_ATM = put_ATM[put_ATM.forward_price == put_ATM.forward_price.min()]

            assert len( put_ATM ) == 1
            calls = option_universe[ option_universe.call_put == 'C' ]
            call_ATM = calls[ calls.strike == put_ATM.strike.values[0] ]

            # handle delta = 0 case that occurs close to expiry
            if abs( call_ATM.delta.min() ) < 0.00000001:
                call_ATM['rel_strike'] =  [ abs( x / market.get_spot( varswap.underlying ) - 1 ) for x in call_ATM.strike ]
                call_ATM = call_ATM[call_ATM.rel_strike == call_ATM.rel_strike.min()]
            call_ATM = call_ATM.drop_duplicates()

            if len(call_ATM.stock_id.unique()) == 1 and \
                    len(call_ATM.strike.unique()) == 1 and \
                    len(call_ATM.call_put.unique()) == 1 and \
                    len(call_ATM.expiration.unique()) == 1  and \
                    call_ATM.forward_price.min() > 0:
                call_ATM = call_ATM[call_ATM.forward_price == call_ATM.forward_price.min()]

            assert len( call_ATM ) == 1

            synth_fwd = pd.concat([put_ATM, call_ATM])
            assert len( synth_fwd.strike.unique() ) == 1
            assert len( synth_fwd ) == 2
            varswap.synth_fwd = synth_fwd
            return synth_fwd



