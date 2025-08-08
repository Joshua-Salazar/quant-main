from enum import Enum

class MarketItemType(Enum):
    BORROWCURVE = "BorrowCurve"
    DISCOUNTCURVE = "DiscountCurve"
    INTERESTRATE = "InterestRate"
    FORWARDRATECURVE = "ForwardRateCurve"
    SPOT = "Spot"
    DIVIDEND = "Dividend"
    CORPACTION = "CorpAction"
    FXSPOT = "FXSpot"
    FXFORWARD = "FXForward"
    FXFORWARDPOINT = "FXForwardPoint"
    FUTURE = "Future"   # future contract with specific expiry date
    VOLATILITY = "Volatility"
    FXVOLATILITY = "FXVolatility"
    CRVOLATILITY = "CRVolatility"
    SWAPTIONVOLATILITY = "SwaptionVolatility"
    OPTIONDATACONTAINER = "OptionDataContainer"
    FUTUREDATACONTAINER = "FutureDataContainer"
    FUTUREINTRADAYDATACONTAINER = "FutureIntradayDataContainer"
    SPOTRATECURVE = "SpotRateCurve"
    PORTFOLIO = 'Portfolio'
    XCCYBASIS = "XCcyBasis"
    FIXING = "Fixing"
    FIXINGTABLE = "FixingTable"
    BONDYIELD = "BondYield"
    HOLIDAYCENTER = "HolidayCenter"
    CORRELATIONMATRIX = "CorrelationMatrix"
