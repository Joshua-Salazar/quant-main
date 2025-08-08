from enum import Enum


class Ccy(Enum):
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    CNY = "CNY"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    USD = "USD"
    HKD = "HKD"
    KRW = "KRW"
    MXN = "MXN"
    NOK = "NOK"
    NZD = "NZD"
    SEK = "SEK"
    SGD = 'SGD'
    TWD = 'TWD'
    CNH = 'CNH'

    @classmethod
    def contains(cls, value):
        return value in cls._value2member_map_