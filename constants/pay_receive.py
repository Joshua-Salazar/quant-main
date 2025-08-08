from enum import Enum


class PayReceive(Enum):
    PAY = "PAY"
    RECEIVE = "RECEIVE"

    def inverse(self):
        return PayReceive.RECEIVE if self == PayReceive.PAY else PayReceive.PAY
