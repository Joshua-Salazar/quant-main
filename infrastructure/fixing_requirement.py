from datetime import date


class FixingRequirement:
    """
    Fixings Requirement for underlying, start date and end date.
    Only support daily fixing requirement so we use date rather datetime here.
    """
    def __init__(self, underlying: str, start_date: date, end_date: date):
        self.underlying = underlying
        self.start_date = start_date
        self.end_date = end_date
