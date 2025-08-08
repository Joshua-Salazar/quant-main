from datetime import date


class HolidayRequirement:
    """
    Holiday Requirement for code, start date and end date.
    Only support requirement date rather datetime here.
    """
    def __init__(self, code: set, start_date: date, end_date: date):
        self.code = code
        self.start_date = start_date
        self.end_date = end_date

    def __eq__(self, other):
        if not isinstance(other, HolidayRequirement):
            return False
        if self.code != other.code:
            return False
        if self.start_date != other.start_date:
            return False
        return self.end_date == other.end_date

    def __hash__(self):
        return hash((self.code, self.start_date, self.end_date))
