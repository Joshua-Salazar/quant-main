class HolidayCenter:
    def __init__(self, holidays: dict):
        self.holidays = holidays

    def get_holidays(self, code, start_date, end_date):
        hols = list(filter(lambda dt: start_date <= dt.date() <= end_date, self.holidays[code]))
        return hols

    def clone(self):
        return HolidayCenter(self.holidays)
