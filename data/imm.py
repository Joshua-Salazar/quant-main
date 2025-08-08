from datetime import datetime, timedelta


def find_next_imm_date(dt):
    month = dt.month
    year = dt.year

    if month in [1, 2, 3]:
        imm_month = 3
    elif month in [4, 5, 6]:
        imm_month = 6
    elif month in [7, 8, 9]:
        imm_month = 9
    else:
        imm_month = 12

    next_imm = datetime(year, imm_month, 15)
    while next_imm.weekday() != 2:
        next_imm += timedelta(days=1)

    if next_imm < dt:
        if imm_month + 3 <= 12:
            next_imm = datetime(year, imm_month + 3, 15)
        else:
            next_imm = datetime(year + 1, imm_month + 3 - 12, 15)
        while next_imm.weekday() != 2:
            next_imm += timedelta(days=1)

    return next_imm
