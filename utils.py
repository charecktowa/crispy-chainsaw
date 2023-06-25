import math


def round_number(num):
    decimal_part = num - int(num)  # Extract the decimal part of the number
    return math.floor(num) if decimal_part <= 0.4 else math.ceil(num)
