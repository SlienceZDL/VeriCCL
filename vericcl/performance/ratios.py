"""Lossless JSON representation for exact times with very large integers."""


def _checked_pair(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int for item in value) or value[1] <= 0):
        raise ValueError('exact ratio requires two integers and a positive denominator')
    return tuple(value)


def ratio_to_json(value):
    if value is None:
        return None
    numerator, denominator = _checked_pair(value)
    # 2048 bits fit below Python's minimum configurable decimal limit (640
    # digits). The encoding does not depend on process-global JSON settings.
    if max(numerator.bit_length(), denominator.bit_length()) <= 2048:
        return [numerator, denominator]
    return {'encoding': 'hex_integer_ratio_v1',
            'numerator': hex(numerator), 'denominator': hex(denominator)}


def ratio_from_json(value):
    if isinstance(value, (list, tuple)):
        return _checked_pair(value)
    if (not isinstance(value, dict) or set(value) != {'encoding', 'numerator', 'denominator'}
            or value['encoding'] != 'hex_integer_ratio_v1'):
        raise ValueError('unknown exact ratio encoding')
    result = []
    for key in ('numerator', 'denominator'):
        text = value[key]
        if not isinstance(text, str):
            raise ValueError('exact hexadecimal ratio components must be strings')
        number = int(text, 16)
        if hex(number) != text:
            raise ValueError('exact ratio requires canonical hexadecimal integers')
        result.append(number)
    return _checked_pair(result)
