import datetime


def datetime_from_string(
    datetime_string: str,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
) -> datetime.datetime:
    """Converts string to datetime object."""
    return datetime.datetime.strptime(datetime_string, datetime_format)


def datetime_to_string(
    datetime_object: datetime.datetime,
    string_format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """Converts datetime object to string."""
    return datetime.datetime.strftime(datetime_object, string_format)


def get_timestamp_string() -> str:
    """Returns timestamp in `YYYYmmdd_HHMMSS_ffffff` format."""
    dt = datetime.datetime.now()
    dts = datetime.datetime.strftime(dt, "%Y%m%d_%H%M%S_%f")
    return dts
