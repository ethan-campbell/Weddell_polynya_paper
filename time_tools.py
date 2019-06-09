# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import matplotlib.dates as mdates


def convert_tuple_to_datetime(tuple_datetime):
    """ Converts a tuple of arbitrary length [e.g. (Y,M,D) or (Y,M,D,H) or (Y,M,D,H,M,S)] to a Datetime object.
    """
    return datetime(*tuple_datetime)


def convert_tuple_to_8_int(tuple_date):
    """ Converts a date tuple (Y,M,D) to 8-digit integer date (e.g. 20161231).
    """
    return int('{0}{1:02}{2:02}'.format(*tuple_date))


def convert_8_int_to_tuple(int_date):
    """ Converts an 8-digit integer date (e.g. 20161231) to a date tuple (Y,M,D).
    """
    return (int(str(int_date)[0:4]), int(str(int_date)[4:6]), int(str(int_date)[6:8]))


def convert_14_to_datetime(int_date_no_delims):
    """ Converts 14-digit datetime (e.g. 20160130235959) to datetime (e.g. datetime(2016,1,30,23,59,59)).
    """
    return convert_tuple_to_datetime(convert_14_to_tuple(int_date_no_delims))


def convert_14_to_tuple(datetime_no_delims):
    """ Converts 14-digit datetime (e.g. 20160130235959) to (Y,M,D,H,M,S) tuple [e.g. (2016,01,30,23,59,59)].
    """
    dtnd_str = str(datetime_no_delims)
    return (int(dtnd_str[0:4]), int(dtnd_str[4:6]), int(dtnd_str[6:8]),
            int(dtnd_str[8:10]), int(dtnd_str[10:12]), int(dtnd_str[12:14]))


def convert_datetime_to_14(datetime_obj):
    """ Converts Datetime object to 14-digit datetime integer.
    """
    return int(datetime_obj.strftime('%Y%m%d%H%M%S'))


def convert_days_since_ref_to_datetime(days_since,year_ref,mon_ref,day_ref,subtract_days=0):
    """ Returns Datetime object representing given number of days since reference date.
        Note: set 'subtract_days' to 367 for dates given since date 0000-00-00 (old MATLAB format).
    """
    return datetime(year_ref,mon_ref,day_ref,0,0,0) + timedelta(days_since) - timedelta(subtract_days)


def convert_datetime_to_total_seconds(datetimestamp):
    return (datetimestamp - datetime(1,1,1)).total_seconds()


def datetime_to_datenum(datetime_vector):
    return mdates.datestr2num([str(dt) for dt in datetime_vector])


def convert_date_to_365(date_tuple):
    date = convert_tuple_to_datetime(date_tuple)
    start_date = datetime(date.year-1,12,31)
    return (date - start_date).days


def is_time_in_range(start_date, end_date, time_in_question):
    """ Is datetime tuple within range of date tuples? Responds 'True' or 'False'.
    """
    return convert_tuple_to_datetime(start_date) <= convert_tuple_to_datetime(time_in_question) <= convert_tuple_to_datetime(end_date)


def dates_in_range(start_date, end_date):
    """ Returns dates within two boundary dates (inclusive).
    
    Args:
        start_date and end_date: (Y,M,D), with start/end inclusive
    Returns:
        list of date tuples (y,m,d)
        
    """
    
    start_date = convert_tuple_to_datetime(start_date)
    end_date = convert_tuple_to_datetime(end_date)
    all_dates = []
        
    date_iter = start_date
    while date_iter <= end_date:
        all_dates.append((date_iter.year,date_iter.month,date_iter.day))
        date_iter += timedelta(days=1)
    
    return all_dates


def days_between(starting_date,ending_date):
    """ Returns days elapsed between date tuples 'starting_date' and 'ending_date'. Can return negative days.
    """
    return (convert_tuple_to_datetime(ending_date) - convert_tuple_to_datetime(starting_date)).days


def date_offset(orig_date, plus_x_days):
    """ Returns date tuple of orig_date offset by <<plus_x_days>> days.

    If plus_x_days < 0, then function subtracts days accordingly.

    """
    new_datetime = (convert_tuple_to_datetime(orig_date) + timedelta(days = plus_x_days))
    return (new_datetime.year,new_datetime.month,new_datetime.day)


def now(include_time=False):
    """ Returns tuple of current date (YYYY,MM,DD) or datetime (YYYY,MM,DD,HH,MM,SS) if 'include_time' is True.
    """
    if include_time:
        return (datetime.now().year,datetime.now().month,datetime.now().day,datetime.now().hour,datetime.now().minute,datetime.now().second)
    else:
        return (datetime.now().year, datetime.now().month, datetime.now().day)
