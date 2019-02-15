import datetime

def now():
    tdiff = datetime.timedelta(hours=9)
    now = datetime.datetime.now() + tdiff
    time = str(now).split()[1].split('.')[0]
    return time

