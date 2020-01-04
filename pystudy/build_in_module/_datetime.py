# 获取当前日期和时间
from datetime import datetime, timedelta, timezone

now = datetime.now() # 获取当前datetime
print(now)  # 2019-10-29 17:49:25.340693

print(type(now))    # <class 'datetime.datetime'>

# 获取指定日期和时间
dt = datetime(2015, 4, 19, 12, 20)  # 用指定日期时间创建datetime
print(dt)   # 2015-04-19 12:20:00

# datetime转换为timestamp
dt = datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
print(dt.timestamp()) # 把datetime转换为timestamp 1429417200.0  Python的timestamp是一个浮点数。如果有小数位，小数位表示毫秒数


# timestamp转换为datetime
t = 1429417200.0
print(datetime.fromtimestamp(t))    # 2015-04-19 12:20:00

# str转换为datetime
cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday)  # 2015-06-01 18:19:59


# datetime转换为str
now = datetime.now()
print(now.strftime('%a, %b %d %H:%M'))  # Tue, Oct 29 17:55

# datetime加减
print(now + timedelta(hours=10))    # 2019-10-30 03:56:42.105026 10小时之后
print(now - timedelta(days=1))     # 2019-10-28 17:57:01.217311 一天之前
print(now + timedelta(days=2, hours=12))    # 2019-11-01 05:57:28.948672 两天12小时之后


# 本地时间转换为UTC时间
tz_utc_8 = timezone(timedelta(hours=8))  # 创建时区UTC+8:00
dt = now.replace(tzinfo=tz_utc_8)  # 强制设置为UTC+8:00
print(dt)   # 2019-10-29 17:58:45.388433+08:00

# 时区转换  拿到UTC时间，并强制设置时区为UTC+0:00:
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
print(utc_dt)   # 2019-10-29 09:59:38.128747+00:00
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))  # astimezone()将转换时区为北京时间:
print(bj_dt)    # 2019-10-29 18:00:37.233671+08:00
tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9))) # astimezone()将转换时区为东京时间:
print(tokyo_dt)     # 2019-10-29 19:01:42.655118+09:00  #  astimezone()将bj_dt转换时区为东京时间:
tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt2)    # 2019-10-29 19:02:16.562020+09:00





