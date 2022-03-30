import time
from datetime import timedelta

start_time = time.time()
time.sleep(3)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    print(time_dif)
    print('----------')
    print(round(3.145))
    print(round(3.469))
    print(type(round(time_dif)))
    return timedelta(seconds=int(round(time_dif)))


time_dif = get_time_dif(start_time)
print(time_dif)

print(timedelta(seconds=3600))
