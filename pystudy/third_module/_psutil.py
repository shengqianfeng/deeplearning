"""
用Python来编写脚本简化日常的运维工作
在Linux下，有许多系统命令可以让我们时刻监控系统运行的状态，如ps，top，free等等。要获取这些系统信息，
Python可以通过subprocess模块调用并获取结果。但这样做显得很麻烦，尤其是要写很多解析代码。

在Python中获取系统信息的另一个好办法是使用psutil这个第三方模块。
顾名思义，psutil = process and system utilities，它不仅可以通过一两行代码实现系统监控，
还可以跨平台使用，支持Linux／UNIX／OSX／Windows等，是系统管理员和运维小伙伴不可或缺的必备模块。
"""



# 获取cpu的信息
import psutil

print(psutil.cpu_count() )# CPU逻辑数量 12

print(psutil.cpu_count(logical=False) )# CPU物理核心 6


# 统计CPU的用户／系统／空闲时间
print(psutil.cpu_times())
# scputimes(user=119453.49999999999, system=229711.95312499814, idle=11974918.0, interrupt=29256.65625, dpc=5198.9375)


# 再实现类似top命令的CPU使用率，每秒刷新一次，累计10次
# for x in range(10):
#     print(psutil.cpu_percent(interval=1, percpu=True))


# 获取内存信息
# 使用psutil获取物理内存和交换内存信息
print(psutil.virtual_memory())  # svmem(total=17001713664, available=4784467968, percent=71.9, used=12217245696, free=4784467968)
print(psutil.swap_memory())     # sswap(total=32628056064, used=30999482368, free=1628573696, percent=95.0, sin=0, sout=0)

# 获取磁盘信息
#
# 可以通过psutil获取磁盘分区、磁盘使用率和磁盘IO信息
print(psutil.disk_partitions() )  # 磁盘分区信息

print(psutil.disk_usage('/')) # 磁盘使用情况

print(psutil.disk_io_counters())  # 磁盘IO



"""
获取网络信息

psutil可以获取网络接口和网络连接信息：
"""
# # 获取网络读写字节／包的个数
print(psutil.net_io_counters())
print( psutil.net_if_addrs() )# 获取网络接口信息
print( psutil.net_if_stats() )# 获取网络接口状态

# 要获取当前网络连接信息，使用net_connections()
print(psutil.net_connections())


"""
获取进程信息

通过psutil可以获取到所有进程的详细信息
"""
print(psutil.pids()) # 所有进程ID
p = psutil.Process(100)  # 获取指定进程ID=3776，其实就是当前Python交互环境
print(p)    # psutil.Process(pid=100, name='svchost.exe', started='2019-10-14 09:50:10')

# print( p.exe() )# 进程exe路径


"""
psutil还提供了一个test()函数，可以模拟出ps命令的效果：
"""
print(psutil.test())
