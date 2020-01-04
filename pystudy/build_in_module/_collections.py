from collections import namedtuple

"""
namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。
这样一来，我们用namedtuple可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便
"""
# namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x)  # 1

# 可以验证创建的Point对象是tuple的一种子类：
print(isinstance(p, Point))     # True
print(isinstance(p, tuple))     # True

# 类似的，如果要用坐标和半径表示一个圆，也可以用namedtuple定义
# namedtuple('名称', [属性list]):
Circle = namedtuple('Circle', ['x', 'y', 'r'])

# deque
#
# 使用list存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，因为list是线性存储，数据量大的时候，插入和删除效率很低。
#
# deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：deque除了实现list的append()和pop()外，还支持appendleft()和popleft()，这样就可以非常高效地往头部添加或删除元素
from collections import deque

q = deque(['a', 'b', 'c'])
q.append('x')
q.appendleft('y')
print(q)    # deque(['y', 'a', 'b', 'c', 'x'])



"""
defaultdict

使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict
除了在Key不存在时返回默认值，defaultdict的其他行为跟dict是完全一样的。
"""
from collections import defaultdict

dd = defaultdict(lambda: 'N/A')

dd['key1'] = 'abc'
print(dd['key1'])   # key1存在
print(dd['key2'])   # key2不存在，返回默认值N/A


"""
OrderedDict

使用dict时，Key是无序的。在对dict做迭代时，我们无法确定Key的顺序。

如果要保持Key的顺序，可以用OrderedDict
"""
from collections import OrderedDict

d = dict([('a', 1), ('b', 2), ('c', 3)])
print(d)  # dict的Key是无序的
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])

print(od)  # # OrderedDict的Key是有序的
"""
OrderedDict的Key会按照插入的顺序排列，不是Key本身排序：
"""

od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
print(list(od.keys()))  # 按照插入的Key的顺序返回 ['z', 'y', 'x']

"""
OrderedDict可以实现一个FIFO（先进先出）的dict，当容量超出限制时，先删除最早添加的Key
"""


class LastUpdatedOrderedDict(OrderedDict):
    def __init__(self, capacity):
        super(LastUpdatedOrderedDict, self).__init__()
        self._capacity = capacity

    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        # 不存在且容量已满则剔除最初的（存在的话不可能进去）
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print('remove:', last)
        # 存在则更新
        if containsKey:
            del self[key]
            print('set:', (key, value))
        else:
            print('add:', (key, value))
        OrderedDict.__setitem__(self, key, value)


lru = LastUpdatedOrderedDict(5)
lru.__setitem__('jeff1',1)
lru.__setitem__('jeff2',2)
lru.__setitem__('jeff3',3)
lru.__setitem__('jeff4',4)
lru.__setitem__('jeff5',5)
print(lru)
lru.__setitem__('jeff6',6)
print(lru)


"""
ChainMap

ChainMap可以把一组dict串起来并组成一个逻辑上的dict。ChainMap本身也是一个dict，但是查找的时候，会按照顺序在内部的dict依次查找。

场景：
    应用程序往往都需要传入参数，参数可以通过命令行传入，可以通过环境变量传入，还可以有默认参数。
我们可以用ChainMap实现参数的优先级查找，即先查命令行参数，如果没有传入，再查环境变量，如果没有，就使用默认参数
                命令行参数---->环境变量---->默认参数
"""
from collections import ChainMap
import os, argparse

# 构造缺省参数:
defaults = {
    'color': 'red',
    'user': 'guest'
}

# 构造命令行参数:
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = {k: v for k, v in vars(namespace).items() if v }

# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)

# 打印参数:
print('color=%s' % combined['color'])   # color=red
print('user=%s' % combined['user'])     # user=guest


"""
Counter

Counter是一个简单的计数器，例如，统计字符出现的个数：
"""

from collections import Counter

c = Counter()
for ch in 'programming':
    c[ch] = c[ch] + 1

print(c)    # Counter({'r': 2, 'g': 2, 'm': 2, 'p': 1, 'o': 1, 'a': 1, 'i': 1, 'n': 1})
# Counter实际上也是dict的一个子类，上面的结果可以看出，字符'g'、'm'、'r'各出现了两次，其他字符各出现了一次。
