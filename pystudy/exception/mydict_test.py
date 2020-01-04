import unittest


from .mydict import Dict


"""
1 编写单元测试时，我们需要编写一个测试类，从unittest.TestCase继承
2 以test开头的方法就是测试方法，不以test开头的方法不被认为是测试方法，测试的时候不会被执行
3 对每一类测试都需要编写一个test_xxx()方法。由于unittest.TestCase提供了很多内置的条件判断，我们只需要调用这些方法就可以断言输出是否是我们所期望的。
    最常用的断言就是assertEqual()


"""


class TestDict(unittest.TestCase):

    def test_init(self):
        d = Dict(a=1, b='test')
        self.assertEqual(d.a, 1)    # 断言函数返回的结果与1相等
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyerror(self):
        d = Dict()
        with self.assertRaises(KeyError):   # 期待抛出指定类型的Error
            value = d['empty']

    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')
"""
运行单元测试的方法之一
    这样就可以把mydict_test.py当做正常的python脚本运行 
        python mydict_test.py
    
另一种方法是在命令行通过参数-m unittest直接运行单元测试
    python -m unittest mydict_test
    
这是推荐的做法，因为这样可以一次批量运行很多单元测试，并且，有很多工具可以自动来运行这些单元测试
"""
if __name__ == '__main__':
    unittest.main()


"""
setUp与tearDown
可以在单元测试中编写两个特殊的setUp()和tearDown()方法。
这两个方法会分别在每调用一个测试方法的前后分别被执行
"""




