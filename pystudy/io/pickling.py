"""
序列化picking
反序列化unpicking
"""
# 尝试把一个对象序列化并写入文件
import pickle
d = dict(name='Bob', age=20, score=88)
print(pickle.dumps(d))
# pickle.dumps()方法把任意对象序列化成一个bytes，然后，就可以把这个bytes写入文件。或者用另一个方法pickle.dump()直接把对象序列化后写入一个file-like Object?
f = open('D://dump.txt', 'wb')
pickle.dump(d, f)
f.close()


"""
反序列化
    当我们要把对象从磁盘读到内存时，可以先把内容读到一个bytes，然后用pickle.loads()方法反序列化出对象，也可以直接用pickle.load()方法从一个file-like Object中直接反序列化出对象
"""
f = open('D://dump.txt', 'rb')
d = pickle.load(f)
f.close()
print(d)    # {'name': 'Bob', 'age': 20, 'score': 88}


"""
如何把Python对象变成一个JSON
    dumps()方法返回一个str，内容就是标准的JSON。
    类似的，dump()方法可以直接把JSON写入一个file-like Object

    
要把JSON反序列化为Python对象，用loads()或者对应的load()方法，前者把JSON的字符串反序列化，后者从file-like Object中读取字符串并反序列化
"""
import json

d = dict(name='Bob', age=20, score=88)
print(json.dumps(d))    # {"name": "Bob", "age": 20, "score": 88}

json_str = '{"age": 20, "score": 88, "name": "Bob"}'
print(json.loads(json_str))


"""
对象的序列化
"""
class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }


# 默认情况下，dumps()方法不知道如何将Student实例变为一个JSON的{}对象
s = Student('Bob', 20, 88)
# 可选参数default就是把任意一个对象变成一个可序列为JSON的对象，我们只需要为Student专门写一个转换函数，再把函数传进去即可
print(json.dumps(s, default=student2dict))  # {"name": "Bob", "age": 20, "score": 88}

"""
下次如果遇到一个Teacher类的实例，照样无法序列化为JSON。我们可以偷个懒，把任意class的实例变为dict
因为通常class的实例都有一个__dict__属性，它就是一个dict，用来存储实例变量
"""
print(json.dumps(s, default=lambda obj: obj.__dict__))  # {"name": "Bob", "age": 20, "score": 88}



"""
对象反序列化
    同样的道理，如果我们要把JSON反序列化为一个Student对象实例，loads()方法首先转换出一个dict对象，然后，我们传入的object_hook函数负责把dict转换为Student实例
"""
def dict2student(d):
    return Student(d['name'], d['age'], d['score'])


json_str = '{"age": 20, "score": 88, "name": "Bob"}'
print(json.loads(json_str, object_hook=dict2student))   # <__main__.Student object at 0x000001C946D77E48>



