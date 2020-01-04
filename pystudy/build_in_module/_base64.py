# Python内置的base64可以直接进行base64的编解码
import base64

print(base64.b64encode(b'binary\x00string'))    # b'YmluYXJ5AHN0cmluZw=='

print(base64.b64decode(b'YmluYXJ5AHN0cmluZw=='))    # b'binary\x00string'


print(base64.b64encode(b'i\xb7\x1d\xfb\xef\xff'))   # b'abcd++//'

print( base64.urlsafe_b64encode(b'i\xb7\x1d\xfb\xef\xff'))  # b'abcd--__'

print(base64.urlsafe_b64decode('abcd--__')) # b'i\xb7\x1d\xfb\xef\xff'

