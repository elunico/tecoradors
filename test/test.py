import typing
from math import sin as _sin, radians

with open('./tecoradors/tecoradors.py', 'r') as f:
    exec(f.read())


@precompute([(radians(i), ) for i in range(360)], storage=PrecomputeStorage.EXCLUSIVE)
def sin(x):
    print('computing', x)
    return _sin(x)


print(sin(100))
