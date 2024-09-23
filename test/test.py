import typing
from tecoradors import precompute, PrecomputeStorage, tattle
from math import sin as _sin, radians, pi


# with open('./tecoradors/tecoradors.py', 'r') as f:
# exec(f.read())


@precompute([(radians(i),) for i in range(360)], storage=PrecomputeStorage.PRESERVING)
def sin(x):
    print('computing', x)
    return _sin(x)


print(sin(1))


@tattle
def fib(n):
    if n <= 0:
        return 1
    return fib(n - 1) + fib(n - 2)


print(fib(10))
