import sys

sys.path.insert(1, "tecoradors")
from tecoradors import exc_to_bool, resultify, EXC_SELF


@resultify
def readall(filename):
    with open(filename) as f:
        return f.read()


print(readall("test/test.py"))

print(readall("test/noexist.txt"))


@exc_to_bool(exception_yields=EXC_SELF)
def funct(b):
    if b:
        raise ValueError("An error occurred")
    else:
        return -3.14


print(repr(funct(False)))

print(repr(funct(True)))
