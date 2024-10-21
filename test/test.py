import re
import typing

from tecoradors.tecoradors import EnforceAnnotations


@EnforceAnnotations
def binify(i: int, *args: float) -> str:
    """Sample docstring"""
    return bin(i)


print(binify(128, 20.0, 3.20))

print(dir(binify))
print(binify.__name__)
print(binify.__doc__)
print(binify.__annotations__)
