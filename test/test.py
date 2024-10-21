import re
import typing

from tecoradors.tecoradors import enforce_annotations


@enforce_annotations
def binify(i: int, *args: float) -> str:
    return bin(i)


print(binify(128, 20.0, 3.20))
