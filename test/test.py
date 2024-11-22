import re
import typing

from tecoradors.tecoradors import enforce_annotations, Enforcer, CompositeEnforcer


@enforce_annotations
def add(a: int, b: int) -> int:
    return a + b


@enforce_annotations
def mul(a: int, b: int) -> int:
    return a * b


@enforce_annotations
def div(a: int, b: int) -> float:
    return a / b


print(add(119, add(26, 74)))
