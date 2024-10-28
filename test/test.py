import re
import typing

from tecoradors.tecoradors import enforce_annotations, Enforcer, CompositeEnforcer


class MyEnforcer(Enforcer):
    def type_check(self, argument: object, name: str, annotation: type) -> bool:
        print(argument, name, annotation)
        if annotation is int:
            return True
        return False

    def return_check(self, value: object, annotation: type) -> bool:
        return True


class MyOtherEnforcer(Enforcer):
    def type_check(self, argument: object, name: str, annotation: type) -> bool:
        if annotation == str and name == "a":
            return True
        return False

    def return_check(self, value: object, annotation: type) -> bool:
        return False


@enforce_annotations(CompositeEnforcer([MyEnforcer(), MyOtherEnforcer()]))
def add(a: int, b: int) -> int:
    return a + b


"""
TODO: I would like to change the EnforceAnnotations class to work differently
I would like the decorator to be a function always
There should be the default behavior of @enforce_annotations but you should be also
able to customize and @enforce_annotations(Enforcer()) and pass an enforcer class
that has the type_check and return_check methods on it that the enforce annotations
fucnction can use as the class does

The only reason I want to do this is because @decorating a fn with a cls breaks mypy
whereas, for some reason, if the decorator is also a function, mypy can still type check
the parameters and the whole point of the enforce_annotations decorator replacing
accepts and returns decorators is so that you can have static type checking always and
runtime type checking as well
"""


print(add("hello", " world"))
