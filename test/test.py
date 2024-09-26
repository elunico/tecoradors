import typing
from tecoradors import precompute, PrecomputeStorage, tattle
from math import sin as _sin, radians, pi
import re

from tecoradors.tecoradors import PredicateType, builder, stringable, equatable, json_serializable


class IPAddressPredicate(PredicateType):
    @classmethod
    def isacceptable(self, value: str) -> bool:
        return re.match(r"^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$", value) is not None


class PortPredicate(PredicateType):
    @classmethod
    def isacceptable(self, value: int) -> bool:
        return 0 < value < 65536


@stringable
@equatable
@json_serializable
@builder
class HTTPServerOptions:
    ip: IPAddressPredicate
    port: PortPredicate
    username: str


print(HTTPServerOptions().setip("127.0.0.1").setport(8080).setusername("admin"))
