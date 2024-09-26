
import re

from tecoradors.tecoradors import json_serializable, builder, stringable, hashable, final, PredicateType


class IPAddress(PredicateType):
    @classmethod
    def isacceptable(self, value: str) -> bool:
        return re.match(r"^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$", value) is not None


class INETPort(PredicateType):
    @classmethod
    def isacceptable(self, value: int) -> bool:
        return 0 < value < 65536


class UsernameString(PredicateType):
    @classmethod
    def isacceptable(self, value: str) -> bool:
        return re.match(r"^[a-zA-Z0-9_-]{3,20}$", value) is not None


@final
@stringable
@hashable
@json_serializable
@builder
class HTTPServerOptions:
    ip: IPAddress
    port: INETPort
    username: UsernameString


local_server = HTTPServerOptions()\
    .setip("127.0.0.1")\
    .setport(8080)\
    .setusername("admin")

remote_server = HTTPServerOptions()\
    .setip("192.168.1.1")\
    .setport(80)\
    .setusername("admin")

print(local_server)
print(remote_server)
print(local_server == remote_server)
