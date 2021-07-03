import functools
import inspect
import typing


# taken from https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3
def _get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return _get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
            inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__,
                                                                                                '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


class Self(type):
    """
    This class stands in for the type of the class being defined. Since decorators require evaluation
    before the defintion end of a class, you cannot use a class's type in the @accepts for one of its
    methods. Instead, you should use the type Self. This will be replaced by the class type at runtime.
    This does not make sense for unbound methods (free functions) or static methods and will raise an
    error if passed to such a function or method.

    NOTE: PASS THE CLASS Self TO THE ACCEPTS DECORATOR

    Example:

        class Vector2D:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

            @accepts(Self)
            def plus(self, other):
                self.x += other.x
                self.y += other.y
                self.z += other.z

         class State:
            registry = {}

            @classmethod
            @accepts((Self, str))
            def get(cls, s):
                if isinstance(s, State):
                    return s
                else:
                    if s not in registry:
                        registry[s] = State(s)
                    return registry[s]

            @accepts(str)
            def __init__(self, name):
                self.name = name
    """

    def __new__(mcs, *args):
        raise TypeError("Do not construct Self(), use the class Self instead")


def _isiterable(t):
    try:
        iter(t)
        return True
    except TypeError:
        return False


def accepts(*types: typing.Union[type, typing.Tuple[type]]):
    """
    Provides a declaration-site and run-time check of the types of the arguments passed to a function or method

    Pass 1 instance of a type per argument or, if multiple types are acceptable, pass a tuple of types for the argument.
    Ultimately, these types or tuples of types will be used in isinstance() checks. So subclasses will be accepted

    Note that when annotating methods (instance or class), DO NOT PASS A TYPE FOR 'self' OR 'cls' parameters.
    The parameters 'self' and 'cls' are NEVER CHECKED by this decorator if they appear as the first
    parameter in a method.

    See Also: Self

    :param types: a splat of types or tuples of types to be matched 1 to 1 against the types of the args to the function
    :return: a decorator which wraps a function and does a run time type check on all arguments against the types
             given EXCEPT for 'self' and 'cls' first args

    """
    import enum

    def check_accepts(f):
        vnames = f.__code__.co_varnames
        is_bound = check_self_or_cls(vnames)
        argcount = f.__code__.co_argcount - (0 if not is_bound else 1)
        assert len(types) == argcount, f"Not enough types for arg count, expected {argcount} but got {len(types)}"

        @functools.wraps(f)
        def new_f(*args, **kwds):
            for_args = args[1:] if is_bound else args
            for (a, t) in zip(for_args, types):
                if inspect.isfunction(t):
                    _check_callable(a, t)
                else:
                    _check_raw_type(a, t)
            return f(*args, **kwds)

        def _check_raw_type(a, t):
            if _isiterable(t) and not isinstance(t, enum.EnumMeta):
                t = tuple([_get_class_that_defined_method(f) if i is Self else i for i in t])
                assert all(i is not None for i in t), f"Cannot accept Self on non-bound method {f.__name__}"
            else:
                t = _get_class_that_defined_method(f) if t is Self else t
                assert t is not None, f"Cannot accept Self on non-bound method {f.__name__}"
            assert isinstance(a, t), f"{f.__name__}: got argument {a} (type of {type(a)}) " + \
                                     f"but expected argument of type(s) {t}"

        def _check_callable(a, t):
            try:
                assert t(a), f'function received {a} which did pass the type check {t}'
            except AssertionError:
                raise
            except Exception as e:
                raise AssertionError(f"Function could not validate parameter {a} with function {t}") from e

        return new_f

    def check_self_or_cls(var_names):
        return len(var_names) > 0 and (var_names[0] == 'self' or var_names[0] == 'cls')

    return check_accepts


def json_serializable(cls):
    """
    Adds a 'to_json' method to instances the class that it annotates. This method
    uses json.dumps and a JSONEncoder to create a JSON string from the
    object's __dict__ attribute

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.
    """
    import json

    class MyEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__

    def to_json(self) -> str:
        return json.dumps(self.__dict__, cls=MyEncoder)

    setattr(cls, 'to_json', to_json)
    return cls


def spread(times):
    """
    creates a new function that takes the same arguments as the original function
    and runs the original function `times` times storing the result in a list
    each time and then returning the list of all the results back to the caller

    :param times: the number of times the decorated function should be run
    """

    def inner(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                results.append(fn(*args, **kwargs))
            return results

        return wrapper

    return inner


def timed(fn):
    """
    Wraps a function using time.time() in order to time the execution of the function

    Returns the a tuple of original function result and the time elapsed in
    seconds
    """
    import time

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        return result, end - start

    return wrapper


TupleOfExceptionTypes = typing.Tuple[typing.Type[BaseException]]
AnyFunction = typing.Callable[[typing.Any], typing.Any]
ExceptionCallback = typing.Callable[[BaseException], typing.Any]


def squash(exceptions: typing.Union[TupleOfExceptionTypes, AnyFunction] = (Exception,),
           on_squashed: typing.Union[typing.Optional[typing.Any], ExceptionCallback] = None):
    """
    returns a function that handles exceptions differently.

    if squash is used without calling it, then the decorated function that is
    returned does the following: if the new function raises an
    exception of type Exception or its derivatives, it is ignored and None is returned

    Using squash wihtout parameters looks like this
    @squash
    def some_function(arg1, arg2):
        ...

    and is equivalent to writing

    @squash((Exception,), on_squashed=None)
    def some_function(arg1, arg2):
        ...

    This will cause all Exceptions that inherit from Exception to be ignored
    and for the function to return None rather than raise the exception.
    Note that any Exceptions that are raised that DO NOT inherit Exception
    (such as those derived from BaseException) will be re-raised and not squashed
    in the function

    If squash is used with parameters, then:

    If an exception is raised in the function that is not of a type listed in
    the `exceptions` parameter, then the exception is raised normally

    If an exception is raised in the function that IS of a type listed in
    the `exceptions` parameter, then..

        If `on_squashed` is a value (including None) then that is returned
        If `on_squashed` is callable (has a __call__ method) then the
          result of calling `on_squashed` with the instance of the
          raised exception is returned
    """
    import types

    if type(exceptions) is types.FunctionType:
        @functools.wraps(exceptions)
        def wrapper(*args, **kwargs):
            try:
                return exceptions(*args, **kwargs)
            except Exception:
                return None

        return wrapper
    else:
        def decor(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except BaseException as e:
                    if _isiterable(exceptions):
                        squashed = tuple(exceptions)
                    else:
                        squashed = exceptions,
                    if not isinstance(e, squashed):
                        raise
                    else:
                        return on_squashed(e) if hasattr(on_squashed, '__call__') else on_squashed

            return wrapper

        return decor


# decorator function
def stringable(cls):
    """
    Adds a __str__ and __repr__ method to a class' instances that creates a human-readable JSON-style string
    out of the object's __dict__ attribute and includes the class name at the beginning

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.
    """

    def __str__(self):
        items = ['{}={}'.format(k, repr(v)) for (k, v) in self.__dict__.items()]
        items_string = ', '.join(items)
        return '{}[{}]'.format(self.__class__.__name__, items_string)

    setattr(cls, '__str__', __str__)
    setattr(cls, '__repr__', __str__)

    return cls


def equatable(cls):
    """
    Adds an __eq__ method that compares all the values in an object's __dict__ to all the values in another instance
    of that object's dict. Note keys are NOT checked, however types are. Only *identical* types (not subclasses)
    are accepted.

    NOTE: this method will also set __hash__ to None in accordance with the general Python langauge contract around
    implementing __eq__. See the hashable decorator if you need hashing

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.

    See Also: hashable
    """

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        pairs = zip(self.__dict__.values(), other.__dict__.values())
        return all([i[0] == i[1] for i in pairs])

    setattr(cls, '__eq__', __eq__)
    setattr(cls, '__hash__', None)
    return cls


def hashable(cls):
    """
    Implicitly calls 'equatable' for the class and also generates a __hash__
    method for the class so it can be used in a dictionary

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.
    """
    cls = equatable(cls)

    def hasher(a, i):
        return ((a << 8) | (a >> 56)) ^ hash(i)

    def __hash__(self):
        for (name, value) in self.__dict__.items():
            if type(value).__hash__ is None:
                fmt_str = "value of type {} can't be hashed because the field {}={} (type={}) is not hashable"
                str_format = fmt_str.format(repr(cls.__name__), repr(name), repr(value), repr(type(value).__name__))
                raise TypeError(str_format)
        return super(cls, self).__hash__() ^ functools.reduce(hasher, self.__dict__.values(), 0)

    setattr(cls, '__hash__', __hash__)
    return cls


# decorator function
def dataclass(cls, **kwargs):
    """
    Wraps the built-in dataclass.dataclass annotation in order to also give the
    class a __str__ method. All options provided in kwargs are forwarded to the
    dataclasses.dataclass decorator. After that the stringable decorator is used
    to give the class a __str__ method

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.

    See Also: dataclasses.dataclass
    """
    import dataclasses

    cls = dataclasses.dataclass(cls, **kwargs)
    cls = stringable(cls)
    return cls


def final(cls):
    """
    Prevents classes from being subclassed, by raising an Exception in __init_subclass__

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.
    """

    def error(*args, **kwargs):
        raise TypeError("Cannot inherit from final class {}".format(repr(cls.__name__)))

    setattr(cls, '__init_subclass__', error)
    return cls


class FrozenClassError(Exception):
    """
    Raised by the @frozen decorator if a class attribute is set or del'd after class initialization
    """
    pass


def freeze(cls):
    """
    Makes a class frozen. All instances of Frozen classes are allowed to set attributes exactly once during __init__.
    After that, attempting to mutate any attribute in the class will raise a TypeError. Additionally, the
    __setitem__ method will raise a FrozenClassError if any attempt to call it is made outside of __init__.

    It is recommended to combine this decorator with the @final decorator as well since subclasses can change the
    behavior of the __setattr__ and __setitem__ methods. This could lead to a situation where some object isinstance
    of a frozen class, but is not frozen

    Finally it also changes __delattr__ so attributes cannot be unset

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.

    Args:
        cls: Class object to freeze

    Returns: Class object that is now frozen. No subclassing or wrapping takes place

    """

    def create_frozen_method(method_name):
        """
        Returns a function that will raise an error when called on a class that is already initialized, otherwise
        it will delegate to the appropriate super method defined by method_name
        """
        def error(self, *args, **kwargs):
            if hasattr(self, '__gu__') and self.__gu__:
                raise FrozenClassError("Class {!r} is frozen and cannot be "
                                       "changed after instantiation".format(cls.__name__))
            else:
                getattr(super(cls, self), method_name)(*args, **kwargs)

        return error

    cls.__setattr__ = create_frozen_method('__setattr__')
    cls.__delattr__ = create_frozen_method('__delattr__')
    cls.__setitem__ = create_frozen_method('__setitem__')

    # preserve cls's init but wrap it so that we can hook in and set a property indicated initialization happened
    old_init = getattr(cls, '__init__')

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.__gu__ = True

    setattr(cls, '__init__', new_init)

    return cls


def deprecated(reason: str,
               replacement: str,
               starting_version: typing.Optional[str] = None,
               removed_version: typing.Optional[str] = None):
    """
    Marks a method or function as deprecated. Will print a DeprecationWarning
    with reason and the given replacement. Replacement should be whatever
    can be used to replace the deprecated method or function
    """
    import warnings

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(f'function {fn.__name__!r} is deprecated: {reason!r}. Use {replacement!r} instead.',
                          category=DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        setattr(wrapper, 'deprecation_reason', reason)
        setattr(wrapper, 'deprecation_starting_in', starting_version)
        setattr(wrapper, 'deprecation_remove_in', removed_version)
        return wrapper

    return decorator
