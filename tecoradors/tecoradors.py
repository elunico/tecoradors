import enum
import functools
import inspect
import sys
import typing
import os


def chained(obj, *fields):
    for field in fields:
        obj = getattr(obj, field)
    return obj


def apply(fns, *arguments, **kwarguments):
    return tuple(fn(*arguments, **kwarguments) for fn in fns)


def _get_class_that_defined_method(meth: typing.Callable) -> typing.Optional[type]:
    # taken from https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3
    if isinstance(meth, functools.partial):
        return _get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in meth.__self__.__class__.__mro__:
            if meth.__name__ in cls.__dict__:
                return cls
        # fallback to __qualname__ parsing
        meth = getattr(meth, "__func__", meth)
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls
    # handle special descriptor objects
    return getattr(meth, "__objclass__", None)


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


def _is_special_annot(annotation: object) -> bool:
    return annotation in {typing.Optional, typing.Union, typing.Any}


def _special_annot_is_ok(value, annotation) -> bool:
    if annotation == typing.Optional:
        legal = typing.get_args(annotation)
        if value is None or isinstance(value, legal):
            return True
        else:
            return False
    if annotation == typing.Union:
        legals = typing.get_args(annotation)
        if not any(isinstance(value, i) for i in legals):
            return False
        else:
            return True
    if annotation == typing.Any:
        return True
    raise TypeError(f"{annotation} is not a special type")


def _isinstance_special(value, annotation):
    if (a := typing.get_origin(annotation)) is not None:
        annotation = a
    return (
        _is_special_annot(annotation) and _special_annot_is_ok(value, annotation)
    ) or (not _is_special_annot(annotation) and isinstance(value, annotation))


def _enforce_impl_wrapper(hook1, hook2, fn):
    def _enforce_impl(*args, **kwargs):
        if os.environ.get("TECORADORS_OVERRIDE_NO_ENFORCE_ANNOTATION", False):
            return fn(*args, **kwargs)

        NO_RET_TYPE = object()
        """
        This function depends on __annotations__ having the types of arguments in the same order as their listing in the function signature
        TODO: it might be a good idea to migrate to something like inspect.getfullargspec(fn) to look up types for positional args
        using a dictionary lookup
        """
        annots = fn.__annotations__

        # check keyword arguments by name
        # kwargs passed to the function come in as a dictionary whose names can be looked up
        # directly in the annotations dictionary
        for ka, val in kwargs:
            # defer to type_check method but if it refuses to handle it do the default
            if not hook1(val, ka, annots[ka]):
                if not _isinstance_special(val, annots[ka]):
                    raise TypeError(
                        f"expected **kwargs argument {repr(ka)} to have type {annots[ka]} but got type {type(val)}"
                    )

        # prevent return type from being zipped up with args types
        if "return" in annots:
            return_type: type = annots["return"]
            del annots["return"]
        else:
            # I know, I know, but just trust me
            return_type: type = typing.cast(type, NO_RET_TYPE)

        # arguments and annotations from signature
        for idx, (arg, (name, typ)) in enumerate(zip(args, annots.items())):
            # first defer to user defined type_check function
            if not hook1(arg, name, typ):
                if not _isinstance_special(arg, typ):
                    raise TypeError(
                        f"expected argument {repr(name)} at pos {idx} to have type {typ} but found type {type(arg)}"
                    )

        # execute the function so we can check return type
        result = fn(*args, **kwargs)

        if return_type is not NO_RET_TYPE and not hook2(result, return_type):
            if return_type is not NO_RET_TYPE and not _isinstance_special(
                result, return_type
            ):
                raise TypeError(
                    f"expected function {repr(fn)} to return {return_type} but got {type(result)}"
                )
        return result

    return _enforce_impl


class EnforceAnnotations:
    """
    Customizable class for enforcing type annotations in a function
    Using the @EnforceAnnotations class by itself gives normal type checking behavior for parameters and annotations
    If you want more customizable type checking you can override the type_check and return_check methods in a subclass
    and then use your subclass as a decorator

    The methods can choose to handle any situation based on value, name, or annotation type. See the methods for more info

    Using this runtime type checking can incur a *significant* runtime cost. This can be overriden on all callables
    decorated with this class using the TECORADORS_OVERRIDE_NO_ENFORCE_ANNOTATION environment variable. Setting this
    variable to any value will cause the __call__ method to just return a call to the function and skip all type checking.
    This allows for expensive runtime enforcement in dev situations but also allows for efficiency in prod
    """

    def __init__(self, fn):
        """
        Initializer for function decorator. fn is the function whose type annotations will be
        checked at runtime
        """
        self.fn = fn
        functools.wraps(fn)(self)

    def __call__(self, *args, **kwargs):
        """
        Invoked in place of self.fn to perform the runtime checking of type annotations of self.fn
        """
        return _enforce_impl_wrapper(self.type_check, self.return_check, self.fn)(
            *args, **kwargs
        )

    def type_check(self, argument: object, name: str, annotation: type) -> bool:
        """
        This method is called *before* the default type checking of the
        @enforce_annotations function.

        Based on the parameters passed to the function, it should decide if it is willing to
        do the type_checking. If it does it should return True. If it does not want
        to handle the type checking, it should return False to get the default behavior

        The default behavior is an isinstance check with value and annotation, except for the
        special cases, typing.Optional, typing.Union, and typing.Any which are special cased
        to behave intuitively based on their name. Note that type arguments are *not*
        checked by default so list[str] will check for list but not for any particular type
        inside the list. To do this see below

        An example for this would be to specially deal with list[str] or tuple[int] or
        any other generic types which normal isinstance checks cannot deal with. In this
        example you might handle the type checking for list, tuple, etc. but choose to default
        to normal type checks for everything else.

        If a type check fails this method *must* raise a TypeError

        @returns True if it handles the check or False if not

        @raises TypeError if the type_check fails
        """
        return False

    def return_check(self, value: object, annotation: type) -> bool:
        """
        This method is called *before* the default type checking of the return value

        Based on the parameters passed to the function, it should decide if it is willing to
        do the type checking for the return type.
        If it does it should return True. If it does not want
        to handle the type checking, it should return False to get the default behavior

        The default behavior is an isinstance check with value and annotation, except for the
        special cases, typing.Optional, typing.Union, and typing.Any which are special cased
        to behave intuitively based on their name. Note that type arguments are *not*
        checked by default so list[str] will check for list but not for any particular type
        inside the list. To do this see below

        An example for this would be to specially deal with list[str] or tuple[int] or
        any other generic types which normal isinstance checks cannot deal with. In this
        example you might handle the type checking for list, tuple, etc. but choose to default
        to normal type checks for everything else.

        If a type check fails this method *must* raise a TypeError

        @returns True if it handles the check or False if not

        @raises TypeError if the type_check fails
        """
        return False


def enforce_annotations(fn):
    """
    Convenience function wrapper for the EnforceAnnotations class-based function decorator
    More information can be found there.
    This function implements the default behavior of the class version. See the class
    documentation for default behavior.

    The advantage to the function form is the return value is an actual function
    rather than replacing the function with a class instance with a __call__ method
    """
    if os.environ.get("TECORADORS_OVERRIDE_NO_ENFORCE_ANNOTATION", False):
        return fn

    def false(*args):
        return False

    NO_RET_TYPE = object()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _enforce_impl_wrapper(false, false, fn)(*args, **kwargs)

    return wrapper


def deprecated(
    reason: str,
    replacement: str,
    starting_version: typing.Optional[str] = None,
    removed_version: typing.Optional[str] = None,
):
    """
    Marks a method or function as deprecated. Will print a DeprecationWarning
    with reason and the given replacement. Replacement should be whatever
    can be used to replace the deprecated method or function
    """
    import warnings

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"function {fn.__name__!r} is deprecated: {reason!r}. Use {replacement!r} instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

        setattr(wrapper, "deprecation_reason", reason)
        setattr(wrapper, "deprecation_starting_in", starting_version)
        setattr(wrapper, "deprecation_remove_in", removed_version)
        return wrapper

    return decorator


@deprecated(
    reason="This decorator is buggy and hard to maintain",
    replacement="@enforce_annotations and add annotations to the function parameters",
    starting_version="6.3.0",
)
def accepts(*types: type | tuple[type, ...]):
    """
    Provides a declaration-site and run-time check of the types of the arguments passed to a function or method

    Pass 1 instance of a type per argument or, if multiple types are acceptable, pass a tuple of types for the argument.
    Ultimately, these types or tuples of types will be used in isinstance() checks. So subclasses will be accepted

    Note that when annotating methods (instance or class), DO NOT PASS A TYPE FOR 'self' OR 'cls' parameters.
    The parameters 'self' and 'cls' are NEVER CHECKED by this decorator if they appear as the first
    parameter in a method.

    NOTE: you must place the @accepts() decorator closer to the function than the @returns()


    See Also: Self, @returns()

    :param types: a splat of types or tuples of types to be matched 1 to 1 against the types of the args to the function
    :return: a decorator which wraps a function and does a run time type check on all arguments against the types
             given EXCEPT for 'self' and 'cls' first args

    """

    import enum

    def check_self_or_cls(var_names):
        # determine if the first arg is self or cls (we will not check this param)
        return len(var_names) > 0 and (var_names[0] == "self" or var_names[0] == "cls")

    def check_accepts(f):
        """
        accepts a function and retruns a new function that has runtime type checks on all its parameters
        Args:
            f: a function for type checking

        Returns: a new wrapped function with run time type checking
        """
        if hasattr(f, "_knows_returns") and f._knows_returns:
            raise TypeError(
                "You must decorate a function with accepts() before returns()"
            )

        vnames = f.__code__.co_varnames
        is_bound = check_self_or_cls(vnames)
        argcount = f.__code__.co_argcount - (
            0 if not is_bound else 1
        )  # remove self or cls if present
        assert (
            len(types) == argcount
        ), f"Incorrect number of types for function, expected {argcount} but got {len(types)}. "

        def _check_raw_type(pos: int, a: typing.Any, t: tuple[type, ...]) -> None:
            """
            Determines if argument a is of type t. If t is iterable checks if a is of any of the types in t.
            If t is Self or contains Self, there is a check to get the class that defined the method and determine
            if that class is the type of a

            Raises AssertionError if a is not of type t or of one of the types in t or Self holds

            Args:
                a: argument to check
                t: type to check a is

            Returns: None

            """
            # can pass in many types that are valid for a, but if Enum is passed in this breaks, so only iterate t
            # if it not an enum otherwise treat it as a single class
            if not isinstance(t, enum.EnumMeta):
                # if Self is given, check a against Self using _get_class_that_defined_method otherwise just check a
                ds = [_get_class_that_defined_method(f) if i is Self else i for i in t]
                if not all(i is not None for i in ds):
                    raise TypeError(
                        f"Self is meaningless on non-bound method {f.__name__}"
                    )
            else:
                # if t is a single type check for Self and check for Self otherwise just check a against t
                ds = [_get_class_that_defined_method(f) if t is Self else t]
                if any(i is None for i in ds):
                    raise TypeError(
                        f"Self is meaningless on non-bound method {f.__name__}"
                    )
            t = tuple(typing.cast(tuple[type], ds))
            assert isinstance(a, t), (
                f"{f.__name__}: got value {repr(a)} (type of {type(a)}) for argument {pos} "
                + f"but expected argument of type(s) {t}"
            )

        def _check_callable(
            a: typing.Any, predicate: typing.Callable[[type], bool]
        ) -> None:
            """
            Instead of checking a directly against a type, t, this function can check a against a predicate.
            t is a predicate which takes the argument a and determines if it is valid. t should raise an AssertionError
            if it is invalid
            Args:
                a: argument to check
                t: callable predicate to check a with

            Returns: None

            """
            try:
                assert predicate(
                    a
                ), f"function received {a} which did pass the type check {predicate}"
            except AssertionError:
                raise
            except Exception as e:
                raise AssertionError(
                    f"Function could not validate parameter {a} with function {predicate}"
                ) from e

        @functools.wraps(f)
        def new_f(*args, **kwds):
            if os.environ.get("TECORADORS_OVERRIDE_NO_TYPING", False):
                return f(*args, **kwds)
            for_args = args[1:] if is_bound else args
            for pos, a, t in zip(range(1, len(args) + 1), for_args, types):
                if not _isiterable(t):
                    t = (t,)
                if inspect.isfunction(t):
                    _check_callable(a, t)
                else:
                    _check_raw_type(pos, a, t)
            return f(*args, **kwds)

        return new_f

    return check_accepts


@deprecated(
    reason="This decorator is buggy and hard to maintain",
    replacement="@enforce_annotations and add annotations to the function return type",
    starting_version="6.3.0",
)
def returns(*types: type):
    """
    Run time assertions for the return type of a function. Use this decorator by annotating a function with @returns()
    and the types that you expect the function to return. Some important notes about syntax:
        1. If you provide 1, single type. There should be exactly 1 return type from the function. This can be any
        type except for tuples. You may return one int, str, object, list, dict, etc.
        2. You may provide a tuple of types as an argument to the function. This is like a Union type and allows many
        values to type check successfully for a single return value.
        3. You can pass several types to the *types splat (individually as varargs, not in a tuple). This has the effect
        of type checking multiple return types. This is the reason, a simple tuple cannot be the only return type.
        Tuple returns are type checked for all the elements in the tuple.

        Note the subtle distinction between passing a tuple and passing varargs to the decorator.

    Like with @accepts(), a method can have a @returns(Self) if it returns an instance of the class it is defined on

    In any case, type check failure will occur at function call time with an AssertionError.

    NOTE: you must place the @accepts() decorator closer to the function than the @returns()

    See Also:
        @accepts, Self

    Args:
        *types: a splat of types or tuples of types that specify which types are acceptable return values for the
        function

    Returns:
        a decorator to implement the runtime type-checking

    """

    def decorator(fn):
        # noinspection PyTypeHints
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if os.environ.get("TECORADORS-OVERRIDE-NO-TYPING", False):
                return fn(*args, **kwargs)
            nonlocal types
            result = fn(*args, **kwargs)
            ds = [
                (_get_class_that_defined_method(fn) if i is Self else i) for i in types
            ]

            assert all(
                i is not None for i in ds
            ), "Cannot have return type of Self on non-method object"
            types = typing.cast(tuple[type, ...], ds)
            if isinstance(result, tuple):
                if len(result) != len(types):
                    raise AssertionError(
                        "Expected {} values returned but got {}".format(
                            len(types), len(result)
                        )
                    )
                for value, cls in zip(result, types):
                    if _isiterable(cls) and not isinstance(cls, enum.EnumMeta):
                        cls = tuple(
                            j
                            for j in (
                                _get_class_that_defined_method(fn) if i is Self else i
                                for i in typing.cast(typing.Iterable[type], cls)
                            )
                            if j is not None
                        )
                    if not isinstance(cls, enum.EnumMeta):
                        assert isinstance(value, cls), (
                            "Return type expected {} but "
                            "received {} of type {}".format(
                                cls, repr(value), type(value)
                            )
                        )
            else:
                t = types
                if _isiterable(t) and not isinstance(t, enum.EnumMeta):
                    t: tuple[type, ...] = tuple(
                        j
                        for j in (
                            _get_class_that_defined_method(fn) if i is Self else i
                            for i in t
                        )
                        if j is not None
                    )
                assert isinstance(
                    result, t
                ), "Return type expected {} but " "received {} of type {}".format(
                    t, repr(result), type(result)
                )
            return result

        # used to check for the correct order of accepts and returns.
        setattr(wrapper, "_knows_returns", True)
        return wrapper

    return decorator


def interruptable(fn: typing.Union[typing.Callable, str]):
    """
    a decorator that can be used to allow a function to seemlessly accept
    a KeyboardInterrupt. When a function is decorated with only
    @interruptable, then it will return None on KeyboardInterrupt but
    not propogate the exception

    If the function is decorated with @interruptable(string) then it will
    print string and return None on KeyboardInterrupt and not propogate
    the exception
    """
    if callable(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except KeyboardInterrupt as e:
                return None

        return wrapper

    else:
        if not isinstance(fn, str):
            # not unreachable - python does not enforce typing
            raise TypeError(
                "@interruptable must be passed a function or a string as its argument"
            )

        def inner(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt as e:
                    print(fn, file=sys.stderr)
                    return None

            return wrapper

        return inner


def json_serializable(cls: typing.Type):
    """
    Adds a 'to_json' method to instances the class that it annotates. This method
    uses json.dumps and a JSONEncoder to create a JSON string from the
    object's __dict__ attribute

    Note that this decorator will return the original cls value passed into it. You will get the same object out that
    gets put in, though, obviously, it will be mutated.

    >>> @json_serializable
    ... class User:
    ...     def __init__(self, name, views, liked_ids):
    ...         self.name = name
    ...         self.views = views
    ...         self.liked_ids = liked_ids
    >>> user1 = User("Alice", 2010, [13, 27, 201, 333])
    >>> user1.to_json()
    '{"name": "Alice", "views": 2010, "liked_ids": [13, 27, 201, 333]}'
    """

    import json

    class MyEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__

    def to_json(self) -> str:
        return json.dumps(self.__dict__, cls=MyEncoder)

    setattr(cls, "to_json", to_json)
    return cls


def spread(times: int):
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


class PredicateType:
    """
    A PredicateType is subclassed as a means to provide custom type checking. A PredicateType is used for value checking
    and not instance checking. A TypeError will be raised if the value does not satisfy the predicate.

    Subclasses of PredicateType are used for attribute type checking in classes decorated with @builder.
    When an annotation is a subclass of PredicateType, the builder will use the PredicateType.check method to check
    if the value satisfies the predicate. Otherwise, instance checking will be used, which means that the value must be of
    an instance of the type specified by the annotation.

    To use this class you must subclass and override the isacceptable method to provide the custom type checking logic.
    The isacceptable method should return True if the value satisfies the predicate and False otherwise.
    The isacceptable method should be a class method.
    """

    @classmethod
    def check(cls, attr: str, value: typing.Any):
        if not cls.isacceptable(value):
            raise TypeError(
                f"Excepted attribute '{attr}' to satisfy predicate {cls} but got {value}"
            )

    @classmethod
    def isacceptable(cls, value: typing.Any) -> bool:
        raise NotImplementedError("Subclass must implement abstract method")


def builder(typechecking: bool | typing.Callable = True) -> typing.Any:
    """
    The builder decorator allows you to create builder classes based on the desired attributes and optionally types.

    You can define a class with attributes and types and decorate with the builder decorator and you will get
        -  An __init__ that takes no arguments and initializes all fields to their given values or None if no value is provided
        -  A setX method for every attribute defined in the class which sets the corresponding field and returns self

    If you decorate a class @builder you get type checking by default. This is equivalent to @builder(True) and if you
    want to turn off type checking you can use @builder(False) or @builder(typechecking=False)

    An example of how to use it

    >>> @builder
    >>> class HTTPServerOptions:
    >>>     ip: str
    >>>     port: int
    >>>     username: str = 'admin'

    This would generate the equivalent of

    >>> class HTTPServerOptions:
    >>>     def __init__(self):
    >>>         self.ip = None
    >>>         self.port = None
    >>>         self.username = 'admin'
    >>>
    >>>     def setip(self, value):
    >>>         if not isinstance(value, str):
    >>>             raise TypeError("Excepted attribute ip of type {} but got type {}".format(str, type(value)))
    >>>         self.ip = value
    >>>         return self
    >>>
    >>>     def setport(self, value):
    >>>         if not isinstance(value, int):
    >>>             raise TypeError("Excepted attribute port of type {} but got type {}".format(int, type(value)))
    >>>         self.port = value
    >>>         return self
    >>>
    >>>     def setusername(self, value):
    >>>         if not isinstance(value, str):
    >>>             raise TypeError("Excepted attribute username of type {} but got type {}".format(str, type(value)))
    >>>         self.username = value
    >>>         return self

    You may also annotate the attributes with a subclass of PredicateType  to customize the type checking.
    For example, if you want to make sure that the ip address is a valid ip address you can do the following:

    >>> class IPAddressPredicate(PredicateType):
    >>>     @classmethod
    >>>     def isacceptable(self, value: str) -> bool:
    >>>         return re.match(r"^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}$", value) is not None

    >>> @builder
    >>> class HTTPServerOptions:
    >>>     ip: IPAddressPredicate
    >>>     port: int
    >>>     username: str = 'admin'

    Only subclasses of PredicateType can be used to do this predicate-based checking. Any other type will be
    used for isinistance checking.

    *Predicate checking is **NOT** done for the initial value on the class only in the setter*
    """

    def builder_interior(cls):
        attributes = cls.__annotations__

        def cls_init(self):
            for attr, typename in attributes.items():
                setattr(self, attr, getattr(cls, attr, None))

        def setter_maker(attr):
            def setter(self, value):
                if typechecking:
                    # cannot type check dict[str] only dict, etc.
                    # very poor solution for Callable, but ¯\_(ツ)_/¯
                    if typing.get_args(attributes[attr]):
                        attr_type = typing.get_origin(attributes[attr])
                    else:
                        attr_type = attributes[attr]
                    if isinstance(attr_type, str):
                        if not isinstance(value, eval(attr_type)):
                            raise TypeError(
                                "Excepted attribute {} of type {} but got type {}".format(
                                    attr, typename, type(value)
                                )
                            )
                    elif isinstance(attr_type, type) and issubclass(
                        attr_type, PredicateType
                    ):
                        attr_type.check(attr, value)
                    elif not isinstance(value, attr_type):
                        raise TypeError(
                            "Excepted attribute {} of type {} but got type {}".format(
                                attr, typename, type(value)
                            )
                        )
                setattr(self, attr, value)
                return self

            return setter

        for attr, typename in attributes.items():
            setattr(cls, "set{}".format(attr), setter_maker(attr))

        setattr(cls, "__init__", cls_init)
        return cls

    if callable(typechecking):
        return builder_interior(typechecking)
    else:
        return builder_interior


@builder
class TattleOptions:
    """
    Determines what happens when a function is decorated with @tattle. See there for more. See also @builder for
    how to construct a TattleOptions

    Each option can either be None or a Callable.

    The onenter property is called before calling the function and it is passed a *args **kwargs splat equivalent
    to what will be passed into the function when it is called. This is never used if onexit is None

    The onexit property is called after the function returns and it is passed (result, *args, **kwargs) that is the
    result of the function call and the same args, kwargs splat that went into the function. This is never used if
    onexit is None *or* if an exception occurs before the function returns

    The onexception property is only called if an exception is raised. The callable is passed (exception, *args, **kwargs)
    where exception is the instance of the caught exception and *args, and **kwargs are the splat arguments passed to
    the function when it was called. This is never used if onexception is None or if the function returns normally
    without raising an Exception.
    """

    onenter: typing.Callable
    onexit: typing.Callable
    onexception: typing.Callable


def tattle(options: typing.Union[TattleOptions, typing.Callable]):
    """
    Function decorator that allows the reporting of events. A function decorated with @tattle
    can have the entrance, exit, and exceptions of the function reported. The `options` parameter is a TattleOptions
    instance that can is given callables for `onenter`, `onexit`, and `onexception`. These callables are called
    with arguments at the appropriate time. These options can also be None and no action on that event will be taken
    """
    if callable(options):

        @functools.wraps(options)
        def wrapper(*args, **kwargs):
            print("Call start: {}({}, {})".format(options.__name__, args, kwargs))
            try:
                result = options(*args, **kwargs)
            except Exception as e:
                print(
                    "Call exception [{}]: {}({}, {})".format(
                        e, options.__name__, args, kwargs
                    )
                )
            else:
                print(
                    "Call finished: {}({}, {})".format(options.__name__, args, kwargs)
                )
                return result

        return wrapper

    def interior(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if options.onenter is not None:
                options.onenter(*args, **kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exception:
                if options.onexception is not None:
                    options.onexception(exception, *args, **kwargs)
                else:
                    raise
            else:
                if options.onexit is not None:
                    options.onexit(result, *args, **kwargs)
                return result

        return wrapper

    return interior


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


def squash(
    exceptions: typing.Union[TupleOfExceptionTypes, AnyFunction] = (Exception,),
    on_squashed: typing.Union[typing.Optional[typing.Any], ExceptionCallback] = None,
):
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

    >>> @squash
    ... def atoi(a):
    ...     return int(a)
    >>> atoi('5013')
    5013
    >>> atoi('hello') is None
    True

    >>> @squash(on_squashed=0)
    ... def atoi(a):
    ...     return int(a)
    >>> atoi('37')
    37
    >>> atoi('4al')
    0
    """
    import types

    if type(exceptions) is types.FunctionType:
        return squash()(exceptions)

    def decor(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                if _isiterable(exceptions):
                    squashed = tuple(typing.cast(TupleOfExceptionTypes, exceptions))
                else:
                    squashed = (exceptions,)
                if not isinstance(e, squashed):
                    raise
                else:
                    return (
                        on_squashed(e)
                        if hasattr(on_squashed, "__call__")
                        else on_squashed
                    )

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
        items = [
            "{}={}".format(k, repr(v))
            for (k, v) in self.__dict__.items()
            if not k.startswith("__") and not k.endswith("__")
        ]
        items_string = ", ".join(items)
        return "{}[{}]".format(self.__class__.__name__, items_string)

    setattr(cls, "__str__", __str__)
    setattr(cls, "__repr__", __str__)

    return cls


def _values_of(self: typing.Any) -> list[typing.Any]:
    if hasattr(self, "__dict__"):
        return list(self.__dict__.values())
    else:
        return [getattr(self, attr) for attr in self.__slots__]


def equatable(cls):
    """
    Adds an __eq__ method that compares all the values in an object's __dict__ to all the values in another instance
    of that object's dict. Note keys are NOT checked, however types are. Note that if the class has a __slots__ attribute,
    then the values in the __slots__ are compared.
    Note that subclasses are necessarily accepted
    because of how the decorators work
    """

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        pairs = zip(_values_of(self), _values_of(other))
        return all([i[0] == i[1] for i in pairs])

    setattr(cls, "__eq__", __eq__)
    return cls


def hashable(cls):
    """
    Implicitly calls 'equatable' for the class and also generates a __hash__
    method for the class so it can be used in a dictionary

    Note that this decorator will return a proxy class type for the cls class passed into it. Type
    checks must assume @hashable classes are descedants of the class they decorated

    >>> @hashable
    ... class Person:
    ...     def __init__(self, name, age):
    ...         self.name = name
    ...         self.age = age
    >>> alice = Person("Alice", 23)
    >>> bob = Person("Bob", 25)
    >>> bob2 = Person("Bob", 25)
    >>> other_bob = Person("Bob", 20)
    >>> third_bob = Person("Bobbert", 25)
    >>> (hash(alice) != hash(bob))
    True
    >>> (hash(alice) == hash(bob))
    False
    >>> (hash(bob) == hash(bob2))
    True
    >>> (hash(bob) != hash(other_bob))
    True
    >>> hash(other_bob) != hash(third_bob)
    True
    >>> hash(third_bob) != hash(bob)
    True
    """

    cls = equatable(cls)

    @squash((TypeError, AttributeError), on_squashed=0)
    def super_hasher(cls, self):
        return super(cls, self).__hash__()

    def hasher(a, i):
        return ((a << 8) | (a >> 56)) ^ hash(i)

    def __hash__(self):
        for name, value in self.__dict__.items():
            if not hasattr(type(value), "__hash__") or type(value).__hash__ is None:
                fmt_str = "value of type {} can't be hashed because the field {}={} (type={}) is not hashable"
                str_format = fmt_str.format(
                    repr(cls.__name__),
                    repr(name),
                    repr(value),
                    repr(type(value).__name__),
                )
                raise TypeError(str_format)
        return super_hasher(cls, self) ^ functools.reduce(hasher, _values_of(self), 0)

    setattr(cls, "__hash__", __hash__)
    return cls


def orderable(cls):
    """
    Modifies a class to be orderable. Provides a __eq__ and __hash__ method using @hashable.
    Requires an implementation of one of the 6 magic comparison methods.
    Uses @functools.total_ordering to accomplish the ordering.

    See the descriptions of @hashable and @functools.total_ordering for the
    caveats around using this decorator
    """
    cls = hashable(cls)
    return functools.total_ordering(cls)


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

    setattr(cls, "__init_subclass__", error)
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
            if hasattr(self, "__gu__") and self.__gu__:
                raise FrozenClassError(
                    "Class {!r} is frozen and cannot be "
                    "changed after instantiation".format(cls.__name__)
                )
            else:
                getattr(super(cls, self), method_name)(*args, **kwargs)

        return error

    cls.__setattr__ = create_frozen_method("__setattr__")
    cls.__delattr__ = create_frozen_method("__delattr__")
    cls.__setitem__ = create_frozen_method("__setitem__")

    # preserve cls's init but wrap it so that we can hook in and set a property indicated initialization happened
    old_init = getattr(cls, "__init__")

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.__gu__ = True

    setattr(cls, "__init__", new_init)

    return cls


def log(destination: typing.IO, include_results: bool = False):
    """
    Logs calls to a function to a particular IO Stream (can be StringIO or a file). Always logs the function name and
    the arguments and kwargs to a function. If an exception is raised when calling the function, the exception is
    also logged. If the function completes successfully AND `include_results` is True, then the result of the
    function call will also be logged.

    Args:
        destination: IO destination for log information
        include_results: whether or not to include the result of the call in addition to the function name and
        arguments

    Returns: a log decorator

    """
    import sys

    if callable(destination) and include_results is False:
        return log(sys.stdout)(destination)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            destination.write(
                f"Call to {fn.__name__!r} with arguments: args={args!r}, kwargs={kwargs!r}\n"
            )
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                destination.write(f"Exception raised during function call: {e!r}")
                raise
            else:
                if include_results:
                    destination.write(f"|-- result={result!r}")

            destination.write("\n")
            return result

        return wrapper

    return decorator


def synchronized(lock):
    """
    Synchronizes a function call access on the given lock object. `lock` must have an `acquire()` and a `release()`
    method. Function call cannot proceed until the lock is acquired. Deadlocks are possible be careful
    Args:
        lock: the lock upon which to synchronize function call access

    Returns: a decorator for wrapping a function

    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            lock.acquire()
            try:
                return fn(*args, **kwargs)
            finally:
                lock.release()

        return wrapper

    return decorator


def count_calls(with_args: bool = False, with_kwargs: bool = False):
    """
    Count the number of times a function is called, optionally keep track of how many times it is called with a specific
    set of arguments or key-word arguments or both. Information can be retrieved using the `call_count()` method on the
    function object which was decorated with this function. This decorator also attaches a `reset_call_count()` method
    to the function object which resets the call count to 0. args and kwargs can be passed to reset_call_count()
    to reset count for that combination. Note that you cannot pass an arg or kwarg to `reset_call_count()`
    unless you also passed True for with_args or with_kwargs, respectively. Calling with no arguments will ALWAYS reset
    ALL counts

    Args:
        with_args: keep track of number of calls separately by argument
        with_kwargs: keep track of number of calls separately by key-word argument

    Returns:
        a decorator to track call counts

    Raises:
        ValueError if `reset_call_count()` is passed args or kwargs, but with_args or with_kwargs, respectively,
        are False
    """
    if callable(with_args) and with_kwargs is False:
        return count_calls()(with_args)

    from collections import Counter

    def decorator(fn):
        def key(*args, **kwargs):
            return (
                (args if with_args else (args, str(kwargs)))
                if with_args or with_kwargs
                else fn
            )

        counts = Counter()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            counts[key(*args, **kwargs)] += 1
            return result

        if not with_args:

            def call_count1(*args, **kwargs):
                check_args(*args, **kwargs)
                return sum(counts.values())

            setattr(wrapper, "call_count", call_count1)
        else:

            def call_count(*args, **kwargs):
                check_args(*args, **kwargs)
                if not args and not kwargs:
                    return counts
                return counts[key(*args, **kwargs)]

            setattr(wrapper, "call_count", call_count)

        def reset_count(*args, **kwargs):
            nonlocal counts
            check_args(*args, **kwargs)
            if not args and not kwargs:
                counts = Counter()
            else:
                counts[key(*args, **kwargs)] = 0

        def check_args(*args, **kwargs):
            if args and not with_args:
                raise ValueError(
                    "args passed to a count_calls() function, but with_args was not passed to "
                    "count_calls()"
                )
            if kwargs and not with_kwargs:
                raise ValueError(
                    "kwargs passed to a count_calls() function, but with_kwargs was not passed to "
                    "count_calls()"
                )

        setattr(wrapper, "reset_call_count", reset_count)
        return wrapper

    return decorator


T = typing.TypeVar("T")
R = typing.TypeVar("R")


class Descriptor(typing.Protocol):
    def __get__(
        this,
        self: typing.Optional[R],
        owner: typing.Optional[typing.Any] = None,
        *args,
        **kwargs,
    ) -> typing.Union[object, typing.Self]:
        ...


def lazy(method: typing.Callable[[typing.Any], T]) -> Descriptor:
    """
    Similar to the `@property` decorator, lazy allows you to access the value computed by a method
    call as if it were a simple attribute on an instance. The main difference between `@property`
    and `@lazy` is that `@property` performs the computation on every attribute access, where as
    `@lazy` performs the calculation once on the first access, and then overwrites the attribute
    field with the value of the property upon its first computation. Subsequent accesses of the
    property will only return the initial value that computed on the first access.

    In this way, it creates a lazy attribute, storing the code to initialize the attribute
    in a function, and only evaluating the function when accessed. It also elimited the overhead of
    other similar solutions like `@functools.cache` by re-writing the attribute after first access,
    removing the descriptor and obviating the need for an `if arg in cache` check
    """

    class descriptor(Descriptor):
        def __get__(
            self,
            receiver: typing.Optional[R],
            owner: typing.Optional[typing.Any] = None,
            *args,
            **kwargs,
        ) -> typing.Union[T, typing.Self]:
            if receiver is None:
                return self
            value = method(receiver, *args, **kwargs)
            setattr(receiver, method.__name__, value)
            return value

    return descriptor()


class PrecomputeStorage(enum.Enum):
    EXCLUSIVE = enum.auto()
    LIMITED = enum.auto()
    PRESERVING = enum.auto()


class NoSuchValue(ValueError):
    pass


def precompute(
    argument_tuples: typing.Iterable[tuple],
    storage: PrecomputeStorage = PrecomputeStorage.PRESERVING,
    max: typing.Optional[int] = None,
):
    """
    Function decorator used to declaratively state precomputed values for a function

    On defintion of a function decorated with precompute, the decorated function will be called once for each item
    in argument_tuples. On each call, that item of argument_tuples is splatted into the function.

    The decorator can also storage method as the storage parameter. This indicates how the decorated function
    should behave regarding its values
        EXCLUSIVE indicates the decorated function should only allow the precomputed values to be retrieved and raise an Exception otherwise
        LIMITED indicates the decorated function will return pre-computed values and will perform computation for *every* call for a non precomputed value
        PRESERVING indicates the decorated function will return pre-computed values where application, perform computations for non-precomputed values, and then store those results so no computation is done more than once, like @functools.lru_cache

    The max argument works the same as maxsize for @functools.lru_cache and only makes sense when storage == PRESERVING
    """
    if storage != PrecomputeStorage.PRESERVING and max is not None:
        raise ValueError("storage must be PRESERVING or max must be None")
    if storage == PrecomputeStorage.PRESERVING:

        def wrapper1(fn):
            @functools.lru_cache(maxsize=max)
            def decorator(*args):
                return fn(*args)

            for args in argument_tuples:
                decorator(*args)  # cached by functools
            return decorator

        return wrapper1
    elif storage == PrecomputeStorage.LIMITED:

        def wrapper(fn):
            cache = {}
            for args in argument_tuples:
                cache[args] = fn(*args)

            def decorator(*args):
                if args in cache:
                    return cache[args]
                else:
                    return fn(*args)

            return decorator

        return wrapper
    else:

        def wrapper(fn):
            cache = {}
            for args in argument_tuples:
                cache[args] = fn(*args)

            def decorator(*args):
                if args not in cache:
                    raise NoSuchValue(
                        "{} was not precomputed for the given function".format(args)
                    )
                return cache[args]

            return decorator

        return wrapper
