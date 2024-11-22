# Tecoradors

Python decorators that I like to use a lot.

[Find the github repo here](https://github.com/elunico/tecoradors)

[Find the pypi package here](https://pypi.org/project/tecoradors-elunico/)

Named based on my name which starts with a T. You can find more information reading the docstrings of the functions.

**Information on how to use these decorators can be found in their docstring**
Using `help(tecorador)` is a good way to learn about them. You can also read their docs on Github

### Decorators are

- Enforcer ***(new in 7.0.0!)**: subclassable and customizable annotation type checking enforcer: pass instance to @enforce_annotations decorator to customize its behavior
- CompositeEnforcer ***(new in 7.0.0!)**: Allows multiple Enforcer objects to work together in sequence*
- ~~EnforceAnnotations~~ ***(Removed in 7.0.0)**: use pluggable Enforcer objects and @enforce_annotations function decorator*
- enforce_annotations ***(new in 6.3.0!)**: Enforces type annotations on arguments and return types at runtime. Replaces @accepts and @returns. Customizable with custom Enforcer sublcasses
- deprecated
- accepts
- returns
- interruptable
- json_serializable
- spread
- builder
- tattle
- timed
- squash
- stringable
- equatable
- hashable
- orderable
- dataclass
- final
- freeze
- log
- synchronized
- count_calls
- lazy
- precompute
- resultify
- exc_to_bool

### Support from types

- Self
- PredicateType
- TattleOptions
- FrozenClassError
- PrecomputeStorage
- NoSuchValue
