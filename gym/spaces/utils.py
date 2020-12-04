from collections import OrderedDict
from functools import singledispatch
from typing import Union

import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import (Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space,
                        Tuple)

@singledispatch
def flatdim(space: Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces`` and doesn't have a registered
    function.
    """
    raise NotImplementedError


@flatdim.register(Box)
def flatdim_box(space: Box) -> int:
    return int(np.prod(space.shape))


@flatdim.register(Discrete)
def flatdim_discrete(space: Discrete) -> int:
    return int(space.n)


@flatdim.register(Tuple)
def flatdim_tuple(space: Tuple) -> int:
    return int(sum([flatdim(s) for s in space.spaces]))


@flatdim.register(Dict)
def flatdim_dict(space: Dict) -> int:
    return int(sum([flatdim(s) for s in space.spaces.values()]))


@flatdim.register(MultiBinary)
def flatdim_multi_binary(space: MultiBinary) -> int:
    return int(space.n)


@flatdim.register(MultiDiscrete)
def flatdim_multi_discrete(space: MultiDiscrete) -> int:
    return int(np.prod(space.shape))


@flatdim.register(Space)
def flatdim_custom(space: Space) -> int:
    raise CustomSpaceError(
        f"No registered handler for space {space} of type `{type(space)}`. "
        f"This only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `flatdims.register({type(space).__name__})`."
    )


@singledispatch
def flatten(space: Space, x) -> np.ndarray:
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises a ``CustomSpaceError`` if the space is not defined in
    ``gym.spaces`` and if there isn't a function registered for that type.
    """
    raise CustomSpaceError(
        f"No registered handler for space {space} of type `{type(space)}`. "
        f"This only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `flatten.register({type(space).__name__})`."
    )


@flatten.register(Box)
def _flatten_box(space: Box, x) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).flatten()

   
@flatten.register(Discrete)
def _flatten_discrete(space: Discrete, x) -> np.ndarray:
    onehot = np.zeros(space.n, dtype=space.dtype)
    onehot[x] = 1.0
    return onehot


@flatten.register(Tuple)
def _flatten_tuple(space: Tuple, x) -> np.ndarray:
    return np.concatenate([
        flatten(s, x_part) for s, x_part in zip(space.spaces, x)
    ])


@flatten.register(Dict)
def _flatten_dict(space: Dict, x: dict) -> np.ndarray:
    return np.concatenate([
        flatten(s, x[key]) for key, s in space.spaces.items()
    ])


@flatten.register(MultiBinary)
@flatten.register(MultiDiscrete)
def _flatten_multi_binary(space: Union[MultiBinary, MultiDiscrete], x) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).flatten()


@singledispatch
def unflatten(space: Space, x: np.ndarray):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space.
    Raises a ``CustomSpaceError`` if the space is not defined in
    ``gym.spaces`` and if there isn't a function registered for that type.
    """
    raise CustomSpaceError(
        f"No registered handler for space {space} of type `{type(space)}`. "
        f"This only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `unflatten.register({type(space).__name__})`."
    )


@unflatten.register(Box)
def _unflatten_box(space: Box, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: np.ndarray) -> int:
    return int(np.nonzero(x)[0][0])


@unflatten.register(Tuple)
def _unflatten_tuple(space: Tuple, x: np.ndarray) -> tuple:
    dims = [flatdim(s) for s in space.spaces]
    list_flattened = np.split(x, np.cumsum(dims)[:-1])
    list_unflattened = [
        unflatten(s, flattened)
        for flattened, s in zip(list_flattened, space.spaces)
    ]
    return tuple(list_unflattened)


@unflatten.register(Dict)
def _unflatten_dict(space: Dict, x: np.ndarray) -> OrderedDict:
    dims = [flatdim(s) for s in space.spaces.values()]
    list_flattened = np.split(x, np.cumsum(dims)[:-1])
    list_unflattened = [
        (key, unflatten(s, flattened))
        for flattened, (key,
                        s) in zip(list_flattened, space.spaces.items())
    ]
    return OrderedDict(list_unflattened)


@unflatten.register(MultiBinary)
@unflatten.register(MultiDiscrete)
def _unflatten_multi(space: Union[MultiBinary, MultiDiscrete],
                    x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@singledispatch
def flatten_space(space: Space) -> Box:
    """Flatten a space into a single ``Box``.

    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Raises a ``CustomSpaceError`` if the space is not defined in
    ``gym.spaces`` and if there isn't a function registered for that type.

    Example::

        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that flattens a discrete space::

        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that recursively flattens a dict::

        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    raise CustomSpaceError(
        f"No registered handler for space {space} of type `{type(space)}`. "
        f"This only supports default Gym spaces (e.g. `Box`, `Tuple`, "
        f"`Dict`, etc...) out-of-the-box. To use custom spaces, register a "
        f"function to use for spaces of type {type(space)} by decorating it "
        f" with `flatten_space.register({type(space).__name__})`."
    )


@flatten_space.register(Box)
def _flatten_box_space(space: Box) -> Box:
    return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)


@flatten_space.register(Discrete)
def _flatten_discrete_space(space: Discrete) -> Box:
    return Box(low=0, high=1, shape=(space.n, ), dtype=space.dtype)


@flatten_space.register(Tuple)
def _flatten_tuple_space(space: Tuple) -> Box:
    spaces = [flatten_space(s) for s in space.spaces]
    return Box(
        low=np.concatenate([s.low for s in spaces]),
        high=np.concatenate([s.high for s in spaces]),
        dtype=np.result_type(*[s.dtype for s in spaces]),
    )


@flatten_space.register(Dict)
def _flatten_dict_space(space: Dict) -> Box:
    spaces = [flatten_space(s) for s in space.spaces.values()]
    return Box(
        low=np.concatenate([s.low for s in spaces]),
        high=np.concatenate([s.high for s in spaces]),
        dtype=np.result_type(*[s.dtype for s in spaces]),
    )

@flatten_space.register(MultiBinary)
def _flatten_multi_binary_space(space: MultiBinary) -> Box:
    return Box(low=0, high=1, shape=(space.n,), dtype=space.dtype)


@flatten_space.register(MultiDiscrete)
def _flatten_multi_discrete_space(space: MultiDiscrete) -> Box:
    return Box(
        low=np.zeros_like(space.nvec),
        high=space.nvec,
        dtype=space.dtype,
    )
