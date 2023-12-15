from functools import partial
from typing import ClassVar, Callable

import jax
import jax.numpy as jnp
from pydantic import BaseModel

A = 100


# @partial(jax.jit, static_argnums=(1,))s
def func(x, f):
    y = x + 5 + A
    # print(x)
    # print(y)
    # print(f)
    return f(y)


@jax.jit
def func2(z):
    c = 7
    f = lambda y: y**2
    return func(z, f)


def expand_tuple(cont):
    # a, b, c, d = cont
    return sum(cont)


# print(jax.make_jaxpr(func2)(5))
print(jax.make_jaxpr(expand_tuple)((1, 2, 3, 4)))
print(jax.make_jaxpr(func2)(5).consts)


class AbstractJoint(BaseModel):
    nc: ClassVar[int] = 0
    eq: ClassVar[tuple[int, ...]] = None

    name: str
    l: tuple[int, ...]
    func: Callable[[float], float]


class Joint(AbstractJoint):
    nc = 5
    eq = None


print(Joint.nc)

c = Joint(name="c", l=(1, 2, 3), func=lambda t, y: None)
print(c)
