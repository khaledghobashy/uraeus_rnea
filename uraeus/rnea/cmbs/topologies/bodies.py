from __future__ import annotations
import functools
from typing import ClassVar
from pydantic import BaseModel

import ruamel.yaml

yaml = ruamel.yaml.YAML(typ="unsafe")


class Array(BaseModel):
    shape: tuple[int, ...]


class BodyConfigData(BaseModel):
    pose: list[float] = [0, 0, 0, 0, 0, 0]
    mass: float = 0
    inertia: list[float] = [0, 0, 0, 0, 0, 0]

    @classmethod
    def to_yaml(cls, representor, node: BodyConfigData):
        tag = getattr(cls, "yaml_tag", "!" + cls.__name__)
        attrs = node.model_dump()
        return representor.represent_mapping(tag, attrs)


class AbstractBody(BaseModel):
    prefix: ClassVar[str] = None
    prefix: ClassVar[str] = None
    name: str

    @classmethod
    def to_yaml(cls, representor, node: AbstractBody):
        tag = getattr(cls, "yaml_tag", "!" + cls.__name__)
        # attrs = {}
        # for x in node.__annotations__:
        #     v = getattr(node, x)
        #     if callable(v):
        #         continue
        #     attrs[x] = v
        attrs = node.model_dump()
        return representor.represent_mapping(tag, attrs)

    # @classmethod
    # def from_yaml(cls, constructor, node):
    #     data = constructor.construct_mapping(node, deep=True)
    #     instance = cls(**data)
    #     return instance

    def __hash__(self):
        return hash(self.name)


class RigidBody(AbstractBody):
    prefix: ClassVar[str] = "rb"
    config: BodyConfigData = BodyConfigData()
    pose: list[float] = [0, 0, 0, 0, 0, 0]
    mass: float = 0
    inertia: list[float] = [0, 0, 0, 0, 0, 0]

    def get_config_inputs(self) -> tuple[str, str, str]:
        variables = ("pose", "mass", "inertia")
        return tuple(f"{self.name}.{var}" for var in variables)


class VirtualBody(AbstractBody):
    prefix: ClassVar[str] = "vb"

    def get_config_inputs(self) -> str:
        return None


# yaml.register_class(BodyConfigData)
# yaml.register_class(AbstractBody)
# yaml.register_class(RigidBody)
# yaml.register_class(VirtualBody)

if __name__ == "__main__":
    # print(RigidBody(name="body"))
    import sys

    body = RigidBody(name="body")
    print(body)
    # quit()
    yaml.dump(body, sys.stdout)
    txt = """
!RigidBody
config:
  inertia: [0, 0, 0, 0, 0, 0]
  mass: 0
  pose: [0, 0, 0, 0, 0, 0]
name: body2
"""
    print(yaml.load(txt))
