import functools
from typing import ClassVar, Union

from pydantic import BaseModel

from uraeus.rnea.cmbs.topologies.bodies import AbstractBody
import ruamel.yaml

yaml = ruamel.yaml.YAML(typ="unsafe")


class AbstractConfigInputs(BaseModel):
    pass


class AbstractJoint(BaseModel):
    prefix: ClassVar[str] = None
    config: ClassVar[AbstractConfigInputs] = None
    nc: ClassVar[int] = None
    nj: ClassVar[int] = None

    name: str
    pred: Union[str, AbstractBody]
    succ: Union[str, AbstractBody]

    # @classmethod
    # def to_yaml(cls, representor, node):
    #     tag = getattr(cls, "yaml_tag", "!" + cls.__name__)
    #     # attrs = node.model_dump()
    #     attrs = {
    #         "name": node.name,
    #         "pred": node.pred,
    #         "succ": node.succ,
    #     }
    #     return representor.represent_mapping(tag, attrs)

    def get_config_inputs(self) -> tuple[str, str, str]:
        variables = ("loc", "ax1", "ax2")
        return tuple(f"{self.name}.{var}" for var in variables)


class SphericalJoint(AbstractJoint):
    prefix = "sph"
    nc = 3
    nj = 3


class UniversalJoint(AbstractJoint):
    prefix = "uni"
    nc = 4
    nj = 2


class CylindricalJoint(AbstractJoint):
    prefix = "cyl"
    nc = 4
    nj = 2


class TranslationalJoint(AbstractJoint):
    prefix = "trn"
    nc = 5
    nj = 1


class RevoluteJoint(AbstractJoint):
    prefix = "rev"
    nc = 5
    nj = 1


# yaml.register_class(RevoluteJoint)
# yaml.register_class(SphericalJoint)
# yaml.register_class(UniversalJoint)
# yaml.register_class(CylindricalJoint)
# yaml.register_class(TranslationalJoint)
