from __future__ import annotations

import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Literal
from typing import Union

import yaml
from pydantic import BaseModel as BaseModel_
from pydantic import Field
from typing_extensions import Self

from itcl_quantizer.equalizers.adaround.annealer import RoundingAnnealer
from qubo_adaround.qubo import QUBOAnnealer


class Init(metaclass=ABCMeta):
    def tflite(self):
        for attribute in self.__dict__.values():
            if isinstance(attribute, Init):
                attribute.tflite()


class BaseModel(BaseModel_, Init):
    pass


class IBuildable(metaclass=ABCMeta):
    @abstractmethod
    def build(self) -> Self:
        pass


#
# region AdaRound:
#
class RoundingAnnealerCfg(BaseModel, IBuildable):
    optimizer: Literal["annealer"] = "annealer"
    t_min: float = 0.1
    t_max: float = 25000
    updates: int = 100
    steps: int = 5000
    max_retries: int = 100

    def build(self):
        return RoundingAnnealer(
            t_min=self.t_min,
            t_max=self.t_max,
            updates=self.updates,
            steps=self.steps,
            max_retries=self.max_retries,
        )


class RoundingQUBOCfg(BaseModel, IBuildable):
    
    optimizer: Literal["qubo"] = "qubo"
    random_adaround_coefficients : bool 
    cuantization_to_round_nearest : bool
    qubo_sampler: Literal["neal", "qaoa", "dwave", "hybrid", "brute_force"] 
    qaoa_num_reps: int = 1
    dwave_num_reads : int = 30
    dwave_annealing_time : int = 20
    dwave_chain_strength: Union[int, Literal["max"]] = 0
    dictionary_subspace: bool = False
    random_qubo_num_variables : Union[bool, int] = False

    def __init__(self, 
                 random_adaround_coefficients: bool = False,
                 cuantization_to_round_nearest : bool = False,
                 qubo_sampler: Literal["neal", "qaoa", "dwave", "hybrid"] = "neal",
                 qaoa_num_reps: int = 1,
                 dwave_num_reads : int = 30,
                 dwave_annealing_time : int = 20,
                 dwave_chain_strength: Union[int, Literal["max"]] = 0,
                 dictionary_subspace: bool = False,
                 random_qubo_num_variables : Union[bool, int] = False):

        super().__init__(random_adaround_coefficients = random_adaround_coefficients,
                         cuantization_to_round_nearest = cuantization_to_round_nearest,
                         qubo_sampler = qubo_sampler,
                         qaoa_num_reps = qaoa_num_reps,
                         dwave_num_reads = dwave_num_reads,
                         dwave_annealing_time = dwave_annealing_time,
                         dwave_chain_strength = dwave_chain_strength,
                         dictionary_subspace = dictionary_subspace,
                         random_qubo_num_variables = random_qubo_num_variables)
        

    def build(self):
        return QUBOAnnealer(self.random_adaround_coefficients,
                            self.cuantization_to_round_nearest,
                             self.qubo_sampler,
                             self.qaoa_num_reps,
                             self.dwave_num_reads,
                             self.dwave_annealing_time,
                             self.dwave_chain_strength, 
                             self.dictionary_subspace,
                             self.random_qubo_num_variables)


class RoundingMinimaCfg(BaseModel, IBuildable):
    optimizer: Literal["base"] = "base"
    max_retries: int = 100

    def build(self):
        return RoundingAnnealer(
            t_min=0.1, t_max=0.1, updates=100, steps=0, max_retries=self.max_retries
        )


_adround_optimizers = RoundingMinimaCfg | RoundingAnnealerCfg | RoundingQUBOCfg
# endregion AdaRound


class ActivationCfg(BaseModel):
    reduced_depth: int = 3


#
# region layers
#
class KerasDenseCfg(BaseModel):
    activation_dtype: str = "int8"
    bias_add_dtype: str = "int8"
    bias_dtype: str = "int32"
    kernel_dtype: str = "int8"
    kernel_symmetric: bool = False
    adaround_optimizer: _adround_optimizers | None = None
    lut_cfg: ActivationCfg = Field(default_factory=ActivationCfg)

    def tflite(self):
        self.kernel_symmetric = True

        if opt := self.adaround_optimizer:
            opt.tflite()
        super().tflite()

    class Config:
        arbitrary_types_allowed = True


class QuantizeCfg(BaseModel):
    dtype: str = "int8"


class DequantizeCfg(BaseModel):
    pass


KerasLayer = KerasDenseCfg | QuantizeCfg | DequantizeCfg  # This is a Union
# endregion layers


class ParamEqualizerCfg(BaseModel):
    optimizer: Literal["base"] = "base"
    max_retries: int = 100


class QuantizerCfg(BaseModel):
    dense: KerasDenseCfg = Field(default_factory=KerasDenseCfg)
    quantize: QuantizeCfg = Field(default_factory=QuantizeCfg)
    dequantize: DequantizeCfg = Field(default_factory=DequantizeCfg)
    ada_round_net: _adround_optimizers | None = None
    param_equalizer: ParamEqualizerCfg | None = None
    specific_layers: dict[str, KerasLayer] = {}

    class Config:
        arbitrary_types_allowed = True

    def tflite(self):
        self.ada_round_net = None
        self.param_equalizer = None
        return super().tflite()

    @classmethod
    def from_json(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return cls(**cfg)

    @classmethod
    def from_yml(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cls(**cfg)


if __name__ == "__main__":
    QuantizerCfg()
