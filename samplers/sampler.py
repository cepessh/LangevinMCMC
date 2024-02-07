from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import SamplableDistribution, GaussianMixture, Distribution


@dataclass
class Params:
    target_dist: Union[torchDist, Distribution]
    starting_point: Optional[Tensor]
    proposal_dist: Union[torchDist, Distribution]

    meta: dict = field(default_factory=dict)


@dataclass
class Cache:
    params: Params

    samples: list = field(default_factory=list)

    point: Optional[Tensor] = None
    logp: Optional[Tensor] = None
    grad: Optional[Tensor] = None


def update_params(params: Params, cache: Cache):
    if cache is None:
        return

    params.starting_point = cache.point
    params.meta = cache.params.meta


@dataclass
class Iteration(ABC):
    cache: Cache

    def init(self) -> None:
        self.cache.point = self.cache.params.starting_point.requires_grad_()
        self.cache.logp = self.cache.params.target_dist.log_prob(self.cache.point)
        self.cache.grad = torch.autograd.grad(
            self.cache.logp.sum(),
            self.cache.point,
        )[0].detach()

    @abstractmethod
    def run(self) -> Cache:
        raise NotImplementedError


@dataclass
class SampleBlock:
    iteration: Iteration
    iteration_count: int
    stopping_rule: Optional[Callable] = None
    probe_period: Optional[int] = None

    def run(self, cache: Cache = None) -> Cache:
        update_params(self.iteration.cache.params, cache)
        self.iteration.init()

        for iter_step in range(self.iteration_count):
            self.iteration.run()

            if self.stopping_rule and (iter_step + 1) % self.probe_period == 0:
                is_stop = self.stopping_rule(self.iteration.cache)
                if is_stop:
                    break

        return self.iteration.cache


@dataclass
class Pipeline:
    sample_blocks: list[SampleBlock]
    cache: Optional[Cache] = None

    def run(self) -> Cache:
        for block in self.sample_blocks:
            self.cache = block.run(self.cache)
            
        return self.cache