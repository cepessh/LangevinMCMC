from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import SamplableDistribution, GaussianMixture, Distribution
from samplers.base_sampler import Cache
from tools.metrics import tv_threshold


@dataclass
class TVStop:
    threshold: float = 0.1
    projection_count: int = 25
    density_probe_count: int = 1000

    def __call__(self, cache: Cache):
        tv_mean, tv_std = tv_threshold(
            jnp.array(cache.true_samples), jnp.array(cache.samples),
            self.density_probe_count, self.projection_count
        )

        return tv_mean < self.threshold