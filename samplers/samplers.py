from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import SamplableDistribution, GaussianMixture, Distribution
from samplers import base_sampler


@dataclass
class MALAIter(base_sampler.Iteration):
    def run(self):
        noise = self.cache.params.proposal_dist.sample(self.cache.point.shape[:-1])

        proposal_point = (
            self.cache.point + 
            0.5 * self.cache.grad * self.cache.params.meta["sigma"] ** 2 + 
            noise * self.cache.params.meta["sigma"]
        ).detach().requires_grad_()

        logp_y = self.cache.params.target_dist.log_prob(proposal_point)
        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
        )[0].detach()

        with torch.no_grad():
            log_qyx = self.cache.params.proposal_dist.log_prob(noise)
            log_qxy = self.cache.params.proposal_dist.log_prob(
                (self.cache.point - proposal_point - 
                0.5 * self.cache.params.meta["sigma"] ** 2 * grad_y) / self.cache.params.meta["sigma"]
            )
            
            accept_prob = torch.clamp((logp_y + log_qxy - self.cache.logp - log_qyx).exp(), max=1).detach()
            mask = torch.rand_like(accept_prob) < accept_prob

            self.cache.point[mask] = proposal_point[mask]
            self.cache.logp[mask] = logp_y[mask]
            self.cache.grad[mask] = grad_y[mask]

            self.cache.params.meta["sigma"] *= (
                1 + self.cache.params.meta["rho"] * (
                    accept_prob[..., None] - self.cache.params.meta["alpha"]
                )
            ) ** 0.5

        self.cache.samples.append(self.cache.point.detach().clone())