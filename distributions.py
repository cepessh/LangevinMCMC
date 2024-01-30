
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

class Distribution(ABC):
    """Abstract class for distribution"""

    @abstractmethod
    def log_prob(self, z: Tensor) -> Tensor:
        """Computes log probability of input z"""
        raise NotImplementedError


class SamplableDistribution(Distribution):

    @abstractmethod
    def sample(self, sample_count) -> Tensor:
        pass


class GaussianMixture(SamplableDistribution):
    def __init__(self, means, covs, weights: Tensor):
        self.weights = weights
        self.category = torch.distributions.Categorical(self.weights)
        self.means = means
        self.covs = covs

    def sample(self, sample_count: int) -> Tensor:
        # which_gaussian = self.category.sample(torch.Size((sample_count,)))
        which_gaussian = self.category.sample(torch.Size((sample_count,)))
        multivariate_normal = torch.distributions.MultivariateNormal(
            self.means[which_gaussian], self.covs[which_gaussian])
        
        return Tensor(multivariate_normal.sample())
    
    def log_prob(self, z: Tensor) -> Tensor:
        logs = torch.distributions.MultivariateNormal(
            self.means, self.covs).log_prob(z[:, None, :])
        logs += torch.log(self.weights)

        return Tensor(torch.logsumexp(logs, dim=1))