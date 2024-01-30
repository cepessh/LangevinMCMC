from distutils.fancy_getopt import longopt_pat
import tqdm
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import Distribution


def mala(
    starting_points: Tensor,
    target_dist: Union[Distribution, torchDist],
    sample_count: int,
    burn_in: int,
    project: Callable = lambda x: x,
    *,
    sigma_init: float = 1.,
    rho: float = 0.015,
    alpha: float = 0.574,
    verbose: bool = False,
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    
    if sample_count + burn_in <= 0:
        raise ValueError("Number of steps might be positive")

    chains = []
    point = starting_points.clone()
    point.requires_grad_()
    point.grad = None

    device = point.device

    # Reassigning of the argument proposal_dist
    proposal_dist = torch.distributions.MultivariateNormal(
        torch.zeros(point.shape[-1], device=device),
        torch.eye(point.shape[-1], device=device),
    )
    
    logp_x = target_dist.log_prob(point)
    sigma = torch.full(point.shape[:-1], sigma_init)[..., None]

    meta = meta or dict()
    meta["mh_accept"] = meta.get("mh_accept", [])
    
    meta["logp"] = meta.get("logp", logp_x)
    meta["sigma"] = meta.get("sigma", sigma)

    if "grad" not in meta:
        if keep_graph:
            grad_x = torch.autograd.grad(
                meta["logp"].sum(),
                point,
                create_graph=keep_graph,
                retain_graph=keep_graph,
            )[0]
        else:
            grad_x = torch.autograd.grad(logp_x.sum(), point)[0].detach()
        meta["grad"] = grad_x
    else:
        grad_x = meta["grad"]

    pbar = tqdm.trange if verbose else range

    for step_id in pbar(sample_count + burn_in):
        noise = proposal_dist.sample(point.shape[:-1])
        # print("noise", noise.shape)

        proposal_point = point + 0.5 * grad_x * sigma ** 2 + noise * sigma 
        proposal_point = project(proposal_point)
        # print("nan proposal", proposal_point.shape)

        if not keep_graph:
            proposal_point = proposal_point.detach().requires_grad_()

        logp_y = target_dist.log_prob(proposal_point)
        # print("logp_y", logp_y.shape)
        # print("nan logp_y", torch.isnan(logp_y).sum())

        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
            create_graph=keep_graph,
            retain_graph=keep_graph,
        )[
            0
        ]  # .detach()
        # print("grad_y", grad_y.shape)
        # print("logp_y", logp_y)
        # print("nan grad_y", torch.isnan(grad_y).sum(dim=1))
        # print("sigma", sigma.squeeze())

        log_qyx = proposal_dist.log_prob(noise)
        # print("log_qyx", log_qyx.shape)

        # print("nan num", torch.isnan(point - proposal_point - sigma ** 2 * grad_y).sum())
        log_qxy = proposal_dist.log_prob(
            (point - proposal_point - 0.5 * sigma ** 2 * grad_y) / sigma
        )
        # print("log_qxy", log_qxy.shape)
        
        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob

        if keep_graph:
            mask_f = mask.float()[..., None]
            point = point * (1 - mask_f) + proposal_point * mask_f
            logp_x = logp_x * (1 - mask_f).squeeze() + logp_y * mask_f.squeeze()
            grad_x = grad_x * (1 - mask_f) + grad_y * mask_f
        else:
            with torch.no_grad():
                point[mask] = proposal_point[mask]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]

        # print("point", point.shape)
        # print("logpx", logp_x.shape)
        # print("grad_x", grad_x.shape)
        meta["mh_accept"].append(accept_prob.detach())
        # print('accept', accept_prob[..., None].shape)
        
        if not keep_graph:
            accept_prob = accept_prob.detach()

        sigma *= (1 + rho * (accept_prob[..., None] - alpha)) ** 0.5

        if not keep_graph:
            sigma = sigma.detach()

        # print("sigma", sigma.shape)
        
        meta["sigma"] = sigma.detach().cpu().clone()

        if not keep_graph:
            point = point.detach().requires_grad_()
            
        if step_id >= burn_in:
            chains.append(point.detach().cpu().clone())

        # print()
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["grad"] = grad_x

    return chains, meta