import tqdm
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import Distribution


def heuristics_step_size(
    mean_acceptance: float,
    target_acceptance: float,
    step_size: float,
    factor: float = 1.05,
    tol: float = 0.03,
):
    if mean_acceptance - target_acceptance > tol:
        return step_size * factor
    if target_acceptance - mean_acceptance > tol:
        return step_size / factor
    return step_size

def mala(
    starting_points: Tensor,
    target_dist: Union[Distribution, torchDist],
    sample_count: int,
    burn_in: int,
    project: Callable = lambda x: x,
    *,
    step_size: float = 1,
    verbose: bool = False,
    target_acceptance=None,
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    Metropolis-Adjusted Langevin Algorithm with Normal proposal_dist

    Args:
        starting_points - strating points of shape [n_chains x dim]
        target_dist - target_dist distribution instance with method "log_prob"
        step_size - step size for drift term
        verbose - whether show iterations' bar

    Returns:
        sequence of slices per each iteration, meta
    """
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

    meta = meta or dict()
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])
    meta["logp"] = logp_x = target_dist.log_prob(point)

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
        proposal_point = point + step_size * grad_x + noise * (2 * step_size) ** 0.5
        proposal_point = project(proposal_point)
        if not keep_graph:
            proposal_point = proposal_point.detach().requires_grad_()

        logp_y = target_dist.log_prob(proposal_point)
        print("logp_y", logp_y)
        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
            create_graph=keep_graph,
            retain_graph=keep_graph,
        )[
            0
        ]  # .detach()

        log_qyx = proposal_dist.log_prob(noise)
        log_qxy = proposal_dist.log_prob(
            (point - proposal_point - step_size * grad_y) / (2 * step_size) ** 0.5
        )

        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob
        mask = mask.detach()

        if keep_graph:
            mask_f = mask.float()
            point = point * (1 - mask_f) + proposal_point * mask_f
            logp_x = logp_x * (1 - mask_f) + logp_y * mask_f
            grad_x = grad_x * (1 - mask_f) + grad_y * mask_f
        else:
            with torch.no_grad():
                point[mask] = proposal_point[mask]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]

        meta["mh_accept"].append(mask.float().mean().item())
        if target_acceptance:
            step_size = heuristics_step_size(
                meta["mh_accept"][-1], target_acceptance, step_size
            )
        meta["step_size"].append(step_size)

        if not keep_graph:
            point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point.cpu().clone())
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["grad"] = grad_x
    meta["mask"] = mask.detach().cpu() # type: ignore

    return chains, meta