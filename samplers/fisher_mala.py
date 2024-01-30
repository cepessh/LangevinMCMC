from functools import partial
import tqdm
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as torchDist

from distributions import Distribution
from samplers.mala_modified import mala


def h(z: Tensor, v: Tensor, sigma: Tensor, prec_factors: list[Tensor], 
      target_dist: Union[Distribution, torchDist], keep_graph: bool) -> Tensor:
    """
    z, v (sample_count, n_dim)
    sigma (sample_count)
    prec_factors List[(sample_count, n_dim, n_dim)]
    """

    logp_v = target_dist.log_prob(v)
    if keep_graph:
        grad_v = torch.autograd.grad(
            logp_v.sum(),
            v,
            create_graph=keep_graph,
            retain_graph=keep_graph,
        )[0]
    else:
        grad_v = torch.autograd.grad(logp_v.sum(), v)[0].detach()
    
    grad_v_img = prec_factors[-1] @ grad_v[..., None]
    for factor in reversed(prec_factors[:-1]):
        grad_v_img = factor @ grad_v_img

    grad_v_img = grad_v_img.squeeze()

    return 0.5 * (grad_v[:, None, :] @ 
                  (z - v - 0.25 * grad_v_img * sigma[..., None] ** 2)[..., None]
                 ).squeeze()


def fisher_mala(
    starting_points: torch.Tensor,
    target_dist: Union[Distribution, torchDist],
    sample_count: int,
    burn_in_prec: int,
    burn_in_sigma: int = 500,
    project: Callable = lambda x: x,
    *,
    sigma_init: float = 1.,
    damping: float = 10.,
    rho: float = 0.015,
    alpha: float = 0.574,
    verbose: bool = False,
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    starting_points (sample_count, n_dim)
    sigma (sample_count)
    """

    chains = []
    point = starting_points.clone()
    point.requires_grad_()
    point.grad = None
    device = point.device

    proposal_dist = torch.distributions.MultivariateNormal(
        torch.zeros(point.shape[-1], device=device),
        torch.eye(point.shape[-1], device=device),
    )

    samples, meta = mala(point, target_dist, sample_count=1, burn_in=burn_in_sigma-1,
                         project=project, sigma_init=sigma_init, meta=meta,
                         keep_graph=keep_graph, verbose=verbose)  
    
    point = samples[0]
    if not keep_graph:
        point = point.detach().requires_grad_()

    R = torch.eye(point.shape[-1]).repeat(*point.shape[:-1], 1, 1)
    sigma_R = meta["sigma"][..., None]
    # print("sigma_R", sigma_R)

    grad_x = meta["grad"]
    logp_x = meta["logp"]

    sigma_ = sigma_R.clone()

    h_ = partial(h, prec_factors=[R, R.permute(0, 2, 1)], keep_graph=keep_graph,
                 target_dist=target_dist)
    
    pbar = tqdm.trange if verbose else range
    for step_id in pbar(sample_count + burn_in_prec):

        h_ = partial(h, prec_factors=[R, R.permute(0, 2, 1)], keep_graph=keep_graph,
                     target_dist=target_dist, sigma=sigma_R.squeeze())

        # print("step", step_id)
        noise = proposal_dist.sample(point.shape[:-1])

        grad_x_img = grad_x[..., None]
        grad_x_img = R @ (R.permute(0, 2, 1) @ grad_x_img)

        # print("grad_transf", grad_x_img.shape)

        proposal_point = point + (
            0.5 * grad_x_img * sigma_R ** 2 + R @ noise[..., None] * sigma_R
        ).squeeze()
        # print("nan proposal_point", torch.isnan(proposal_point).sum())

        # print("proposal point", proposal_point.shape)

        if not keep_graph:
            proposal_point = proposal_point.detach().requires_grad_()

        logp_y = target_dist.log_prob(proposal_point)
        # print("logpy", logp_y.shape)
        # print("nan logp_y", torch.isnan(logp_y).sum())        
        # print("logp_y", logp_y)
        
        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
            create_graph=keep_graph,
            retain_graph=keep_graph,
        )[
            0
        ]  # .detach()

        # Inefficient O(d^3) calculation of the inverse 
        # grad_y_img = grad_y[..., None]
        # grad_y_img = R @ (R.permute(0, 2, 1) @ grad_y_img)

        # log_qyx = proposal_dist.log_prob(noise)
        # log_qxy = proposal_dist.log_prob(
        #     (
        #         (R * sigma_R).inverse() @ 
        #         (point - proposal_point - (0.5 * grad_y_img * sigma_R ** 2).squeeze())[..., None]
        #     ).squeeze()
        # )
        #
        # accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        accept_prob = torch.clamp(
            torch.exp(
                logp_y + h_(point, proposal_point) - logp_x - h_(proposal_point, point)
            ),
            max=1
        )

        if not keep_graph:
            accept_prob = accept_prob.detach()
        # print("accept_prob", accept_prob)
        # print("nan accept_prob", torch.isnan(accept_prob).sum())


        # print("accept", accept_prob.shape)
        # print("grad_y - grad_x", (grad_y - grad_x).shape)
        
        signal_adaptation = torch.sqrt(accept_prob)[..., None] * (grad_y - grad_x)

        #if not keep_graph: 
        #    signal_adaptation = signal_adaptation.detach()
        # print("nan signal_adaptation", torch.isnan(signal_adaptation).sum())
        # print("sqrt accept", torch.sqrt(accept_prob))
        
        # print("sig adapt", signal_adaptation)

        phi_n = R.permute(0, 2, 1) @ signal_adaptation[..., None]
        #if not keep_graph: 
        #    phi_n = phi_n.detach()
        # print("nan phi_n", torch.isnan(phi_n).sum())
        # print("phi_n", phi_n)

        gramm_diag = phi_n.permute(0, 2, 1) @ phi_n
        #if not keep_graph:
        #    gramm_diag = gramm_diag.detach()

        # print("gramm_diag", gramm_diag)

        if step_id == 0:
            r_1 = 1. / (1 + torch.sqrt(damping / (damping + gramm_diag)))
            shift = phi_n @ phi_n.permute(0, 2, 1)
            # print("shift", shift)
            R = 1. / damping ** 0.5 * (R - shift * r_1 / (damping + gramm_diag))
        else:
            r_n = 1. / (1 + torch.sqrt(1 / (1 + gramm_diag)))
            # print("nan rn", torch.isnan(r_n).sum())
            # print("r_n", r_n)

            # print((R @ phi_n).shape)
            # print((phi_n.permute(0, 2, 1)).shape)
            shift = (R @ phi_n) @ phi_n.permute(0, 2, 1)
            # print("shift", shift)
            R = R - shift * r_n / (1 + gramm_diag)

        
        # print("R", R)
        
        # print("sigma update", (1 + rho * (accept_prob - alpha)))
        sigma_[..., 0, 0] *= (1 + rho * (accept_prob - alpha)) ** 0.5
        # sigma_R = torch.full_like(sigma_R, 1)

        # print("sigma_R before norm", sigma_R)

        trace_prec = (R[..., None, :] @ R[..., None]).sum(dim=1)
        # print("trace", trace_prec)
        normalizer = (1. / point.shape[-1]) * trace_prec
        # print("normalizer", normalizer)
        sigma_R = sigma_ / normalizer ** 0.5
        # print("sigma_R", sigma_R)

        # A_n = R * sigma_R
        # print("R * sigma", A_n)

        # A_n = A_n @ A_n.permute(0, 2, 1)
        # trace_A = [A.trace() for A in A_n]

        # print("trace_A", trace_A)

        # sigma_R = torch.full_like(sigma_R, 1)
        # R = torch.eye(point.shape[-1]).repeat(*point.shape[:-1], 1, 1)
        # print("normalizer", normalizer)
        # print("sigma_R", sigma_R)  

        # print()

        mask = torch.rand_like(accept_prob) < accept_prob
        mask = mask.detach()

        if keep_graph:
            mask_f = mask.float()[..., None]
            point = point * (1 - mask_f) + proposal_point * mask_f
            logp_x = logp_x * (1 - mask_f) + logp_y * mask_f
            grad_x = grad_x * (1 - mask_f) + grad_y * mask_f
        else:
            with torch.no_grad():
                point[mask] = proposal_point[mask]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]

        meta["mh_accept"].append(accept_prob)
        meta["sigma"] = sigma_R

        if not keep_graph:
            point = point.detach().requires_grad_()

        if step_id >= burn_in_prec:
            chains.append(point.detach().cpu().clone())
        
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x.detach()
    meta["grad"] = grad_x.detach()

    return chains, meta