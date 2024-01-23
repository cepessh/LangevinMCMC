import jax
import jax.numpy as jnp
import ot

from ex2mcmc.metrics.total_variation import average_total_variation
from ex2mcmc.metrics.chain import ESS, acl_spectrum, autocovariance

def compute_metrics(
    xs_true,
    xs_pred,
    name=None,
    sample_count=1000,
    scale=1.0,
    trunc_chain_len: int = 0,
    ess_rar=1,
    max_iter_ot=1_000_000,
):
    metrics = dict()
    key = jax.random.PRNGKey(0)
    n_steps = 25
    # sample_count = 100

    ess = ESS(
        acl_spectrum(
            xs_pred[::ess_rar] - xs_pred[::ess_rar].mean(0)[None, ...],
        ),
    ).mean()
    metrics["ess"] = ess

    xs_pred = xs_pred[-trunc_chain_len:]
    
    #print("avg total variation")
    tracker = average_total_variation(
        key,
        xs_true,
        xs_pred,
        n_steps=n_steps,
        sample_count=sample_count,
    )

    metrics["tv_mean"] = tracker.mean()
    metrics["tv_conf_sigma"] = tracker.std_of_mean()

    mean = tracker.mean()
    std = tracker.std()

    metrics["wasserstein"] = 0
    #Cost_matr_isir = ot.dist(x1 = isir_res[j][i], x2=gt_samples[i], metric='sqeuclidean', p=2, w=None)
    #print("wasserstein")
    for b in range(xs_pred.shape[1]):
        M = jnp.array(ot.dist(xs_true / scale, xs_pred[:, b] / scale))
        emd = ot.lp.emd2([], [], M, numItermax=max_iter_ot)
        metrics["wasserstein"] += emd / xs_pred.shape[1]

    return metrics