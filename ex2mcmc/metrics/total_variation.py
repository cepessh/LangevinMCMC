from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp
from scipy.stats import gaussian_kde
from tqdm.auto import trange


class MeanTracker:
    def __init__(self):
        self.values = []

    def update(self, value: float) -> None:
        self.values.append(value)

    def __len__(self):
        return len(self.values)

    def mean(self) -> jnp.ndarray:
        return jnp.mean(jnp.array(self.values))

    def std(self) -> jnp.ndarray:
        return jnp.std(jnp.array(self.values), ddof=1)

    def std_of_mean(self) -> jnp.ndarray:
        return jnp.std(jnp.array(self.values)) / jnp.sqrt(len(self))

    def last(self) -> float:
        return self.values[-1]


@dataclass
class Projector:
    x0: jnp.ndarray
    v: jnp.ndarray

    def project(self, xs: jnp.ndarray) -> jnp.ndarray:
        return (xs - self.x0[None]) @ self.v


def create_random_projection(key: jnp.ndarray, xs: jnp.ndarray) -> Projector:
    x0 = jnp.mean(xs, axis=0)
    v = jax.random.normal(key, [len(x0)])
    v = v / jnp.linalg.norm(v)

    return Projector(x0, v)


def create_random_2d_projection(
    key: jnp.ndarray,
    xs: jnp.ndarray,
) -> Projector:
    x0 = jnp.mean(xs, 0)
    v = jax.random.normal(key, [len(x0), len(x0)])
    v = v / jnp.linalg.norm(v)

    return Projector(x0, v)


def average_total_variation(
    key: jnp.ndarray,
    true: jnp.ndarray,
    other: jnp.ndarray,
    density_probe_count: int,
    projection_count: int,
) -> MeanTracker:
    tracker = MeanTracker()
    keys = jax.random.split(key, projection_count)

    for chain_index in range(other.shape[1]):
        for i in range(projection_count):
            tracker.update(total_variation(
                keys[i], true, other[:, chain_index], density_probe_count))

    return tracker


def total_variation(
    key: jnp.ndarray,
    xs_true: jnp.ndarray,
    xs_pred: jnp.ndarray,
    density_probe_count: int,
):
    proj = create_random_projection(key, xs_true)
    return total_variation_1d(
        proj.project(xs_true),
        proj.project(xs_pred),
        density_probe_count,
    )


def total_variation_1d(xs_true, xs_pred, density_probe_count):
    true_density = gaussian_kde(xs_true)
    pred_density = gaussian_kde(xs_pred)

    x_min = min(xs_true.min(), xs_pred.min())
    x_max = max(xs_true.max(), xs_pred.max())

    points = np.linspace(x_min, x_max, density_probe_count)

    return (
        0.5
        * np.abs(true_density(points) - pred_density(points)).mean()
        * (x_max - x_min)
    )


# def average_emd(
#     key: jnp.ndarray, true: jnp.ndarray, other: jnp.ndarray, sample_count: int, n_steps: int
# ) -> MeanTracker:
#     tracker = MeanTracker()
#     keys = jax.random.split(key, n_steps)
#     for i in trange(n_steps, leave=False):
#         tracker.update(emd_2d(keys[i], true, other, sample_count))
#     return tracker


# def emd_2d(key: jnp.ndarray, xs_true: jnp.ndarray, xs_pred: jnp.ndarray, sample_count: int):
#     proj = create_random_2d_projection(key, xs_true)
#     return earth_movers_distance_2d(proj.project(xs_true), proj.project(xs_pred), sample_count)


# def earth_movers_distance_2d(xs_true, xs_pred, sample_count):
#     print(xs_true.shape, xs_pred.shape)
#     M = np.linalg.norm(xs_true[None, :, :] - xs_pred[:, None, :], axis=-1, ord=2)**2
#     emd = ot.lp.emd2([], [], M)
#     return emd
