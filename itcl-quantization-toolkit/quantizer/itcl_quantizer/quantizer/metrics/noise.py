import numpy as np

# import kl_divergence
from scipy.special import kl_div


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:

    p[p == 0] = 1e-10
    q[q == 0] = 1e-10
    return kl_div(p, q).sum()


def mse(p: np.ndarray, q: np.ndarray) -> float:
    return (np.square(p - q)).mean(axis=0)  # typing: ignore


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    return -np.sum(p * np.log(q))  # typing: ignore # typing: ignore
