import numpy as np


def calc_collisions(fp_tensor: np.ndarray, q_tensor: np.ndarray, epsilon: int) -> int:
    """Calculates the collisions between two tensors

    Collision: Number of equal elements in the Q tensor that are not equal in the FP tensor

    Args:
        fp_tensor (np.ndarray): Floating point tensor
        q_tensor (np.ndarray): Quantized tensor
        epsilon (int): Decimal tolerance of the floating point tensor

    Returns:
        int: Number of collisions
    """
    rounded_fp = np.around(fp_tensor, decimals=epsilon) if epsilon > 0 else fp_tensor

    _, counts_fp = np.unique(rounded_fp, return_counts=True)
    _, counts_q = np.unique(q_tensor, return_counts=True)

    return (counts_q * (counts_q != 1)).sum() - (counts_fp - 1).sum()
