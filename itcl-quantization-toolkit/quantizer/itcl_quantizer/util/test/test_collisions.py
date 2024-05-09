import numpy as np

from ..collisions import calc_collisions


def test_calc_collisions():
    fp_a = np.array([[1.2, 5.6, 0.9], [-2.9, 1.24, 7.8], [0.7, -4.7, 3.6]])

    q_a = np.array([[67, 128, 67], [-12, 67, 128], [32, -89, 111]])

    assert (
        calc_collisions(
            fp_a,
            q_a,
            0,
        )
        == 5
    )

    assert (
        calc_collisions(
            fp_a,
            q_a,
            1,
        )
        == 4
    )
