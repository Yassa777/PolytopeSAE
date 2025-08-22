import numpy as np
from polytope_hsae.logistic_regression import (
    fit_logistic_regressions,
    fit_logistic_regressions_parallel,
)


def test_parallel_matches_sequential():
    rng = np.random.default_rng(0)
    parent_data = {}
    for i in range(3):
        X = rng.standard_normal((200, 5))
        y = rng.integers(0, 3, 200)
        parent_data[f"p_{i}"] = (X, y)

    seq = fit_logistic_regressions(parent_data)
    par = fit_logistic_regressions_parallel(parent_data, n_jobs=2)

    assert seq.keys() == par.keys()
    for k in seq:
        np.testing.assert_allclose(seq[k]['coef'], par[k]['coef'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(seq[k]['intercept'], par[k]['intercept'], rtol=1e-6, atol=1e-6)
