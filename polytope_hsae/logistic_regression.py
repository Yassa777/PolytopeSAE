import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

def _fit_single(X: np.ndarray, y: np.ndarray):
    """Fit multinomial logistic regression for one parent."""
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        penalty="l2",
        max_iter=1000,
        random_state=0,
        n_jobs=1,
    )
    # Limit threads so parallel processes do not oversubscribe CPUs
    with threadpool_limits(limits=1):
        clf.fit(X, y)
    return clf.coef_, clf.intercept_

def fit_logistic_regressions(parent_data: dict) -> dict:
    """Fit logistic regression models sequentially for each parent.

    Args:
        parent_data: Mapping from parent id to tuple (X, y)
                     where X is shape [n_samples, dim].

    Returns:
        Mapping from parent id to dict with keys 'coef' and 'intercept'.
    """
    results = {}
    for pid, (X, y) in parent_data.items():
        if len(np.unique(y)) < 2:
            continue
        coef, intercept = _fit_single(X, y)
        results[pid] = {"coef": coef, "intercept": intercept}
    return results

def fit_logistic_regressions_parallel(parent_data: dict, n_jobs: int = -1) -> dict:
    """Fit logistic regression models in parallel for each parent.

    Args:
        parent_data: Mapping from parent id to tuple (X, y).
        n_jobs: Number of parallel jobs. Defaults to -1 (all cores).

    Returns:
        Mapping from parent id to dict with keys 'coef' and 'intercept'.
    """
    items = [
        (pid, X, y)
        for pid, (X, y) in parent_data.items()
        if len(np.unique(y)) >= 2
    ]
    parallel_out = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_single)(X, y) for _, X, y in items
    )
    results = {}
    for (pid, _, _), (coef, intercept) in zip(items, parallel_out):
        results[pid] = {"coef": coef, "intercept": intercept}
    return results
