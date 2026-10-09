import numpy as np
import pytest
from scripts.information_probe.core import output_basis, fit_pca, project, fit_ridge, predict, mismatch_indices


def test_modes_separate_cycle_levels_and_shapes():
    q = output_basis(96)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(q[:, 2:].reshape(4, 24, 2).mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(np.diff(q[:, :2].reshape(4, 24, 2), axis=1), 0, atol=1e-12)


def test_capacity_match_does_not_depend_on_feature_covariance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(500, 11))
    b = a @ rng.normal(size=(11, 11))
    y = a[:, :4] + rng.normal(size=(500, 4))*.1
    for x in (a, b):
        fitted = fit_ridge(x, y, 4.)
        assert fitted['readout_parameters'] == 48
        assert fitted['slope_edf'] == pytest.approx(4., abs=1e-10)
        assert np.square(predict(x, fitted)-y).mean() < np.square(y).mean()


def test_mismatch_rejects_overlapping_forecast_supports():
    ids = np.arange(0, 4000, 24)
    mapping = mismatch_indices(ids, ids, 3, 1440, same_role=True)
    assert np.all(mapping.reshape(-1, 3) % 3 == np.arange(3))
    with pytest.raises(ValueError):
        mismatch_indices(np.arange(0, 2000, 24), np.arange(0, 2000, 24), 3, 1440, same_role=True)


def test_preprocessing_is_frozen_for_later_samples():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(80, 12))
    transform = fit_pca(train)
    x = rng.normal(size=(5, 12))
    before = project(x, transform)
    project(x*100, transform)
    np.testing.assert_array_equal(before, project(x, transform))
