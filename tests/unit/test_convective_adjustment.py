import gsw
import numpy as np
import xarray as xr

from OceanOSSE.gridding.convective_adjustment import (
    _convective_adjustment,
    convective_adjustment,
)


def is_stable(SA, CT, p):
    """Return True if density increases monotonically down the column."""

    rho = gsw.density.rho(SA, CT, p)

    return np.all(rho[:-1] <= rho[1:])


def test_stable_column_is_unchanged():
    """A stable column should pass through completely unchanged."""

    SA = np.array([35.0, 35.1, 35.2])
    CT = np.array([15.0, 10.0, 5.0])
    p = np.array([0.0, 100.0, 200.0])
    h = np.array([100.0, 100.0, 100.0])

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    np.testing.assert_array_equal(SA_new, SA)
    np.testing.assert_array_equal(CT_new, CT)


def test_unstable_two_layer_column_is_mixed():
    """An unstable two-layer column should be completely mixed."""

    # Warm, relatively fresh water below colder, saltier water.
    SA = np.array([35.0, 35.0])
    CT = np.array([10.0, 15.0])
    p = np.array([0.0, 100.0])
    h = np.array([100.0, 100.0])

    assert not is_stable(SA, CT, p)

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    expected_SA = np.average(SA, weights=h)
    expected_CT = np.average(CT, weights=h)

    np.testing.assert_allclose(
        SA_new,
        [expected_SA, expected_SA],
    )

    np.testing.assert_allclose(
        CT_new,
        [expected_CT, expected_CT],
    )

    assert is_stable(SA_new, CT_new, p)


def test_unstable_column_conserves_properties():
    """Convective adjustment should conserve h-weighted SA and CT."""

    SA = np.array([35.0, 35.0])
    CT = np.array([10.0, 15.0])
    p = np.array([0.0, 100.0])
    h = np.array([100.0, 100.0])

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    np.testing.assert_allclose(
        np.sum(SA_new * h),
        np.sum(SA * h),
    )

    np.testing.assert_allclose(
        np.sum(CT_new * h),
        np.sum(CT * h),
    )


def test_partially_unstable_column():
    """
    An unstable pair in the middle of the water column should be mixed,
    while the surrounding stable layers should remain unchanged.
    """

    # First construct a known stable column.
    SA = np.array(
        [
            35.0,
            35.0,
            35.0,
            35.0,
        ]
    )

    CT = np.array(
        [
            15.0,
            10.0,
            5.0,
            0.0,
        ]
    )

    p = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ]
    )

    h = np.full(4, 100.0)

    assert is_stable(SA, CT, p)

    # Make the middle two layers unstable by making the
    # third layer substantially warmer.
    CT[2] = 15.0

    assert not is_stable(SA, CT, p)

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    # Surface layer should remain unchanged.
    np.testing.assert_allclose(SA_new[0], SA[0])
    np.testing.assert_allclose(CT_new[0], CT[0])

    # Layers 1 and 2 should be mixed.
    expected_SA = np.average(SA[1:3], weights=h[1:3])
    expected_CT = np.average(CT[1:3], weights=h[1:3])

    np.testing.assert_allclose(
        SA_new[1:3],
        expected_SA,
    )

    np.testing.assert_allclose(
        CT_new[1:3],
        expected_CT,
    )

    # Deepest layer should remain unchanged.
    np.testing.assert_allclose(SA_new[3], SA[3])
    np.testing.assert_allclose(CT_new[3], CT[3])

    assert is_stable(SA_new, CT_new, p)


def test_nan_padding_is_preserved():
    """NaN values below the wet column should remain NaN."""

    SA = np.array(
        [
            35.0,
            35.0,
            35.0,
            np.nan,
            np.nan,
        ]
    )

    CT = np.array(
        [
            15.0,
            10.0,
            5.0,
            np.nan,
            np.nan,
        ]
    )

    p = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
        ]
    )

    h = np.array(
        [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ]
    )

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    assert np.all(np.isfinite(SA_new[:3]))
    assert np.all(np.isfinite(CT_new[:3]))

    assert np.all(np.isnan(SA_new[3:]))
    assert np.all(np.isnan(CT_new[3:]))


def test_variable_layer_thickness():
    """Mixing should use layer-thickness weighting."""

    SA = np.array([35.0, 36.0])
    CT = np.array([10.0, 20.0])

    p = np.array([0.0, 100.0])

    h = np.array(
        [
            100.0,
            300.0,
        ]
    )

    assert not is_stable(SA, CT, p)

    SA_new, CT_new = _convective_adjustment(SA, CT, p, h)

    expected_SA = np.average(SA, weights=h)
    expected_CT = np.average(CT, weights=h)

    np.testing.assert_allclose(
        SA_new,
        expected_SA,
    )

    np.testing.assert_allclose(
        CT_new,
        expected_CT,
    )


def test_xarray_wrapper():
    """The public wrapper should operate along the specified dimension."""

    SA = xr.DataArray(
        [
            [35.0, 35.0, 35.0],
            [35.0, 35.0, 35.0],
        ],
        dims=("x", "lev"),
    )

    CT = xr.DataArray(
        [
            [15.0, 10.0, 5.0],
            [10.0, 15.0, 5.0],
        ],
        dims=("x", "lev"),
    )

    p = xr.DataArray(
        [0.0, 100.0, 200.0],
        dims="lev",
    )

    h = xr.DataArray(
        [100.0, 100.0, 100.0],
        dims="lev",
    )

    SA_new, CT_new = convective_adjustment(
        SA,
        CT,
        p,
        h,
        dim="lev",
    )

    assert SA_new.dims == SA.dims
    assert CT_new.dims == CT.dims

    assert SA_new.shape == SA.shape
    assert CT_new.shape == CT.shape

    # First column should be unchanged.
    np.testing.assert_array_equal(
        SA_new[0],
        SA[0],
    )

    np.testing.assert_array_equal(
        CT_new[0],
        CT[0],
    )

    # Second column should have been adjusted.
    assert not np.array_equal(
        CT_new[1],
        CT[1],
    )
