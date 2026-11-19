"""
vertical_anomaly_taper.py

Description: Fast vertical tapering utilities for ocean anomaly fields using
a fused NumPy/xarray ufunc kernel.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

_VALID_METHODS = ("linear", "double_tanh", "cosine", "quintic")


# -- Helper Functions -- #
def _resolve_dim_and_depth(
    data: xr.DataArray | xr.Dataset,
    dim: str,
    depth_coord: str | None,
) -> tuple[str, str]:
    """Resolve the vertical dimension name and 1D depth coordinate name."""
    if dim in data.dims:
        vert_dim = dim
        if depth_coord is None:
            if dim in data.coords:
                z_coord = dim
            elif "depth" in data.coords and data["depth"].dims == (vert_dim,):
                z_coord = "depth"
            else:
                raise ValueError(
                    f"No 1D depth coordinate found for dimension {dim!r}; "
                    "pass ``depth_coord`` explicitly."
                )
        else:
            z_coord = depth_coord
    elif dim in data.coords and data[dim].ndim == 1:
        z_coord = dim
        vert_dim = str(data[dim].dims[0])
    else:
        raise ValueError(
            f"{dim!r} must be a vertical dimension or a 1D depth coordinate in input data."
        )

    if z_coord not in data.coords or data[z_coord].dims != (vert_dim,):
        raise ValueError(
            f"Depth coordinate {z_coord!r} must be a 1D coordinate along dimension {vert_dim!r}."
        )

    return vert_dim, z_coord


def _compute_weights_np(
    s: np.ndarray,
    method: str,
    tanh_steepness: float,
) -> np.ndarray:
    """Compute taper weights for normalized depth ``s`` in ``[0, 1]``."""
    if method == "linear":
        return 1.0 - s
    if method == "double_tanh":
        a = tanh_steepness
        return (np.tanh(a) - np.tanh(a * (2.0 * s - 1.0))) / (2.0 * np.tanh(a))
    if method == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * s))
    if method == "quintic":
        return 1.0 - (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3)

    raise ValueError(
        f"Unknown tapering method {method!r}. Expected one of {_VALID_METHODS}."
    )


def _taper_numpy(
    arr: np.ndarray,
    z: np.ndarray,
    lengthscale: float,
    method: str,
    tanh_steepness: float,
    min_depth: float | None,
    max_depth: float | None,
) -> np.ndarray:
    """Fused NumPy kernel to spread and taper along the last axis."""
    descending = z.size > 1 and z[-1] < z[0]
    if descending:
        arr = arr[..., ::-1]
        z = z[::-1]

    out = arr.copy()
    n_lev = arr.shape[-1]
    finite_mask = np.isfinite(arr)
    valid = finite_mask & (arr != 0.0)
    has_obs = valid.any(axis=-1)

    if np.any(has_obs):
        # Deepest observed (non-zero, finite) index, depth, and value per column
        k_deep = (n_lev - 1) - np.argmax(valid[..., ::-1], axis=-1)
        z_deep = z[k_deep]
        v_deep = np.take_along_axis(arr, k_deep[..., None], axis=-1)[..., 0]

        col_active = has_obs
        if min_depth is not None:
            col_active = col_active & (z_deep >= min_depth)

        z_deep_b = z_deep[..., None]
        spread_mask = (
            col_active[..., None]
            & (z > z_deep_b)
            & (z <= z_deep_b + lengthscale)
            & (arr == 0.0)
        )

        if np.any(spread_mask):
            v_broad = np.broadcast_to(v_deep[..., None], arr.shape)[spread_mask]
            z_broad = np.broadcast_to(z, arr.shape)[spread_mask]
            zd_broad = np.broadcast_to(z_deep_b, arr.shape)[spread_mask]
            s = np.clip((z_broad - zd_broad) / lengthscale, 0.0, 1.0)
            w = _compute_weights_np(s, method=method, tanh_steepness=tanh_steepness)
            out[spread_mask] = (v_broad * w).astype(out.dtype, copy=False)

    if max_depth is not None:
        out[(z > max_depth) & finite_mask] = 0.0

    if descending:
        out = out[..., ::-1]

    return out


def _taper_dataarray(
    da: xr.DataArray,
    lengthscale: float,
    dim: str,
    depth_coord: str,
    method: str,
    tanh_steepness: float,
    min_depth: float | None,
    max_depth: float | None,
) -> xr.DataArray:
    """Apply ``_taper_numpy`` across ``da`` while preserving dimension order."""
    # Core dim must live in a single Dask chunk for apply_ufunc
    if da.chunks is not None and len(da.chunks[da.get_axis_num(dim)]) > 1:
        da = da.chunk({dim: -1})

    out = xr.apply_ufunc(
        _taper_numpy,
        da,
        da[depth_coord],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        kwargs={
            "lengthscale": lengthscale,
            "method": method,
            "tanh_steepness": tanh_steepness,
            "min_depth": min_depth,
            "max_depth": max_depth,
        },
        dask="parallelized",
        output_dtypes=[da.dtype],
        keep_attrs=True,
    )

    out = out.transpose(*da.dims)
    assert out.shape == da.shape
    return out


# -- Public Functions -- #
def taper_anomalies(
    data: xr.DataArray | xr.Dataset,
    lengthscale: float = 500.0,
    dim: str = "lev",
    depth_coord: str | None = None,
    method: Literal["linear", "double_tanh", "cosine", "quintic"] = "cosine",
    tanh_steepness: float = 3.0,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Spread the deepest observed anomaly downward over ``lengthscale`` and taper to zero.

    Unobserved deep ocean cells are assumed to be ``0.0`` and bathymetry/land
    cells are assumed to be ``NaN``. For each water column, the deepest finite,
    non-zero value (at depth ``z_deep``) is propagated downward across
    ``(z_deep, z_deep + lengthscale]`` and smoothly tapered from 1 to 0.
    Bathymetry ``NaN`` masks and all-zero columns are preserved.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Anomaly field containing vertical dimension ``dim`` (or 1D depth
        coordinate ``dim``, e.g., ``"depth"`` on dimension ``"lev"``).
    lengthscale : float, optional
        Vertical distance over which the deepest observed anomaly is spread
        and tapered to zero, by default 500.0.
    dim : str, optional
        Name of the vertical dimension (e.g., ``"lev"``) or 1D depth
        coordinate (e.g., ``"depth"``), by default ``"lev"``.
    depth_coord : str or None, optional
        Name of the 1D depth coordinate along ``dim`` (positive downward).
        If ``None``, inferred from ``dim`` or ``"depth"``, by default ``None``.
    method : {"linear", "double_tanh", "cosine", "quintic"}, optional
        Tapering profile to apply across ``lengthscale``, by default ``"cosine"``.
    tanh_steepness : float, optional
        Steepness parameter for ``"double_tanh"`` (ignored by other methods),
        by default 3.0.
    min_depth : float or None, optional
        Minimum ``z_deep`` required to trigger downward spreading (e.g., ``1500.0``
        to avoid spreading shallow-only profiles). If ``None``, any deepest
        non-zero value is spread, by default ``None``.
    max_depth : float or None, optional
        Hard cutoff depth below which ocean anomalies are forced to ``0.0``
        (bathymetry ``NaN`` values remain ``NaN``), by default ``None``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Tapered anomaly field with the same type, dimensions, and shape as ``data``.

    Raises
    ------
    ValueError
        If ``dim``/``depth_coord`` cannot be resolved, ``lengthscale`` or
        ``tanh_steepness`` is not positive, ``method`` is unknown, or
        ``max_depth <= min_depth``.
    """
    # -- Validate Inputs -- #
    vert_dim, z_coord = _resolve_dim_and_depth(data, dim=dim, depth_coord=depth_coord)
    if lengthscale <= 0.0:
        raise ValueError("``lengthscale`` must be strictly positive.")
    if tanh_steepness <= 0.0:
        raise ValueError("``tanh_steepness`` must be strictly positive.")
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown tapering method {method!r}. Expected one of {_VALID_METHODS}."
        )
    if min_depth is not None and max_depth is not None and max_depth <= min_depth:
        raise ValueError("``max_depth`` must be strictly greater than ``min_depth``.")

    # -- Apply Tapering -- #
    if isinstance(data, xr.Dataset):
        return data.map(
            lambda da: (
                _taper_dataarray(
                    da,
                    lengthscale=lengthscale,
                    dim=vert_dim,
                    depth_coord=z_coord,
                    method=method,
                    tanh_steepness=tanh_steepness,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
                if vert_dim in da.dims
                else da
            )
        )

    return _taper_dataarray(
        data,
        lengthscale=lengthscale,
        dim=vert_dim,
        depth_coord=z_coord,
        method=method,
        tanh_steepness=tanh_steepness,
        min_depth=min_depth,
        max_depth=max_depth,
    )
