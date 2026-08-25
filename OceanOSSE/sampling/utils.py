"""
utils.py

Description: Sampling utility module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree


def _collect_obs_profiles(
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: tuple | None = None
    ) -> xr.Dataset:
    """
    Create an xarray.Dataset of historical Argo profile observations.

    Parameters
    ----------
    start_date : str | None, optional
        Start date for the time period (inclusive), by default None
    end_date : str | None, optional
        End date for the time period (inclusive), by default None
    bbox : tuple | None, optional
        Subregion of the model grid with lat/lon bounding box in the form 
        (min_lon, min_lat, max_lon, max_lat), by default None

    Returns
    -------
    xr.Dataset
        Historical Argo profile times & locations.
    """
    # --- Open Argo Profiling DataFrame --- #
    # Open Argo profile database from JASMIN OS:
    df = pd.read_parquet("https://noc-msm-o.s3-ext.jc.rl.ac.uk/ocean-obs/OceanOSSE/EN.4.2.2.f.profiles.g10.2001_2026.parquet")
    df["month"] = df["JULD"].dt.month

    if start_date is not None:
        df = df[df["JULD"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["JULD"] <= pd.Timestamp(end_date)]

    if bbox is not None:
        df = df[
            (df["LATITUDE"] >= bbox[1])
            & (df["LATITUDE"] <= bbox[3])
            & (df["LONGITUDE"] >= bbox[0])
            & (df["LONGITUDE"] <= bbox[2])
        ]

    if len(df) == 0:
        raise ValueError(
            f"No Argo profiles found within time period {start_date or df['JULD'].min()} to {end_date or df['JULD'].max()}."
        )

    # -- Construct xarray Dataset of Historical Argo profiles -- #
    ds_out = xr.Dataset(data_vars=dict(lon=(("profile_id"), df["LONGITUDE"]),
                                       lat=(("profile_id"), df["LATITUDE"]),
                                       time=(("profile_id"), df["JULD"])),
                        coords=dict(profile_id=df["ID"])
                        )

    return ds_out

def _collect_obs_probability(
    ds_mdl : xr.Dataset,
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: tuple | None = None
    ) -> xr.Dataset:
    """
    Create an xarray.Dataset of monthly count & occurrence probability maps
    defined on a target ocean model grid from Argo profiling observations.

    Parameters
    ----------
    ds_mdl : xr.Dataset
        Ocean model dataset defining the target ocean model grid. Must contain
        longitude ("lon"), latitude ("lat"), and land-sea mask ("mask") variables.
    start_date : str | None, optional
        Start date for the time period (inclusive), by default None.
    end_date : str | None, optional
        End date for the time period (inclusive), by default None.
    bbox : tuple | None, optional
        Geographical region to select Argo profiles from given by a bounding
        box in the form (min_lon, min_lat, max_lon, max_lat), by default None.

    Returns
    -------
    xr.Dataset
        Count & probability maps defined on a target ocean model grid.
    """
    # -- Collect Ocean Model Variables -- #
    longitude = ds_mdl["lon"].values
    latitude = ds_mdl["lat"].values
    mask = ds_mdl["mask"].isel(lev=0).values

    # --- Optionally Apply Region to Land-Sea Mask --- #
    # Defining shape of geographical coordinates:
    if longitude.shape != latitude.shape:
        raise ValueError(
            "Longitude and latitude coordinates of the ocean model grid must have the same shape."
        )
    ny, nx = longitude.shape

    # --- Collect Ocean Grid Points --- #
    # Collect indices of wet grid points within regional subdomain:
    index_flat_idx = np.flatnonzero(mask.ravel())
    grid_points = np.column_stack(
        [latitude.ravel()[index_flat_idx], longitude.ravel()[index_flat_idx]]
    )

    # GeoBallTree: haversine-distance BallTree over wet ocean points only:
    geoballtree = BallTree(np.deg2rad(grid_points), metric="haversine")

    # --- Open Argo Profiling DataFrame --- #
    # Open Argo profile database from JASMIN OS:
    df = pd.read_parquet("https://noc-msm-o.s3-ext.jc.rl.ac.uk/ocean-obs/OceanOSSE/EN.4.2.2.f.profiles.g10.2001_2026.parquet")
    df["month"] = df["JULD"].dt.month

    if start_date is not None:
        df = df[df["JULD"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["JULD"] <= pd.Timestamp(end_date)]

    if bbox is not None:
        df = df[
            (df["LATITUDE"] >= bbox[1])
            & (df["LATITUDE"] <= bbox[3])
            & (df["LONGITUDE"] >= bbox[0])
            & (df["LONGITUDE"] <= bbox[2])
        ]

    if len(df) == 0:
        raise ValueError(
            f"No Argo profiles found within time period {start_date or df['JULD'].min()} to {end_date or df['JULD'].max()}."
        )

    # -- Identify ocean model grid  (j, i) indices for each Argo profile -- #
    # Define Argo profile coordinates:
    obs_points = np.column_stack([df["LATITUDE"], df["LONGITUDE"]])
    # Find nearest wet grid cell for each Argo profile:
    ind = geoballtree.query(np.deg2rad(obs_points), k=1, return_distance=False)

    # Transform (regional subdomain) wet grid point indices to ocean model grid indices:
    flat_idx_full = index_flat_idx[ind[:, 0]]
    iy, ix = np.unravel_index(flat_idx_full, latitude.shape)
    df["iy"] = iy
    df["ix"] = ix

    # -- Count Argo profiles for each month -- #
    counts = np.zeros((12, ny, nx), dtype=np.int32)
    for m in range(1, 13):
        df_month = df[df["month"] == m]
        if len(df_month) == 0:
            continue
        # Add counts of Argo profiles to wet grid cells:
        np.add.at(counts[m - 1], (df_month["iy"].values, df_month["ix"].values), 1)

    # -- Construct xarray Dataset of monthly probabilities -- #
    # Create Dataset with total monthly Argo profile counts:
    ds_out = xr.Dataset(
        {
         "counts": (["month", "j", "i"], counts)
         },
        coords={
            "month": np.arange(1, 13),
            "lat": (["j", "i"], latitude),
            "lon": (["j", "i"], longitude),
        },
    )
    # Add `count` variable attributes:
    ds_out["counts"].attrs = {
        "long_name": "Monthly Argo Profile Count",
        "units": "1",
    }

    # Compute monthly Argo profile occurrence probability:
    ds_out["probability"] = ds_out["counts"] / ds_out["counts"].sum(dim=("j", "i"))
    # Add `probability` variable attributes:
    ds_out["probability"].attrs = {
        "long_name": "Monthly Argo Profile Occurrence Probability",
        "units": "1",
    }

    return ds_out

def _extract_locations_ij(ds_mdl, i_index, j_index, time_index):
    """
    Extract a model profile at the specified model index. The input indices
    need to have dims="profile_id", coords={"profile_id": prof_id}.

    Parameters
    ----------
    ds_mdl : xarray.Dataset
        Gridded ocean model dataset.
    i_index : observation index on model grid in i direction
    j_index : observation index on model grid in j direction
    time_index : observation index in time

    Return
    ------
    xarray.Dataset
        Synthetic profile dataset.
    """

    ds_model_profile = ds_mdl.isel(i=i_index, j=j_index, time=time_index)

    return ds_model_profile
