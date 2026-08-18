import xarray as xr

def extract_locations_ij(ds, i_index, j_index, t_index):
    """
    Extract a model profile at the specified model index. The input indices
    need to have dims="profile_id", coords={"profile_id": prof_id}.

    Parameters
    ----------
    ds : xarray.Dataset
        Gridded ocean model dataset.
    i_index : observation index on model grid in i direction
    j_index : observation index on model grid in j direction
    t_index : observation index in time

    Return
    xarray.Dataset
        Model profile dataset
    """

    ds_model_profile = ds.isel(i=i_index, j=j_index, t=t_index)

    return ds_model_profile