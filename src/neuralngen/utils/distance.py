# src/neuralngen/utils/distance.py

import numpy as np
import torch


def haversine_distance(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """
    Compute great-circle distance (Haversine formula) between two points in degrees.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 (in degrees)
    lat2, lon2 : float
        Latitude and longitude of point 2 (in degrees)
    radius_km : float
        Radius of the Earth in kilometers.

    Returns
    -------
    float
        Distance in kilometers
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius_km * c


def compute_distance_matrix(x_info, normalize=False):
    """
    Compute pairwise distance matrix between all sites in x_info.

    Parameters
    ----------
    x_info_list : list of dict
        Each dict contains:
        - "gauge_id"
        - "gauge_lat"
        - "gauge_lon"
    normalize : bool
        If True, returns matrix scaled between 0 and 1.

    Returns
    -------
    torch.Tensor
        Distance matrix of shape (n_sites, n_sites), dtype=float32
    """
    # Convert lat/lon to numpy arrays
    lats = x_info["gauge_lat"].cpu().numpy()
    lons = x_info["gauge_lon"].cpu().numpy()

    n_sites = len(lats)

    dist_mat = np.zeros((n_sites, n_sites), dtype=np.float32)

    for i in range(n_sites):
        for j in range(n_sites):
            dist_mat[i, j] = haversine_distance(lats[i], lons[i], lats[j], lons[j])

    if normalize:
        max_val = dist_mat.max()
        if max_val > 0:
            dist_mat = dist_mat / max_val

    return torch.tensor(dist_mat, dtype=torch.float32)
