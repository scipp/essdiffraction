import numpy as np
import scipp as sc

from .types import DetectorTofData, RunType, StreakClusteredData


def compute_tof_in_each_cluster(
    da: StreakClusteredData[RunType],
) -> DetectorTofData[RunType]:
    '''Fits a line through each cluster, the intercept of the line is t0.
    The line is fitted using linear regression with an outlier removal procedure.

    The algorithm is:

    1. Use least squares to fit line through clusters.
    2. Mask points that are "outliers" based on the criteria that they are too far
       from the line in the ``t`` variable.
       This means they don't seem to have the same time of flight origin as the rest
       of the points in the cluster, and probably should belong to another cluster or
       are part of the background.
    3. Go back to 1) and iterate until convergence. A few iterations should be enough.
    '''
    sin_theta_L = sc.sin(da.bins.coords['two_theta'] / 2) * da.bins.coords['L']
    t = da.bins.coords['t']
    for _ in range(5):
        s, t0 = _linear_regression_by_bin(sin_theta_L, t, da)

        s_left = sc.array(dims=s.dims, values=np.roll(s.values, 1), unit=s.unit)
        s_right = sc.array(dims=s.dims, values=np.roll(s.values, -1), unit=s.unit)
        t0_left = sc.array(dims=t0.dims, values=np.roll(t0.values, 1), unit=t0.unit)
        t0_right = sc.array(dims=t0.dims, values=np.roll(t0.values, -1), unit=t0.unit)

        # Distance from point to line through cluster
        distance_to_self = sc.abs(sc.values(t0) + sc.values(s) * sin_theta_L - t)
        # Distance from this cluster line to next before cluster line
        distance_self_to_left = sc.abs(
            sc.values(t0_left)
            + sc.values(s_left) * sin_theta_L
            - (sc.values(t0) + sc.values(s) * sin_theta_L)
        )
        # Distance from this cluster line to next after cluster line
        distance_self_to_right = sc.abs(
            sc.values(t0_right)
            + sc.values(s_right) * sin_theta_L
            - (sc.values(t0) + sc.values(s) * sin_theta_L)
        )

        da = da.bins.assign_masks(
            # TODO: Find suitable masking parameters for other chopper settings
            too_far_from_center=(distance_to_self > sc.scalar(3e-4, unit='s')).data,
            too_close_to_other=(
                (distance_self_to_left < sc.scalar(8e-4, unit='s'))
                | (distance_self_to_right < sc.scalar(8e-4, unit='s'))
            ).data,
        )

    da = da.assign_coords(t0=sc.values(t0).data)
    da = da.bins.assign_coords(tof=(t - sc.values(t0)).data)
    return da


def _linear_regression_by_bin(
    x: sc.Variable, y: sc.Variable, w: sc.DataArray
) -> tuple[sc.DataArray, sc.DataArray]:
    w = sc.values(w)
    tot_w = w.bins.sum()

    avg_x = (w * x).bins.sum() / tot_w
    avg_y = (w * y).bins.sum() / tot_w

    cov_xy = (w * (x - avg_x) * (y - avg_y)).bins.sum() / tot_w
    var_x = (w * (x - avg_x) ** 2).bins.sum() / tot_w

    b1 = cov_xy / var_x
    b0 = avg_y - b1 * avg_x

    return b1, b0


providers = (compute_tof_in_each_cluster,)
