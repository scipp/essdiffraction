import scipp as sc
from scipy.signal import find_peaks


def cluster_events_by_streak(da: sc.DataArray) -> sc.DataArray:
    h = da.hist(coarse_d=1000)
    i_peaks, _ = find_peaks(h.data.values, height=40, distance=3)
    i_valleys, _ = find_peaks(
        h.data.values.max() - h.data.values, distance=3, height=h.data.values.max() / 2
    )

    valleys = sc.array(
        dims=['coarse_d'],
        values=h.coords['coarse_d'].values[i_valleys],
        unit=h.coords['coarse_d'].unit,
    )
    peaks = sc.array(
        dims=['coarse_d'],
        values=h.coords['coarse_d'].values[i_peaks],
        unit=h.coords['coarse_d'].unit,
    )

    has_peak = peaks.bin(coarse_d=valleys).bins.size().data.to(dtype='bool')
    has_peak_left = sc.concat(
        (has_peak, sc.array(dims=['coarse_d'], values=[False], unit=None)), 'coarse_d'
    )
    has_peak_right = sc.concat(
        (
            sc.array(dims=['coarse_d'], values=[False], unit=None),
            has_peak,
        ),
        'coarse_d',
    )
    filtered_valleys = valleys[has_peak_left | has_peak_right]
    has_peak = peaks.bin(coarse_d=filtered_valleys).bins.size().data
    b = da.bin(coarse_d=filtered_valleys).assign_masks(
        no_peak=has_peak != sc.scalar(1, unit=None)
    )
    b.coords['peak'] = (
        peaks.bin(coarse_d=filtered_valleys).bins.coords['coarse_d'].bins.mean()
    )
    return b


def compute_tof_in_each_cluster(b: sc.DataArray) -> None:
    '''Fits a line through each cluster, the intercept of the line is t0.
    The line is fitted using linear regression with an outlier removal procedure.

    The algorithm is:

    1. Fit line through clusters.
    2. Mask points that are "outliers" based on the criteria that they are too far
       from the line in the ``t`` variable.
       This means they don't seem to have the same time of flight origin as the rest
       of the points in the cluster.
    3. Go back to 1. and iterate until convergence. A few iterations should be enough.
    '''
    for _ in range(3):
        s, t0 = _linear_regression_by_bin(
            b.bins.coords['sin_theta_L'], b.bins.coords['t'], b
        )
        b.bins.masks['too_far_from_center'] = (
            sc.abs(
                sc.values(t0)
                + sc.values(s) * b.bins.coords['sin_theta_L']
                - b.bins.coords['t']
            )
            > sc.scalar(3e-4, unit='s')
        ).data

    b.coords['t0'] = sc.values(t0).data
    b.bins.coords['tof'] = (b.bins.coords['t'] - sc.values(t0)).data


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
