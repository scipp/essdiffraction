import scipp as sc
import scipp.constants

from .types import (
    DetectorData,
    DetectorTofData,
    DHKLList,
    MaxTimeOffset,
    ModDt,
    ModShift,
    ModTwidth,
    RunType,
    StreakClusteredData,
    Time0,
    TofCoordTransformGraph,
)


def compute_tof_in_each_cluster(
    da: StreakClusteredData[RunType],
    max_distance_from_streak_line: MaxTimeOffset,
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
    if isinstance(da, sc.DataGroup):
        return sc.DataGroup(
            {
                k: compute_tof_in_each_cluster(v, max_distance_from_streak_line)
                for k, v in da.items()
            }
        )

    sin_theta_L = sc.sin(da.bins.coords['two_theta'] / 2) * da.bins.coords['Ltotal']
    t = da.bins.coords['event_time_offset']
    for _ in range(15):
        s, t0 = _linear_regression_by_bin(sin_theta_L, t, da.data)

        # Distance from point to line through cluster
        distance_to_self = sc.abs(sc.values(t0) + sc.values(s) * sin_theta_L - t)

        da = da.bins.assign_masks(
            too_far_from_center=(distance_to_self > max_distance_from_streak_line),
        )

    da = da.assign_coords(t0=sc.values(t0))
    da = da.bins.assign_coords(tof=(t - sc.values(t0)))
    return da


def _linear_regression_by_bin(
    x: sc.Variable, y: sc.Variable, w: sc.Variable
) -> tuple[sc.Variable, sc.Variable]:
    '''Performs a weighted linear regression of the points
    in the binned variables ``x`` and ``y`` weighted by ``w``.
    Returns ``b1`` and ``b0`` such that ``y = b1 * x + b0``.
    '''
    w = sc.values(w)
    tot_w = w.bins.sum()

    avg_x = (w * x).bins.sum() / tot_w
    avg_y = (w * y).bins.sum() / tot_w

    cov_xy = (w * (x - avg_x) * (y - avg_y)).bins.sum() / tot_w
    var_x = (w * (x - avg_x) ** 2).bins.sum() / tot_w

    b1 = cov_xy / var_x
    b0 = avg_y - b1 * avg_x

    return b1, b0


def _compute_d(
    event_time_offset: sc.Variable,
    theta: sc.Variable,
    dhkl_list: sc.Variable,
    mod_twidth: sc.Variable,
    mod_shift: sc.Variable,
    L0: sc.Variable,
) -> sc.Variable:
    """Determines the ``d_hkl`` peak each event belongs to,
    given a list of known peaks."""
    # Source: https://www2.mcstas.org/download/components/3.4/contrib/NPI_tof_dhkl_detector.comp
    sinth = sc.sin(theta)
    t = event_time_offset

    d = sc.empty(dims=sinth.dims, shape=sinth.shape, unit=dhkl_list[0].unit)
    d[:] = sc.scalar(float('nan'), unit=dhkl_list[0].unit)
    dtfound = sc.empty(dims=sinth.dims, shape=sinth.shape, dtype='float64', unit=t.unit)
    dtfound[:] = sc.scalar(float('nan'), unit=t.unit)

    const = (
        2 * (1.0 + mod_shift) * sinth * L0 / (scipp.constants.h / scipp.constants.m_n)
    ).to(unit=f'{event_time_offset.unit}/angstrom')

    for dhkl in dhkl_list:
        dt = sc.abs(t - dhkl * const)
        dt_in_range = dt < mod_twidth / 2
        no_dt_found = sc.isnan(dtfound)
        dtfound = sc.where(dt_in_range, sc.where(no_dt_found, dt, dtfound), dtfound)
        d = sc.where(
            dt_in_range,
            sc.where(no_dt_found, dhkl, sc.scalar(float('nan'), unit=dhkl.unit)),
            d,
        )

    return d


def _tof_from_dhkl(
    event_time_offset: sc.Variable,
    theta: sc.Variable,
    coarse_dhkl: sc.Variable,
    Ltotal: sc.Variable,
    mod_shift: sc.Variable,
    mod_dt: sc.Variable,
    time0: sc.Variable,
) -> sc.Variable:
    '''Computes tof for BEER given the dhkl peak that the event belongs to'''
    # Source: https://www2.mcstas.org/download/components/3.4/contrib/NPI_tof_dhkl_detector.comp
    # tref = 2 * (1.0 + mod_shift) * d_hkl * sin(theta) / hm * Ltotal
    # tc = event_time_zero - time0 - tref
    # dt = floor(tc / mod_dt + 0.5) * mod_dt + time0
    # tof = event_time_offset - dt
    c = (-2 * (1.0 + mod_shift) / (scipp.constants.h / scipp.constants.m_n)).to(
        unit=f'{event_time_offset.unit}/m/angstrom'
    )
    out = sc.sin(theta)
    out *= c
    out *= coarse_dhkl
    out *= Ltotal
    out += event_time_offset
    out -= time0
    out /= mod_dt
    out += 0.5
    sc.floor(out, out=out)
    out *= mod_dt
    out += time0
    out *= -1
    out += event_time_offset
    return out


def tof_from_known_dhkl_graph(
    mod_shift: ModShift,
    mod_dt: ModDt,
    mod_twidth: ModTwidth,
    time0: Time0,
    dhkl_list: DHKLList,
) -> TofCoordTransformGraph:
    return {
        'mod_twidth': lambda: mod_twidth,
        'mod_dt': lambda: mod_dt,
        'time0': lambda: time0,
        'mod_shift': lambda: mod_shift,
        'tof': _tof_from_dhkl,
        'coarse_dhkl': _compute_d,
        'theta': lambda two_theta: two_theta / 2,
        'dhkl_list': lambda: dhkl_list,
    }


def compute_tof_from_known_peaks(
    da: DetectorData[RunType], graph: TofCoordTransformGraph
) -> DetectorTofData[RunType]:
    return da.transform_coords(('tof',), graph=graph, keep_intermediate=False)


convert_from_known_peaks_providers = (
    tof_from_known_dhkl_graph,
    compute_tof_from_known_peaks,
)
providers = (compute_tof_in_each_cluster,)
