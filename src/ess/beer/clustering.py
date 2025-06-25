import scipp as sc
from scipp import constants
from scipy.signal import find_peaks

from .types import DetectorData, RunType, StreakClusteredData


def cluster_events_by_streak(da: DetectorData[RunType]) -> StreakClusteredData[RunType]:
    da = da.copy(deep=False)
    da.coords['coarse_d'] = (
        constants.h
        / constants.m_n
        * (da.coords['t'] - da.coords['approximate_t0'])
        / sc.sin(da.coords['two_theta'] / 2)
        / da.coords['L']
        / 2
    ).to(unit='angstrom')

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
    b = b.drop_coords(('coarse_d',))
    b = b.bins.drop_coords(('coarse_d',))
    b = b.rename_dims(coarse_d='streak')
    return b


providers = (cluster_events_by_streak,)
