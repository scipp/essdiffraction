# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import numpy as np
import scipp as sc

from .types import (
    CorrectedDetector,
    CorrectedDspacing,
    DspacingBins,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    KeepEvents,
    NormalizedDspacing,
    RunType,
    TwoThetaBins,
)


def _reconstruct_wavelength(
    dspacing_bins: DspacingBins, two_theta_bins: TwoThetaBins
) -> sc.Variable:
    dspacing = dspacing_bins
    two_theta = sc.midpoints(two_theta_bins)
    return (2 * dspacing * sc.sin(two_theta / 2)).to(unit='angstrom')


def focus_data_dspacing(
    data: CorrectedDetector[RunType],
    dspacing_bins: DspacingBins,
    # keep_events: KeepEvents[RunType],
) -> CorrectedDspacing[RunType]:
    """
    Reduce the pixel-based data to d-spacing dimension.
    We add a two-theta coordinate to the events, as it is needed further down the
    workflow for focussing in two-theta.

    Parameters
    ----------
    data:
        The input data to be reduced, which must have 'wavelength', 'dspacing',
        'two_theta' coordinates.
    dspacing_bins:
        The bins to use for the d-spacing dimension.
    # keep_events:
    #     Whether to keep the events in the output. If `False`, the output will be
    #     histogrammed instead of binned.

    Returns
    -------
    :
        The reduced data with 'dspacing' and 'two_theta' dimensions.
    """
    with_two_theta = data.bins.assign_coords(
        two_theta=sc.bins_like(data.bins.coords['dspacing'], data.coords['two_theta'])
    )
    # Use the `dim` argument to remove all existing dims and keep only dspacing
    return CorrectedDspacing[RunType](
        with_two_theta.bin({dspacing_bins.dim: dspacing_bins}, dim=data.dims)
    )

    ttheta = data.coords['two_theta']
    ttheta_min = ttheta.nanmin()
    ttheta_max = ttheta.nanmax()
    ttheta_max.value = np.nextafter(ttheta_max.value, np.inf)
    twotheta_bins = sc.linspace(
        'two_theta',
        start=ttheta_min,
        stop=ttheta_max,
        num=1024,
        unit=ttheta.unit,
    )
    args = {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins}
    if keep_events.value:
        result = data.bin(args)
    else:
        # Reconstructing the wavelength results in an inconsistency if dspacing was
        # computed with a calibration table. Another option would be to use, e.g., the
        # mean wavelength in each bin, but this leads to random wavelength values that
        # break stream processing.
        result = data.hist(args).assign_coords(
            wavelength=_reconstruct_wavelength(
                dspacing_bins=dspacing_bins, two_theta_bins=twotheta_bins
            )
        )

    return CorrectedDspacing[RunType](result)


def integrate_two_theta(
    data: NormalizedDspacing[RunType],
) -> FocussedDataDspacing[RunType]:
    """Integrate the two-theta dimension of the data."""
    if 'two_theta' not in data.dims:
        raise sc.DimensionError("Data does not have a 'two_theta' dimension.")
    return FocussedDataDspacing[RunType](
        data.nansum(dim='two_theta')
        if data.bins is None
        else data.bins.concat('two_theta')
    )


def group_two_theta(
    data: NormalizedDspacing[RunType],
    two_theta_bins: TwoThetaBins,
    keep_events: KeepEvents[RunType],
) -> FocussedDataDspacingTwoTheta[RunType]:
    """Group the data by two-theta bins."""
    out = data.bin(two_theta=two_theta_bins)
    if not keep_events.value:
        out = out.hist()
    return FocussedDataDspacingTwoTheta[RunType](
        out.transpose(['two_theta', 'dspacing'])
    )

    # if 'two_theta' not in data.dims:
    #     raise ValueError("Data does not have a 'two_theta' dimension.")
    # data = data.assign_coords(two_theta=sc.midpoints(data.coords['two_theta']))
    # return FocussedDataDspacingTwoTheta[RunType](
    #     data.groupby('two_theta', bins=two_theta_bins).nansum('two_theta')
    #     if data.bins is None
    #     else data.bin(two_theta=two_theta_bins)
    # )


def collect_detectors(*detectors: sc.DataArray) -> sc.DataGroup:
    """Store all inputs in a single data group.

    This function is intended to be used to reduce a workflow which
    was mapped over detectors.

    Parameters
    ----------
    detectors:
        Data arrays for each detector bank.
        All arrays must have a scalar "detector" coord containing a ``str``.

    Returns
    -------
    :
        The inputs as a data group with the "detector" coord as the key.
    """
    return sc.DataGroup({da.coords.pop('detector').value: da for da in detectors})


providers = (
    focus_data_dspacing,
    integrate_two_theta,
    group_two_theta,
)
"""Sciline providers for grouping pixels."""
