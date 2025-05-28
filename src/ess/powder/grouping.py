# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import scipp as sc

from .types import (
    DspacingBins,
    DspacingData,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    KeepEvents,
    NormalizedDspacing,
    NormalizedDspacingTwoTheta,
    NormalizedRunData,
    RunType,
    TwoThetaBins,
)


def focus_data_dspacing(
    data: DspacingData[RunType], dspacing_bins: DspacingBins
) -> FocussedDataDspacing[RunType]:
    return FocussedDataDspacing[RunType](
        data.bin({dspacing_bins.dim: dspacing_bins}, dim=data.dims)
    )


def focus_data_dspacing_and_two_theta(
    data: DspacingData[RunType],
    dspacing_bins: DspacingBins,
    twotheta_bins: TwoThetaBins,
    keep_events: KeepEvents[RunType],
) -> FocussedDataDspacingTwoTheta[RunType]:
    # TODO Use finer binning for two-theta if available. Or just use limits?
    args = {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins}
    if keep_events.value:
        result = data.bin(args)
    else:
        result = data.hist(args)
        # result.coords['wavelength'] = result.bins.coords['wavelength'].bins.nanmean()
        # result.coords['wavelength_delta'] = (
        #    result.bins.coords['wavelength'].bins.nanmax()
        #    - result.bins.coords['wavelength'].bins.nanmin()
        # )
        # result = result.hist()
    return FocussedDataDspacingTwoTheta[RunType](result)


def integrate_two_theta(
    data: NormalizedRunData[RunType],
) -> NormalizedDspacing[RunType]:
    """Integrate the two-theta dimension of the data."""
    if 'two_theta' not in data.dims:
        raise ValueError("Data does not have a 'two_theta' dimension.")
    return NormalizedDspacing[RunType](
        data.nansum(dim='two_theta')
        if data.bins is None
        else data.bins.concat('two_theta')
    )


def group_two_theta(
    data: NormalizedRunData[RunType],
    two_theta_bins: TwoThetaBins,
) -> NormalizedDspacingTwoTheta[RunType]:
    """Group the data by two-theta bins."""
    if 'two_theta' not in data.dims:
        raise ValueError("Data does not have a 'two_theta' dimension.")
    # two_theta_bins = sc.linspace('two_theta_group', 0.0, 180.0, num=10, unit='deg')
    data = data.assign_coords(two_theta=sc.midpoints(data.coords['two_theta']))
    # TODO make sure that for event data we are using the two_theta event coord
    groups = data.groupby('two_theta', bins=two_theta_bins)
    return NormalizedDspacingTwoTheta[RunType](
        groups.nansum('two_theta') if data.bins is None else groups.concat('two_theta')
    )


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
    focus_data_dspacing_and_two_theta,
    integrate_two_theta,
    group_two_theta,
)
"""Sciline providers for grouping pixels."""
