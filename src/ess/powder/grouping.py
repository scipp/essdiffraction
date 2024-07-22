# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import scipp as sc

from .types import (
    DspacingBins,
    DspacingData,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    RunType,
    TwoThetaBins,
)


def _drop_grouping_and_bin(
    data: sc.DataArray, *, dims_to_reduce: tuple[str, ...] | None = None, edges: dict
) -> sc.DataArray:
    all_pixels = data if dims_to_reduce == () else data.bins.concat(dims_to_reduce)
    # all_pixels may just have a single bin now, which currently yields
    # inferior performance when binning (no/bad multi-threading?).
    # We operate on the content buffer for better multi-threaded performance.
    if all_pixels.ndim == 0:
        return (
            all_pixels.value.bin(**edges)
            .assign_coords(all_pixels.coords)
            .assign_masks(all_pixels.masks)
        )
    else:
        return all_pixels.bin(**edges)


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
) -> FocussedDataDspacingTwoTheta[RunType]:
    return FocussedDataDspacingTwoTheta[RunType](
        data.bin(
            {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins},
            dim=data.dims,
        )
    )


providers = (focus_data_dspacing, focus_data_dspacing_and_two_theta)
"""Sciline providers for grouping pixels."""
