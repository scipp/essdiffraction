# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

from .types import (
    DspacingBins,
    DspacingData,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
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
) -> FocussedDataDspacingTwoTheta[RunType]:
    # Note the binning in two steps: Since two_theta is only pixel-dependent it is
    # more efficient, since it avoids having to map two_theta to each event, or a
    # costly many-to-many binning operation.
    return FocussedDataDspacingTwoTheta[RunType](
        data.bin({twotheta_bins.dim: twotheta_bins}).bin(
            {dspacing_bins.dim: dspacing_bins}
        )
    )


providers = (focus_data_dspacing, focus_data_dspacing_and_two_theta)
"""Sciline providers for grouping pixels."""
