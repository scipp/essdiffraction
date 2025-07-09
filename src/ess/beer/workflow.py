# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl
import scipp as sc

from .clustering import providers as clustering_providers
from .conversions import convert_from_known_peaks_providers
from .conversions import providers as conversion_providers
from .io import load_beer_mcstas
from .types import (
    DetectorData,
    Filename,
    MaxTimeOffset,
    ModDt,
    ModShift,
    ModTwidth,
    RunType,
    SampleRun,
    Time0,
    TwoThetaMaskFunction,
)


def load_mcstas(
    fname: Filename[SampleRun], two_theta_mask: TwoThetaMaskFunction
) -> DetectorData[SampleRun]:
    da = DetectorData[SampleRun](load_beer_mcstas(fname))
    da = (
        sc.DataGroup(
            {
                k: v.assign_masks(two_theta=two_theta_mask(v.coords['two_theta']))
                for k, v in da.items()
            }
        )
        if isinstance(da, sc.DataGroup)
        else da.assign_masks(two_theta=two_theta_mask(da.coords['two_theta']))
    )
    return da


default_parameters = {
    ModTwidth: sc.scalar(0.003, unit='s'),
    ModShift: sc.scalar(5e-4),
    ModDt: sc.scalar(4.464e-4, unit='s'),
    Time0: sc.scalar(1.16 * 17 / 360 / 28, unit='s'),
    MaxTimeOffset: sc.scalar(3e-4, unit='s'),
    TwoThetaMaskFunction: lambda two_theta: (
        (two_theta >= sc.scalar(105, unit='deg').to(unit='rad', dtype='float64'))
        | (two_theta <= sc.scalar(75, unit='deg').to(unit='rad', dtype='float64'))
    ),
}


def BeerMcStasWorkflow():
    return sl.Pipeline(
        (load_mcstas, *clustering_providers, *conversion_providers),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )


def BeerMcStasWorkflowKnownPeaks():
    return sl.Pipeline(
        (load_mcstas, *convert_from_known_peaks_providers),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )
