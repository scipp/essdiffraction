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
    MinTimeToNextStreak,
    ModDt,
    ModShift,
    ModTwidth,
    RunType,
    SampleRun,
    Time0,
)


def load_mcstas(fname: Filename[SampleRun]) -> DetectorData[SampleRun]:
    return DetectorData[SampleRun](load_beer_mcstas(fname))


default_parameters = {
    ModTwidth: sc.scalar(0.003, unit='s'),
    ModShift: sc.scalar(5e-4),
    ModDt: sc.scalar(4.464e-4, unit='s'),
    Time0: sc.scalar(1.16 * 17 / 360 / 28, unit='s'),
    MaxTimeOffset: sc.scalar(3e-4, unit='s'),
    MinTimeToNextStreak: sc.scalar(8e-4, unit='s'),
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
