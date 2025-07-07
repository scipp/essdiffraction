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
    DHKLList,
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
    DHKLList: sc.array(
        dims=('peaks',),
        values=[
            2.1055,
            2.0407,
            1.8234,
            1.443,
            1.2893,
            1.1782,
            1.0996,
            1.0527,
            1.0204,
            0.9126,
            0.9117,
            0.8366,
            0.8331,
            0.8154,
            0.7713,
            0.7444,
            0.7215,
            0.7018,
            0.7018,
            0.6802,
            0.6802,
            0.6453,
            0.6447,
            0.6164,
            0.6153,
            0.6078,
            0.6078,
            0.5891,
            0.5766,
            0.566,
            0.566,
            0.5561,
            0.5498,
            0.5269,
            0.5264,
            0.5107,
            0.5107,
            0.5102,
            0.5057,
        ],
        unit='angstrom',
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
