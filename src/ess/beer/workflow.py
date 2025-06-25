# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl

from .clustering import providers as clustering_providers
from .conversions import providers as conversion_providers
from .io import load_beer_mcstas
from .types import DetectorData, Filename, RunType, SampleRun


def load_mcstas(fname: Filename[SampleRun]) -> DetectorData[SampleRun]:
    return DetectorData[SampleRun](load_beer_mcstas(fname))


default_parameters = {}


def BeerMcStasWorkflow():
    return sl.Pipeline(
        (load_mcstas, *clustering_providers, *conversion_providers),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )
