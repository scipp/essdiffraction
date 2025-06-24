# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl

from ess.reduce.nexus.types import DetectorData, Filename, SampleRun
from ess.reduce.time_of_flight.types import DetectorTofData

from .clustering import cluster_events_by_streak, compute_tof_in_each_cluster
from .io import load_beer_mcstas


def load_data(fname: Filename[SampleRun]) -> DetectorData[SampleRun]:
    return DetectorData[SampleRun](load_beer_mcstas(fname))


def cluster_events_and_compute_tof(
    da: DetectorData[SampleRun],
) -> DetectorTofData[SampleRun]:
    da = cluster_events_by_streak(da)
    compute_tof_in_each_cluster(da)
    return DetectorTofData[SampleRun](da)


default_parameters = {}


def BeerMcStasWorkflow():
    return sl.Pipeline((load_data, cluster_events_and_compute_tof))
