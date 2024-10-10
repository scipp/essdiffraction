# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc

import ess.dream.data  # noqa: F401
from ess import dream
from ess.reduce.nexus.types import (
    CalibratedDetector,
    CalibratedMonitor,
    DetectorData,
    Filename,
    Monitor1,
    NeXusDetectorName,
    SampleRun,
)
from ess.reduce.nexus.types import NeXusName as NeXusMonitorName

bank_dims = {'wire', 'module', 'segment', 'strip', 'counter'}
hr_sans_dims = {'strip', 'other'}


@pytest.fixture
def nexus_workflow() -> sciline.Pipeline:
    return dream.io.nexus.LoadNeXusWorkflow()


@pytest.fixture(
    params=[
        'mantle_detector',
        'endcap_backward_detector',
        'endcap_forward_detector',
        'high_resolution_detector',
        # TODO: the 'sans_detector' is strange in the current files
    ]
)
def params(request):
    params = {
        Filename[SampleRun]: dream.data.get_path('DREAM_nexus_sorted-2023-12-07.nxs'),
        NeXusDetectorName: request.param,
    }
    return params


def test_can_load_nexus_detector_data(nexus_workflow, params):
    for key, value in params.items():
        nexus_workflow[key] = value
    result = nexus_workflow.compute(CalibratedDetector[SampleRun])
    assert (
        set(result.dims) == hr_sans_dims
        if params[NeXusDetectorName]
        in (
            'high_resolution_detector',
            'sans_detector',
        )
        else bank_dims
    )

    assert sc.identical(result.data, result.coords['detector_number'])


def test_can_load_nexus_monitor_data(nexus_workflow):
    nexus_workflow[Filename[SampleRun]] = dream.data.get_path(
        'DREAM_nexus_sorted-2023-12-07.nxs'
    )
    nexus_workflow[NeXusMonitorName[Monitor1]] = 'monitor_cave'
    result = nexus_workflow.compute(CalibratedMonitor[SampleRun, Monitor1])
    assert result.sizes == {'event_time_zero': 0}


def test_assemble_nexus_detector_data(nexus_workflow, params):
    for key, value in params.items():
        nexus_workflow[key] = value
    result = nexus_workflow.compute(DetectorData[SampleRun])
    assert (
        set(result.dims) == hr_sans_dims
        if params[NeXusDetectorName]
        in (
            'high_resolution_detector',
            'sans_detector',
        )
        else bank_dims
    )
    assert "source_position" in result.coords
    assert "sample_position" in result.coords
