# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx
from scippneutron.io.cif import Author

import ess.dream.data  # noqa: F401
from ess import dream, powder
from ess.dream.workflow import (
    DreamGeant4MonitorHistogramWorkflow,
    DreamGeant4MonitorIntegratedWorkflow,
    DreamGeant4ProtonChargeWorkflow,
)
from ess.powder.types import (
    AccumulatedProtonCharge,
    BackgroundRun,
    CalibrationFilename,
    CaveMonitorPosition,
    CIFAuthors,
    DspacingBins,
    DspacingData,
    Filename,
    IofDspacing,
    IofDspacingTwoTheta,
    IofTof,
    MaskedData,
    MonitorFilename,
    NeXusDetectorName,
    NormalizedRunData,
    Position,
    ReducedTofCIF,
    SampleRun,
    TofMask,
    TwoThetaBins,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMask,
)
from ess.reduce import workflow as reduce_workflow

sample = sc.vector([0.0, 0.0, 0.0], unit='mm')
source = sc.vector([-3.478, 0.0, -76550], unit='mm')
charge = sc.scalar(1.0, unit='µAh')

params = {
    Filename[SampleRun]: dream.data.simulated_diamond_sample(),
    Filename[VanadiumRun]: dream.data.simulated_vanadium_sample(),
    Filename[BackgroundRun]: dream.data.simulated_empty_can(),
    MonitorFilename[SampleRun]: dream.data.simulated_monitor_diamond_sample(),
    MonitorFilename[VanadiumRun]: dream.data.simulated_monitor_vanadium_sample(),
    MonitorFilename[BackgroundRun]: dream.data.simulated_monitor_empty_can(),
    CalibrationFilename: None,
    UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 201, unit='angstrom'),
    TofMask: lambda x: (x < sc.scalar(0.0, unit='ns'))
    | (x > sc.scalar(86e6, unit='ns')),
    Position[snx.NXsample, SampleRun]: sample,
    Position[snx.NXsample, VanadiumRun]: sample,
    Position[snx.NXsource, SampleRun]: source,
    Position[snx.NXsource, VanadiumRun]: source,
    AccumulatedProtonCharge[SampleRun]: charge,
    AccumulatedProtonCharge[VanadiumRun]: charge,
    TwoThetaMask: None,
    WavelengthMask: None,
    CaveMonitorPosition: sc.vector([0.0, 0.0, -4220.0], unit='mm'),
    CIFAuthors: CIFAuthors(
        [
            Author(
                name="Jane Doe", email="jane.doe@ess.eu", orcid="0000-0000-0000-0001"
            ),
        ]
    ),
}


@pytest.fixture(params=["mantle", "endcap_backward", "endcap_forward"])
def params_for_det(request):
    # Not available in simulated data
    return {**params, NeXusDetectorName: request.param}


@pytest.fixture
def workflow(params_for_det):
    return make_workflow(params_for_det, run_norm=powder.RunNormalization.proton_charge)


def make_workflow(params_for_det, *, run_norm):
    wf = dream.DreamGeant4Workflow(run_norm=run_norm)
    for key, value in params_for_det.items():
        wf[key] = value
    return wf


def test_pipeline_can_compute_dspacing_result(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_pipeline_can_compute_dspacing_result_with_hist_monitor_norm(params_for_det):
    workflow = make_workflow(
        params_for_det, run_norm=powder.RunNormalization.monitor_histogram
    )
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_pipeline_can_compute_dspacing_result_with_integrated_monitor_norm(
    params_for_det,
):
    workflow = make_workflow(
        params_for_det, run_norm=powder.RunNormalization.monitor_integrated
    )
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_workflow_is_deterministic(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = workflow.get(IofTof, scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_can_compute_intermediate_results(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    results = workflow.compute((NormalizedRunData[SampleRun], NeXusDetectorName))
    result = results[NormalizedRunData[SampleRun]]

    detector_name = results[NeXusDetectorName]
    expected_dims = {'segment', 'wire', 'counter', 'strip', 'module'}
    if detector_name in ('endcap_backward', 'endcap_forward'):
        expected_dims.add('sumo')

    assert set(result.dims) == expected_dims


def test_pipeline_group_by_two_theta(workflow):
    two_theta_bins = sc.linspace(
        dim='two_theta', unit='rad', start=0.8, stop=2.4, num=17
    )
    workflow[TwoThetaBins] = two_theta_bins
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacingTwoTheta)
    assert result.sizes == {
        'two_theta': 16,
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
    assert sc.allclose(result.coords['two_theta'], two_theta_bins)


def test_pipeline_wavelength_masking(workflow):
    wmin = sc.scalar(0.18, unit="angstrom")
    wmax = sc.scalar(0.21, unit="angstrom")
    workflow[WavelengthMask] = lambda x: (x > wmin) & (x < wmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(MaskedData[SampleRun])
    assert 'wavelength' in masked_sample.bins.masks
    sum_in_masked_region = (
        masked_sample.bin(wavelength=sc.concat([wmin, wmax], dim='wavelength'))
        .sum()
        .data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


def test_pipeline_two_theta_masking(workflow):
    tmin = sc.scalar(1.0, unit="rad")
    tmax = sc.scalar(1.2, unit="rad")
    workflow[TwoThetaMask] = lambda x: (x > tmin) & (x < tmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(MaskedData[SampleRun])
    assert 'two_theta' in masked_sample.masks
    sum_in_masked_region = (
        masked_sample.bin(two_theta=sc.concat([tmin, tmax], dim='two_theta')).sum().data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


def test_use_workflow_helper(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_pipeline_can_save_data(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(ReducedTofCIF)

    buffer = io.StringIO()
    result.save(buffer)
    buffer.seek(0)
    content = buffer.read()

    assert content.startswith(r'#\#CIF_1.1')
    _assert_contains_source_info(content)
    _assert_contains_author_info(content)
    _assert_contains_beamline_info(content)
    _assert_contains_tof_data(content)


def _assert_contains_source_info(cif_content: str) -> None:
    assert 'diffrn_source.beamline DREAM' in cif_content


def _assert_contains_author_info(cif_content: str) -> None:
    assert "audit_contact_author.name 'Jane Doe'" in cif_content
    assert 'audit_contact_author.email jane.doe@ess.eu' in cif_content
    assert 'audit_contact_author.id_orcid 0000-0000-0000-0001' in cif_content


def _assert_contains_beamline_info(cif_content: str) -> None:
    assert 'diffrn_source.beamline DREAM' in cif_content
    assert 'diffrn_source.facility ESS' in cif_content


def _assert_contains_tof_data(cif_content: str) -> None:
    assert 'pd_meas.time_of_flight' in cif_content
    assert 'pd_proc.intensity_norm' in cif_content
    assert 'pd_proc.intensity_norm_su' in cif_content


def test_dream_workflow_registers_subclasses():
    # Because it was imported
    for wf in (
        DreamGeant4MonitorHistogramWorkflow,
        DreamGeant4MonitorIntegratedWorkflow,
        DreamGeant4ProtonChargeWorkflow,
    ):
        assert wf in reduce_workflow.workflow_registry
    count = len(reduce_workflow.workflow_registry)

    @reduce_workflow.register_workflow
    class MyWorkflow: ...

    assert MyWorkflow in reduce_workflow.workflow_registry
    assert len(reduce_workflow.workflow_registry) == count + 1


def test_dream_workflow_parameters_returns_filtered_params():
    wf = DreamGeant4ProtonChargeWorkflow()
    parameters = reduce_workflow.get_parameters(wf, (DspacingData[SampleRun],))
    assert Filename[SampleRun] in parameters
    assert Filename[BackgroundRun] not in parameters
