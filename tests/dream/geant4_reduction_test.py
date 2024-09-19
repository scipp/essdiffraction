# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import pytest
import sciline
import scipp as sc
import scipp.testing

import ess.dream.data  # noqa: F401
from ess import dream, powder
from ess.dream.io.cif import CIFAuthor, CIFAuthors
from ess.powder.types import (
    AccumulatedProtonCharge,
    BackgroundRun,
    CalibrationFilename,
    DspacingBins,
    Filename,
    IofDspacing,
    IofDspacingTwoTheta,
    MaskedData,
    NeXusDetectorName,
    NeXusSample,
    NeXusSource,
    NormalizedByProtonCharge,
    OutFilename,
    SampleRun,
    TofMask,
    TwoThetaBins,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMask,
)

sample = sc.DataGroup(position=sc.vector([0.0, 0.0, 0.0], unit='mm'))
source = sc.DataGroup(position=sc.vector([-3.478, 0.0, -76550], unit='mm'))
charge = sc.scalar(1.0, unit='µAh')

params = {
    Filename[SampleRun]: dream.data.simulated_diamond_sample(),
    Filename[VanadiumRun]: dream.data.simulated_vanadium_sample(),
    Filename[BackgroundRun]: dream.data.simulated_empty_can(),
    CalibrationFilename: None,
    UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 201, unit='angstrom'),
    TofMask: lambda x: (x < sc.scalar(0.0, unit='ns'))
    | (x > sc.scalar(86e6, unit='ns')),
    NeXusSample[SampleRun]: sample,
    NeXusSample[VanadiumRun]: sample,
    NeXusSource[SampleRun]: source,
    NeXusSource[VanadiumRun]: source,
    AccumulatedProtonCharge[SampleRun]: charge,
    AccumulatedProtonCharge[VanadiumRun]: charge,
    TwoThetaMask: None,
    WavelengthMask: None,
    CIFAuthors: CIFAuthors(
        [
            CIFAuthor(
                name="Jane Doe", email="jane.doe@ess.eu", id_orcid="0000-0000-0000-0001"
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
    wf = dream.DreamGeant4Workflow()
    for key, value in params_for_det.items():
        wf[key] = value
    return wf


def test_pipeline_can_compute_dspacing_result(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_workflow_is_deterministic(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = workflow.get(IofDspacing, scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_can_compute_intermediate_results(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(NormalizedByProtonCharge[SampleRun])
    assert set(result.dims) == {'segment', 'wire', 'counter', 'strip', 'module'}


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
    def get_result(da: IofDspacing) -> IofDspacing:
        return da

    buffer = io.StringIO()
    workflow[OutFilename] = buffer
    workflow = powder.with_pixel_mask_filenames(workflow, [])

    result, expected = workflow.bind_and_call(
        [dream.io.save_reduced_dspacing, get_result]
    )
    sc.testing.assert_identical(result, expected)

    buffer.seek(0)
    content = buffer.read()
    # print(content)
    # assert False

    assert content.startswith(r'#\#CIF_1.1')
    _assert_contains_source_info(content)
    _assert_contains_author_info(content)


def _assert_contains_source_info(cif_content: str) -> None:
    assert 'diffrn_source.beamline DREAM' in cif_content


def _assert_contains_author_info(cif_content: str) -> None:
    assert "audit_contact_author.name 'Jane Doe'" in cif_content
    assert 'audit_contact_author.email jane.doe@ess.eu' in cif_content
