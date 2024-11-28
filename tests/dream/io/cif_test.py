# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import pytest
import scipp as sc
from scippneutron.io import cif

import ess.dream.io.cif
from ess.powder.calibration import OutputCalibrationData
from ess.powder.types import CIFAuthors, IofTof


@pytest.fixture
def ioftof() -> IofTof:
    return IofTof(
        sc.DataArray(
            sc.array(dims=['tof'], values=[2.1, 3.2], variances=[0.3, 0.4]),
            coords={'tof': sc.linspace('tof', 0.1, 1.2, 3, unit='us')},
        )
    )


@pytest.fixture
def cal() -> OutputCalibrationData:
    return OutputCalibrationData(
        {
            0: sc.scalar(0.2, unit='us'),
            1: sc.scalar(1.2, unit='us/angstrom'),
            2: sc.scalar(-1.4, unit='us/angstrom^2'),
        }
    )


def save_reduced_tof_to_str(cif_: cif.CIF) -> str:
    buffer = io.StringIO()
    cif_.save(buffer)
    buffer.seek(0)
    return buffer.read()


def test_save_reduced_tof(ioftof: IofTof, cal: OutputCalibrationData) -> None:
    from ess.dream import __version__

    author = cif.Author(name='John Doe')
    cif_ = ess.dream.io.cif.prepare_reduced_tof_cif(
        ioftof, authors=CIFAuthors([author]), calibration=cal
    )
    result = save_reduced_tof_to_str(cif_)

    assert "_audit_contact_author.name 'John Doe'" in result
    assert f"_computing.diffrn_reduction 'ess.dream v{__version__}'" in result
    assert '_diffrn_source.beamline DREAM' in result
    assert 'ZERO 0 0.2' in result
    assert 'DIFC 1 1.2' in result
    assert 'DIFA 2 -1.4' in result

    loop_header = """loop_
_pd_data.point_id
_pd_meas.time_of_flight
_pd_proc.intensity_norm
_pd_proc.intensity_norm_su
"""
    assert loop_header in result
