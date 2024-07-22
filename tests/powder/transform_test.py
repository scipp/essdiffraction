# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scipp.testing
from ess.reduce.uncertainty import UncertaintyBroadcastMode

from ess.powder.transform import (
    compute_pdf_from_structure_factor,
)


def test_pdf_structure_factor_needs_q_coord():
    da = sc.DataArray(sc.ones(sizes={'Q': 3}))
    r = sc.array(dims='r', values=[2, 3, 4, 5.0])
    with pytest.raises(KeyError):
        compute_pdf_from_structure_factor(
            da,
            r,
        )


@pytest.mark.parametrize(
    'uncertainty_mode',
    [
        UncertaintyBroadcastMode.drop,
        UncertaintyBroadcastMode.upper_bound,
        UncertaintyBroadcastMode.fail,
    ],
)
def test_pdf_structure_factor(uncertainty_mode):
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da, r, uncertainty_broadcast_mode=uncertainty_mode
    )
    assert v.data.unit == '1/angstrom^2'
    sc.testing.assert_identical(v.coords['r'], r)
    assert v.variances is not None


def test_pdf_structure_factor_result_unchanged():
    # Note: bogus data
    da = sc.DataArray(
        sc.array(dims='Q', values=[1, 2, 4.0]),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da,
        r,
    )
    sc.testing.assert_allclose(
        v.data,
        sc.array(dims='r', values=[-0.616322, 1.51907, -3.11757], unit='1/angstrom^2'),
        rtol=sc.scalar(1e-5),
    )
