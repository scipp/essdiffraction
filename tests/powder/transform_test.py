# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scipp.testing

from ess.powder.transform import (
    compute_pdf_from_structure_factor,
)


def test_pdf_structure_factor_needs_q_coord():
    da = sc.DataArray(sc.ones(sizes={'Q': 3}))
    r = sc.array(dims='r', values=[2, 3, 4, 5.0])
    with pytest.raises(KeyError):
        compute_pdf_from_structure_factor(
            da,
            rho=sc.scalar(1.0, unit='1/angstrom'),
            r=r,
        )


def test_pdf_structure_factor():
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}, unit='angstrom**4'),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da, rho=sc.scalar(1.0, unit='1/angstrom**3'), r=r
    )
    assert v.data.unit == '1/angstrom'
    sc.testing.assert_identical(v.coords['r'], r)


def test_pdf_structure_factor_result_unchanged():
    # Note: bogus data
    da = sc.DataArray(
        sc.array(dims='Q', values=[1, 2, 4.0], unit='angstrom**4'),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da,
        rho=sc.scalar(1.0, unit='1/angstrom**3'),
        r=r,
    )
    sc.testing.assert_allclose(
        v.data,
        sc.array(
            dims='r', values=[-7.74492875, 19.08923564, -39.17659565], unit='1/angstrom'
        ),
        rtol=sc.scalar(1e-5),
    )
