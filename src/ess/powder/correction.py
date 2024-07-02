# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Correction algorithms for powder diffraction."""

from typing import Any, Dict, Optional

import scipp as sc
from scippneutron.conversion.graph import beamline, tof

from ._util import event_or_outer_coord
from .logging import get_logger
from .smoothing import lowpass
from .types import (
    AccumulatedProtonCharge,
    FilteredData,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    IofDspacing,
    IofDspacingTwoTheta,
    NormalizedByProtonCharge,
    RunType,
    SampleRun,
    UncertaintyBroadcastMode,
    VanadiumRun,
)
from .uncertainty import broadcast_uncertainties


def normalize_by_monitor(
    data: sc.DataArray,
    *,
    monitor: sc.DataArray,
    wavelength_edges: Optional[sc.Variable] = None,
    smooth_args: Optional[Dict[str, Any]] = None,
) -> sc.DataArray:
    """
    Normalize event data by a monitor.

    The input is converted to wavelength if it does not already contain wavelengths.

    Parameters
    ----------
    data:
        Input event data.
    monitor:
        A histogrammed monitor.
    wavelength_edges:
        If given, rebin the monitor with these edges.
    smooth_args:
        If given, the monitor histogram is smoothed with
        :func:`ess.powder.lowpass` before dividing into `data`.
        `smooth_args` is passed as keyword arguments to
        :func:`ess.powder.lowpass`. If ``None``, the monitor is not smoothed.

    Returns
    -------
    :
        `data` normalized by a monitor.
    """
    if "wavelength" not in monitor.coords:
        monitor = monitor.transform_coords(
            "wavelength",
            graph={**beamline.beamline(scatter=False), **tof.elastic("tof")},
            keep_inputs=False,
            keep_intermediate=False,
            keep_aliases=False,
        )

    if wavelength_edges is not None:
        monitor = monitor.rebin(wavelength=wavelength_edges)
    if smooth_args is not None:
        get_logger().info(
            "Smoothing monitor for normalization using "
            "ess.powder.smoothing.lowpass with %s.",
            smooth_args,
        )
        monitor = lowpass(monitor, dim="wavelength", **smooth_args)
    return data.bins / sc.lookup(func=monitor, dim="wavelength")


def _normalize_by_vanadium(
    data: sc.DataArray,
    vanadium: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    vanadium = broadcast_uncertainties(vanadium, uncertainty_broadcast_mode)
    norm = vanadium.hist()
    # Converting to unit 'one' because the division might produce a unit
    # with a large scale if the proton charges in data and vanadium were
    # measured with different units.
    return (data / norm).to(unit="one", copy=False)


def normalize_by_vanadium_dspacing(
    data: FocussedDataDspacing[SampleRun],
    vanadium: FocussedDataDspacing[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IofDspacing:
    """
    Normalize sample data by a vanadium measurement and return intensity vs d-spacing.

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    uncertainty_broadcast_mode:
        Choose how uncertainties of vanadium are broadcast to the sample data.
        Defaults to ``UncertaintyBroadcastMode.fail``.
    """
    return IofDspacing(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_vanadium_dspacing_and_two_theta(
    data: FocussedDataDspacingTwoTheta[SampleRun],
    vanadium: FocussedDataDspacingTwoTheta[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IofDspacingTwoTheta:
    """
    Normalize sample data by a vanadium measurement and return intensity vs
    (d-spacing, 2theta).

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    uncertainty_broadcast_mode:
        Choose how uncertainties of vanadium are broadcast to the sample data.
        Defaults to ``UncertaintyBroadcastMode.fail``.
    """
    return IofDspacingTwoTheta(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_proton_charge(
    data: FilteredData[RunType], proton_charge: AccumulatedProtonCharge[RunType]
) -> NormalizedByProtonCharge[RunType]:
    """Normalize data by an accumulated proton charge.

    Parameters
    ----------
    data:
        Un-normalized data array as events or a histogram.
    proton_charge:
        Accumulated proton charge over the entire run.

    Returns
    -------
    :
        ``data / proton_charge``
    """
    return NormalizedByProtonCharge[RunType](data / proton_charge)


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    """
    Return a scipp.DataArray containing calibration metadata as coordinates.

    Parameters
    ----------
    into:
        Base data and metadata for the returned object.
    calibration:
        Calibration parameters.

    Returns
    -------
    :
        Copy of `into` with additional coordinates and masks
        from `calibration`.

    See Also
    --------
    ess.snspowder.powgen.calibration.load_calibration
    """
    for name, coord in calibration.coords.items():
        if not sc.identical(into.coords[name], coord):
            raise ValueError(
                f"Coordinate {name} of calibration and target dataset do not agree."
            )
    out = into.copy(deep=False)
    for name in ("difa", "difc", "tzero"):
        if name in out.coords:
            raise ValueError(
                f"Cannot add calibration parameter '{name}' to data, "
                "there already is metadata with the same name."
            )
        out.coords[name] = calibration[name].data
    if "calibration" in out.masks:
        raise ValueError(
            "Cannot add calibration mask 'calibration' tp data, "
            "there already is a mask with the same name."
        )
    out.masks["calibration"] = calibration["mask"].data
    return out


def apply_lorentz_correction(da: sc.DataArray) -> sc.DataArray:
    """Perform a Lorentz correction for ToF powder diffraction data.

    This function uses this definition:

    .. math::

        L = d^4 \\sin\\theta

    where :math:`d` is d-spacing, :math:`\\theta` is half the scattering angle
    (note the definitions in
    https://scipp.github.io/scippneutron/user-guide/coordinate-transformations.html).

    The Lorentz factor as defined here is suitable for correcting time-of-flight data
    expressed in wavelength or d-spacing.
    It follows the definition used by GSAS-II, see page 140 of
    https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf

    Parameters
    ----------
    da:
        Input data with coordinates ``two_theta`` and ``dspacing``.

    Returns
    -------
    :
        ``da`` multiplied by :math:`L`.
        Has the same dtype as ``da``.
    """
    # The implementation is optimized under the assumption that two_theta
    # is small and dspacing and the data are large.
    out = _shallow_copy(da)
    dspacing = event_or_outer_coord(da, "dspacing")
    two_theta = event_or_outer_coord(da, "two_theta")
    theta = 0.5 * two_theta

    d4 = dspacing.broadcast(sizes=out.sizes) ** 4
    if out.bins is None:
        out.data = d4.to(dtype=out.dtype, copy=False)
        out_data = out.data
    else:
        out.bins.data = d4.to(dtype=out.bins.dtype, copy=False)
        out_data = out.bins.data
    out_data *= sc.sin(theta, out=theta)
    out_data *= da.data if da.bins is None else da.bins.data
    return out


def _shallow_copy(da: sc.DataArray) -> sc.DataArray:
    # See https://github.com/scipp/scipp/issues/2773
    out = da.copy(deep=False)
    if da.bins is not None:
        out.data = sc.bins(**da.bins.constituents)
    return out


def compute_pdf_from_structure_factor(
    s: sc.DataArray,
    rho: sc.Variable,
    r: sc.Variable,
    use_filter: bool,
) -> sc.DataArray:
    '''
    Computes the pair-distribution-function :math:`D(r)` as defined in
    `Review: Pair distribution functions from neutron total scattering for the study of local structure in disordered materials <https://www.sciencedirect.com/science/article/pii/S2773183922000374>`_
    from :math:`S(Q)`.

    The input to the algorithm is a histogram representing :math:`S(Q)`,
    the density :math:`\\rho`, and a bin-edge grid over :math:`r` for the output.

    In each output bin, the output is computed as:

    .. math::
        D_{i+\\frac{1}{2}} = \\frac{2}{\\pi(r_{i+1}-r_i)} \\int_{r_i}^{r_{i+1}} \\int_{0}^\\infty i(Q) Q sin(Q r) dQ \\ dr  \\\\
        \\approx \\frac{2}{\\pi(r_{i+1}-r_i)} \\sum_{j=1}^{N} i(Q_j) (cos(Q_j r_{i})-cos(Q_j r_{i+1})) \\Delta Q_j

    Parameters
    ----------
    s:
        :math:`S(Q)` with bin-edge coordinate :math:`Q`
    rho:
        density of sample
    r:
        bin-edges of output grid
    use_filter:
        if ``True`` the Lorch filter is applied

    Returns
    -------
    :
        :math:`D(r)` for each bin in the provided output grid

    '''  # noqa: E501
    for i in range(s.size):
        if not sc.isnan(s[i]).value:
            minbound = i
            break
    for i in range(s.size - 1, -1, -1):
        if not sc.isnan(s[i]).value:
            maxbound = i
            break

    s = s[minbound : maxbound + 1]
    q = s.coords['Q']
    qm = sc.midpoints(q)
    dq = q[1:] - q[:-1]
    dr = r[1:] - r[:-1]

    v = sc.cos(qm * r * sc.scalar(1, unit='rad'))
    v = v[r.dim, :-1] - v[r.dim, 1:]

    ioq = (4 * sc.constants.pi * rho) * (s - 1)
    ioq.variances = None
    ioq *= dq

    if use_filter:
        qm_pi_q_max = qm * sc.constants.pi / q.max()
        ioq *= sc.sin(qm_pi_q_max * sc.scalar(1, unit='rad'))
        ioq /= qm_pi_q_max

    c = 2 / sc.constants.pi / dr
    g = c * (v * ioq).sum('Q')
    return sc.DataArray(g.data, coords={'r': r})


providers = (
    normalize_by_proton_charge,
    normalize_by_vanadium_dspacing,
    normalize_by_vanadium_dspacing_and_two_theta,
)
"""Sciline providers for powder diffraction corrections."""
