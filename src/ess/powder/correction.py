# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Correction algorithms for powder diffraction."""

import enum
from typing import TypeVar
from uuid import uuid4

import sciline
import scipp as sc

from ess.reduce.uncertainty import broadcast_uncertainties

from ._util import event_or_outer_coord
from .types import (
    AccumulatedProtonCharge,
    CaveMonitor,
    EmptyCanRun,
    EmptyCanSubtractedIofDspacing,
    EmptyCanSubtractedIofDspacingTwoTheta,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    IofDspacing,
    IofDspacingTwoTheta,
    ReducedCountsDspacing,
    RunType,
    SampleRun,
    ScaledCountsDspacing,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMonitor,
)


def normalize_by_monitor_histogram(
    detector: ReducedCountsDspacing[RunType],
    *,
    monitor: WavelengthMonitor[RunType, CaveMonitor],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ScaledCountsDspacing[RunType]:
    """Normalize detector data by a histogrammed monitor.

    Parameters
    ----------
    detector:
        Input event data in wavelength.
    monitor:
        A histogrammed monitor in wavelength.
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        `detector` normalized by a monitor.
    """
    norm = broadcast_uncertainties(
        monitor, prototype=detector, mode=uncertainty_broadcast_mode
    )
    lut = sc.lookup(norm, dim="wavelength")
    if detector.bins is None:
        result = (
            detector / lut[sc.midpoints(detector.coords['wavelength'], dim='dspacing')]
        )
    else:
        result = detector.bins / lut
    return ScaledCountsDspacing[RunType](result)


def normalize_by_monitor_integrated(
    detector: ReducedCountsDspacing[RunType],
    *,
    monitor: WavelengthMonitor[RunType, CaveMonitor],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ScaledCountsDspacing[RunType]:
    """Normalize detector data by an integrated monitor.

    The monitor is integrated according to

    .. math::

        M = \\sum_{i=0}^{N-1}\\, m_i (x_{i+1} - x_i) I(x_i, x_{i+1}),

    where :math:`m_i` is the monitor intensity in bin :math:`i`,
    :math:`x_i` is the lower bin edge of bin :math:`i`, and
    :math:`I(x_i, x_{i+1})` selects bins that are within the range of the detector.

    Parameters
    ----------
    detector:
        Input event data in wavelength.
    monitor:
        A histogrammed monitor in wavelength.
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        `detector` normalized by a monitor.
    """
    dim = monitor.dim
    if not monitor.coords.is_edges(dim):
        raise sc.CoordError(
            f"Monitor coordinate '{dim}' must be bin-edges to integrate the monitor."
        )

    # Clip `monitor` to the range of `detector`, where the bins at the boundary
    # may extend past the detector range (how label-based indexing works).
    if detector.bins is not None and dim in detector.bins.coords:
        det_coord = detector.bins.coords.get(dim)
        lo = det_coord.nanmin()
        hi = det_coord.nanmax()
    else:
        # Mask zero count bins, which are an artifact from the rectangular 2-D binning.
        # The wavelength of those bins must be excluded when determining the integration
        # range.
        counts = (
            detector.copy(deep=False) if detector.bins is None else detector.bins.size()
        )
        counts.masks[uuid4().hex] = counts.data == sc.scalar(0.0, unit=counts.unit)
        det_coord = detector.coords[dim]
        edge_dims = [
            dim
            for dim, size in counts.sizes.items()
            if size + 1 == det_coord.sizes[dim]
        ]
        if len(edge_dims) != 1:
            raise sc.CoordError(
                f"Cannot determine edge dimension of coordinate '{dim}'."
            )
        edge_dim = edge_dims[0]
        lo = counts.assign(det_coord[edge_dim, :-1]).nanmin().data
        hi = counts.assign(det_coord[edge_dim, 1:]).nanmax().data

    if monitor.coords[dim].min() > lo or monitor.coords[dim].max() < hi:
        raise ValueError(
            f"Cannot normalize by monitor: The wavelength range of the monitor "
            f"({monitor.coords[dim].min().value} to {monitor.coords[dim].max().value}) "
            f"is smaller than the range of the detector ({lo.value} to {hi.value})."
        )
    monitor = monitor[dim, lo:hi]
    # Strictly limit `monitor` to the range of `detector`.
    edges = sc.concat([lo, monitor.coords[dim][1:-1], hi], dim=dim)
    monitor = sc.rebin(monitor, {dim: edges})

    coord = monitor.coords[dim]
    norm = sc.sum(monitor.data * (coord[1:] - coord[:-1]))
    norm = broadcast_uncertainties(
        norm, prototype=detector, mode=uncertainty_broadcast_mode
    )
    return ScaledCountsDspacing[RunType](detector / norm)


def _normalize_by_vanadium(
    data: sc.DataArray,
    vanadium: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    norm = vanadium.hist() if vanadium.bins is not None else vanadium
    norm = broadcast_uncertainties(
        norm, prototype=data, mode=uncertainty_broadcast_mode
    )
    # Converting to unit 'one' because the division might produce a unit
    # with a large scale if the proton charges in data and vanadium were
    # measured with different units.
    normed = (data / norm).to(unit="one", copy=False)
    mask = norm.data == sc.scalar(0.0, unit=norm.unit)
    if mask.any():
        normed.masks['zero_vanadium'] = mask
    return normed


_RunTypeNoVanadium = TypeVar("_RunTypeNoVanadium", SampleRun, EmptyCanRun)


def normalize_by_vanadium_dspacing(
    data: FocussedDataDspacing[_RunTypeNoVanadium],
    vanadium: FocussedDataDspacing[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IofDspacing[_RunTypeNoVanadium]:
    """
    Normalize sample data by a vanadium measurement and return intensity vs. d-spacing.

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    uncertainty_broadcast_mode:
        Choose how uncertainties of vanadium are broadcast to the sample data.
        Defaults to ``UncertaintyBroadcastMode.fail``.

    Returns
    -------
    :
        ``data / vanadium``.
        May contain a mask "zero_vanadium" which is ``True``
        for bins where vanadium is zero.
    """
    return IofDspacing(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_vanadium_dspacing_and_two_theta(
    data: FocussedDataDspacingTwoTheta[_RunTypeNoVanadium],
    vanadium: FocussedDataDspacingTwoTheta[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IofDspacingTwoTheta[_RunTypeNoVanadium]:
    """
    Normalize sample data by a vanadium measurement and return intensity vs.
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

    Returns
    -------
    :
        ``data / vanadium``.
        May contain a mask "zero_vanadium" which is ``True``
        for bins where vanadium is zero.
    """
    return IofDspacingTwoTheta(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_proton_charge(
    data: ReducedCountsDspacing[RunType],
    proton_charge: AccumulatedProtonCharge[RunType],
) -> ScaledCountsDspacing[RunType]:
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
    return ScaledCountsDspacing[RunType](data / proton_charge)


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


def subtract_empty_can(
    data: IofDspacing[SampleRun],
    background: IofDspacing[EmptyCanRun],
) -> EmptyCanSubtractedIofDspacing[SampleRun]:
    return EmptyCanSubtractedIofDspacing(data.bins.concatenate(-background))


def subtract_empty_can_two_theta(
    data: IofDspacingTwoTheta[SampleRun],
    background: IofDspacingTwoTheta[EmptyCanRun],
) -> EmptyCanSubtractedIofDspacingTwoTheta[SampleRun]:
    return EmptyCanSubtractedIofDspacingTwoTheta(data.bins.concatenate(-background))


class RunNormalization(enum.Enum):
    """Type of normalization applied to each run."""

    monitor_histogram = enum.auto()
    monitor_integrated = enum.auto()
    proton_charge = enum.auto()


def insert_run_normalization(
    workflow: sciline.Pipeline, run_norm: RunNormalization
) -> None:
    """Insert providers for a specific normalization into a workflow."""
    match run_norm:
        case RunNormalization.monitor_histogram:
            workflow.insert(normalize_by_monitor_histogram)
        case RunNormalization.monitor_integrated:
            workflow.insert(normalize_by_monitor_integrated)
        case RunNormalization.proton_charge:
            workflow.insert(normalize_by_proton_charge)


providers = (
    normalize_by_proton_charge,
    normalize_by_vanadium_dspacing,
    normalize_by_vanadium_dspacing_and_two_theta,
    subtract_empty_can,
    subtract_empty_can_two_theta,
)
"""Sciline providers for powder diffraction corrections."""
