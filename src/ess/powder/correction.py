# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Correction algorithms for powder diffraction."""

from typing import Any

import scipp as sc
from scippneutron.absorption import compute_transmission_map
from scippneutron.absorption.cylinder import Cylinder
from scippneutron.absorption.material import Material
from scippneutron.atoms import ScatteringParams
from scippneutron.conversion.graph import beamline, tof

from ess.reduce.uncertainty import broadcast_uncertainties

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
    NormalizedByAbsorption,
    NormalizedByProtonCharge,
    RunType,
    SampleRun,
    UncertaintyBroadcastMode,
    VanadiumRun,
)


def normalize_by_monitor(
    data: sc.DataArray,
    *,
    monitor: sc.DataArray,
    wavelength_edges: sc.Variable | None = None,
    smooth_args: dict[str, Any] | None = None,
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
    norm = vanadium.hist()
    norm = broadcast_uncertainties(
        norm, prototype=data, mode=uncertainty_broadcast_mode
    )
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


def normalize_by_absorption(
    da: NormalizedByProtonCharge[RunType],
) -> NormalizedByAbsorption[RunType]:
    """Correct data by absorption probability.

    Parameters
    ----------
    data:
        Non-corrected data array as events or a histogram.

    Returns
    -------
    :
        ``data / transmission_probability``
    """
    # For now hard coding material and sample shape.
    # In the finished implementation those will be parameters to the function.
    material = Material(
        scattering_params=ScatteringParams.for_isotope('V'),
        effective_sample_number_density=sc.scalar(0.07192, unit='1/angstrom**3'),
    )
    sample_shape = Cylinder(
        symmetry_line=sc.vector([0, 1, 0]),
        center_of_base=sc.vector([0, -0.5, 0], unit='cm'),
        radius=sc.scalar(1, unit='cm'),
        height=sc.scalar(5.0, unit='cm'),
    )

    # The strategy is a bit messy unfortunately, is there a better way?
    #
    # Goal is to evaluate the transmission probability as a function
    # of pixel position and wavelength and to then scale each event weight
    # by its associated transmission probability.
    # Issue is, we cannot evaluate the correction for each pixel x wavelength bin.
    # Instead we have have to
    #    1. evaluate the correction factor on a grid,
    #    2. bin the events on that grid,
    #    3. apply the correction, and
    #    4. re-bin the events in their original pixel binning.

    # The grid is defined in a cylindrical coordinate system
    # because the mantle detector is cylindrical-shaped.
    def r(x, y, z):
        return sc.sqrt(x**2 + y**2)

    def th(x, y, z):
        return sc.atan2(x=x, y=y)

    def z(x, y, z):
        return z

    # pixel positions in grid coordinate system
    p = da.coords['position'].flatten(da.coords['position'].dims, to='_')
    rs = r(p.fields.x, p.fields.y, p.fields.z)
    ths = th(p.fields.x, p.fields.y, p.fields.z)  # codespell:ignore ths
    zs = z(p.fields.x, p.fields.y, p.fields.z)

    # grid bin-edges
    r_be = sc.linspace('r', rs.nanmin(), rs.nanmax(), 5)
    th_be = sc.linspace('th', ths.nanmin(), ths.nanmax(), 51)  # codespell:ignore ths
    z_be = sc.linspace('z', zs.nanmin(), zs.nanmax(), 20)
    wav_be = sc.linspace(
        'wavelength',
        da.bins.coords['wavelength'].nanmin(),
        da.bins.coords['wavelength'].nanmax(),
        20,
        unit='angstrom',
    )

    # grid midpoints
    rm, thm, zm, wavm = map(sc.midpoints, (r_be, th_be, z_be, wav_be))

    # Transmission fraction evaluated on grid (1)
    transmission_fraction = compute_transmission_map(
        sample_shape,
        material,
        beam_direction=sc.vector([0, 0, 1.0]),
        wavelength=wavm,
        detector_position=sc.spatial.as_vectors(sc.cos(thm) * rm, sc.sin(thm) * rm, zm),
        quadrature_kind="medium",
    )
    transmission_fraction.coords['wavelength'] = wav_be
    transmission_fraction.coords['r'] = r_be
    transmission_fraction.coords['th'] = th_be
    transmission_fraction.coords['z'] = z_be

    # To be able to recreate the pixel binning we have to add a coordinate
    # keeping track of what event belongs to what pixel.
    da.bins.coords['detector_id'] = sc.bins_like(
        da, sc.arange('_', 0, da.size, unit=None).fold(dim='_', sizes=da.sizes)
    )

    # Events binned on grid (2)
    da.coords['r'] = rs
    da.coords['th'] = ths  # codespell:ignore ths
    da.coords['z'] = zs
    # Need copy here for some reason
    # "DimensionError: View over subspan can only be created for contiguous range of data."  #noqa: E501
    # if we don't.
    bda = da.copy().bin(r=r_be, th=th_be, z=z_be, wavelength=wav_be)
    # Events corrected by factor (3)
    bda /= transmission_fraction.transpose(bda.dims)

    # Re-binning events in original pixel binning (4)
    bda = bda.bins.concat().group('detector_id').fold(sizes=da.sizes)
    for c, v in da.coords.items():
        bda.coords[c] = v
    for c, m in da.masks.items():
        bda.masks[c] = m
    return NormalizedByAbsorption[RunType](bda)


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


providers = (
    normalize_by_proton_charge,
    normalize_by_vanadium_dspacing,
    normalize_by_vanadium_dspacing_and_two_theta,
)
"""Sciline providers for powder diffraction corrections."""
