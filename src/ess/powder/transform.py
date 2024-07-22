# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Signal transformation algorithms for powder diffraction."""

import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties


def compute_pdf_from_structure_factor(
    s: sc.DataArray,
    r: sc.Variable,
    *,
    uncertainty_broadcast_mode=UncertaintyBroadcastMode.drop,
) -> sc.DataArray:
    '''
    Compute a pair distribution function from a structure factor.

    Computes the pair distribution function :math:`D(r)` as defined in
    `Review: Pair distribution functions from neutron total scattering for the study of local structure in disordered materials <https://www.sciencedirect.com/science/article/pii/S2773183922000374>`_
    from the overall scattering function :math:`S(Q)`.

    The inputs to the algorithm are:

    * A histogram representing :math:`S(Q)` with :math:`N` bins on a bin-edge grid with :math:`N+1` edges :math:`Q_j` for :math:`j=0\\ldots N`.
    * The bin-edge grid over :math:`r` the output histogram representing :math:`D(r)` will be computed on.

    In each output bin, the output is computed as:

    .. math::
        D_{i+\\frac{1}{2}} &= \\frac{2}{\\pi(r_{i+1}-r_i)} \\int_{r_i}^{r_{i+1}} \\int_{0}^\\infty (S(Q) - 1) Q \\sin(Q r) dQ \\ dr  \\\\
        &\\approx \\frac{2}{\\pi(r_{i+1}-r_i)} \\sum_{j=0}^{N-1} (S(Q)_{j+\\frac{1}{2}} - 1) (\\cos(\\bar{Q}_{j+\\frac{1}{2}} r_{i})-\\cos(\\bar{Q}_{j+\\frac{1}{2}} r_{i+1})) \\Delta Q_{j+\\frac{1}{2}}

    Note that in the above expression the subscript :math:`_{j+\\frac{1}{2}}` is used to denote
    quantities belonging to the :math:`j^\\text{th}` bin of a histogram, :math:`\\bar{Q}_{j+\\frac{1}{2}} = \\frac{Q_j + Q_{j+1}}{2}` and :math:`\\Delta Q_{j+\\frac{1}{2}} = Q_{j+1} - Q_{j}`.

    Parameters
    ----------
    s:
        1D DataArray representing :math:`S(Q)` with a bin-edge coordinate called :math:`Q`
    r:
        1D array, bin-edges of output grid
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
        Choose how uncertainties in S(Q) are broadcast to D(r).
        Defaults to ``UncertaintyBroadcastMode.drop``.

    Returns
    -------
    :
        1D DataArray representing :math:`D(r)` with a bin-edge coordinate called :math:`r` that is the provided output grid

    '''  # noqa: E501
    q = s.coords['Q']
    qm = sc.midpoints(q)
    dq = q[1:] - q[:-1]
    dr = r[1:] - r[:-1]

    v = sc.cos(qm * r * sc.scalar(1, unit='rad'))
    v = v[r.dim, :-1] - v[r.dim, 1:]

    ioq = (s - sc.scalar(1.0, unit=s.unit)) * dq
    if uncertainty_broadcast_mode in (
        UncertaintyBroadcastMode.fail,
        UncertaintyBroadcastMode.drop,
    ):
        # avoid scipp uncertainty propagation exception when multiplying v and ioq
        # the variances are correct after summing over the `Q` dim
        ioq = ioq.broadcast(sizes=v.sizes).copy()
    else:
        ioq = broadcast_uncertainties(ioq, prototype=v, mode=uncertainty_broadcast_mode)
    c = 2 / sc.constants.pi / dr
    g = c * (v * ioq).sum('Q')
    return sc.DataArray(g.data, coords={'r': r})
