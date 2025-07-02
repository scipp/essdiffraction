# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with BEER."""

import scipp as sc

_version = "1"

__all__ = ["mcstas_duplex_medium_resolution", "mcstas_silicon_medium_resolution"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/beer"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/beer/{version}/",
        version=_version,
        registry={
            "duplex.h5": "md5:ebb3f9694ffdd0949f342bd0deaaf627",
            "silicon.h5": "md5:3425ae2b2fe7a938c6f0a4afeb9e0819",
            "silicon-dhkl.tab": "md5:90bedceb23245b045ce1ed0170c1313b",
            "duplex-dhkl.tab": "md5:b4c6c2fcd66466ad291f306b2d6b346e",
        },
    )


_pooch = _make_pooch()


def mcstas_duplex_medium_resolution() -> str:
    """
    Simulated intensity from duplex sample with
    medium resolution chopper configuration.
    """
    return _pooch.fetch('duplex.h5')


def mcstas_silicon_medium_resolution() -> str:
    """
    Simulated intensity from silicon sample with
    medium resolution chopper configuration.
    """
    return _pooch.fetch('silicon.h5')


def duplex_peaks() -> str:
    return _pooch.fetch('duplex-dhkl.tab')


def silicon_peaks() -> str:
    return _pooch.fetch('silicon-dhkl.tab')


def duplex_peaks_array() -> sc.Variable:
    with open(duplex_peaks()) as f:
        return sc.array(
            dims='d',
            values=sorted([float(x) for x in f.read().split('\n') if x != '']),
            unit='angstrom',
        )


def silicon_peaks_array() -> sc.Variable:
    with open(silicon_peaks()) as f:
        return sc.array(
            dims='d',
            values=sorted([float(x) for x in f.read().split('\n') if x != '']),
            unit='angstrom',
        )
