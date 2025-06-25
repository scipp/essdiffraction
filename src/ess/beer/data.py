# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with BEER."""

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
