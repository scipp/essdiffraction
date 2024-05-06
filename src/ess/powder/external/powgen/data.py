# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Utilities for loading example data for POWGEN."""

import scipp as sc

from ...types import (
    AccumulatedProtonCharge,
    CalibrationFilename,
    Filename,
    FilenameType,
    FilePath,
    ProtonCharge,
    RawCalibrationData,
    RawDataAndMetadata,
    RawDetectorData,
    RunType,
    SampleRun,
)
from .types import DetectorInfo

_version = "1"

__all__ = ["_get_path"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/powgen"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/powgen/{version}/",
        version=_version,
        registry={
            # Files loadable with Mantid
            "PG3_4844_event.nxs": "md5:d5ae38871d0a09a28ae01f85d969de1e",
            "PG3_4866_event.nxs": "md5:3d543bc6a646e622b3f4542bc3435e7e",
            "PG3_5226_event.nxs": "md5:58b386ebdfeb728d34fd3ba00a2d4f1e",
            "PG3_FERNS_d4832_2011_08_24.cal": "md5:c181221ebef9fcf30114954268c7a6b6",
            # Zipped Scipp HDF5 files
            "PG3_4844_event.zip": "md5:a644c74f5e740385469b67431b690a3e",
            "PG3_4866_event.zip": "md5:5bc49def987f0faeb212a406b92b548e",
            "PG3_FERNS_d4832_2011_08_24.zip": "md5:0fef4ed5f61465eaaa3f87a18f5bb80d",
        },
    )


_pooch = _make_pooch()


def _get_path(name: str, unzip: bool = False) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    import pooch

    return _pooch.fetch(name, processor=pooch.Unzip() if unzip else None)


def mantid_sample_file() -> str:
    return _get_path("PG3_4844_event.nxs")


def mantid_vanadium_file() -> str:
    return _get_path("PG3_4866_event.nxs")


def mantid_empty_instrument_file() -> str:
    return _get_path("PG3_5226_event.nxs")


def mantid_calibration_file() -> str:
    return _get_path("PG3_FERNS_d4832_2011_08_24.cal")


def sample_file() -> str:
    (path,) = _get_path("PG3_4844_event.zip", unzip=True)
    return path


def vanadium_file() -> str:
    (path,) = _get_path("PG3_4866_event.zip", unzip=True)
    return path


def calibration_file() -> str:
    (path,) = _get_path("PG3_FERNS_d4832_2011_08_24.zip", unzip=True)
    return path


def get_path(filename: FilenameType) -> FilePath[FilenameType]:
    """Translate any filename to a path to the file obtained from pooch registry."""
    if filename.endswith(".zip"):
        (path,) = _get_path(filename, unzip=True)
    else:
        path = _get_path(filename)
    return FilePath[FilenameType](path)


def _load_hdf5(filename: str) -> sc.DataArray:
    return sc.io.load_hdf5(filename)


def pooch_load(filename: FilePath[Filename[RunType]]) -> RawDataAndMetadata[RunType]:
    """Load a file with pooch.

    If the file is a zip archive, it is extracted and a path to the contained
    file is returned.

    The loaded data holds both the events and any metadata from the file.
    """
    return RawDataAndMetadata[RunType](_load_hdf5(filename))


def pooch_load_calibration(
    filename: FilePath[CalibrationFilename],
) -> RawCalibrationData:
    """Load the calibration data for the POWGEN test data."""
    # if filename.endswith(".zip"):
    #     (path,) = _get_path(filename, unzip=True)
    # else:
    #     path = _get_path(filename)
    return RawCalibrationData(_load_hdf5(filename))


def extract_raw_data(dg: RawDataAndMetadata[RunType]) -> RawDetectorData[RunType]:
    """Return the events from a loaded data group."""
    return RawDetectorData[RunType](dg["data"])


def extract_detector_info(dg: RawDataAndMetadata[SampleRun]) -> DetectorInfo:
    """Return the detector info from a loaded data group."""
    return DetectorInfo(dg["detector_info"])


def extract_proton_charge(dg: RawDataAndMetadata[RunType]) -> ProtonCharge[RunType]:
    """Return the proton charge from a loaded data group."""
    return ProtonCharge[RunType](dg["proton_charge"])


def extract_accumulated_proton_charge(
    data: RawDetectorData[RunType],
) -> AccumulatedProtonCharge[RunType]:
    """Return the stored accumulated proton charge from a loaded data group."""
    return AccumulatedProtonCharge[RunType](data.coords["gd_prtn_chrg"])


providers = (
    get_path,
    pooch_load,
    pooch_load_calibration,
    extract_accumulated_proton_charge,
    extract_detector_info,
    extract_proton_charge,
    extract_raw_data,
)
"""Sciline Providers for loading POWGEN data."""
