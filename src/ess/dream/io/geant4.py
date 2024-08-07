# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import sciline
import scipp as sc
from ess.reduce.nexus import extract_detector_data

from ess.powder.types import (
    CalibrationData,
    CalibrationFilename,
    Filename,
    NeXusDetector,
    NeXusDetectorDimensions,
    NeXusDetectorName,
    RawDetector,
    RawSample,
    RawSource,
    ReducibleDetectorData,
    RunType,
    SamplePosition,
    SampleRun,
    SourcePosition,
)

MANTLE_DETECTOR_ID = sc.index(7)
HIGH_RES_DETECTOR_ID = sc.index(8)
SANS_DETECTOR_ID = sc.index(9)
ENDCAPS_DETECTOR_IDS = tuple(map(sc.index, (3, 4, 5, 6)))


class AllRawDetectors(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data for all detectors."""


def load_geant4_csv(file_path: Filename[RunType]) -> AllRawDetectors[RunType]:
    """Load a GEANT4 CSV file for DREAM.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - URL of a CSV or zipped CSV file.
        - Path to a CSV or zipped CSV file on disk.
        - File handle or buffer for reading text or binary data.

    Returns
    -------
    :
        A :class:`scipp.DataGroup` containing the loaded events.
    """
    events = _load_raw_events(file_path)
    detectors = _split_detectors(events)
    for det in detectors.values():
        _adjust_coords(det)
    detectors = _group(detectors)

    return AllRawDetectors[RunType](
        sc.DataGroup({"instrument": sc.DataGroup(detectors)})
    )


def extract_geant4_detector(
    detectors: AllRawDetectors[RunType], detector_name: NeXusDetectorName
) -> NeXusDetector[RunType]:
    """Extract a single detector from a loaded GEANT4 simulation."""
    return NeXusDetector[RunType](detectors["instrument"][detector_name])


def extract_geant4_detector_data(
    detector: NeXusDetector[RunType],
) -> RawDetector[RunType]:
    """Extract the histogram or event data from a loaded GEANT4 detector."""
    return RawDetector[RunType](extract_detector_data(detector))


def _load_raw_events(file_path: str) -> sc.DataArray:
    table = sc.io.load_csv(
        file_path, sep="\t", header_parser="bracket", data_columns=[]
    )
    table = table.rename_dims(row="event")
    table.coords.pop("lambda")
    return sc.DataArray(
        sc.ones(sizes=table.sizes, with_variances=True, unit="counts"),
        coords=table.coords,
    )


def _adjust_coords(da: sc.DataArray) -> None:
    da.coords["position"] = sc.spatial.as_vectors(
        da.coords.pop("x_pos"), da.coords.pop("y_pos"), da.coords.pop("z_pos")
    )


def _group(detectors: dict[str, sc.DataArray]) -> dict[str, sc.DataGroup]:
    elements = ("module", "segment", "counter", "wire", "strip")

    def group(key: str, da: sc.DataArray) -> sc.DataArray:
        if key in ["high_resolution", "sans"]:
            # Only the HR and SANS detectors have sectors.
            res = da.group("sector", *elements)
        else:
            res = da.group(*elements)
        res.coords['position'] = res.bins.coords.pop('position').bins.mean()
        res.bins.coords.pop("sector", None)
        return res

    return {key: sc.DataGroup(events=group(key, da)) for key, da in detectors.items()}


def _split_detectors(
    data: sc.DataArray, detector_id_name: str = "det ID"
) -> dict[str, sc.DataArray]:
    groups = data.group(
        sc.concat(
            [
                MANTLE_DETECTOR_ID,
                HIGH_RES_DETECTOR_ID,
                SANS_DETECTOR_ID,
                *ENDCAPS_DETECTOR_IDS,
            ],
            dim=detector_id_name,
        )
    )
    detectors = {}
    if (
        mantle := _extract_detector(groups, detector_id_name, MANTLE_DETECTOR_ID)
    ) is not None:
        detectors["mantle"] = mantle.copy()
    if (
        high_res := _extract_detector(groups, detector_id_name, HIGH_RES_DETECTOR_ID)
    ) is not None:
        detectors["high_resolution"] = high_res.copy()
    if (
        sans := _extract_detector(groups, detector_id_name, SANS_DETECTOR_ID)
    ) is not None:
        detectors["sans"] = sans.copy()

    endcaps_list = [
        det
        for i in ENDCAPS_DETECTOR_IDS
        if (det := _extract_detector(groups, detector_id_name, i)) is not None
    ]
    if endcaps_list:
        endcaps = sc.concat(endcaps_list, data.dim)
        endcaps = endcaps.bin(
            z_pos=sc.array(
                dims=["z_pos"],
                values=[-np.inf, 0.0, np.inf],
                unit=endcaps.coords["z_pos"].unit,
            )
        )
        detectors["endcap_backward"] = endcaps[0].bins.concat().value.copy()
        detectors["endcap_forward"] = endcaps[1].bins.concat().value.copy()

    return detectors


def _extract_detector(
    detector_groups: sc.DataArray, detector_id_name: str, detector_id: sc.Variable
) -> sc.DataArray | None:
    events = detector_groups[detector_id_name, detector_id].value
    if len(events) == 0:
        return None
    return events


def get_source_position(raw_source: RawSource[RunType]) -> SourcePosition[RunType]:
    return SourcePosition[RunType](raw_source["position"])


def get_sample_position(raw_sample: RawSample[RunType]) -> SamplePosition[RunType]:
    return SamplePosition[RunType](raw_sample["position"])


def patch_detector_data(
    detector_data: RawDetector[RunType],
    source_position: SourcePosition[RunType],
    sample_position: SamplePosition[RunType],
) -> ReducibleDetectorData[RunType]:
    return ReducibleDetectorData[RunType](
        detector_data.assign_coords(
            source_position=source_position, sample_position=sample_position
        )
    )


def geant4_detector_dimensions(
    data: RawDetector[SampleRun],
) -> NeXusDetectorDimensions:
    # For geant4 data, we group by detector identifier, so the data already has
    # logical dimensions, so we simply return the dimensions of the detector.
    return NeXusDetectorDimensions(data.sizes)


def geant4_load_calibration(
    filename: CalibrationFilename,
) -> CalibrationData:
    if filename is not None:
        # Needed to build a complete pipeline.
        raise NotImplementedError(
            "Calibration data loading is not implemented for DREAM GEANT4 data."
        )
    return CalibrationFilename(None)


providers = (
    extract_geant4_detector,
    extract_geant4_detector_data,
    load_geant4_csv,
    get_sample_position,
    get_source_position,
    patch_detector_data,
    geant4_detector_dimensions,
    geant4_load_calibration,
)
"""Geant4-providers for Sciline pipelines."""
