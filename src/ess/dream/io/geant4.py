# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import sciline
import scipp as sc
import scippneutron as scn
import scippnexus as snx

from ess.powder.types import (
    CalibratedDetector,
    CalibratedMonitor,
    CalibrationData,
    CalibrationFilename,
    CaveMonitor,
    CaveMonitorPosition,
    DetectorData,
    DetectorLtotal,
    Filename,
    MonitorData,
    MonitorFilename,
    MonitorLtotal,
    MonitorType,
    NeXusComponent,
    NeXusDetectorName,
    Position,
    RunType,
    SampleRun,
    VanadiumRun,
)
from ess.reduce.nexus.types import CalibratedBeamline
from ess.reduce.nexus.workflow import GenericNeXusWorkflow

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
) -> NeXusComponent[snx.NXdetector, RunType]:
    """Extract a single detector from a loaded GEANT4 simulation."""
    return NeXusComponent[snx.NXdetector, RunType](
        detectors["instrument"][detector_name]
    )


def get_calibrated_geant4_detector(
    detector: NeXusComponent[snx.NXdetector, RunType],
) -> CalibratedDetector[RunType]:
    """
    Replacement for :py:func:`ess.reduce.nexus.workflow.get_calibrated_detector`.

    Since the Geant4 detectors already have computed positions as well as logical shape,
    this just extracts the relevant event data.
    """
    return detector["events"].copy(deep=False)


def _load_raw_events(file_path: str) -> sc.DataArray:
    table = sc.io.load_csv(
        file_path, sep="\t", header_parser="bracket", data_columns=[]
    )
    table.coords["sumo"] = table.coords["det ID"]
    table.coords.pop("lambda", None)
    table = table.rename_dims(row="event")
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
        elif key in ["endcap_backward", "endcap_forward"]:
            # Other banks only have a single SUMO.
            res = da.group("sumo", *elements)
        else:
            res = da.group(*elements)
        res.coords["position"] = res.bins.coords.pop("position").bins.mean()
        res.bins.coords.pop("sector", None)
        res.bins.coords.pop("sumo", None)
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


def _to_edges(centers: sc.Variable) -> sc.Variable:
    interior_edges = sc.midpoints(centers)
    return sc.concat(
        [
            2 * centers[0] - interior_edges[0],
            interior_edges,
            2 * centers[-1] - interior_edges[-1],
        ],
        dim=centers.dim,
    )


def load_mcstas_monitor(
    file_path: MonitorFilename[RunType],
    position: CaveMonitorPosition,
) -> NeXusComponent[CaveMonitor, RunType]:
    """Load a monitor from a McStas file.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
    position:
        The position of the monitor.

        Note that the files contain the position. But we don't know what coordinate
        system they are define din. So we cannot reliably use them.

    Returns
    -------
    :
        A :class:`scipp.DataArray` containing the loaded histogram.
    """

    tof, counts, err = np.loadtxt(file_path, usecols=(0, 1, 2), unpack=True)

    tof = _to_edges(sc.array(dims=["tof"], values=tof, unit="us"))
    return NeXusComponent[CaveMonitor, RunType](
        sc.DataGroup(
            data=sc.DataArray(
                sc.array(dims=["tof"], values=counts, variances=err**2, unit="counts"),
                coords={
                    "tof": tof,
                },
            ),
            position=position,
        )
    )


def geant4_load_calibration(filename: CalibrationFilename) -> CalibrationData:
    if filename is not None:
        # Needed to build a complete pipeline.
        raise NotImplementedError(
            "Calibration data loading is not implemented for DREAM GEANT4 data."
        )
    return CalibrationData(None)


def assemble_detector_data(
    detector: CalibratedBeamline[RunType],
) -> DetectorData[RunType]:
    """
    In the raw data, the tofs extend beyond 71ms, this is thus not an event_time_offset.
    We convert the detector data to data which resembles NeXus data, with
    event_time_zero and event_time_offset coordinates.

    Parameters
    ----------
    detector:
        The calibrated detector data.
    """

    da = detector.copy(deep=False)
    da.bins.coords["tof"] = da.bins.coords["tof"].to(unit="us")

    period = (1.0 / sc.scalar(14.0, unit="Hz")).to(unit="us")
    # Bin the data into bins with a 71ms period.
    npulses = int((da.bins.coords["tof"].max() / period).ceil().value)
    da = da.bin(tof=sc.arange("tof", npulses + 1) * period)
    # npulses = int((da.bins.coords["tof"].max() / period).value) + 1
    # print('npulses', npulses)
    # da = da.bin(tof=sc.arange("tof", npulses) * period)
    # Add a event_time_zero coord for each bin, but not as bin edges,
    # as all events in the same pulse have the same event_time_zero, hence the `[:2]`
    # We need to pick a start time. The actual value does not matter. We chose the
    # random date of Friday, November 1, 2024 8:40:34.078
    da.coords["event_time_zero"] = (
        sc.scalar(1730450434078980000, unit="ns").to(unit="us") + da.coords["tof"]
    )[:npulses]
    # Remove the meaningless tof coord at the top level
    del da.coords["tof"]
    da = da.rename_dims(tof="event_time_zero")
    # Compute a event_time_offset as tof % period
    da.bins.coords["event_time_offset"] = (da.bins.coords.pop("tof") % period).to(
        unit="us"
    )
    # Add a event_time_zero coord for each event
    da.bins.coords["event_time_zero"] = sc.bins_like(
        da.bins.coords["event_time_offset"], da.coords["event_time_zero"]
    )
    da = da.bins.concat('event_time_zero')
    # Add a useful Ltotal coordinate
    graph = scn.conversion.graph.beamline.beamline(scatter=True)
    da = da.transform_coords("Ltotal", graph=graph)
    return DetectorData[RunType](da)


def assemble_monitor_data(
    monitor: CalibratedMonitor[RunType, MonitorType],
) -> MonitorData[RunType, MonitorType]:
    """
    Dummy assembly of monitor data, monitor already contains neutron data.
    We simply add a Ltotal coordinate necessary to calculate the time-of-flight.

    Parameters
    ----------
    monitor:
        The calibrated monitor data.
    """
    graph = scn.conversion.graph.beamline.beamline(scatter=False)
    return MonitorData[RunType, MonitorType](
        monitor.transform_coords("Ltotal", graph=graph)
    )


def dummy_source_position() -> Position[snx.NXsource, RunType]:
    return Position[snx.NXsource, RunType](
        sc.vector([np.nan, np.nan, np.nan], unit="mm")
    )


def dummy_sample_position() -> Position[snx.NXsample, RunType]:
    return Position[snx.NXsample, RunType](
        sc.vector([np.nan, np.nan, np.nan], unit="mm")
    )


def extract_detector_ltotal(detector: DetectorData[RunType]) -> DetectorLtotal[RunType]:
    """
    Extract Ltotal from the detector data.
    TODO: This is a temporary implementation. We should instead read the positions
    separately from the event data, so we don't need to re-load the positions every time
    new events come in while streaming live data.
    """
    return DetectorLtotal[RunType](detector.coords["Ltotal"])


def extract_monitor_ltotal(
    monitor: MonitorData[RunType, MonitorType],
) -> MonitorLtotal[RunType, MonitorType]:
    """
    Extract Ltotal from the monitor data.
    TODO: This is a temporary implementation. We should instead read the positions
    separately from the event data, so we don't need to re-load the positions every time
    new events come in while streaming live data.
    """
    return MonitorLtotal[RunType, MonitorType](monitor.coords["Ltotal"])


def LoadGeant4Workflow() -> sciline.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = GenericNeXusWorkflow(
        run_types=[SampleRun, VanadiumRun], monitor_types=[CaveMonitor]
    )
    wf.insert(extract_geant4_detector)
    wf.insert(load_geant4_csv)
    wf.insert(load_mcstas_monitor)
    wf.insert(geant4_load_calibration)
    wf.insert(get_calibrated_geant4_detector)
    wf.insert(assemble_detector_data)
    wf.insert(assemble_monitor_data)
    wf.insert(dummy_source_position)
    wf.insert(dummy_sample_position)
    wf.insert(extract_detector_ltotal)
    wf.insert(extract_monitor_ltotal)
    return wf
