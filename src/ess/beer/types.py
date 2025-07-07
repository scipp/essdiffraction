from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus.types import DetectorData, Filename, RunType, SampleRun
from ess.reduce.time_of_flight.types import DetectorTofData


class StreakClusteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned by streak"""


DetectorData = DetectorData
Filename = Filename
SampleRun = SampleRun
DetectorTofData = DetectorTofData


MaxTimeOffset = NewType('MaxTimeOffset', sc.Variable)
MinTimeToNextStreak = NewType('MinTimeToNextStreak', sc.Variable)

PeakList = NewType('PeakList', sc.Variable)

ElasticCoordTransformGraph = NewType("ElasticCoordTransformGraph", dict)

ModShift = NewType('ModShift', sc.Variable)
ModTwidth = NewType('ModTwidth', sc.Variable)
ModDt = NewType('ModDt', sc.Variable)
Time0 = NewType('Time0', sc.Variable)
DHKLList = NewType('DHKLList', sc.Variable)
