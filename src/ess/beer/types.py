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
