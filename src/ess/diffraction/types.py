# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types uses in ess.diffraction.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from typing import NewType, TypeVar

import sciline
import scipp as sc

# 1 TypeVars used to parametrize the generic parts of the workflow

# 1.1 Run types
EmptyInstrumentRun = NewType('EmptyInstrumentRun', int)
"""Empty instrument run."""
SampleRun = NewType('SampleRun', int)
"""Sample run."""
VanadiumRun = NewType('VanadiumRun', int)
"""Vanadium run."""
RunType = TypeVar('RunType', EmptyInstrumentRun, SampleRun, VanadiumRun)
"""TypeVar used for specifying the run."""


# 2 Workflow parameters


class Filename(sciline.Scope[RunType, str], str):
    """Filename of a run."""


ValidTofRange = NewType('ValidTofRange', sc.Variable)
"""Min and max tof value of the instrument."""

# 3 Workflow (intermediate) results

# This is Mantid-specific and can probably be removed when the POWGEN
# workflow is removed.
DetectorInfo = NewType('DetectorInfo', sc.Dataset)
"""Mapping between detector numbers and spectra."""


class FilteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data without invalid events."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class TofCroppedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data cropped to the valid TOF range."""