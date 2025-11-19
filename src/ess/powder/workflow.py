from .conversion import convert_to_dspacing
from .correction import apply_lorentz_correction
from .masking import apply_masks
from .types import (
    CalibrationData,
    DspacingDetector,
    ElasticCoordTransformGraph,
    MaskedDetectorIDs,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthDetector,
    WavelengthMask,
)


def add_coords_masks_and_corrections(
    da: WavelengthDetector[RunType],
    masked_pixel_ids: MaskedDetectorIDs,
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
    graph: ElasticCoordTransformGraph[RunType],
    calibration: CalibrationData,
) -> DspacingDetector[RunType]:
    masked = apply_masks(
        data=da,
        masked_pixel_ids=masked_pixel_ids,
        tof_mask_func=tof_mask_func,
        wavelength_mask_func=wavelength_mask_func,
        two_theta_mask_func=two_theta_mask_func,
    )

    with_dspacing = convert_to_dspacing(masked, graph, calibration)

    corrected = apply_lorentz_correction(with_dspacing)
    return DspacingDetector[RunType](corrected)


providers = (add_coords_masks_and_corrections,)
