import NCrystal as NC
import scipp as sc


def dspacing_peak_positions_from_cif(cif, intensity_threshold=None) -> sc.Variable:
    info = NC.NCMATComposer.from_cif(cif).load('comp=bragg').info
    min_intensity = (
        intensity_threshold.to(unit='dimensionless').value
        if intensity_threshold is not None
        else 0
    )
    return sc.array(
        dims=['peaks'],
        values=[
            hkl.d for hkl in info.hklObjects() if (hkl.f2 * hkl.mult) >= min_intensity
        ],
        unit='angstrom',
    )
