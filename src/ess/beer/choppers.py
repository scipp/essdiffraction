#!/usr/bin/env python3
"""
TOF diagrams for the BEER instrument

The functions for plotting TOF diagrams and the predefined modes of operation
for the BEER instrument.

@author: premysl.beran@ess.eu
"""

# %% Basic import

import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc

# import scippneutron as scn
import tof
from scippneutron.chopper import disk_chopper
from tof.chopper import AntiClockwise, Clockwise
from tof.utils import wavelength_to_speed as w2s

mpl.rcParams.update({'font.size': 12})

# %% Definition of the units (needed for Scipp)

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
m = sc.Unit('m')
A = sc.Unit('angstrom')
s = sc.Unit('second')
ms = sc.Unit('ms')

# %% ESS pulse plot


def plot_ess_pulse(save_fig=False):
    """
    Plot the basic ESS pulse

    Parameters
    ----------
    save_fig : Boolean, optional, The default is False.
        Do you want to save the plot as PDF?

    Returns
    -------
    ess_pulse : matplotlib figure
        The figure of the ESS pulse in Matplotlib figure format

    """

    # definition of the ESS pulse
    pulse = tof.Source(neutrons=1_000_000, facility='ess', pulses=1)

    # plot the chart
    pulse_plot = pulse.plot(bins=1000)
    ess_pulse = pulse_plot.fig

    ess_pulse.set_size_inches(9, 3.5)
    ess_pulse.suptitle('ESS source pulse')
    ess_pulse.set_layout_engine(layout='constrained')
    pulse_plot.show()

    if save_fig:
        ess_pulse.savefig('ESS_source_pulse.pdf', format='pdf')

    return ess_pulse


# %% Definition of the "modes" class and some functions


# setup the BEER mode
class beer_mode:
    """
    Class containing the choppers setup.
    """

    def __init__(self, lambda_):
        self.choppers = []
        self.caption = ''
        self.lambda_0 = lambda_
        self.chopper_offset = []
        self.t0_dist = []

    def __repr__(self) -> str:
        t = ''
        for i, chopper in enumerate(self.choppers):
            off = (
                self.chopper_offset[i] - chopper.close[0] / 2
                if "PS" not in chopper.name
                else self.chopper_offset[i]
            )
            t += (
                f'{chopper.name:4s} -> freq: {chopper.frequency.value:3.0f} Hz; '
                f'dis: {chopper.distance.value:6.3f} m; '
                f'phase: {chopper.phase.value:6.2f}º; '
                f'dir: {"-->" if chopper.direction == Clockwise else "<--"}; '
                f'offset: {off.value:3.0f}º '
                f'{"*" if "PS" not in chopper.name else ""} \n'
            )

        return (
            f'Mode description: {self.caption}\n'
            f'Number of choppers: {len(self.choppers)}\n'
            f'Nominal wavelength: {self.lambda_0.value} Å\n{t}'
            '* - frame overlap offset expressed as the shift '
            'from the opening centre\n'
        )

    def get_tof(self, distance, wavelength):
        """
        Calculate time-of-flight of neutrons of given wavelength at given distance
        """
        return (distance.to(unit='m') / w2s(wavelength.to(unit='angstrom'))).to(
            unit='s'
        )

    def get_phase(self, distance, frequency, ang_offset):
        """
        Calculate the angular phase shift of the chopper
        """
        return (
            (self.get_tof(distance, self.lambda_0) * frequency.to(unit='Hz'))
            * (360 * deg)
            + ang_offset.to(unit='deg')
        ).to(unit='deg')

    def add_chopper(
        self,
        dist,
        freq,
        offset,
        caption,
        starts,
        stops,
        rotation=None,
        dist_for_phase=None,
    ):
        """
        fill up the chopper variable
        dist_for_phase -
            for PSC choppers, the phase is calculated at the middle distance
        """
        if dist_for_phase is None:
            dist_for_phase = dist

        self.chopper_offset.append(offset.to(unit='deg'))
        self.t0_dist.append(dist_for_phase.to(unit='m'))
        self.choppers.append(
            tof.Chopper(
                frequency=freq.to(unit='Hz'),
                open=sc.array(dims=['cutout'], unit='deg', values=starts),
                close=sc.array(dims=['cutout'], unit='deg', values=stops),
                distance=dist.to(unit='m'),
                name=caption,
                direction=Clockwise if rotation is None else rotation,
                phase=self.get_phase(
                    dist_for_phase,
                    freq,
                    (0.5 * 0.00286 * s) * (360 * deg) * freq  # middle of the pulse
                    - offset.to(unit='deg'),
                ),
            )
        )  # user offset

    def update_chopper_phases(self):
        for i, chopper in enumerate(self.choppers):
            off = self.chopper_offset[i]
            chopper.phase = self.get_phase(
                self.t0_dist[i],
                chopper.frequency,
                (0.5 * 0.00286 * s)
                * (360 * deg)
                * chopper.frequency  # middle of the pulse
                - off,
            )  # user offset


# %% Detector definition

# define the detectors and monitors
detectors = [
    tof.Detector(distance=10.0 * m, name='monitor-10m'),
    tof.Detector(distance=28.198 * m, name='bunker exit'),
    tof.Detector(distance=82.0 * m, name='80 m chopper'),
    tof.Detector(distance=158.0 * m, name='sample position'),
]

# %% Modes definition

###### initialisation of all modes
modes = []

# definition of the nominal wavelength
lambda_0 = 2.1 * A


def load_modes():
    global modes
    modes = []

    ### Maximum intensity - F0 ###
    # No choppers - all choppers stops open

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    mode.caption = 'maximum intensity - F0'
    modes.append(mode)

    ### Maximum experimental beam - F1/F2 ###
    # Only FC2A is running

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(3.1 * A)  # lambda_0)
    mode.caption = 'maximum experimental beam - F1+F2'
    #                distance  freq.     offset  title  opening/closing
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Pulse shaping mode - high flux - single frame - PS0/PS1 ###
    # This mode runs **PSC1**, **PSC3**, **FC1A** and **FC2A**.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    mode.caption = 'pulse shaping (HF) - PS0+PS1'
    # distance  freq. offset  title opening/closing distance for phase calculation
    mode.add_chopper(
        6.450 * m, 168 * Hz, 0 * deg, 'PSC1', [0], [144], AntiClockwise, 6.9125 * m
    )
    mode.add_chopper(
        7.375 * m, 168 * Hz, 0 * deg, 'PSC3', [0], [144], Clockwise, 6.9125 * m
    )
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Pulse shaping mode - medium resolution - single frame - PS2 ###
    # This mode runs **PSC1**, **PSC2**, **FC1A** and **FC2A**.

    # **distance for phase** corresponds to the distance between PSC choppers which is
    # used for phase shift calculation, if not provided, primary distance is used

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    mode.caption = 'pulse shaping (MR) - PS2'
    # distance  freq. offset title opening/closing distance for phase calculation
    mode.add_chopper(
        6.450 * m, 168 * Hz, 0 * deg, 'PSC1', [0], [144], AntiClockwise, 6.650 * m
    )
    mode.add_chopper(
        6.850 * m, 168 * Hz, 0 * deg, 'PSC2', [0], [144], Clockwise, 6.650 * m
    )
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Pulse shaping mode - high resolution - single frame - PS3 ###
    # This mode runs **PSC1**, **PSC2**, **FC1A** and **FC2A**.

    # **distance for phase** corresponds to the distance between PSC choppers which is
    # used for phase shift calculation, if not provided, primary distance is used

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    mode.caption = 'pulse shaping (HR) - PS3'
    # distance freq. offset title opening/closing distance for phase calculation
    mode.add_chopper(
        6.450 * m, 168 * Hz, 0 * deg, 'PSC1', [0], [144], AntiClockwise, 6.550 * m
    )
    mode.add_chopper(
        6.650 * m, 168 * Hz, 0 * deg, 'PSC2', [0], [144], Clockwise, 6.550 * m
    )
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (HF) - 8 opening - single frame - M0/M1 ###
    # This mode runs **MCA** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCA** is 70 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 70

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (HF) 8X - M0+M1'
    #                distance  freq.    offset    title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.300 * m,
        fMC * Hz,
        2.5 * deg,
        'MCA',
        list(np.arange(0, 360, 45)),  # opening
        list(np.arange(5, 360, 45)),
    )  # closing
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (MR) - 8 opening - single frame - M2 ###
    # This mode runs **MCA** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCA** is 140 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 140

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (MR) 8X - M2'
    #                distance  freq.    offset    title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.300 * m,
        fMC * Hz,
        2.5 * deg,
        'MCA',
        list(np.arange(0, 360, 45)),  # opening
        list(np.arange(5, 360, 45)),
    )  # closing
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (HR) - 8 opening - single frame - M3 ###
    # This mode runs **MCA** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCA** is 280 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 280

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (HR) 8X - M3'
    #                distance  freq.    offset    title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.300 * m,
        fMC * Hz,
        2.5 * deg,
        'MCA',
        list(np.arange(0, 360, 45)),  # opening
        list(np.arange(5, 360, 45)),
    )  # closing
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode - 8 opening - double frame ###
    # This mode runs **MCA** with 8 openings and **FC1A** and **FC2A**
    # in reduced speed to double the wavelength frame.
    # The frequency of the **MCA** chopper can be adjusted as necessary.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 280

    # clear up the choppers and adjust the proper for the mode
    mode.caption = f'modulation 8X ({fMC} Hz - double frame)'
    #                distance  freq.     offset        title   opening/closing
    mode.add_chopper(8.283 * m, 7 * Hz, (72 / 2 + 12) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.300 * m,
        fMC * Hz,
        2.5 * deg,
        'MCA',
        list(np.arange(0, 360, 45)),
        list(np.arange(5, 360, 45)),
    )
    mode.add_chopper(79.975 * m, 7 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (HF)- 16 opening - single frame - M0/M1 ###
    # This mode runs **MCB** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCB** is 70 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 70

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (HF) 16X - M0+M1'
    #                distance  freq.     offset        title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.350 * m,
        fMC * Hz,
        2.5 * deg,
        'MCB',
        list(np.arange(0, 360, 22.5)),
        list(np.arange(5, 360, 22.5)),
    )
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (MR) - 16 opening - single frame - M2 ###
    # This mode runs **MCB** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCB** is 140 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 140

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (MR) 16X - M2'
    #                distance  freq.     offset        title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.350 * m,
        fMC * Hz,
        2.5 * deg,
        'MCB',
        list(np.arange(0, 360, 22.5)),
        list(np.arange(5, 360, 22.5)),
    )
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode (HR) - 16 opening - single frame - M3 ###
    # This mode runs **MCB** with 8 openings, **FC1A** and **FC2A**.
    # The frequency of the **MCB** is 280 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 280

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'modulation (HR) 16X - M3'
    #                distance  freq.     offset        title   opening/closing
    mode.add_chopper(8.283 * m, 28 * Hz, (72 / 2 + 6) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.350 * m,
        fMC * Hz,
        2.5 * deg,
        'MCB',
        list(np.arange(0, 360, 22.5)),
        list(np.arange(5, 360, 22.5)),
    )
    mode.add_chopper(79.975 * m, 14 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### Modulation mode - 16 opening - double frame ###
    # This mode runs **MCB** with 16 openings and **FC1A** and **FC2A**
    # in reduced speed to double the wavelength frame.
    # The frequency of the **MCB** chopper can be adjusted as necessary.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    fMC = 280

    # clear up the choppers and adjust the proper for the mode
    mode.caption = f'modulation 16X ({fMC} Hz, double frame)'
    #                distance  freq.     offset        title   opening/closing
    mode.add_chopper(8.283 * m, 7 * Hz, (72 / 2 + 12) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.350 * m,
        fMC * Hz,
        2.5 * deg,
        'MCB',
        list(np.arange(0, 360, 22.5)),
        list(np.arange(5, 360, 22.5)),
    )
    mode.add_chopper(79.975 * m, 7 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])

    modes.append(mode)

    ### SANS + modulation mode - double frame - DS0 ###
    # This mode runs **MCC** with 8 openings (180 + 7x4) and **FC1A** and **FC2A**
    # in reduced speed to double the wavelength frame.
    # The frequency of the ***MCC*** chopper should be 70 Hz.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(4.0 * A)
    fMC = 70

    # clear up the choppers and adjust the proper for the mode
    mode.caption = 'SANS + modulation - DS0'
    # distance freq. offset title opening/closing
    mode.add_chopper(8.283 * m, 7 * Hz, (72 / 2 + 12) * deg, 'FC1A', [0], [72])
    mode.add_chopper(
        9.875 * m,
        fMC * Hz,
        (180 + 2.5) * deg,
        'MCC',
        list(np.arange(0, 150, 22.5)) + [160.0],  # noqa: RUF005
        list(np.arange(5, 150, 22.5)) + [340.0],  # noqa: RUF005
    )
    mode.add_chopper(79.975 * m, 7 * Hz, 175 / 2 * deg, 'FC2A', [0], [175])
    modes.append(mode)

    ### SANS + pulse shaping mode - alternating frame - DS1 ###
    # This mode runs **PSC1**, **PSC3**, **FC1A**, **FC1B** and **FC2B**.

    # create the mode and add choppers with the proper setting for the mode
    mode = beer_mode(lambda_0)
    mode.caption = 'SANS + pulse shaping (HF) - DS1'
    # distance freq. offset title opening/closing distance for phase calculation
    mode.add_chopper(
        6.450 * m, 168 * Hz, 0 * deg, 'PSC1', [0], [144], AntiClockwise, 6.9125 * m
    )
    mode.add_chopper(
        7.375 * m, 168 * Hz, 0 * deg, 'PSC3', [0], [144], Clockwise, 6.9125 * m
    )
    mode.add_chopper(8.283 * m, 14 * Hz, (72 / 2 - 9) * deg, 'FC1A', [0], [72])
    mode.add_chopper(8.317 * m, 63 * Hz, 180 / 2 * deg, 'FC1B', [0], [180])
    mode.add_chopper(80.025 * m, 7 * Hz, 85 / 2 * deg, 'FC2B', [0], [85])
    modes.append(mode)


# loading the modes
load_modes()
# %% Mode info print


def print_modes_info(mode_id='all', index=-1):
    """
    Print the summary of the loaded modes

    Parameters
    ----------
    mode_id : String, optional, default 'all'
        Mode ID string included in the caption. Try following F0, F1, F2, PS0, PS1, PS2,
        PS3, M0, M1, M2, M3, IM0, IM1, DS0, DS1, or print the modes first.

    index : Integer, optional, default -1
        Number of pulses to be simulated

    """
    print(f'In total {len(modes)} modes were loaded.')  # noqa: T201
    if (index == -1) and (mode_id == 'all'):
        for i, mode in enumerate(modes):
            print(f'Mode {i + 1} (index: {i})\n{mode}')  # noqa: T201
    elif (index != -1) and (mode_id == 'all'):
        print(f'Mode {index + 1} (index: {index})\n{modes[index]}')  # noqa: T201
    elif (index == -1) and (mode_id != 'all'):
        i = get_mode_index(mode_id)
        print(f'Mode {i + 1} (index: {i})\n{modes[i]}')  # noqa: T201


def get_mode_index(mode_id, verbose=False):
    """
    Search the index of the mode based on the mode ID

    Parameters
    ----------
    mode_id : String
        String contained in the caption of the mode
    verbose : Boolean
        Whether the info is printed in the output

    Returns
    -------
    mode_index : Integer
        Index of the mode in the all modes list (zero-based)

    """
    result = -1
    for i, mode in enumerate(modes):
        if mode_id in mode.caption:
            if verbose:
                print(f'Mode {mode_id} has index: {i}')  # noqa: T201
            result = i
            break  # get just the first appearance

    if result == -1:
        result = 0
        print(f'No mode has ID: "{mode_id}". Selecting mode "{modes[result].caption}"')  # noqa: T201

    return result


# %% TOF diagrams


# definition of the function for running and plotting
def run_and_plot_mode(
    mode_id, n_pulses, neutrons=100_000, wmax=15.0, mode_index=-1, save_fig=False
):
    """
    Run selected mode and plot the TOF diagram. Number of neutrons is set to smaller
    value by default (100_000) to get data quickly. For better statistic increase it.

    Parameters
    ----------
    mode_id : String
        Mode ID string included in the caption. Try following F0, F1, F2, PS0, PS1, PS2,
        PS3, M0, M1, M2, M3, IM0, IM1, DS0, DS1, or print the modes first.

    n_pulses : Integer
        Number of pulses to be simulated

    neutrons : Integer, optional, default 100_000
        Number of neutron to simulate per pulse.

    wmax : Float, optional, default 15.0
        Maximum wavelength simulated.

    mode_index : Integer, optional, default -1
        Alternatively mode index can be used instead of mode ID.
        Useful when ID has not specific string sequence as double frame ones.
        If index is provided them mode_id is overwritten.
        mode_index is zero based!

    save_fig :  Boolean, optional, default False
        Save chart to the PDF file?

    Returns
    -------
    mode_index : Integer
        Index of selected mode ID, useful for plotting detector's and chopper's info
        Zero based!

    tof_res : tof Results
        Information of choppers and detectors, useful for plotting

    tof_fig : tof matplotlib figure
        TOF diagram in Matplotlib format
    """
    tof_res = []

    # search the index of the mode based on the mode string
    if mode_index == -1:
        mode_index = get_mode_index(mode_id)

    print('Running and plotting the model ... please wait.')  # noqa: T201

    # we have to run the model
    pulse_ = tof.Source(
        neutrons=neutrons, facility='ess', pulses=n_pulses, wmax=wmax * A
    )
    model_tof = tof.Model(
        source=pulse_, choppers=modes[mode_index].choppers, detectors=detectors
    )
    tof_res = model_tof.run()

    # plotting the tof
    tof_plot = tof_res.plot(visible_rays=5000, blocked_rays=1000, figsize=(9, 6))
    tof_fig = tof_plot.fig
    tof_ax = tof_plot.ax

    # some adjustment
    tof_ax.set_title(f'BEER TOF - {modes[mode_index].caption}')
    tof_fig.set_layout_engine(layout='tight')
    tof_fig.canvas.manager.set_window_title(f'BEER TOF - {modes[mode_index].caption}')

    plt.show()

    print(f'Number of simulated pulses: {n_pulses}')  # noqa: T201
    print(modes[mode_index])  # noqa: T201

    if save_fig:
        tof_fig.savefig(f'BEER_TOF_{modes[mode_index].caption}.pdf', format='pdf')

    return mode_index, tof_res, tof_fig


# %% Plotting the results from detectors


def plot_detectors(tof_res, mode_index, det_index=0, pulse_list=None, save_fig=False):
    """
    Plotting the chart with information about the pulses at the detectors

    Parameters
    ----------
    tof_res : tof Results
        Output of the function plot

    mode_index : Integer
        Index of the plotted and run mode. Zero-based!

    det_index : Integer
        Index of the detector to plot. Use the print_detectors function to get info
        about available detectors. Zero-based!

    pulse_list : [Integer] default [0]
        List of indexes of the pulses to plot. Zero-based!

    save_fig : Boolean, optional default False
        Save the chart to the PDF file?

    Returns
    -------
    fig_det : matplotlib figure
        The figure of the detector charts in Matplotlib figure format
    """
    if pulse_list is None:
        pulse_list = [0]
    # chart definition
    fig_det, ax_det = plt.subplots(1, 2, figsize=(9, 4))

    # getting the data as the histograms
    det_name = detectors[det_index].name
    da_toa = tof_res.detectors[det_name].toa.data.copy()
    # create DataGroup where a DataArray is associated with a pulse
    tofs = sc.DataGroup()
    for i in range(da_toa.sizes['pulse']):
        if da_toa['pulse', i].data.sum('event').value:
            tofs[f'pulse:{i}'] = da_toa['pulse', i].hist(toa=3000)
        else:
            tofs[f'pulse:{i}'] = da_toa['pulse', i]

    da_wlgth = tof_res.detectors[det_name].wavelength.data.copy()
    waves = sc.DataGroup()
    for i in range(da_wlgth.sizes['pulse']):
        if da_wlgth['pulse', i].data.sum('event').value:
            waves[f'pulse:{i}'] = da_wlgth['pulse', i].hist(wavelength=300)
        else:
            waves[f'pulse:{i}'] = da_wlgth['pulse', i]

    info = ''

    # converting toa to ms and consider midpoints of tof and wlgth
    for value in pulse_list:
        tofs[f'pulse:{value}'].coords['toa'] = sc.midpoints(
            tofs[f'pulse:{value}'].coords['toa'].to(unit='ms')
        )
        waves[f'pulse:{value}'].coords['wavelength'] = sc.midpoints(
            waves[f'pulse:{value}'].coords['wavelength']
        )

    # plotting data
    pp.plot(
        {f'Pulse: {value + 1}': tofs[f'pulse:{value}'] for value in pulse_list},
        ax=ax_det[0],
        linestyle='solid',
        marker='',
    )
    pp.plot(
        {f'Pulse: {value + 1}': waves[f'pulse:{value}'] for value in pulse_list},
        ax=ax_det[1],
        linestyle='solid',
        marker='',
    )

    # order the pulses by TOF
    order = [
        [value, tofs[f'pulse:{value}'].coords['toa'].mean().value]
        for value in pulse_list
    ]
    order = sorted(order, key=lambda column: column[1], reverse=False)

    # calculate info about the frame crossing
    for index, _ in enumerate(order):
        if len(order) > 1 and index < len(order) - 1:
            y0 = tofs[f'pulse:{order[index][0]}'].data.values
            tof0 = tofs[f'pulse:{order[index][0]}'].coords['toa'].values[y0 > 0].max()
            y1 = tofs[f'pulse:{order[index + 1][0]}'].data.values
            tof1 = (
                tofs[f'pulse:{order[index + 1][0]}'].coords['toa'].values[y1 > 0].min()
            )
            info += (
                f'P{order[index][0] + 1}:{order[index + 1][0] + 1} '
                f'{"gap" if tof1 - tof0 >= 0 else "overlap"} = {tof1 - tof0:0.3f} ms'
            )
            if index != len(order) - 2:
                info += '\n'

    # show the info
    ax_det[0].text(
        0.03,
        0.96,
        f'Frame crossing info\n{info}',
        fontsize=8,
        ha='left',
        va='top',
        transform=ax_det[0].transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round', alpha=0.6),  # noqa: C408
    )

    # info about wavelength
    info = ''
    for index, value in enumerate(pulse_list):
        count_wave = waves[f'pulse:{value}'].data.values
        wave = waves[f'pulse:{value}'].coords['wavelength'].values[count_wave > 0]
        w_max = wave.max()
        w_min = wave.min()
        w_band = w_max - w_min
        w_center = (w_max + w_min) / 2
        info += (
            f'P{value + 1}: '
            r'$\Delta \lambda$ = '
            f'{w_band:0.3f} Å;'
            r' $\lambda_{mid}$'
            f' = {w_center:0.3f} Å'
        )
        if index != len(pulse_list) - 1:
            info += '\n'

    ax_det[1].text(
        0.03,
        0.96,
        f'{info}',
        fontsize=8,
        ha='left',
        va='top',
        transform=ax_det[1].transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round', alpha=0.6),  # noqa: C408
    )

    # chart adjustment
    ax_det[0].set_ylabel('[counts]')
    ax_det[0].legend(fontsize=9, loc='upper right')

    ax_det[1].set_ylabel('[counts]')
    ax_det[1].legend(fontsize=9, loc='upper right')

    fig_det.canvas.manager.set_window_title(f'{det_name} - {modes[mode_index].caption}')
    fig_det.suptitle(f'{det_name} - {modes[mode_index].caption}')
    fig_det.tight_layout()

    if save_fig:
        fig_det.savefig(f'{det_name}-{modes[mode_index].caption}.pdf', format='pdf')

    plt.show()

    return fig_det


def print_detectors_info():
    """
    Print the detectors info with indexes
    """
    print(f'In total {len(detectors)} detectors are loaded.')  # noqa: T201
    for i, detector in enumerate(detectors):
        print(  # noqa: T201
            f'Detector [index {i}], name: "{detector.name}", '
            f'distance: {detector.distance.value} [m]'
        )


# %% Plot the results from the choppers


def plot_choppers(tof_res, mode_index, chop_index, pulse_list=None, save_fig=False):
    """
    Plotting the chart with information about the pulses at the choppers

    Parameters
    ----------
    tof_res : tof Results
        Output of the function plot

    mode_index : Integer
        Index of the plotted and run mode. Zero based!

    chop_index : Integer
        Index of the chopper to plot. Use print_choppers function to get info
        about available choppers. Zero based!

    pulse_list : [Integer] default [0]
        List of indexes of the pulses to plot. Zero based!

    save_fig : Boolean, optional default False
        Save the chart to the PDF file?

    Returns
    -------
    fig_chop : matplotlib figure
        The figure of the chopper chart in Matplotlib figure format
    """
    if pulse_list is None:
        pulse_list = [0]
    # chart definition
    fig_chop, ax_chops = plt.subplots(1, 2, figsize=(9, 4))

    info = ''
    det_name = modes[mode_index].choppers[chop_index].name

    for i in pulse_list:
        # getting the tofs data not binned yet
        tofs = tof_res.choppers[det_name].toa.data['pulse', i].copy()
        tofs.coords['toa'] = tofs.coords['toa'].to(unit='ms')

        # blocked
        tofs_ = tofs.copy()
        # tofs_ = sc.DataArray(data=tofs.data, coords=tofs.coords)
        # tofs_.masks['visible'] = ~reduce(lambda a, b: a | b, tofs.masks.values())

        del tofs.masks['blocked_by_me']

        # get proper binning to align visible and blocked
        toa_bins = sc.linspace(
            dim='toa',
            num=300,
            unit=tofs.coords['toa'].unit,
            start=tofs.coords['toa'].min().value,
            stop=tofs.coords['toa'].max().value,
        )
        tofs = tofs.hist(toa=toa_bins)
        tofs_ = tofs_.hist(toa=toa_bins)

        # getting the wavelength data not binned yet visible and blocked
        waves = tof_res.choppers[det_name].wavelength.data['pulse', i].copy()
        # blocked
        waves_ = waves.copy()

        del waves.masks['blocked_by_me']

        # get proper binning to align visible and blocked
        wlgth_bins = sc.linspace(
            dim='wavelength',
            num=300,
            unit='angstrom',
            start=waves.coords['wavelength'].min().value,
            stop=waves.coords['wavelength'].max().value,
        )
        waves = waves.hist(wavelength=wlgth_bins)
        waves_ = waves_.hist(wavelength=wlgth_bins)

        ax_chops[0].plot(tofs, label=f'transmitted - P{i + 1}')
        ax_chops[0].plot(tofs_, label=f'blocked - P{i + 1}')
        ax_chops[1].plot(waves, label=f'transmitted - P{i + 1}')
        ax_chops[1].plot(waves_, label=f'blocked - P{i + 1}')

        info += f'(P{i + 1}) = {tofs_.sum().value / (tofs_.sum().value + tofs.sum().value) * 100:0.1f}% '  # noqa: E501

    # put some info
    ax_chops[0].text(
        0.5,
        1.06,
        f'Blocked ratio {info}',
        fontsize=8,
        ha='center',
        va='top',
        transform=ax_chops[0].transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round', alpha=0.6),  # noqa: C408
    )
    # chart adjustment
    ax_chops[0].set_xlabel('toa [ms]')
    ax_chops[0].set_ylabel('[counts]')
    ax_chops[0].legend(fontsize=9)

    ax_chops[1].set_xlabel('Wavelength [Å]')
    ax_chops[1].set_ylabel('[counts]')
    ax_chops[1].legend(fontsize=9)

    fig_chop.canvas.manager.set_window_title(
        f'{det_name} - {modes[mode_index].caption}'
    )
    fig_chop.suptitle(f'{det_name} - {modes[mode_index].caption}')
    fig_chop.tight_layout()

    if save_fig:
        fig_chop.savefig(f'{det_name}-{modes[mode_index].caption}.pdf', format='pdf')

    plt.show()

    return fig_chop


def print_choppers_info(mode_index):
    """
    Print the chopper information for selected mode index

    Parameters
    ----------
    mode_index : Integer
        Index of the mode for which to print the choppers' info. Zero-based!

    """
    print(  # noqa: T201
        f'For selected mode [{mode_index}] "{modes[mode_index].caption}": '
        f'{len(modes[mode_index].choppers)} chopper(s) is/are defined.'
    )
    for i, chopper in enumerate(modes[mode_index].choppers):
        print(  # noqa: T201
            f'Chopper[{i}] name: "{chopper.name}", '
            f'dist: {chopper.distance.value:6.3f} [m], '
            f'freq: {chopper.frequency.value:3.0f} [Hz], '
            f'phase: {chopper.phase.value:6.2f} [º], '
            f'cutouts: {len(chopper.open):2d}, '
            f'dir: {"-->" if chopper.direction == Clockwise else "<--"}'
        )


def get_chopper_index(mode_index, chopper_id, verbose=False):
    """
    Search the index of the chopper in the mode based on the chopper ID

    Parameters
    ----------
    mode_index : Integer
        Index of the mode in the all modes list (zero-based)
    chopper_id : String
        String contained in the caption of the chopper
    verbose : Boolean
        Whether the info is printed in the output

    Returns
    -------
    chopper_id : String
        String contained in the caption of the chopper

    """
    result = -1
    for i, chopper in enumerate(modes[mode_index].choppers):
        if chopper_id in chopper.name:
            if verbose:
                print(f'Chopper {chopper_id} has index: {i}')  # noqa: T201
            result = i
            break  # get just the first appearance

    if result == -1:
        result = 0
        print(  # noqa: T201
            f'No chopper has ID: "{chopper_id}". Selecting chopper '
            f'"{modes[mode_index].chopper[result].name}"'
        )

    return result


def draw_chopper(mode_index, chopper_index):
    """
    Graphical visualisation of the chopper for selected mode
    Works only in Jupyter notebook!

    Parameters
    ----------
    mode_index : Integer
        Index of the mode. Zero-based!

    chopper_index : Integer
        Index of the chopper for which to display chopper. Zero-based!

    """
    to_show = sc.DataGroup()
    chop = modes[mode_index].choppers[chopper_index]
    shift = chop.phase.value * (1 if chop.direction == Clockwise else -1)
    # shift = chop.phase.value

    to_show[chop.name] = disk_chopper.DiskChopper(  # scn.chopper.DiskChopper(
        axle_position=sc.vector(value=np.array([0, 0, chop.distance.value]), unit='m'),
        frequency=chop.frequency * (-1 if chop.direction == Clockwise else 1),
        beam_position=0.0 * sc.Unit('deg'),
        phase=chop.phase,
        slit_begin=sc.array(
            dims=['slit'], values=[i.value + shift for i in chop.open], unit='deg'
        ),
        slit_end=sc.array(
            dims=['slit'], values=[i.value + shift for i in chop.close], unit='deg'
        ),
        slit_height=0.05 * m,
        radius=0.375 * m,
    )
    return to_show[chop.name]


# %% Comparison of the modes


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(
            f"{prefix}[{'█' * x}{('.' * (size - x))}] {j}/{count} Estimated wait {time_str} ",  # noqa: E501
            end='\r',
            file=out,
            flush=True,
        )

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def get_wave_for_all_modes(neutrons=1_000_000, pulses=2, wmax=15.0):
    """
    Run the simulation for all the modes and store the wavelength related results.

    Parameters
    ----------
    neutrons : Integer, optional
        Number of neutron to simulate per pulse. The default is 1_000_000.

    pulses : Integer, optional
        Number of pulses to simulate, The default is 2.

    wmax : Float, optional
        Maximum wavelength simulated. The default is 15.0.

    Returns
    -------
    waves : List of results
        List of TOF simulated wavelength related results.
    """
    waves = []
    for i in progressbar(range(len(modes)), 'Computing: ', 40):
        pulse_ = tof.Source(
            neutrons=neutrons, facility='ess', pulses=pulses, wmax=wmax * A
        )
        model_ = tof.Model(
            source=pulse_, detectors=detectors, choppers=modes[i].choppers
        )
        res_ = model_.run()

        # data added from sample detector not binned
        # (binning adjusted during comparison)
        waves.append(res_.detectors['sample position'].wavelength.data)
    print(f'{len(waves)} models were run.')  # noqa: T201
    return waves


def plot_comparison(waves, what, pulses=None, save_fig=False):
    """
    Plot the chart comparing wavelength distribution of various modes and pulses.

    Parameters
    ----------
    waves : List of results
        Wavelength events from running all the modes. Use output from
        "get_wave_for_all_modes" function.

    what : List of strings
        Part of the mode's name which you want to compare. Be specific, if two
        occurrences happened then longer text (ex. "8X - M2").

    pulses : List of array, optional, The default is None
        The list of Integer array representing what pulse to plot from each
        mode. Ex: [[0],[1, 2]] - for the first mode in "what" plot the first
        pulse and for the second mode plot pulse 2 and 3 (zero based!).
        If not provided, only first pulse plotted for each mode. The number of
        members has to be the same as for "what".

    save_fig : Boolean, Optional, The default is False.
        Do you want to save the chart as PDF?
    """

    # plot definition
    fig_mode_comp, mode_comp_ax = plt.subplots(1, 1, figsize=(9, 6))

    # adjusting pulses if None
    if pulses is None:
        pulses = []
        for _ in range(len(what)):
            pulses.append([0])

    # getting min and max for proper binning
    w_min, w_max = 100, 0
    index = 0
    for i, wave in enumerate(waves):
        if any(
            select in modes[i].caption for select in what
        ):  # select what you want to compare
            for j in pulses[index]:
                w_min = min(w_min, wave['pulse', j].coords['wavelength'].min().value)
                w_max = max(w_max, wave['pulse', j].coords['wavelength'].max().value)
            index += 1
    bins = sc.linspace(
        dim='wavelength', num=300, unit='angstrom', start=w_min, stop=w_max
    )

    # plotting the comparison
    index = 0
    for i, wave in enumerate(waves):
        if any(
            select in modes[i].caption for select in what
        ):  # select what you want to compare
            for j in pulses[index]:
                info = ''
                if len(pulses[index]) > 1:
                    info = f' - P{j + 1}'

                wave_ = wave['pulse', j].hist(wavelength=bins)
                mode_comp_ax.plot(
                    sc.midpoints(wave_.coords['wavelength']),
                    wave_,
                    ('--' if 'modulation' in modes[i].caption else '-'),
                    label=modes[i].caption + info,
                )
            index += 1

    mode_comp_ax.legend(fontsize=10)
    mode_comp_ax.set_title('BEER modes intensity comparison at sample position')
    mode_comp_ax.set_xlabel('Wavelength [Å]')
    mode_comp_ax.set_ylabel('[counts]')

    fig_mode_comp.canvas.manager.set_window_title(
        'BEER modes intensity comparison at sample position'
    )
    fig_mode_comp.tight_layout()

    if save_fig:
        fig_mode_comp.savefig(
            'BEER_modes_intensity_comparison_at_sample_position.pdf', format='pdf'
        )
