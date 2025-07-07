# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import h5py
import scipp as sc


def _load_h5(group: h5py.Group | str, *paths: str):
    if isinstance(group, str):
        with h5py.File(group) as group:
            yield from _load_h5(group, *paths)
        return
    for path in paths:
        g = group
        for p in path.strip('/').split('/'):
            g = _unique_child_group_h5(g, p) if p.startswith('NX') else g.get(p)
        yield g


def _unique_child_group_h5(
    group: h5py.Group,
    nx_class: str,
) -> h5py.Group | None:
    out = None
    for v in group.values():
        if v.attrs.get("NX_class") == nx_class.encode():
            if out is None:
                out = v
            else:
                raise ValueError(
                    f'Expected exactly one {nx_class} group, but found more'
                )
    return out


def load_beer_mcstas(f):
    if isinstance(f, str):
        with h5py.File(f) as ff:
            return load_beer_mcstas(ff)

    data, events, params, sample_pos, chopper_pos = _load_h5(
        f,
        'NXentry/NXdetector/bank01_events_dat_list_p_x_y_n_id_t',
        'NXentry/NXdetector/bank01_events_dat_list_p_x_y_n_id_t/events',
        'NXentry/simulation/Param',
        '/entry1/instrument/components/0189_sampleMantid/Position',
        '/entry1/instrument/components/0017_cMCA/Position',
    )
    da = sc.DataArray(
        sc.array(dims=['events'], values=events[:, 0], variances=events[:, 0] ** 2),
    )
    for name, value in data.attrs.items():
        da.coords[name] = sc.scalar(value.decode())

    for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
        if label == 'p':
            continue
        da.coords[label] = sc.array(dims=['events'], values=events[:, i])
    for k, v in params.items():
        v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        da.coords[k] = sc.scalar(v)

    da.coords['sample_position'] = sc.vector(sample_pos[:], unit='m')
    da.coords['detector_position'] = sc.vector(
        list(map(float, da.coords['position'].value.split(' '))), unit='m'
    )
    da.coords['chopper_position'] = sc.vector(chopper_pos[:], unit='m')
    da.coords['x'].unit = 'm'
    da.coords['y'].unit = 'm'
    da.coords['t'].unit = 's'

    z = sc.norm(da.coords['detector_position'] - da.coords['sample_position'])
    L1 = sc.norm(da.coords['sample_position'] - da.coords['chopper_position'])
    L2 = sc.sqrt(da.coords['x'] ** 2 + da.coords['y'] ** 2 + z**2)
    # Source is assumed to be at the origin
    da.coords['L0'] = L1 + L2 + sc.norm(da.coords['chopper_position'])
    da.coords['Ltotal'] = L1 + L2
    da.coords['two_theta'] = sc.acos(-da.coords['x'] / L2)

    # Save some space
    da.coords.pop('x')
    da.coords.pop('y')
    da.coords.pop('n')
    da.coords.pop('id')

    # TODO: approximate t0 should depend on chopper information
    da.coords['approximate_t0'] = sc.scalar(6e-3, unit='s')

    # TODO: limits should be user configurable
    da.masks['two_theta'] = (
        da.coords['two_theta'] >= sc.scalar(105, unit='deg').to(unit='rad')
    ) | (da.coords['two_theta'] <= sc.scalar(75, unit='deg').to(unit='rad'))

    da.coords['event_time_offset'] = da.coords.pop('t')
    return da
