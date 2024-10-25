# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Detector diagnostics for DREAM."""

import math
from collections.abc import Callable, Iterable, Mapping
from functools import reduce

import ipywidgets as ipw
import numpy as np
import scipp as sc
from plopp.core.typing import FigureLike

# This leads to y-axis: segment, x-axis: module
_STARTING_DIMS = ('segment', 'wire', 'strip', 'module', 'counter', 'sumo', 'sector')


class FlatVoxelViewer(ipw.VBox):
    """Interactive 2D plot of all detector voxels.

    See `DREAM Detector Diagnostics
    <../../user-guide/dream/dream-detector-diagnostics.rst>`_ for explanations
    and an example.
    """

    def __init__(
        self, data: Mapping[str, sc.DataArray], *, rasterized: bool = True
    ) -> None:
        """Create a new viewer.

        Parameters
        ----------
        data:
            Histogrammed data, one entry per bank.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        """
        self._data = self._prepare_data(data)
        self._bank_selector = _make_bank_selector(data.keys())
        self._bank = self._data[self._bank_selector.value]

        self._dim_selector = _DimensionSelector(self._bank.dims, self._update_view)

        self._fig_kwargs = {'rasterized': rasterized}
        self._figure_box = ipw.HBox([self._make_figure()])
        self._bank_selector.observe(self._select_bank, names='value')

        super().__init__(
            [
                ipw.HBox([ipw.Label('Bank:'), self._bank_selector]),
                self._figure_box,
                self._dim_selector,
            ]
        )

    def _select_bank(self, *_args: object, **_kwargs: object) -> None:
        self._bank = self._data[self._bank_selector.value]
        self._dim_selector.set_dims(self._bank.dims)
        self._update_view()

    def _update_view(self, *_args: object, **_kwargs: object) -> None:
        self._figure_box.children = [self._make_figure()]

    def _make_figure(self) -> FigureLike:
        sel = self._dim_selector.value
        fig = _flat_voxel_figure(
            self._bank, sel['horizontal'], sel['vertical'], **self._fig_kwargs
        )
        return fig

    @staticmethod
    def _prepare_data(dg: sc.DataGroup) -> sc.DataGroup:
        return sc.DataGroup(
            {
                name: bank.transpose(
                    [dim for dim in _STARTING_DIMS if dim in bank.dims]
                )
                for name, bank in dg.items()
            }
        )


class _DimensionSelector(ipw.VBox):
    def __init__(self, dims: tuple[str, ...], callback: Callable[[dict], None]) -> None:
        self._lock = False
        self._callback = callback

        self._horizontal_buttons, self._vertical_buttons = self._make_buttons(
            dims, *self._default_dims(dims)
        )

        super().__init__(
            [
                ipw.HBox([ipw.Label('X'), self._horizontal_buttons]),
                ipw.HBox([ipw.Label('Y'), self._vertical_buttons]),
            ]
        )

    def _make_buttons(
        self, dims: tuple[str, ...], h_dim: str, v_dim: str
    ) -> tuple[ipw.ToggleButtons, ipw.ToggleButtons]:
        style = {'button_width': '10em'}
        options = {dim.capitalize(): dim for dim in dims}
        h_buttons = ipw.ToggleButtons(options=options, value=h_dim, style=style)
        v_buttons = ipw.ToggleButtons(options=options, value=v_dim, style=style)
        h_buttons.observe(self.update, names='value')
        v_buttons.observe(self.update, names='value')
        return h_buttons, v_buttons

    def set_dims(self, new_dims: tuple[str, ...]) -> None:
        default_h, default_v = self._default_dims(new_dims)
        options = {dim.capitalize(): dim for dim in new_dims}
        self._lock = True
        self._horizontal_buttons.options = options
        self._vertical_buttons.options = options
        self._horizontal_buttons.value = default_h
        self._vertical_buttons.value = default_v
        self._lock = False

    @staticmethod
    def _default_dims(dims: tuple[str, ...]) -> tuple[str, str]:
        return dims[math.ceil(len(dims) / 2)], dims[0]

    @property
    def value(self):
        return {
            'horizontal': self._horizontal_buttons.value,
            'vertical': self._vertical_buttons.value,
        }

    def update(self, change: dict) -> None:
        if self._lock:
            return
        clicked = change['owner']
        other = (
            self._vertical_buttons
            if clicked is self._horizontal_buttons
            else self._horizontal_buttons
        )
        if other.value == clicked.value:
            self._lock = True  # suppress update from `other`
            other.value = change['old']
            self._lock = False
        self._callback(change)


def _flat_voxel_figure(
    data: sc.DataArray,
    horizontal_dim: str,
    vertical_dim: str,
    *,
    rasterized: bool = True,
) -> FigureLike:
    kept_dims = {horizontal_dim, vertical_dim}

    to_flatten = [dim for dim in data.dims if dim not in kept_dims]
    n = len(to_flatten)
    flatten_to_h = [horizontal_dim, *to_flatten[n // 2 :]]
    flatten_to_v = [vertical_dim, *to_flatten[: n // 2]]

    # Drop unused coordinates
    aux = data.drop_coords(list(set(data.coords.keys()) - kept_dims))
    reordered = aux.transpose(flatten_to_v + flatten_to_h)

    h_coord = reordered.coords.pop(horizontal_dim)
    v_coord = reordered.coords.pop(vertical_dim)

    flat = reordered.flatten(flatten_to_v, to='vertical').flatten(
        flatten_to_h, to='horizontal'
    )
    flat = flat.assign_coords(
        {name: sc.arange(name, flat.sizes[name], unit=None) for name in flat.dims}
    )

    # This relies on the order of flatten_to_h/v
    inner_volume_h = _product(data.sizes[d] for d in flatten_to_h[1:])
    inner_volume_v = _product(data.sizes[d] for d in flatten_to_v[1:])
    h_ticks = np.arange(0, flat.sizes['horizontal'], inner_volume_h)
    v_ticks = np.arange(0, flat.sizes['vertical'], inner_volume_v)

    h_labels = [str(value) for value in h_coord.values]
    v_labels = [str(value) for value in v_coord.values]

    fig = flat.plot(rasterized=rasterized, cbar=True)

    fig.ax.xaxis.set_ticks(ticks=h_ticks, labels=h_labels)
    fig.ax.yaxis.set_ticks(ticks=v_ticks, labels=v_labels)
    fig.canvas.xlabel = horizontal_dim.capitalize()
    fig.canvas.ylabel = vertical_dim.capitalize()

    unwrap_indices = unwrap_flat_indices_2d(
        {dim: reordered.sizes[dim] for dim in flatten_to_h},
        {dim: reordered.sizes[dim] for dim in flatten_to_v},
    )

    def format_coord(x: float, y: float) -> str:
        # Use round because axis coords are in the middle of bins.
        indices = (
            f'{key.capitalize()}: {val}'
            for key, val in unwrap_indices(round(x), round(y)).items()
        )
        return f"{{{', '.join(indices)}}}"

    fig.ax.format_coord = format_coord

    return fig


def _product(it):
    return reduce(lambda a, b: a * b, it)


def unwrap_flat_indices_2d(
    x_sizes: dict[str, int], y_sizes: dict[str, int]
) -> Callable[[int, int], dict[str, int]]:
    def unwrap(x: int, y: int) -> dict[str, int]:
        return {**_unwrap_flat_index(x, x_sizes), **_unwrap_flat_index(y, y_sizes)}

    return unwrap


def _unwrap_flat_index(index: int, sizes: dict[str, int]) -> dict[str, int]:
    res = []
    for key, size in reversed(sizes.items()):
        res.append((key, index % size))
        index //= size
    return dict(reversed(res))  # Reverse to reproduce the input order.


def _make_bank_selector(banks: Iterable[str]) -> ipw.ToggleButtons:
    options = (
        (' '.join(s.capitalize() for s in bank.split('_')), bank) for bank in banks
    )
    return ipw.ToggleButtons(options=options)
