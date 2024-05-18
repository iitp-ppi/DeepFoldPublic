from typing import Optional, Tuple

import matplotlib.cm
import matplotlib.patheffects
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.eval.plot import find_cluster_boundaries, plddt_cmap

PYMOL_COLORS = [
    "#33ff33",
    "#00ffff",
    "#ff33cc",
    "#ffff00",
    "#ff9999",
    "#e5e5e5",
    "#7f7fff",
    "#ff7f00",
    "#7fff7f",
    "#199999",
    "#ff007f",
    "#ffdd5e",
    "#8c3f99",
    "#b2b2b2",
    "#007fff",
    "#c4b200",
    "#8cb266",
    "#00bfbf",
    "#b27f7f",
    "#fcd1a5",
    "#ff7f7f",
    "#ffbfdd",
    "#7fffff",
    "#ffff7f",
    "#00ff7f",
    "#337fcc",
    "#d8337f",
    "#bfff3f",
    "#ff7fff",
    "#d8d8ff",
    "#3fffbf",
    "#b78c4c",
    "#339933",
    "#66b2b2",
    "#ba8c84",
    "#84bf00",
    "#b24c66",
    "#7f7f7f",
    "#3f3fa5",
    "#a5512b",
]

pymol_cmap = ListedColormap(PYMOL_COLORS)


def kabsch(
    a: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, float)
    # a: [N_res, 3]
    b = np.asarray(b, float)
    # b: [N_res, 3]

    if weights is None:
        weights = np.ones(a.shape[:-1])
    else:
        weights = np.asarray(weights, float)

    ab = np.einsum("ji,jk->ik", weights[:, None] * a, b)
    u, _, vh = np.linalg.svd(ab)

    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1.0

    return u @ vh, u


def _plot_pseudo_3d(
    xyz: np.ndarray,
    c: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    chain_break_cut: float | None = 5.0,
    cmap: str | Colormap = "gist_rainbow",
    linewidth: float = 2.0,
    cmin: float | None = None,
    cmax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> plt.Axes:

    def rescale(
        a: np.ndarray,
        amin: Optional[float] = None,
        amax: Optional[float] = None,
    ) -> np.ndarray:
        a = a.copy()
        if amin is None:
            amin = a.min()
        if amax is None:
            amax = a.max()

        a[a < amin] = amin
        a[a > amax] = amax

        return (a - amin) / (amax - amin)

    # Segment
    xyz = np.asarray(xyz)
    # xyz: [N_res, 3]
    seg = np.concatenate([xyz[:-1, None, :], xyz[1:, None, :]], axis=-2)
    # xyz: [N_res - 1, 1, 3]
    seg_xy = seg[..., :2]
    # seg_xy: [..., 2]
    seg_z = seg[..., 2].mean(-1)
    # seg_z: [..., 1]
    order = seg_z.argsort()

    # Colors
    if c is None:
        c = np.arange(len(seg))[::-1]  # [N_res - 1]
    else:
        c = (c[1:] + c[:-1]) * 0.5

    c = rescale(c, cmin, cmax)

    if isinstance(cmap, str):
        if cmap == "gist_rainbow":
            c *= 0.75
        colors = matplotlib.cm.get_cmap(cmap)(c)
    else:
        colors = cmap(c)

    if chain_break_cut is not None:
        dists = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
        colors[..., 3] = (dists < chain_break_cut).astype(float)

    # Add shade and tint based on z-dimension
    z = rescale(seg_z, zmin, zmax)[:, None]
    tint, shade = z / 3, (z + 2) / 3
    colors[:, :3] = colors[:, :3] + (1.0 - colors[:, :3]) * tint
    colors[:, :3] = colors[:, :3] * shade

    set_limit = False
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(5)
        set_limit = True
    else:
        fig = ax.get_figure()
        if ax.get_xlim() == (0, 1):
            set_limit = True

    if set_limit:
        xy_min = xyz[:, :2].min() - linewidth
        xy_max = xyz[:, :2].max() + linewidth
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)

    ax.set_aspect("equal")

    # Linewidths
    width = fig.bbox_inches.width * ax.get_position().width
    linewidths = linewidth * 72 * width / np.diff(ax.get_xlim())
    lines = LineCollection(
        seg_xy[order],
        colors=colors[order],
        linewidths=linewidths,
        path_effects=[matplotlib.patheffects.Stroke(capstyle="round")],
    )

    return ax.add_collection(lines)


def _add_text(text: str, ax: plt.Axes) -> plt.Text:
    return plt.text(
        0.5,
        1.01,
        text,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )


def plot_protein(
    protein: protein.Protein | None = None,
    pos: np.ndarray | None = None,
    plddt: np.ndarray | None = None,
    ls: np.ndarray | None = None,
    dpi: float = 150.0,
    best_view: bool = True,
    linewidth: float = 2.0,
) -> plt.Figure:
    if protein is not None:
        pos = np.asarray(protein.atom_positions[:, rc.atom_order["CA"], :])
        if plddt is None:
            plddt = np.asarray(protein.b_factors[:, 1])

    if best_view:
        pos = _protein_best_view(pos, plddt=plddt)

    if plddt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(6)
        fig.set_figheight(3)
    else:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_figwidth(3)
        fig.set_figheight(3)

    if ls is None:
        cluster = find_cluster_boundaries(protein.chain_index.astype(int))
        ls = np.array([j - i + 1 for i, j, _ in cluster])

    fig.set_dpi(dpi)
    fig.subplots_adjust(top=0.9, bottom=0.1, right=1, left=0, hspace=0, wspace=0)

    if ls is None or len(ls) == 1:
        # Color from NTER to CTER
        plot_protein_bb(pos, coloring="NC", best_view=False, linewidth=linewidth, axes=ax1)
        _add_text("colored by N→C", ax1)
    else:
        # Color by chain
        plot_protein_bb(pos, coloring="chain", best_view=False, ls=ls, linewidth=linewidth, axes=ax1)
        _add_text("colored by chain", ax1)

    if plddt is not None:
        # Color by pLDDT
        plot_protein_bb(pos, coloring="plddt", best_view=False, plddt=plddt, linewidth=linewidth, axes=ax2)
        _add_text("colored by pLDDT", ax2)

    return fig


def _protein_best_view(
    pos: np.ndarray,
    plddt: np.ndarray | None = None,
) -> np.ndarray:
    if plddt is not None:
        weights = plddt / 100
        pos = pos - (pos * weights[:, None]).sum(0, keepdims=True) / weights.sum()
        pos = pos @ kabsch(pos, pos, weights)[1]
    else:
        pos = pos - pos.mean(axis=0, keepdims=True)
        pos = pos @ kabsch(pos, pos)[1]
    return pos


def plot_protein_bb(
    pos: np.ndarray,  # [N_res, 3]
    plddt: np.ndarray | None = None,  # [N_res]
    axes: plt.Axes | None = None,
    coloring: str = "plddt",
    ls: np.ndarray | None = None,
    best_view: bool = True,
    linewidth: float = 2.0,
):
    if plddt is None:
        plddt = np.ones(pos.shape[0]) * 50

    if best_view:
        pos = _protein_best_view(pos, plddt=plddt)

    xy_min = pos[..., :2].min() - linewidth
    xy_max = pos[..., :2].max() + linewidth
    axes.set_xlim(xy_min, xy_max)
    axes.set_ylim(xy_min, xy_max)
    axes.axis(False)

    if coloring == "NC":
        # Color from NTER to CTER
        _plot_pseudo_3d(pos, linewidth=linewidth, ax=axes)
    elif coloring == "plddt":
        # Color by pLDDT
        _plot_pseudo_3d(pos, c=plddt, cmap=plddt_cmap, cmin=50, cmax=90, linewidth=linewidth, ax=axes)
    elif coloring == "chain":
        # Color by chain
        c = np.concatenate([[n] * l for n, l in enumerate(ls)])
        num_res = len(ls)
        if num_res > 40:
            _plot_pseudo_3d(pos, c=c, linewidth=linewidth, ax=axes)
        else:
            _plot_pseudo_3d(pos, c=c, cmap=pymol_cmap, cmin=0, cmax=39, linewidth=linewidth, ax=axes)
