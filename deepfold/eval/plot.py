from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import deepfold.common.residue_constants as rc
from deepfold.eval.distogram import compute_distogram, compute_predicted_distogram
from deepfold.eval.msa import compute_neff_v2 as compute_neff


def _set_size(
    w: float,
    h: float,
    ax: plt.Axes | None = None,
):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def find_cluster_boundaries(a: np.ndarray) -> List[Tuple[int, int, int]]:
    a = np.asarray(a)
    assert a.ndim == 1
    boundaries = []
    start = 0
    current = a[0]

    for i in range(1, len(a)):
        if a[i] != current:
            boundaries.append((start, i - 1, current))
            start = i
            current = a[i]
    boundaries.append((start, len(a) - 1, current))

    return boundaries


def plot_distogram(
    outputs: dict,
    asym_id: Optional[np.ndarray] = None,
    ncols: int = 5,
    sort: bool = False,
    fig_kwargs: dict = dict(),
) -> plt.Figure:
    num_models = len(outputs)
    nrows = int((num_models + ncols + 1) / ncols) * 2
    fig_kwargs.update(
        {
            "figsize": (5 * ncols, 9 * nrows),
            "dpi": 150.0,
        }
    )
    fig = plt.figure(**fig_kwargs)

    if sort:
        sorted_outputs = {k: outputs[k] for k in sorted(outputs.keys())}
    else:
        sorted_outputs = outputs

    for n, (model_name, value) in enumerate(sorted_outputs.items(), start=1):
        if asym_id is not None:
            boundaries = find_cluster_boundaries(asym_id)

        # From the distogram head
        m = n - 1
        ax = fig.add_subplot(2 * nrows, ncols, 2 * (m // ncols) * ncols + m % ncols + 1)
        ax.set_title(f"{model_name} from the Distogram Head")
        distogram = compute_predicted_distogram(value["distogram_logits"])
        im1 = ax.imshow(distogram, cmap="viridis_r", vmin=0, vmax=22)

        # Draw chain breaks
        if asym_id is not None:
            for i, _, _ in boundaries[1:]:
                z = i - 0.5
                ax.axhline(y=z, color="k", linestyle="-", alpha=0.6)
                ax.axvline(x=z, color="k", linestyle="-", alpha=0.6)

        # Pseudo beta positions
        all_atom_positions = value["final_atom_positions"]
        all_atom_mask = value["final_atom_mask"]
        ca_idx = rc.atom_order["CA"]
        is_gly = all_atom_mask[:, rc.restype_order["G"]] < 0.5
        cb_idx = rc.atom_order["CB"]
        pseudo_beta = np.where(
            np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
            all_atom_positions[..., ca_idx, :],
            all_atom_positions[..., cb_idx, :],
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical", ticks=[])

        # From the final atom positions
        ax = fig.add_subplot(2 * nrows, ncols, 2 * (m // ncols) * ncols + ncols + m % ncols + 1)
        ax.set_title(f"{model_name} from the Structure")
        distogram = compute_distogram(pseudo_beta)
        im2 = ax.imshow(np.clip(distogram, a_min=0, a_max=22), cmap="viridis_r", vmin=0, vmax=22)

        # Draw chain breaks
        if asym_id is not None:
            for i, _, _ in boundaries[1:]:
                z = i - 0.5
                ax.axhline(y=z, color="k", linestyle="-", alpha=0.6)
                ax.axvline(x=z, color="k", linestyle="-", alpha=0.6)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax, orientation="vertical", ticks=[])

    return fig


def plot_predicted_alignment_error(
    outputs: dict,
    asym_id: Optional[np.ndarray] = None,
    fig_kwargs: dict = dict(),
) -> plt.Figure:
    num_models = len(outputs)
    fig_kwargs.update(
        {
            "figsize": (5 * num_models, 4),
            "dpi": 150.0,
        }
    )
    fig = plt.figure(**fig_kwargs)
    for n, (model_name, value) in enumerate(outputs.items(), start=1):
        # Draw PAE
        ax = fig.add_subplot(1, num_models, n)
        ax.set_title(model_name)
        im = ax.imshow(value["predicted_aligned_error"], label=model_name, cmap="Greens", vmin=0, vmax=30)

        # Draw chain breaks
        if asym_id is not None:
            boundaries = find_cluster_boundaries(asym_id)
            for i, _, _ in boundaries[1:]:
                z = i - 0.5
                ax.axhline(y=z, color="k", linestyle="-", alpha=0.6)
                ax.axvline(x=z, color="k", linestyle="-", alpha=0.6)
        # fig.colorbar(im)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    return fig


PLDDT_COLORS = [
    (0.0, "#ff7d45"),
    (0.5, "#ffdb13"),
    (0.7, "#65cbf3"),
    (0.9, "#0053d6"),
    (1.0, "#0053d6"),
]

plddt_cmap = LinearSegmentedColormap.from_list(name="plddt", colors=PLDDT_COLORS)


def plot_plddt(
    outputs: dict,
    asym_id: Optional[np.ndarray] = None,
    scale_with_len: bool = False,
    fig_kwargs: dict = dict(),
) -> plt.Figure:
    # Scale with the length
    scale = 1
    if scale_with_len:
        max_len = 0
        for v in outputs.values():
            max_len = max(max_len, v["plddt"].shape[0])
        scale = max(scale, 1 + max_len // 200)

    fig_kwargs.update(
        {
            "figsize": (12 * scale, 5),
            "dpi": 150.0,
        }
    )
    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(rf"Predicted C$\alpha$-lDDT")

    ranking = [(v["plddt"].mean(), k) for k, v in outputs.items()]
    ranking.sort(key=lambda x: x[0], reverse=True)

    for n, (_, key) in enumerate(ranking, start=1):
        model_name = key
        value = outputs[key]

        # Draw plDDT
        x = np.arange(len(value["plddt"])) + 1
        y = value["plddt"]
        ax.scatter(
            x=x,
            y=y,
            c=(y / 100),
            cmap=plddt_cmap,
            marker=".",
            zorder=2,
        )
        ax.plot(x, y, "-", label=f"Rank {n} ({model_name})", zorder=1)

        # Draw chain breaks
        if asym_id is not None:
            boundaries = find_cluster_boundaries(asym_id)
            for i, _, _ in boundaries[1:]:
                z = i - 0.5
                ax.axvline(x=z, color="k", linestyle="-.", alpha=0.6)
    ax.legend(ncols=4, fontsize="smaller")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Positions")
    ax.set_ylabel("plDDT")
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plddt_cmap), ax=ax)

    return fig


def plot_msa(
    feature_dict: Dict[str, np.ndarray],
    sort_lines: bool = True,
    dpi: float = 150.0,
    scale_with_len: bool = False,
) -> plt.Figure:
    scale = 1
    seqlen = feature_dict["msa"][0]
    if scale_with_len:
        scale = max(scale, 1 + seqlen // 200)

    if "asym_id" in feature_dict:  # Multimer
        ls = [0]
        k = feature_dict["asym_id"][0]
        for i in feature_dict["asym_id"]:
            if i == k:
                ls[-1] += 1
            else:
                ls.append(1)
            k = i
    else:
        ls = [len(seqlen)]
    ln = np.cumsum([0] + ls)

    try:
        n = feature_dict["num_alignments"][0]
    except:
        n = feature_dict["num_alignments"]

    msa = feature_dict["msa"][:n]
    gap = msa != 21
    qid = msa == seqlen
    gapid = np.stack([gap[:, ln[i] : ln[i + 1]].max(-1) for i in range(len(ls))], -1)
    lines = []
    nn = []
    for g in np.unique(gapid, axis=0):
        i = np.where((gapid == g).all(axis=-1))
        qid_ = qid[i]
        gap_ = gap[i]
        seqid = np.stack([qid_[:, ln[i] : ln[i + 1]].mean(-1) for i in range(len(ls))], -1).sum(-1) / (g.sum(-1) + 1e-8)
        non_gaps = gap_.astype(float)
        non_gaps[non_gaps == 0] = np.nan
        if sort_lines:
            lines_ = non_gaps[seqid.argsort()] * seqid[::-1, None]
        else:
            lines_ = non_gaps[::-1] * seqid[::-1, None]
        nn.append(len(lines_))
        lines.append(lines_)

    nn = np.cumsum(np.append(0, nn))
    lines = np.concatenate(lines, 0)
    fig = plt.figure(figsize=(12 * scale, 5), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        lines,
        cmap="rainbow_r",
        interpolation="nearest",
        aspect="auto",
        vmax=1.0,
        vmin=0.0,
        origin="lower",
        extent=(0, lines.shape[1], 0, lines.shape[0]),
    )
    for i in ln[1:-1]:
        ax.plot([i - 0.5, i - 0.5], [0, lines.shape[0]], color="black", alpha=0.6)
    for j in nn[1:-1]:
        ax.plot([0, lines.shape[1]], [j - 0.5, j - 0.5], color="black", alpha=0.6)

    neff = compute_neff(msa)
    title = "MSA Depth"
    title += r" ($N_{eff}=$"
    title += f"{neff:6.3f})"
    ax.set_title(title)
    ax.plot((np.isnan(lines) == False).sum(0), color="black")
    ax.set_xlim(0, lines.shape[1])
    ax.set_ylim(0, max(lines.shape[0], 100))
    fig.colorbar(im)
    ax.set_xlabel("Positions")
    ax.set_ylabel("Sequences")

    return fig
