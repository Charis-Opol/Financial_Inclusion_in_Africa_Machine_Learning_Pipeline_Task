"""Static figures for attributions (saved to disk, never shown interactively).

One visual language across every figure, so they can be read side by side:

  - Sign is colour: a contribution that pushes the logit UP is red, DOWN is
    blue, zero is neutral. The same pair everywhere.
  - Magnitude without sign (importance) is a single blue.
  - A feature's own value, where shown, is a light-to-dark ramp of one hue;
    categories with no natural order are grey, because shading them would
    invent an order.
  - Every figure carries a caption stating the unit (logits), the reference,
    the sample size, and that the score is a ranking score, not a calibrated
    probability.

Figures describe what a model learned to rely on; they are not causal.
"""
from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: every plot here is saved to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from fin_inclusion.interpretability.bootstrap import BootstrapResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POSITIVE = "#e34948"  # pushes the logit up
NEGATIVE = "#2a78d6"  # pushes the logit down
MAGNITUDE = "#2a78d6"
UNORDERED = "#898781"
VALUE_RAMP = LinearSegmentedColormap.from_list("value", ["#cde2fb", "#3987e5", "#0d366b"])
# Identity colours for a small number of named series, always in this order
# (validated categorical slots 1-3; colour follows the series, never its rank).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

SCORE_NOTE = "Unit: logits (pre-sigmoid model output). The score is a ranking score, not a calibrated probability."
CAUSAL_NOTE = "Attributions describe the model, not causes of account ownership."

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _style(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)


def _save(fig, path: Path, caption: str) -> None:
    """Adds the caption under the plot and writes the file."""
    lines = textwrap.wrap(f"{caption} {SCORE_NOTE} {CAUSAL_NOTE}", width=int(fig.get_figwidth() * 17))
    fig.tight_layout()
    caption_height = 0.15 * (len(lines) + 1) / fig.get_figheight()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + caption_height)
    fig.text(0.02, 0.01, "\n".join(lines), ha="left", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance_bars(
    panels: Mapping[str, BootstrapResult], feature_names: Sequence[str], path: Path, title: str, caption: str
) -> None:
    """Mean |attribution| per feature with bootstrap intervals, one panel per
    sample (e.g. pooled, Uganda). Features are ordered by the FIRST panel in
    every panel, so a feature sits on the same row across panels and a
    change of rank shows as a change of bar length, not of position."""
    order = np.argsort(next(iter(panels.values())).estimate)
    labels = [feature_names[i] for i in order]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), sharex=True, sharey=True)
    for ax, (panel_title, result) in zip(np.atleast_1d(axes), panels.items()):
        estimate = result.estimate[order]
        ax.barh(labels, estimate, height=0.62, color=MAGNITUDE)
        ax.errorbar(
            estimate, labels, xerr=[estimate - result.lower[order], result.upper[order] - estimate],
            fmt="none", ecolor=INK, elinewidth=1, capsize=2.5,
        )
        ax.set_title(f"{panel_title} (n = {result.n_instances:,})", loc="left")
        ax.set_xlabel("mean |attribution| (logits)")
        _style(ax)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    level = next(iter(panels.values()))
    _save(fig, path, f"{caption} Whiskers: {level.level:.0%} bootstrap interval over respondents "
                     f"({level.reps:,} resamples).")


def plot_beeswarm(
    values: np.ndarray,
    feature_codes: np.ndarray,
    feature_names: Sequence[str],
    ordered_features: Mapping[str, str],
    path: Path,
    title: str,
    caption: str,
    seed: int = 42,
) -> None:
    """Every respondent's attribution for every feature, one row per feature.

    `ordered_features` maps a feature whose values have a meaningful order
    (numeric, or two-level) to a short description of what "dark" means;
    those rows are shaded by the feature's value. Other (nominal) features
    are grey: shading them by an arbitrary category code would suggest an
    order that does not exist.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(np.abs(values).mean(axis=0))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    labels = []
    for row, j in enumerate(order):
        name = feature_names[j]
        jitter = row + rng.uniform(-0.3, 0.3, size=len(values))
        if name in ordered_features:
            codes = feature_codes[:, j]
            span = np.ptp(codes) or 1.0
            ax.scatter(values[:, j], jitter, c=(codes - codes.min()) / span, cmap=VALUE_RAMP, s=5, linewidths=0)
            labels.append(f"{name}\n({ordered_features[name]})")
        else:
            ax.scatter(values[:, j], jitter, color=UNORDERED, s=5, linewidths=0, alpha=0.55)
            labels.append(f"{name}\n(unordered categories)")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(order)), labels, fontsize=7.5)
    ax.set_xlabel("attribution (logits): right pushes towards \"has an account\"")
    ax.set_title(title, loc="left")
    _style(ax)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=VALUE_RAMP), ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["low", "high"])
    colorbar.set_label("feature value (ordered features only)", fontsize=7.5)
    colorbar.outline.set_visible(False)
    _save(fig, path, f"{caption} One dot per respondent (n = {len(values):,}); rows sorted by mean |attribution|.")


def plot_waterfall(
    feature_labels: Sequence[str],
    contributions: np.ndarray,
    base_value: float,
    path: Path,
    title: str,
    caption: str,
) -> None:
    """How one respondent's logit is built up from the base value, feature by feature.

    Bars start at the base value (bottom) and are stacked in order of
    increasing |contribution|, so the largest steps are at the top, ending
    at the respondent's logit.
    """
    order = np.argsort(np.abs(contributions))
    steps = contributions[order]
    starts = base_value + np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    final = base_value + steps.sum()
    labels = [feature_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(range(len(steps)), steps, left=starts, height=0.62,
            color=[POSITIVE if s >= 0 else NEGATIVE for s in steps])
    span = max(np.ptp(np.concatenate([starts, starts + steps])), 1e-9)
    for row, (start, step) in enumerate(zip(starts, steps)):
        end = start + step
        ax.text(end + (0.012 * span if step >= 0 else -0.012 * span), row, f"{step:+.2f}",
                va="center", ha="left" if step >= 0 else "right", fontsize=7.5, color=INK_SECONDARY)
    ax.axvline(base_value, color=AXIS, linewidth=1)
    ax.axvline(final, color=INK, linewidth=1)
    # Both labels sit in their own band, clear of the bars and the axis.
    ax.text(base_value, -0.65, f"base value {base_value:.2f}", ha="center", va="top", fontsize=7.5, color=INK_MUTED)
    ax.text(final, len(steps) - 0.3, f"this respondent {final:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yticks(range(len(steps)), labels, fontsize=8)
    ax.set_ylim(-1.5, len(steps) + 0.7)
    # Explicit limits: bars carry "sticky" edges that defeat ax.margins, which
    # would push the value labels at either extreme off the plot.
    ends = np.concatenate([starts, starts + steps])
    ax.set_xlim(ends.min() - 0.16 * span, ends.max() + 0.16 * span)
    ax.set_xlabel("logit")
    ax.set_title(title, loc="left")
    _style(ax)
    _save(fig, path, f"{caption} Red pushes the logit up (towards \"has an account\"), blue pushes it down.")


def plot_dependence(
    feature: str,
    raw_values: np.ndarray,
    attributions: np.ndarray,
    is_categorical: bool,
    path: Path,
    title: str,
    caption: str,
    seed: int = 42,
) -> None:
    """A feature's attribution against the feature's own value.

    Categorical: one row per level (dots = respondents, bar = level mean).
    Numeric: scatter, with the mean attribution at each distinct value.
    Vertical spread at a fixed value is the model's interaction with other
    features: the same answer is worth different amounts for different
    respondents.
    """
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = np.where(attributions >= 0, POSITIVE, NEGATIVE)
    if is_categorical:
        levels = sorted(set(raw_values), key=lambda lv: attributions[raw_values == lv].mean())
        for row, level in enumerate(levels):
            mask = raw_values == level
            ax.scatter(attributions[mask], row + rng.uniform(-0.28, 0.28, size=mask.sum()),
                       c=colors[mask], s=6, linewidths=0, alpha=0.5)
            ax.plot([attributions[mask].mean()] * 2, [row - 0.38, row + 0.38], color=INK, linewidth=2)
        ax.set_yticks(range(len(levels)), [f"{lv}  (n = {(raw_values == lv).sum():,})" for lv in levels], fontsize=8)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlabel(f"attribution of {feature} (logits)")
        _style(ax)
        mark = "Black bar: mean for that level."
    else:
        x = raw_values.astype(float)
        ax.scatter(x + rng.uniform(-0.3, 0.3, size=len(x)), attributions, c=colors, s=6, linewidths=0, alpha=0.5)
        distinct = np.unique(x)
        ax.plot(distinct, [attributions[x == v].mean() for v in distinct], color=INK, linewidth=2)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"attribution of {feature} (logits)")
        _style(ax, grid_axis="y")
        mark = "Black line: mean at each value."
    ax.set_title(title, loc="left")
    _save(fig, path, f"{caption} One dot per respondent (n = {len(attributions):,}); red = positive, "
                     f"blue = negative. {mark}")


def plot_convergence(
    x: Sequence[float], series: Mapping[str, Sequence[float]], path: Path, title: str, xlabel: str,
    ylabel: str, caption: str, threshold: float | None = None,
) -> None:
    """One line per named series on log-log axes, each labelled at its end;
    an optional horizontal threshold line."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, (name, values) in zip(SERIES, series.items()):
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.annotate(name, (x[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_SECONDARY)
    if threshold is not None:
        ax.axhline(threshold, color=INK_MUTED, linewidth=1)
        ax.annotate(f"target {threshold:g}", (x[0], threshold), xytext=(0, 4), textcoords="offset points",
                    fontsize=7.5, color=INK_MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(x), [str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.margins(x=0.15)
    _style(ax, grid_axis="y")
    _save(fig, path, caption)


def plot_paired_shares(
    panels: Mapping[str, tuple[Sequence[str], Mapping[str, np.ndarray]]], path: Path, title: str, caption: str
) -> None:
    """For each panel, every feature's importance share under two (or three)
    alternatives, as dots joined by a line: a long line is a feature whose
    importance depends on the choice. Features are ordered by the first
    alternative of the first panel, and keep that row in every panel."""
    first_names, first_series = next(iter(panels.values()))
    order = np.argsort(next(iter(first_series.values())))
    labels = [first_names[i] for i in order]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels) + 1.2, 4.4), sharey=True, sharex=True)
    for ax, (panel_title, (names, series)) in zip(np.atleast_1d(axes), panels.items()):
        if list(names) != list(first_names):
            raise ValueError("Every panel must cover the same features in the same order")
        stacked = np.vstack([values[order] for values in series.values()])
        rows = np.arange(len(order))
        ax.hlines(rows, stacked.min(axis=0), stacked.max(axis=0), color=AXIS, linewidth=1.5)
        for color, (name, values) in zip(SERIES, series.items()):
            ax.scatter(values[order], rows, color=color, s=34, zorder=3, label=name,
                       edgecolors=SURFACE, linewidths=1)
        ax.set_yticks(rows, labels)
        ax.set_title(panel_title, loc="left")
        ax.set_xlabel("share of total importance")
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    _save(fig, path, caption)


def plot_unit_profiles(
    layers: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str, caption: str
) -> None:
    """Hidden units of each layer sorted by mean |attribution|, one bar per
    unit; units that never activate on the explained rows are marked.
    Shows how concentrated the network's computation is."""
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 3.8))
    for ax, (name, (importance, dead)) in zip(np.atleast_1d(axes), layers.items()):
        order = np.argsort(-importance)
        ax.bar(np.arange(len(order)), importance[order], width=0.8,
               color=np.where(dead[order], UNORDERED, MAGNITUDE))
        top10 = importance[order][:10].sum() / importance.sum()
        ax.set_title(f"{name}: {len(order)} units, {int(dead.sum())} never active", loc="left")
        ax.text(0.98, 0.92, f"top 10 units carry {top10:.0%}", transform=ax.transAxes, ha="right",
                fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("units, sorted")
        ax.set_ylabel("mean |conductance| (logits)")
        ax.set_xticks([])
        _style(ax, grid_axis="y")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    _save(fig, path, f"{caption} Grey: units with zero activation on every explained row.")
