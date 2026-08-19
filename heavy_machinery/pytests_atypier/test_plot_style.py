"""Shared publication-plotting primitives in plot_style.

These guard the properties the DDA/EDA figures rely on for correctness:
reproducible jitter, intervals that stay inside [0, 1], densities that do not
escape the observed support, and integer-aligned histogram bins.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import plot_style as ps


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_prettify_level():
    assert ps.prettify_level(True, "high_grade") == "High grade"
    assert ps.prettify_level(False, "high_grade") == "Low grade"
    # Unregistered boolean columns fall back to a neutral yes/no.
    assert ps.prettify_level(True, "calcification") == "Yes"
    # Ranges and thresholds are already readable, so they pass through.
    for raw in ("60-69", "80+", "<50", ">=3.64"):
        assert ps.prettify_level(raw) == raw
    # Comparator tokens expand.
    assert ps.prettify_level("low_le_4") == "Low ≤ 4"
    assert ps.prettify_level("intermediate_5_9") == "Intermediate 5–9"


def test_prettify_label_and_level_tick_labels():
    assert ps.prettify_label("edema_volume_ge3.64") == "Edema volume ≥ 3.64"
    assert ps.prettify_label("max_diameter_cm_gt6") == "Max diameter (cm) > 6"
    assert ps.prettify_label("age_bins_10") == "Age group"
    assert ps.level_tick_labels(["a", "b"], [10, 200], column="x") == [
        "A\n(n=10)", "B\n(n=200)"]


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def test_deterministic_rng_is_stable_across_calls():
    a = ps.deterministic_rng("dda", "age", "sex").uniform(size=5)
    b = ps.deterministic_rng("dda", "age", "sex").uniform(size=5)
    c = ps.deterministic_rng("dda", "age", "side").uniform(size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_wilson_ci_brackets_the_estimate_and_stays_in_unit_range():
    lo, hi = ps.wilson_ci(5, 20)
    assert 0.0 <= lo < 0.25 < hi <= 1.0
    # Boundary counts must not produce an interval outside [0, 1].
    lo0, hi0 = ps.wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = ps.wilson_ci(10, 10)
    assert hi1 == pytest.approx(1.0) and 0.0 < lo1 < 1.0

    # More patients, a narrower interval.
    small = ps.wilson_ci(1, 2)
    large = ps.wilson_ci(100, 200)
    assert (large[1] - large[0]) < (small[1] - small[0])

    # Nothing observed at all is nan, not a crash.
    lo, hi = ps.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)

    assert (ps.errorbar_lengths([0.5, 0.2], [0.6, 0.1], [0.4, 0.3]) >= 0).all()


def test_kde_curve_stays_inside_the_observed_support():
    rng = np.random.default_rng(0)
    values = rng.uniform(10.0, 20.0, size=200)
    xs, dens = ps.kde_curve(values, clip=(float(values.min()), float(values.max())))
    assert xs.min() >= values.min()
    assert xs.max() <= values.max()
    assert (dens >= 0).all()

    assert ps.kde_curve([1.0, 1.0, 1.0, 1.0]) is None  # zero variance
    assert ps.kde_curve([1.0, 2.0]) is None  # too few points


def test_histogram_bin_edges_suit_integer_and_continuous_data():
    edges = ps.histogram_bin_edges(np.array([1, 1, 2, 3, 6], dtype=float))
    assert edges[0] == pytest.approx(0.5)
    assert np.allclose(np.diff(edges), 1.0)

    rng = np.random.default_rng(1)
    values = rng.normal(size=300)
    edges = ps.histogram_bin_edges(values)
    assert edges[0] == pytest.approx(values.min())
    assert edges[-1] == pytest.approx(values.max())

    assert ps.freedman_diaconis_bins([1.0, 2.0, 3.0]) <= 3


def test_spearman_lowess_and_p_formatting():
    x = np.arange(40, dtype=float)
    y = x + np.random.default_rng(2).normal(scale=5.0, size=40)
    out = ps.spearman_summary(x, y)
    assert out["rho"] > 0.5
    assert out["ci_lo"] < out["rho"] < out["ci_hi"]
    assert out["n"] == 40

    assert ps.lowess_curve([1.0] * 20, [1.0] * 20) is None
    assert ps.lowess_curve([1.0, 2.0], [1.0, 2.0]) is None

    assert ps.format_p(1e-9) == "< 0.001"
    assert ps.format_p(0.0432) == "= 0.043"
    assert ps.format_p(None) == ""


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _scatter_offsets(ax: plt.Axes) -> np.ndarray:
    """Offsets of the jittered raw points only (not the violin polygon)."""
    from matplotlib.collections import PathCollection

    points = [
        c.get_offsets() for c in ax.collections
        if isinstance(c, PathCollection) and len(c.get_offsets())
    ]
    assert points, "raincloud drew no raw points"
    return np.vstack(points)


def test_raincloud_places_points_on_the_value_axis():
    """Raw points must share the value axis with the box, not the group axis."""
    values = np.linspace(10.0, 20.0, 60)

    fig, ax = plt.subplots()
    ps.raincloud(ax, [values], positions=[0.0])
    offsets = _scatter_offsets(ax)
    assert offsets[:, 1].min() >= 10.0 - 1e-6  # y carries the values
    assert offsets[:, 1].max() <= 20.0 + 1e-6
    assert abs(offsets[:, 0]).max() < 1.0  # x stays inside the group slot
    plt.close(fig)

    # Horizontal swaps the two axes.
    fig, ax = plt.subplots()
    ps.raincloud(ax, [values], positions=[0.0], orient="h")
    offsets = _scatter_offsets(ax)
    assert offsets[:, 0].min() >= 10.0 - 1e-6
    assert abs(offsets[:, 1]).max() < 1.0
    plt.close(fig)

    # A group too small for a density still gets its raw points.
    fig, ax = plt.subplots()
    ps.raincloud(ax, [np.array([1.0, 2.0])], positions=[0.0])
    assert len(_scatter_offsets(ax)) == 2
    plt.close(fig)


def test_proportion_bars_titles_and_saving(tmp_path):
    fig, ax = plt.subplots()
    pct = ps.proportion_bars(ax, [5, 15], [20, 20])
    assert pct == pytest.approx([25.0, 75.0])
    assert ax.containers, "no error bars drawn"
    plt.close(fig)

    fig, ax = plt.subplots()
    ps.set_titles(ax, "Age", ps.n_subtitle(42, extra="median 60"))
    # Recorded for the report to print as text, never drawn into the image.
    assert ax.get_title() == ""
    assert [t.get_text() for t in ax.texts] == []
    legend = ps.figure_legend(fig)
    assert legend["title"] == "Age"
    assert legend["note"] == "n = 42 · median 60"
    plt.close(fig)

    fig, (top, bottom) = plt.subplots(2, 1)
    bottom.plot([1, 2], [1, 2])
    path = ps.save_figure(fig, tmp_path / "x", tight_layout=False)
    assert path.exists()
    assert path.suffix == ".png"
    assert not (tmp_path / "x.eps").exists()
    plt.close(fig)


def test_forest_rows_sort_by_estimate_descending():
    """Grey and black rows share one ranking — largest estimate at the top."""
    est = [3.0, 5.0, 0.2, 1.5]
    lo = [2.0, 0.5, 0.1, 0.8]
    hi = [4.0, 10.0, 0.4, 2.0]
    assert list(ps.forest_row_order(est, lo, hi)) == [1, 0, 3, 2]

    fig, ax = ps.forest_lr(
        ["A", "B", "C", "D"], est, lo, hi,
        xlabel="OR",
        value_header="OR (95% CI)",
    )
    pairs = sorted(
        zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]),
        reverse=True,
    )
    assert [lab for _, lab in pairs] == ["B", "A", "D", "C"]
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fast ROC AUC
# ---------------------------------------------------------------------------
# roc_auc replaces sklearn's roc_auc_score inside the bootstrap loops, so the
# published optimism-corrected numbers depend on the two agreeing. These pin
# that down, including the tie handling that is the only place they could part.

@pytest.mark.parametrize("tied", [False, True])
def test_roc_auc_matches_sklearn(tied):
    """Ties are the only place a rank-based AUC could disagree with the ROC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(12 if tied else 11)
    for _ in range(200):
        n = int(rng.integers(20, 400))
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            continue
        score = (rng.integers(0, 3, n).astype(float) if tied
                 else rng.normal(size=n))
        assert ps.roc_auc(y, score) == pytest.approx(
            roc_auc_score(y, score), abs=1e-12
        )


def test_roc_auc_degenerate_cases():
    assert ps.roc_auc([0, 1, 0, 1], [2.0, 2.0, 2.0, 2.0]) == 0.5
    # AUC is undefined with one class; callers skip those resamples.
    assert np.isnan(ps.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(ps.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Figure export profile
# ---------------------------------------------------------------------------

def test_the_figure_profile_decides_formats_and_dpi(monkeypatch):
    """The TIF keeps the 1200-dpi line-art spec; only the report PNG is
    downsampled. Silently falling back on an unknown profile would ship the
    wrong files to a journal."""
    monkeypatch.delenv("ATYPIER_FIGURES", raising=False)
    assert ps.figure_profile() == "report"
    assert ps.figure_formats() == ("png",)
    assert ps._dpi_for("png", "line") == ps.REPORT_PNG_DPI

    monkeypatch.setenv("ATYPIER_FIGURES", "submission")
    assert ps.figure_formats() == ("tif", "png")
    assert ps._dpi_for("tif", "line") == 1200
    assert ps._dpi_for("png", "line") == 1200

    monkeypatch.setenv("ATYPIER_FIGURES", "hi-res")
    with pytest.raises(ValueError, match="figure profile"):
        ps.figure_profile()


def test_explicit_formats_override_the_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("ATYPIER_FIGURES", raising=False)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ps.save_figure(fig, tmp_path / "x", formats=("png", "pdf"))
    assert (tmp_path / "x.pdf").exists()
    plt.close(fig)


def test_a_figure_carrying_only_a_plain_explanation_still_gets_a_sidecar(tmp_path):
    """Each of the three legend fields is enough on its own.

    The per-model ROC / calibration / decision-curve figures carry a
    plain-language line and nothing else — their title is the report's
    name-derived caption. A save guard that checked only title-or-note wrote no
    sidecar for them at all, and the explanation silently never reached the
    page.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ps.set_figure_legend(fig, plain="Higher is better.")
    png = ps.save_figure(fig, tmp_path / "only_plain")
    assert ps.read_figure_legend(png) == {
        "title": "", "plain": "Higher is better.", "note": "",
    }

    for field in ("title", "note"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ps.set_figure_legend(fig, **{field: "solo"})
        png = ps.save_figure(fig, tmp_path / f"only_{field}")
        assert ps.read_figure_legend(png)[field] == "solo"


def test_a_figure_with_no_legend_leaves_no_stale_sidecar(tmp_path):
    """A figure that loses its legend must not keep publishing the old one."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ps.set_figure_legend(fig, title="First run")
    png = ps.save_figure(fig, tmp_path / "fig")
    assert ps.read_figure_legend(png)["title"] == "First run"

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    ps.save_figure(fig, tmp_path / "fig")
    assert ps.read_figure_legend(png) == {}
    assert not ps.figure_legend_path(png).exists()


def test_pruning_only_removes_figures_the_report_actually_contains(tmp_path):
    """The interlock: a figure missing from the page survives.

    Deleting every PNG in a figures directory would turn one failed render into
    silent data loss — the file would be gone and the page would not have it
    either. Only bytes provably present in the HTML are removed.
    """
    import base64
    figs = tmp_path / "figures"
    figs.mkdir()

    def _png(stem, seed):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, seed])
        ps.set_figure_legend(fig, title=f"fig {seed}")
        return ps.save_figure(fig, figs / stem)

    shown, hidden = _png("shown", 1), _png("hidden", 2)
    tif = figs / "shown.tif"
    tif.write_bytes(b"II*\x00 pretend tiff")

    payload = base64.b64encode(shown.read_bytes()).decode("ascii")
    report = tmp_path / "report.html"
    report.write_text(f'<img src="data:image/png;base64,{payload}"/>')

    deleted, freed, kept = ps.prune_embedded_figures(report, roots=[figs])
    assert deleted == 1 and freed > 0
    assert not shown.exists(), "the embedded figure should be reclaimed"
    assert not ps.figure_legend_path(shown).exists(), "its legend goes with it"
    assert hidden.exists(), "a figure the report lacks must NOT be deleted"
    assert ps.figure_legend_path(hidden).exists()
    assert kept == [hidden]
    assert tif.exists(), "TIFs are the journal deliverable and are never pruned"


def test_pruning_dry_run_reports_without_deleting(tmp_path):
    import base64
    figs = tmp_path / "figures"
    figs.mkdir()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png = ps.save_figure(fig, figs / "one")
    payload = base64.b64encode(png.read_bytes()).decode("ascii")
    report = tmp_path / "r.html"
    report.write_text(f'<img src="data:image/png;base64,{payload}"/>')

    deleted, freed, _ = ps.prune_embedded_figures(report, roots=[figs], dry_run=True)
    assert deleted == 1 and freed > 0
    assert png.exists(), "dry run must not delete anything"
