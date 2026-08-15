"""The figure contract — the rules that refuse to ship a non-compliant figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import ajnr_style as aj


@pytest.fixture
def fig():
    figure, ax = aj.new_figure(height_in=2.4)
    ax.plot([0, 1], [0, 1], **aj.series_style(0))
    ax.set_xlabel("x")
    yield figure
    plt.close(figure)


# --- the palette -----------------------------------------------------------
def test_series_differ_by_shade_and_line_style_together():
    """Either cue alone must be enough — colour is never the only difference."""
    a, b = aj.series_style(0), aj.series_style(1)
    assert a["color"] != b["color"]
    assert a["linestyle"] != b["linestyle"]


def test_the_palette_is_pure_greyscale():
    for shade in aj.SHADES:
        r, g, b = (shade[1:3], shade[3:5], shade[5:7])
        assert r == g == b, f"{shade} is not neutral grey"


def test_styles_cycle_rather_than_run_out():
    assert aj.series_style(len(aj.SHADES))["color"] == aj.SHADES[0]


def test_palette_matches_the_existing_or_forest():
    """A reader moving between figures must not have to relearn the greys."""
    from plot_style import OKABE
    assert aj.INK == OKABE["black"]
    assert aj.MUTED == OKABE["grey"].upper()
    assert aj.ROW_BAND == OKABE["lightgrey"].upper()


def test_a_null_crossing_estimate_is_muted_not_dropped():
    assert aj.point_style(muted=True)["color"] == aj.MUTED
    assert aj.point_style()["color"] == aj.INK


def test_points_are_squares_like_the_existing_forest():
    assert aj.point_style()["marker"] == "s"


def test_the_null_line_is_black_and_dashed_not_a_soft_guide():
    assert aj.NULL_LINE["color"] == aj.INK
    assert aj.NULL_LINE["linestyle"] == "--"
    assert aj.REFERENCE != aj.INK


# --- the export contract ---------------------------------------------------
def test_both_formats_come_off_one_figure(fig, tmp_path):
    written = aj.save_figure(fig, tmp_path / "fig_demo")
    assert [p.suffix for p in written] == [".tif", ".png"]
    assert all(p.exists() for p in written)


def test_tif_is_lzw_compressed(fig, tmp_path):
    aj.save_figure(fig, tmp_path / "fig_demo")
    with Image.open(tmp_path / "fig_demo.tif") as im:
        assert im.info.get("compression") == "tiff_lzw"


def test_saved_at_six_hundred_dpi(fig, tmp_path):
    aj.save_figure(fig, tmp_path / "fig_demo")
    with Image.open(tmp_path / "fig_demo.png") as im:
        dpi_x, _ = im.info["dpi"]
        assert round(dpi_x) == aj.DPI


def test_png_carries_no_tooling_metadata(fig, tmp_path):
    aj.save_figure(fig, tmp_path / "fig_demo")
    with Image.open(tmp_path / "fig_demo.png") as im:
        assert not im.info.get("Software")


def test_the_figure_is_at_least_the_journal_floor(fig):
    assert fig.get_size_inches()[0] >= 4.0


# --- the refusals ----------------------------------------------------------
def test_type_below_eight_point_is_refused(fig, tmp_path):
    fig.axes[0].set_xlabel("too small", fontsize=6.5)
    with pytest.raises(aj.FigureContractError, match="below the 8.0 pt minimum"):
        aj.save_figure(fig, tmp_path / "fig_demo")


def test_empty_labels_do_not_trip_the_type_check(fig, tmp_path):
    """Matplotlib leaves blank Text objects at default sizes all over an axes."""
    fig.axes[0].set_title("", fontsize=2.0)
    aj.save_figure(fig, tmp_path / "fig_demo")


def test_an_identifying_filename_is_refused(fig, tmp_path):
    with pytest.raises(aj.FigureContractError, match="identifying text"):
        aj.save_figure(fig, tmp_path / "fig_pskus_cohort")


def test_a_scanner_name_in_the_filename_is_refused(fig, tmp_path):
    with pytest.raises(aj.FigureContractError, match="identifying text"):
        aj.save_figure(fig, tmp_path / "fig_siemens_adc")


def test_an_oversized_figure_is_refused(fig, tmp_path):
    fig.set_size_inches(9.0, 3.0)
    with pytest.raises(aj.FigureContractError, match="broadside"):
        aj.save_figure(fig, tmp_path / "fig_demo")


def test_eps_is_refused_because_it_flattens_the_bands(fig, tmp_path):
    with pytest.raises(aj.FigureContractError, match="not one this phase ships"):
        aj.save_figure(fig, tmp_path / "fig_demo", formats=("eps",))


# --- the style is applied locally, not globally ---------------------------
def test_importing_the_module_does_not_change_global_rcparams():
    before = matplotlib.rcParams["font.size"]
    import importlib
    importlib.reload(aj)
    assert matplotlib.rcParams["font.size"] == before


def test_new_figure_applies_arial():
    figure, _ = aj.new_figure()
    assert matplotlib.rcParams["font.sans-serif"][0] == "Arial"
    plt.close(figure)
