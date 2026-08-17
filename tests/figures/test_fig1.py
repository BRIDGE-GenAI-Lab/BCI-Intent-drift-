import json
from idrift.figures.fig1_drift import make_fig1


def test_fig1_written(tmp_path):
    digest = {"curve_by_class":{"frontier":{0.0:0.0,0.2:0.1,0.4:0.3},"local":{0.0:0.0,0.2:0.08,0.4:0.25}},
              "critical_curve":{0.0:0.0,0.2:0.02,0.4:0.12},
              "reliability":[{"bin_mid":0.1,"acc":0.2,"conf":0.5},{"bin_mid":0.9,"acc":0.7,"conf":0.95}]}
    out = tmp_path/"fig1.pdf"; make_fig1(digest, out); assert out.exists() and out.stat().st_size>0


def test_fig1_writes_sibling_png(tmp_path):
    digest = {"curve_by_class":{"frontier":{0.0:0.0,0.2:0.1,0.4:0.3},"local":{0.0:0.0,0.2:0.08,0.4:0.25}},
              "critical_curve":{0.0:0.0,0.2:0.02,0.4:0.12},
              "reliability":[{"bin_mid":0.1,"acc":0.2,"conf":0.5},{"bin_mid":0.9,"acc":0.7,"conf":0.95}]}
    out = tmp_path/"fig1.pdf"
    make_fig1(digest, out)
    png = out.with_suffix(".png")
    assert png.exists() and png.stat().st_size > 0


def test_fig1_handles_missing_optional_taxonomy(tmp_path):
    # No "taxonomy" key at all -- panel (c) should degrade gracefully rather
    # than crash, still producing the reliability-only two-panel figure.
    digest = {"curve_by_class": {"frontier": {0.0: 0.0, 0.2: 0.1}},
              "critical_curve": {0.0: 0.0, 0.2: 0.02},
              "reliability": [{"bin_mid": 0.5, "acc": 0.5, "conf": 0.5}]}
    out = tmp_path / "fig1_no_taxonomy.pdf"
    make_fig1(digest, out)
    assert out.exists() and out.stat().st_size > 0


def test_fig1_renders_taxonomy_panel_when_present(tmp_path):
    digest = {"curve_by_class": {"frontier": {0.0: 0.0, 0.2: 0.1}},
              "critical_curve": {0.0: 0.0, 0.2: 0.02},
              "reliability": [{"bin_mid": 0.5, "acc": 0.5, "conf": 0.5}],
              "taxonomy": {
                  "0.0": {"omission": 0.5, "substitution": 0.5},
                  "0.2": {"omission": 0.3, "substitution": 0.7},
              }}
    out = tmp_path / "fig1_taxonomy.pdf"
    make_fig1(digest, out)
    assert out.exists() and out.stat().st_size > 0


def test_fig1_no_warnings_emitted(tmp_path, recwarn):
    digest = {"curve_by_class": {"frontier": {0.0: 0.0, 0.2: 0.1, 0.4: 0.3}},
              "critical_curve": {0.0: 0.0, 0.2: 0.02, 0.4: 0.12},
              "reliability": [{"bin_mid": 0.1, "acc": 0.2, "conf": 0.5}]}
    out = tmp_path / "fig1_clean.pdf"
    make_fig1(digest, out)
    assert len(recwarn) == 0
