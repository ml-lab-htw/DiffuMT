"""Aggregate downstream-experiment results into summary figures and LaTeX macros.

Run after tstr_unet.py (or build_diffumt.py) has written JSONs into
``results/downstream/``:

    python experiments_downstream/gather_results.py

Produces:
  - results/downstream/tstr.pdf          (SKIoU per arm)
  - results/downstream/hpo_transfer.pdf  (if hpo_transfer.csv exists)
  - results/downstream/downstream_stats.tex  (LaTeX macros, optional)
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "downstream"

ARMS = ["real", "v1", "v2", "real_v2"]
ARM_LABEL = {"real": "Real", "v1": "SynthMT", "v2": "Ours", "real_v2": "Real + Ours"}
# model -> arm -> macro name
MACROS = {
    "unet":        {"real": "tstrReal",   "v1": "tstrVone",   "v2": "tstrVtwo",   "real_v2": "tstrRealVtwo"},
    "cellposesam": {"real": "tstrCpReal", "v1": "tstrCpVone", "v2": "tstrCpVtwo", "real_v2": "tstrCpRealVtwo"},
}
MODEL_LABEL = {"unet": "U-Net", "cellposesam": "CellposeSAM"}

PLACEHOLDER_BASE = r"""% PLACEHOLDER macros for the downstream experiments (Exp1 TSTR, Exp2 HPO transfer).
% Filled by experiments_downstream/gather_results.py once cluster results land.
\providecommand{\tstrPLACE}{\ensuremath{\bullet}}
\providecommand{\tstrReal}{\tstrPLACE}
\providecommand{\tstrVone}{\tstrPLACE}
\providecommand{\tstrVtwo}{\tstrPLACE}
\providecommand{\tstrRealVtwo}{\tstrPLACE}
\providecommand{\tstrCpReal}{\tstrPLACE}
\providecommand{\tstrCpVone}{\tstrPLACE}
\providecommand{\tstrCpVtwo}{\tstrPLACE}
\providecommand{\tstrCpRealVtwo}{\tstrPLACE}
\providecommand{\skiouHuman}{0.80}
\providecommand{\skiouHumanStd}{0.14}
\providecommand{\tstrNFolds}{4}
\providecommand{\hpoVoneTransfer}{\tstrPLACE}
\providecommand{\hpoVtwoTransfer}{\tstrPLACE}
\providecommand{\hpoNSeeds}{10}
\providecommand{\hpoNTrials}{200}
\providecommand{\hpoModels}{SAM3Text, MicroSAM and CellSAM}
% Per-segmenter HPO medians (match the per-box labels in fig:downstream(b)).
\providecommand{\hpoVoneMedLo}{\tstrPLACE}
\providecommand{\hpoVoneMedHi}{\tstrPLACE}
\providecommand{\hpoVtwoMedLo}{\tstrPLACE}
\providecommand{\hpoVtwoMedHi}{\tstrPLACE}
\providecommand{\hpoSamVone}{\tstrPLACE}
\providecommand{\hpoSamVtwo}{\tstrPLACE}
\providecommand{\hpoSamVoneLo}{\tstrPLACE}
\providecommand{\hpoSamVoneHi}{\tstrPLACE}
"""


def load_tstr(model: str) -> dict[str, list[float]]:
    by_arm: dict[str, list[float]] = {a: [] for a in ARMS}
    for f in sorted(glob.glob(str(RES / f"{model}_*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("model") == model and d.get("arm") in by_arm and "SKIoU" in d:
            by_arm[d["arm"]].append(float(d["SKIoU"]))
    return by_arm


# Inter-annotator agreement on real images (SKIoU), the achievable ceiling.
# Source: microtubule_tracking/results_final_default/per_image_metrics_human_real.csv
# (n=66; SKIoU 0.80 +/- 0.14). Hardcoded -- that CSV lives in a sibling repo.
SKIOU_HUMAN = 0.80
SKIOU_HUMAN_STD = 0.14

# HPO panel (b) is broken out by segmenter; one colour per model.
HPO_MODELS = ["sam3text", "microsam", "cellsam"]
HPO_MODEL_LABEL = {"sam3text": "SAM3Text", "microsam": "MicroSAM", "cellsam": "CellSAM"}
HPO_MODEL_COLOR = {"sam3text": "#EE7733", "microsam": "#009988", "cellsam": "#AA3377"}

# Shared box-plot style; facecolor is overridden per panel/model via _box_kw().
# Fliers are hidden because every data point is drawn as a strip overlay instead.
_BOX_KW = dict(
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="#4477AA", alpha=0.7, linewidth=1.5),
    whiskerprops=dict(linewidth=1.5, color="#333333"),
    capprops=dict(linewidth=1.5, color="#333333"),
    medianprops=dict(linewidth=2.0, color="black"),
)


def _box_kw(face: str) -> dict:
    kw = dict(_BOX_KW)
    kw["boxprops"] = dict(facecolor=face, alpha=0.7, linewidth=1.5)
    return kw


_RNG = np.random.default_rng(0)


def _darken(color, f: float = 0.62):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r * f, g * f, b * f)


def _box_strip(ax, data, positions, face):
    """Box plot with every data point overlaid as a jittered, darker dot."""
    bp = ax.boxplot(data, positions=positions, widths=0.6, **_box_kw(face))
    for med in bp["medians"]:
        med.set_zorder(6)
    for ln in bp["whiskers"] + bp["caps"]:
        ln.set_zorder(5)
    for x, vals in zip(positions, data):
        if vals:
            jit = (_RNG.random(len(vals)) - 0.5) * 0.22
            ax.scatter(jit + x, vals, s=18, color=_darken(face, 0.55), alpha=0.95,
                       edgecolors="white", linewidths=0.5, zorder=5.5)


def _median_labels(ax, positions, datasets):
    """Write each box's median value just above the top of the box/whisker."""
    for x, vals in zip(positions, datasets):
        if vals:
            ax.text(x, max(vals) + 0.025, f"{st.median(vals):.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


def fig_downstream(per_model, hpo_rows) -> dict[str, float]:
    """Combined full-width downstream figure, two panels sharing the y-axis and
    the inter-annotator-agreement baseline: (a) train-synthetic/test-real (TSTR),
    (b) repeated synthetic-to-real hyperparameter transfer, broken out per model.

    Returns the HPO transfer medians (pooled over models) for the macro file.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    arms = [a for a in ARMS if any(per_model[m].get(a) for m in per_model)]
    by_sm: dict[tuple, list[float]] = {}
    for model, src, _seed, iou in (hpo_rows or []):
        by_sm.setdefault((src, model), []).append(iou)
    srcs = [s for s in ("synthetic", "v2dm") if any(k[0] == s for k in by_sm)]
    hpo_models = [m for m in HPO_MODELS if any(k[1] == m for k in by_sm)]
    if not arms and not srcs:
        print("[downstream] nothing to plot -- skipping figure")
        return {}

    # Panel (b) layout: nm boxes per source on an integer grid, with a 1-unit gap
    # between source groups (so the two groups read apart).
    nm = max(len(hpo_models), 1)
    def bpos(i, j):                       # x of model j within source group i
        return i * (nm + 1) + (j + 1)
    b_last = bpos(len(srcs) - 1, nm - 1) if srcs else 1

    # Box width is identical (0.6) in both panels; width_ratios match the number
    # of box-slots per panel so every box renders at the same physical width.
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(10.16, 3.5), sharey=True,
        gridspec_kw=dict(wspace=0.08,
                         width_ratios=[max(len(arms), 1), max(b_last, 1)]))

    # Panel (a): TSTR -- one box per training source over the U-Net folds.
    mdl = next((m for m in ("unet", *per_model) if per_model.get(m)
                and any(per_model[m].get(a) for a in arms)), None)
    if arms and mdl:
        data_a = [per_model[mdl].get(a) or [0.0] for a in arms]
        pos_a = list(range(1, len(arms) + 1))
        _box_strip(axa, data_a, pos_a, "#4477AA")
        _median_labels(axa, pos_a, data_a)
        axa.set_xticks(pos_a)
        axa.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=9)
        axa.set_xlim(0.5, len(arms) + 0.5)
    axa.axhspan(SKIOU_HUMAN - SKIOU_HUMAN_STD, SKIOU_HUMAN + SKIOU_HUMAN_STD,
                color="grey", alpha=0.12, zorder=0, linewidth=0)
    axa.axhline(SKIOU_HUMAN, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    axa.set_ylabel("SKIoU on real", fontsize=11, fontweight="bold")
    axa.set_xlabel("Training data", fontsize=11, fontweight="bold")
    axa.set_ylim(0, 1)
    axa.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.7)
    axa.set_axisbelow(True)
    axa.text(0.0, 1.04, "a", transform=axa.transAxes, fontsize=14,
             fontweight="bold", va="bottom", ha="left")

    # Panel (b): HPO transfer -- one box per (source, model), same width as (a),
    # grouped by source (model = colour).
    for j, m in enumerate(hpo_models):
        pairs = [(bpos(i, j), by_sm[(s, m)])
                 for i, s in enumerate(srcs) if by_sm.get((s, m))]
        if pairs:
            xs, dat = zip(*pairs)
            _box_strip(axb, list(dat), list(xs), HPO_MODEL_COLOR[m])
            _median_labels(axb, xs, dat)
    axb.axhspan(SKIOU_HUMAN - SKIOU_HUMAN_STD, SKIOU_HUMAN + SKIOU_HUMAN_STD,
                color="grey", alpha=0.12, zorder=0, linewidth=0)
    axb.axhline(SKIOU_HUMAN, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    axb.set_xticks([bpos(i, (nm - 1) / 2) for i in range(len(srcs))])
    axb.set_xticklabels(["SynthMT" if s == "synthetic" else "Ours" for s in srcs],
                        fontsize=9)
    axb.set_xlim(0.5, b_last + 0.5)
    axb.set_xlabel("HPO data", fontsize=11, fontweight="bold")
    axb.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.7)
    axb.set_axisbelow(True)
    axb.text(0.0, 1.04, "b", transform=axb.transAxes, fontsize=14,
             fontweight="bold", va="bottom", ha="left")

    # shared legend: U-Net (panel a) + per-model colours (panel b) + baseline.
    handles = [Patch(facecolor="#4477AA", alpha=0.7, label="U-Net")]
    handles += [Patch(facecolor=HPO_MODEL_COLOR[m], alpha=0.7, label=HPO_MODEL_LABEL[m])
                for m in hpo_models]
    handles.append(Line2D([0], [0], color="grey", linestyle="--", linewidth=1.0,
                          label=f"Inter-annotator agreement ({SKIOU_HUMAN:.2f})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    out_pdf = RES / "downstream.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    out_png = RES / "downstream.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    medians = {s: st.median([v for (src, _m), vals in by_sm.items() if src == s for v in vals])
               for s in srcs}
    print(f"[downstream] {out_pdf}  hpo medians={medians}")
    return medians


def load_hpo_transfer():
    f = RES / "hpo_transfer.csv"
    if not f.exists():
        return None
    rows = []
    for line in f.read_text().splitlines()[1:]:
        p = line.split(",")
        if len(p) >= 4:
            rows.append((p[0], p[1], p[2], float(p[3])))
    return rows


def write_macros(per_model, hpo_medians, hpo_rows=None):
    lines = ["% AUTO-GENERATED by experiments_downstream/gather_results.py\n"]

    def cmd(key, val):
        lines.append(f"\\renewcommand{{\\{key}}}{{{val}}}\n")

    for model, arm_macro in MACROS.items():
        # Report the median per arm (matches the box-plot central line in
        # fig:downstream); the box conveys spread, so the text stays point-only.
        # Bold the best (highest-median) arm.
        medians = {a: st.median(per_model[model][a]) for a in ARMS if per_model.get(model, {}).get(a)}
        best_arm = max(medians, key=medians.get) if medians else None
        for a in ARMS:
            vals = per_model.get(model, {}).get(a)
            if vals:
                val_str = f"{st.median(vals):.2f}"
                if a == best_arm:
                    val_str = f"\\textbf{{{val_str}}}"
                cmd(arm_macro[a], val_str)
    if hpo_medians.get("synthetic"):
        cmd("hpoVoneTransfer", f"{hpo_medians['synthetic']:.2f}")
    if hpo_medians.get("v2dm"):
        cmd("hpoVtwoTransfer", f"{hpo_medians['v2dm']:.2f}")

    # Per-segmenter HPO stats, so the text matches the per-box labels in
    # fig:downstream(b) (the pooled median above is not shown on any single box).
    if hpo_rows:
        def med_by_model(src):
            d: dict[str, list[float]] = {}
            for model, source, _seed, iou in hpo_rows:
                if source == src:
                    d.setdefault(model, []).append(iou)
            return {m: st.median(v) for m, v in d.items()}

        v1m, v2m = med_by_model("synthetic"), med_by_model("v2dm")
        if v1m:
            cmd("hpoVoneMedLo", f"{min(v1m.values()):.2f}")
            cmd("hpoVoneMedHi", f"{max(v1m.values()):.2f}")
        if v2m:
            cmd("hpoVtwoMedLo", f"{min(v2m.values()):.2f}")
            cmd("hpoVtwoMedHi", f"{max(v2m.values()):.2f}")

        def sam(src):
            return [iou for model, source, _seed, iou in hpo_rows
                    if source == src and model == "sam3text"]

        sam_v1, sam_v2 = sam("synthetic"), sam("v2dm")
        if sam_v1:
            cmd("hpoSamVone", f"{st.median(sam_v1):.2f}")
            cmd("hpoSamVoneLo", f"{min(sam_v1):.2f}")
            cmd("hpoSamVoneHi", f"{max(sam_v1):.2f}")
        if sam_v2:
            cmd("hpoSamVtwo", f"{st.median(sam_v2):.2f}")

    out = RES / "downstream_stats.tex"
    out.write_text(PLACEHOLDER_BASE + "\n% --- filled by gather_results.py ---\n" + "".join(lines))
    print(f"[macros] updated {out}")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    per_model = {m: load_tstr(m) for m in MACROS}
    hpo_rows = load_hpo_transfer()
    hpo_medians = fig_downstream(per_model, hpo_rows)
    write_macros(per_model, hpo_medians, hpo_rows)
    for m in per_model:
        got = {a: round(st.mean(per_model[m][a]), 3) for a in ARMS if per_model[m].get(a)}
        if got:
            print(f"[{m}] {got}")
    print("[done] gather complete")


if __name__ == "__main__":
    main()
