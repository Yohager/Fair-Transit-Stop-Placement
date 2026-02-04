import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.lines import Line2D
try:
    from scipy.stats import t as student_t
    has_scipy = True
except Exception:
    has_scipy = False

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"


def load_all_runs(output_dir: str) -> pd.DataFrame:
    base = Path(output_dir)
    rows = []

    col_to_display = {
        "GC": "GC-TrSP",
        "ECA": "ECA",
        "Hybrid": "Hybrid"
    }

    for exp_dir in base.iterdir():
        if not exp_dir.is_dir():
            continue

        params_path = exp_dir / "params.json"
        results_path = exp_dir / "results.csv"
        if not params_path.exists() or not results_path.exists():
            continue

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        df = pd.read_csv(results_path)

        for col, display in col_to_display.items():
            if col not in df.columns:
                continue

            tmp = df[["round", col]].copy()
            tmp = tmp.rename(columns={col: "value"})
            tmp["algo"] = display
            tmp["k"] = int(params["k"])
            tmp["transit_scaling"] = int(params["transit_scaling"])
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["transit_scaling", "k", "round", "algo", "value"])

    out = pd.concat(rows, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["round"] = out["round"].astype(int)
    out["k"] = out["k"].astype(int)
    out["transit_scaling"] = out["transit_scaling"].astype(int)
    return out

def plot_JR(
    df_long: pd.DataFrame,
    output_fig_dir: str,
    jitter: float = 0.10,
    seed: int = 0,
    ci_level: float = 0.95,
    show_hybrid: bool = True,
):
    Path(output_fig_dir).mkdir(parents=True, exist_ok=True)

    algos = ["GC-TrSP", "ECA"] + (["Hybrid"] if show_hybrid else [])

    transit_scaling = sorted(df_long["transit_scaling"].unique().tolist())
    if len(transit_scaling) < 4:
        raise ValueError(f"transit scaling is less than 4 types: {transit_scaling}")
    if len(transit_scaling) > 4:
        transit_scaling = transit_scaling[:4]

    rng = np.random.default_rng(seed)

    colors = {
        "GC-TrSP": "tab:blue",
        "ECA": "tab:red",
        "Hybrid": "tab:purple",
    }
    markers = {
        "GC-TrSP": "o",
        "ECA": "s",
        "Hybrid": "*",
    }

    alpha_ci = 0.18
    alpha_scatter = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    axes = axes.ravel()

    for idx, tr in enumerate(transit_scaling):
        ax = axes[idx]
        dft = df_long[df_long["transit_scaling"] == tr].copy()
        ks = sorted(dft["k"].unique().tolist())

        x_pos = {k: i for i, k in enumerate(ks)}
        x_ticks = list(range(len(ks)))
        x_labels = [str(k) for k in ks]

        for algo in algos:
            dfa = dft[dft["algo"] == algo].copy()
            if dfa.empty:
                continue

            stats = dfa.groupby("k")["value"].agg(["mean", "std", "count"]).reindex(ks)

            means = stats["mean"].to_numpy()
            stds = stats["std"].to_numpy()
            ns = stats["count"].to_numpy()

            se = np.divide(stds, np.sqrt(ns), out=np.zeros_like(stds), where=ns > 0)

            if has_scipy:
                alpha = 1.0 - ci_level
                dfree = np.maximum(ns - 1, 1)
                tcrit = student_t.ppf(1.0 - alpha / 2.0, dfree)
            else:
                tcrit = 1.96

            half_width = np.multiply(tcrit, se)
            lower = np.subtract(means, half_width)
            upper = np.add(means, half_width)

            mean_x = [x_pos[k] for k in ks]

            ax.fill_between(
                mean_x,
                lower,
                upper,
                alpha=alpha_ci,
                color=colors[algo],
                linewidth=0.0,
                zorder=1,
            )

            xs_scatter = []
            ys_scatter = []
            for k in ks:
                vals = dfa[dfa["k"] == k]["value"].to_numpy()
                if vals.size == 0:
                    continue
                base_x = x_pos[k]
                offset = rng.uniform(-jitter, jitter, size=vals.size)
                xs_scatter.append(base_x + offset)
                ys_scatter.append(vals)

            if xs_scatter:
                xs_scatter = np.concatenate(xs_scatter)
                ys_scatter = np.concatenate(ys_scatter)
                ax.scatter(
                    xs_scatter,
                    ys_scatter,
                    s=26,
                    alpha=alpha_scatter,
                    marker=markers[algo],
                    facecolors=colors[algo],
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=2,
                )

            ax.plot(
                mean_x,
                means,
                marker=markers[algo],
                markersize=5.5,
                linewidth=2.4,
                color=colors[algo],
                alpha=0.95,
                zorder=3,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=24)
        ax.tick_params(axis="y", labelsize=24)

        ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.8, zorder=0)

        ax.spines["top"].set_alpha(0.35)
        ax.spines["right"].set_alpha(0.35)
        ax.spines["left"].set_alpha(0.6)
        ax.spines["bottom"].set_alpha(0.6)

        if idx in (2, 3):
            ax.set_xlabel(r"Number of stops $k$", fontsize=24)

        ax.text(
            0.03,
            0.96,
            f"Transit cost scaling: {tr}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=20,
        )

    fig.text(
        0.035, 0.5,
        r"Approximation ratio of JR",
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=24,
    )

    handles = []
    handles.append(Line2D([0], [0], color=colors["GC-TrSP"], marker=markers["GC-TrSP"], linewidth=2.4, markersize=6, label="GC-TrSP Average"))
    handles.append(Line2D([0], [0], color=colors["ECA"], marker=markers["ECA"], linewidth=2.4, markersize=6, label="ECA Average"))
    if show_hybrid:
        handles.append(Line2D([0], [0], color=colors["Hybrid"], marker=markers["Hybrid"], linewidth=2.4, markersize=6, label="1/2-Hybrid Average"))

    handles.append(Line2D([0], [0], color=colors["GC-TrSP"], marker=markers["GC-TrSP"], linestyle="None", markersize=7, alpha=alpha_scatter, label="GC-TrSP"))
    handles.append(Line2D([0], [0], color=colors["ECA"], marker=markers["ECA"], linestyle="None", markersize=7, alpha=alpha_scatter, label="ECA"))
    if show_hybrid:
        handles.append(Line2D([0], [0], color=colors["Hybrid"], marker=markers["Hybrid"], linestyle="None", markersize=7, alpha=alpha_scatter, label="1/2-Hybrid"))
    
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3 if show_hybrid else 2,
        frameon=True,
        framealpha=0.9,
        fancybox=True,
        fontsize=24,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=(0.06, 0.03, 0.995, 0.9))

    pdf_path = Path(output_fig_dir) / (
        "approximation_gc_eca_hybrid.pdf" if show_hybrid else "approximation_gc_eca.pdf"
    )
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

if __name__ == "__main__":
    output_dir = "../data/results"
    fig_dir = "../data/plot/"

    df = load_all_runs(output_dir)

    if df.empty:
        print("No results found. Check output_dir path and folder structure.")
    else:
        plot_JR(df, fig_dir, show_hybrid=False)
        print(f"Saved figures to {fig_dir}")

    