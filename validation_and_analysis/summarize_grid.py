"""R@k tables and plots from benchmark_grid.py output (Table 4)."""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [5, 10, 20, 50]
STYLES = ["terse", "casual", "vague", "detailed", "notation"]
METHOD_COLORS = {"ours": "#1a6faa", "pathfinder": "#c98a1e", "s2": "#7a4fa3"}
FORM_STYLES = {"orig": "-", "expanded": "--", "keywords": ":", "fused": "-."}


def load_tidy(path):
    rows = []
    for line in open(path):
        rec = json.loads(line)
        for style, cells in rec["cells"].items():
            for cell_key, c in cells.items():
                method, form = cell_key.split("/")
                rows.append({
                    "arxiv_id": rec["arxiv_id"],
                    "fig_index": rec["fig_index"],
                    "target_in_pf": rec.get("target_in_pf"),
                    "style": style, "method": method, "form": form,
                    "fig_rank": c.get("fig_rank"),
                    "paper_rank": c.get("paper_rank"),
                    "pool_size": c.get("pool_size"),
                    "n_arxiv_ids": c.get("n_arxiv_ids"),
                })
    return pd.DataFrame(rows)


def rank_column(df, level):
    if level == "fig":
        return df.fig_rank
    if level == "paper":
        return df.paper_rank
    # task: figure rank for ours, paper rank for baselines
    return df.fig_rank.where(df.method == "ours", df.paper_rank)


def recall_table(df, level):
    df = df.copy()
    df["rank"] = rank_column(df, level)
    out = []
    for (style, method, form), g in df.groupby(["style", "method", "form"]):
        entry = {"style": style, "method": method, "form": form,
                 "n": len(g)}
        for k in KS:
            entry[f"R@{k}"] = round(
                float((g["rank"].notna() & (g["rank"] <= k)).mean()), 3)
        if g.n_arxiv_ids.notna().any():
            entry["s2_empty"] = round(
                float((g.n_arxiv_ids == 0).mean()), 3)
        out.append(entry)
    t = pd.DataFrame(out)
    t["style"] = pd.Categorical(t["style"], STYLES)
    return t.sort_values(["style", "method", "form"]).reset_index(drop=True)


def print_blocks(t):
    for style in STYLES:
        block = t[t["style"] == style].drop(columns="style")
        if block.empty:
            continue
        print(f"\n=== {style} ===")
        print(block.to_string(index=False))


def plot_heatmap(t, level, k, stem):
    col = f"R@{k}"
    piv = t.pivot_table(index=["method", "form"], columns="style",
                        values=col, observed=True)
    piv = piv.reindex(columns=[s for s in STYLES if s in piv.columns])
    fig, ax = plt.subplots(
        figsize=(1.3 * len(piv.columns) + 3, 0.42 * len(piv) + 1.5))
    im = ax.imshow(piv.to_numpy(), vmin=0, vmax=1, cmap="viridis",
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv)),
                  [f"{m}/{f}" for m, f in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v < 0.55 else "black")
    ax.set_title(f"{col}, level={level}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = f"{stem}_heatmap_{level}_r{k}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_curves(t, level, stem):
    fig, axes = plt.subplots(1, len(STYLES), figsize=(3.2 * len(STYLES), 3.4),
                             sharey=True)
    for ax, style in zip(axes, STYLES):
        block = t[t["style"] == style]
        for _, r in block.iterrows():
            ys = [r[f"R@{k}"] for k in KS]
            ax.plot(KS, ys, color=METHOD_COLORS.get(r.method, "gray"),
                    linestyle=FORM_STYLES.get(r.form, "-"),
                    marker="o", ms=3,
                    label=f"{r.method}/{r.form}")
        ax.set_title(style)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("k")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"recall (level={level})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    out = f"{stem}_curves_{level}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--level", default="task",
                    choices=["task", "fig", "paper"])
    ap.add_argument("--k", type=int, default=20,
                    help="k for the heatmap")
    ap.add_argument("--forms", default=None,
                    help="comma list to restrict, e.g. orig,keywords")
    ap.add_argument("--covered-only", action="store_true",
                    help="restrict pathfinder rows to figures whose target "
                         "paper exists in the pathfinder corpus")
    args = ap.parse_args()

    stem = str(Path(args.jsonl).with_suffix(""))
    df = load_tidy(args.jsonl)
    n_figs = df[["arxiv_id", "fig_index"]].drop_duplicates().shape[0]
    print(f"{n_figs} figures, {len(df)} cells, level={args.level}")

    if args.forms:
        keep = [f.strip() for f in args.forms.split(",")]
        df = df[df.form.isin(keep)]
    if args.covered_only:
        drop = (df.method == "pathfinder") & (df.target_in_pf == False)  # noqa: E712
        print(f"covered-only: dropping {int(drop.sum())} pathfinder cells "
              "whose target is absent from the pathfinder corpus")
        df = df[~drop]

    if df.target_in_pf.notna().any():
        cov = df.drop_duplicates(["arxiv_id", "fig_index"]).target_in_pf
        print(f"pathfinder corpus covers {cov.mean():.1%} of sampled targets")

    t = recall_table(df, args.level)
    print_blocks(t)
    csv = f"{stem}_recall_{args.level}.csv"
    t.to_csv(csv, index=False)
    h = plot_heatmap(t, args.level, args.k, stem)
    c = plot_curves(t, args.level, stem)
    print(f"\nwrote: {csv}\n       {h}\n       {c}")


if __name__ == "__main__":
    main()
