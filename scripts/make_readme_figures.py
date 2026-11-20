"""Render the README figures from the small data files in docs/figures/data/.

Run:  ET_FIG_FONT_DIR=<folder with IBM Plex Mono .ttf> uv run --with matplotlib python scripts/make_readme_figures.py

Data sources (both committed):
- docs/figures/data/text-examples.json: two short generations by Qwen3.5-9B through the
  recorder (condition qwen35-9b-bf16-mlx-1195d88b, seed 7), with H, S and the top-5
  alternatives per token.
- docs/figures/data/smoke-summary.json: aggregate of the 20 smoke apps (exploratory).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures"
PREVIEW = Path(__import__("os").environ.get("ET_FIG_PREVIEW", "/tmp"))

# Colours and type follow the figures on batmanvane.github.io: paper, ink, warm grays,
# one dark green accent, IBM Plex Mono.
SURFACE = "#f2f1ec"  # paper
INK = "#1a1a1a"
INK2 = "#6b6b6b"
GRID = "#dcdad3"
GRAY = "#8f8c84"  # neutral bars
ACCENT = "#1f4d3a"  # dark green: forks, the sampled token, the highlighted series
# Sequential warm-gray ramp for entropy: paper (certain) -> dark gray (open).
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f7f6f2", "#e3e0d8", "#c4c1b8", "#9d9a92", "#77746d", "#4a4844"]
)
H_MAX = 3.0
FORK = 2.0

# IBM Plex Mono if available (set ET_FIG_FONT_DIR to a folder with its .ttf files);
# Menlo / DejaVu Sans cover glyphs Plex lacks (≥, ←, ⏎) and stand in otherwise.
_font_dir = __import__("os").environ.get("ET_FIG_FONT_DIR")
if _font_dir:
    for _f in Path(_font_dir).glob("*.ttf"):
        font_manager.fontManager.addfont(str(_f))

plt.rcParams.update(
    {
        "font.family": ["IBM Plex Mono", "Menlo", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",
    }
)


def _clean(tokens):
    """Drop empty display tokens (EOS, detokenizer space buffers) for the strips."""
    return [t for t in tokens if t["text"] and t["text"].strip() != ""]


def token_strips():
    import numpy as np

    ex = json.loads((DATA / "text-examples.json").read_text())
    rows = [("Sentence", ex["button"]), ("Code", ex["loop"])]
    # lay out first, then size the figure to the content
    placed, y = [], 0.0
    for label, e in rows:
        placed.append(("label", 0.0, y, f'{label}:  "{e["prompt"]}"', None))
        y -= 6.5
        x = 0.0
        for t in _clean(e["tokens"]):
            txt = t["text"].replace("\n", "⏎")
            w = 1.25 * len(txt) + 1.6
            if x + w > 100:
                x, y = 0.0, y - 8.5
            placed.append(("tok", x, y, txt, (w, t["H"])))
            x += w
        y -= 13
    y_legend = y + 4
    height_units = -y_legend + 8
    fig = plt.figure(figsize=(10.5, height_units * 0.085))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-1, 101)
    ax.set_ylim(y_legend - 5, 5)
    ax.axis("off")
    for kind, x, yy, txt, extra in placed:
        if kind == "label":
            ax.text(x, yy, txt, fontsize=8.5, color=INK2, va="center")
            continue
        w, h = extra
        fork = h >= FORK
        ax.add_patch(FancyBboxPatch((x, yy - 2.4), w - 0.6, 4.8,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=SEQ(min(h, H_MAX) / H_MAX),
                                    edgecolor=ACCENT if fork else SURFACE,
                                    linewidth=2.4 if fork else 0.5))
        dark_box = min(h, H_MAX) / H_MAX > 0.55
        ax.text(x + (w - 0.6) / 2, yy, txt, ha="center", va="center", fontsize=8.5,
                color="white" if dark_box else INK)
        ax.text(x + (w - 0.6) / 2, yy - 4.0, f"{h:.1f}", ha="center", va="center",
                fontsize=6.5, color=INK2)
    # legend row
    ax.add_patch(FancyBboxPatch((0, y_legend - 1.3), 3.2, 2.6,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor=SURFACE, edgecolor=ACCENT, linewidth=2.4))
    ax.text(4.3, y_legend, "fork: H ≥ 2 bits     number under each token = H in bits",
            va="center", fontsize=8, color=INK2)
    x0, x1 = 62, 92
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=SEQ,
              extent=[x0, x1, y_legend - 1.0, y_legend + 1.0], zorder=1)
    for v in range(4):
        xv = x0 + (x1 - x0) * v / H_MAX
        ax.text(xv, y_legend - 2.9, ["0", "1", "2", "≥3 bits"][v], ha="center",
                fontsize=7.5, color=INK2)
    ax.text((x0 + x1) / 2, y_legend + 2.6, "entropy H", ha="center", fontsize=8, color=INK2)
    fig.savefig(OUT / "token-strips.svg", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(PREVIEW / "token-strips.png", dpi=110, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fork_distribution():
    ex = json.loads((DATA / "text-examples.json").read_text())
    toks = ex["button"]["tokens"]
    i = next(k for k, t in enumerate(toks) if t["text"] == " triggers")
    t = toks[i]
    alts = t["alts"][:4]
    rest = max(0.0, 1.0 - sum(p for _, p in alts))
    labels = [a.strip() for a, _ in alts] + ["all others"]
    probs = [p for _, p in alts] + [rest]
    prefix = "".join(x["text"] for x in toks[:i]).strip()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = list(range(len(labels)))[::-1]
    for y, lab, p in zip(ys, labels, probs):
        chosen = lab == "triggers"
        ax.barh(y, p, height=0.62, color=ACCENT if chosen else GRAY, zorder=2)
        ax.text(p + 0.008, y, f"{p:.3f}" + ("   ← sampled" if chosen else ""),
                va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("probability of the next token")
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f'"{prefix} ___"\nH = {t["H"]:.2f} bits   ·   S(triggers) = {t["S"]:.2f} bits',
                 fontsize=10, loc="left", color=INK)
    fig.savefig(OUT / "fork-distribution.svg", bbox_inches="tight")
    plt.close(fig)


def entropy_histogram():
    s = json.loads((DATA / "smoke-summary.json").read_text())
    e, c = s["log10H_edges"], s["log10H_counts"]
    total = sum(c)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for lo, hi, n in zip(e[:-1], e[1:], c):
        ax.bar((lo + hi) / 2, 100 * n / total, width=(hi - lo) * 0.85, color=GRAY, zorder=2)
    for x, lab in [(-1.0, "0.1 bits"), (0.301, "2 bits = fork")]:
        ax.axvline(x, color=INK2, linestyle=(0, (3, 3)), linewidth=1, zorder=3)
        ax.text(x + 0.04, ax.get_ylim()[1] * 0.92, lab, fontsize=8.5, color=INK2)
    ticks = [-3, -2, -1, 0, 0.477]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["≤0.001", "0.01", "0.1", "1", "3"])
    ax.set_xlabel("entropy H per token (bits, log scale)")
    ax.set_ylabel("% of tokens")
    ax.grid(axis="y", color=GRID, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{total:,} tokens from {s['n_apps']} apps (same brief: unit converter)",
                 fontsize=10, loc="left")
    fig.savefig(OUT / "entropy-histogram.svg", bbox_inches="tight")
    plt.close(fig)


def forks_by_section():
    s = json.loads((DATA / "smoke-summary.json").read_text())
    secs = ["css", "html", "js", "comment"]
    names = {"css": "CSS", "html": "HTML", "js": "JS code", "comment": "comments"}
    tt, ft = sum(s["section_tokens"].values()), sum(s["section_forks"].values())
    tok = [100 * s["section_tokens"][k] / tt for k in secs]
    frk = [100 * s["section_forks"][k] / ft for k in secs]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(secs))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], tok, width=w, color=GRAY, label="share of all tokens", zorder=2)
    ax.bar([i + w / 2 + 0.01 for i in x], frk, width=w, color=ACCENT, label="share of all forks (H ≥ 2 bits)", zorder=2)
    for i in x:
        ax.text(i - w / 2 - 0.01, tok[i] + 1.2, f"{tok[i]:.0f} %", ha="center", fontsize=8.5)
        ax.text(i + w / 2 + 0.01, frk[i] + 1.2, f"{frk[i]:.0f} %", ha="center", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([names[k] for k in secs])
    ax.set_ylim(0, 75)
    ax.set_ylabel("%")
    ax.grid(axis="y", color=GRID, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Where the forks are (20 smoke apps)", fontsize=10, loc="left")
    fig.savefig(OUT / "forks-by-section.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    token_strips()
    fork_distribution()
    entropy_histogram()
    forks_by_section()
    print("written:", sorted(p.name for p in OUT.glob("*.svg")))
