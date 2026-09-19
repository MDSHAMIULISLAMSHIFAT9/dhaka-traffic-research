"""
04_figures.py - Phase 4: IEEE-ready figures.

    python 04_figures.py

Outputs (figures/):
    fig1_speed_boxplot.pdf / .png   Speed_kmh by corridor          (single column, 3.5 in)
    fig2_timeslot_lines.pdf / .png  Mean speed by time slot, one line per corridor
PDFs are vector graphics with embedded TrueType fonts (Type 42), which
IEEE PDF eXpress accepts. PNGs are rendered at 300 DPI.
"""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import FIG_DIR, SLOT_ORDER, corridor_ids, load_processed

# ---- switch here if you prefer corridor-normalised travel time in Fig. 2 ----
FIG2_Y = "Speed_kmh"                    # or "Estimated_Duration_Min" / "TTI"
FIG2_YLABEL = "Mean speed (km/h)"       # match the column above

COL_W = 3.5     # IEEE single-column width, inches
DPI = 300

# seaborn style first (it resets font.family), then our rcParams
sns.set_style("whitegrid", {"grid.linewidth": 0.4, "grid.color": "#dddddd"})
mpl.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")                # vector
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=DPI)       # 300 DPI raster
    plt.close(fig)
    print(f"saved figures/{stem}.pdf and .png")


def fig_boxplot(df, ids) -> None:
    d = df.copy()
    d["ID"] = d["Corridor_Name"].map(ids)
    order = list(ids.values())                       # C1 (fastest) ... Ck (slowest)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    sns.boxplot(
        data=d, x="ID", y="Speed_kmh", order=order, color="#9ecae1", linewidth=0.6, width=0.65,
        fliersize=1.2,
        flierprops={"marker": "o", "markerfacecolor": "none", "markeredgewidth": 0.3, "alpha": 0.6},
        ax=ax,
    )
    ax.set_xlabel("Corridor (see Table I)")
    ax.set_ylabel("Speed (km/h)")
    save(fig, "fig1_speed_boxplot")


def fig_timeslot(df, ids) -> None:
    d = df.copy()
    d["ID"] = d["Corridor_Name"].map(ids)
    d["Time_Slot"] = d["Time_Slot"].astype(str)
    order = list(ids.values())
    k = len(order)
    marks = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    lines = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    sns.pointplot(
        data=d, x="Time_Slot", y=FIG2_Y, order=SLOT_ORDER, hue="ID", hue_order=order,
        palette=sns.color_palette("colorblind", k),
        markers=[marks[i % len(marks)] for i in range(k)],
        linestyles=[lines[i % len(lines)] for i in range(k)],
        errorbar=("ci", 95), n_boot=1000, seed=42, dodge=0.3 if k > 4 else 0.2,
        err_kws={"linewidth": 0.6}, markersize=2.5, linewidth=0.8, ax=ax,
    )
    ax.set_xticks(range(len(SLOT_ORDER)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in SLOT_ORDER])
    ax.set_xlabel("Time slot (Asia/Dhaka)")
    ax.set_ylabel(FIG2_YLABEL)
    ax.legend(title="Corridor", ncol=2 if k > 4 else 1, frameon=True, framealpha=0.9,
              edgecolor="#bbbbbb", loc="best", fontsize=6, title_fontsize=6.5)
    save(fig, "fig2_timeslot_lines")


def main() -> None:
    df = load_processed()
    ids = corridor_ids(df)          # same C1..Ck labels as the paper tables
    fig_boxplot(df, ids)
    fig_timeslot(df, ids)


if __name__ == "__main__":
    main()
