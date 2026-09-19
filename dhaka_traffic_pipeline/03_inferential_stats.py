"""
03_inferential_stats.py - Phase 3: one-way ANOVA of speed ACROSS CORRIDORS.

    python 03_inferential_stats.py

H0: mean speed is the same on every corridor.  H1: at least one corridor differs.

Primary analysis : Speed_kmh by Corridor_Name (observation level)
Robustness check : Speed_kmh averaged per (Date, Corridor), by Corridor
                   (reduces pseudo-replication from 15-minute autocorrelation)

Volatility       : coefficient of variation (CV) of travel time per corridor,
                   and Levene's test for equal variances across corridors.

Writes CSV tables to tables/ and LaTeX tables + macros to paper/generated/.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from common import GEN_DIR, SLOT_ORDER, TAB_DIR, corridor_ids, load_processed

ALPHA = 0.05
TOP_PAIRS = 6          # pairs shown in the paper's post-hoc table (all pairs go to CSV)


# ---------------------------------------------------------------- helpers
def fmt_p(p: float) -> str:
    """'< 0.001' or '= 0.012' (used as  $p \\AnovaP$ )."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def fmt_p_cell(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def n2(x: float) -> str:
    """2-decimal number with a typographic minus sign for LaTeX text mode."""
    return f"{x:.2f}".replace("-", "$-$")


def tex_escape(s) -> str:
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(c, c) for c in str(s))


def macro(name: str, value) -> str:
    return "\\newcommand{\\%s}{%s}\n" % (name, value)


def games_howell(d: pd.DataFrame, y: str, levels: list) -> pd.DataFrame:
    """Games-Howell pairwise test (unequal variances / unequal n). diff = mean(group2) - mean(group1)."""
    grp = {k: d.loc[d["Group"] == k, y].to_numpy() for k in levels}
    k = len(levels)
    rows = []
    for a, b in itertools.combinations(levels, 2):
        x, z = grp[a], grp[b]
        n1, n2_ = len(x), len(z)
        v1, v2 = x.var(ddof=1), z.var(ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2_)
        diff = z.mean() - x.mean()
        dfw = (v1 / n1 + v2 / n2_) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2_) ** 2 / (n2_ - 1))
        q = abs(diff) / (se / np.sqrt(2))
        p = float(studentized_range.sf(q, k, dfw))
        half = studentized_range.ppf(1 - ALPHA, k, dfw) * se / np.sqrt(2)
        rows.append({"group1": a, "group2": b, "meandiff": diff, "p_adj": p,
                     "lower": diff - half, "upper": diff + half, "reject": p < ALPHA})
    return pd.DataFrame(rows)


def get_pair(tbl: pd.DataFrame, a: str, b: str):
    """Return dict(diff, lo, hi, p) for mean(b) - mean(a), whatever order tbl stored it in."""
    r = tbl[(tbl.group1 == a) & (tbl.group2 == b)]
    if len(r):
        r = r.iloc[0]
        return {"diff": r.meandiff, "lo": r.lower, "hi": r.upper, "p": r.p_adj}
    r = tbl[(tbl.group1 == b) & (tbl.group2 == a)]
    if len(r):
        r = r.iloc[0]
        return {"diff": -r.meandiff, "lo": -r.upper, "hi": -r.lower, "p": r.p_adj}
    return None


# ---------------------------------------------------------------- core suite
def run_suite(d_in: pd.DataFrame, y: str, label: str, order: list) -> dict:
    d = d_in[["Corridor_Name", y]].dropna().rename(columns={"Corridor_Name": "Group"}).copy()
    d["Group"] = d["Group"].astype(str)
    levels = [c for c in order if c in set(d["Group"])]
    arrays = [d.loc[d["Group"] == k, y].to_numpy() for k in levels]
    if len(levels) < 2 or any(len(a) < 2 for a in arrays):
        raise SystemExit(f"[{label}] need >=2 corridors with >=2 observations each.")
    flat = [k for k, a in zip(levels, arrays) if np.std(a, ddof=1) <= 1e-9 * max(1.0, abs(np.mean(a)))]
    if flat:
        raise SystemExit(
            f"[{label}] {y} is CONSTANT within corridor(s): {flat}.\n"
            "  Every reading is identical, so there is no within-group variation and an ANOVA F-ratio\n"
            "  is undefined (division by zero). This is what a static router produces (same duration at\n"
            "  every hour). Fix the data source first - see GUIDE.md, section 0."
        )
    N, k = len(d), len(levels)

    bar = "=" * 72
    print(f"\n{bar}\n{label}   (response: {y}, N = {N:,}, k = {k} corridors)\n{bar}")

    desc = d.groupby("Group")[y].agg(n="size", mean="mean", sd="std", median="median").reindex(levels)
    print("\n[1] Descriptives by corridor\n", desc.round(3).to_string())

    rng = np.random.default_rng(42)
    sw_rows = []
    for name, a in zip(levels, arrays):
        s = a if len(a) <= 5000 else rng.choice(a, 5000, replace=False)
        W, p = stats.shapiro(s)
        sw_rows.append({"Corridor": name, "n": len(a), "W": W, "p": p,
                        "skew": stats.skew(a), "excess_kurtosis": stats.kurtosis(a)})
    sw = pd.DataFrame(sw_rows)
    print("\n[2] Shapiro-Wilk normality (H0: normal)\n", sw.round(4).to_string(index=False))

    lev = stats.levene(*arrays, center="median")
    print(f"\n[3] Levene (median-centred) W({k - 1}, {N - k}) = {lev.statistic:.3f}, p {fmt_p(lev.pvalue)}")

    model = smf.ols(f"{y} ~ C(Group)", data=d).fit()
    aov = sm.stats.anova_lm(model, typ=1)
    ss_b, ss_w = float(aov.iloc[0]["sum_sq"]), float(aov.iloc[1]["sum_sq"])
    df_b, df_w = int(aov.iloc[0]["df"]), int(aov.iloc[1]["df"])
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    F, p_F = float(aov.iloc[0]["F"]), float(aov.iloc[0]["PR(>F)"])
    eta2 = ss_b / (ss_b + ss_w)
    F_sp, p_sp = stats.f_oneway(*arrays)
    if not np.isclose(F, F_sp, rtol=1e-6):
        warnings.warn("statsmodels and scipy F disagree - inspect the data.")
    print("\n[4] One-way ANOVA")
    print(f"    F({df_b}, {df_w}) = {F:.3f},  p {fmt_p(p_F)},  eta^2 = {eta2:.4f}")
    print(f"    SS_between = {ss_b:,.2f}   SS_within = {ss_w:,.2f}   MS_between = {ms_b:,.3f}   MS_within = {ms_w:,.3f}")
    print(f"    (cross-check scipy.f_oneway: F = {F_sp:.3f}, p = {p_sp:.3g})")

    welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)
    if hasattr(welch, "df_num"):
        w_num, w_den = float(welch.df_num), float(welch.df_denom)
    else:
        w_num, w_den = (float(x) for x in welch.df)
    kw = stats.kruskal(*arrays)
    print(f"\n[5] Welch ANOVA  F({w_num:.0f}, {w_den:.1f}) = {welch.statistic:.3f}, p {fmt_p(welch.pvalue)}")
    print(f"    Kruskal-Wallis H({k - 1}) = {kw.statistic:.3f}, p {fmt_p(kw.pvalue)}, epsilon^2 = {kw.statistic / (N - 1):.4f}")

    tk = pairwise_tukeyhsd(endog=d[y].to_numpy(), groups=d["Group"].to_numpy(), alpha=ALPHA)
    gu = [str(x) for x in tk.groupsunique]
    ii, jj = np.triu_indices(len(gu), 1)
    tuk = pd.DataFrame({"group1": [gu[a] for a in ii], "group2": [gu[b] for b in jj],
                        "meandiff": tk.meandiffs, "p_adj": tk.pvalues,
                        "lower": tk.confint[:, 0], "upper": tk.confint[:, 1],
                        "reject": tk.reject.astype(bool)})
    gh = games_howell(d, y, levels)
    print(f"\n[6a] Tukey HSD ({len(tuk)} pairs; diff = mean(group2) - mean(group1))\n", tuk.round(4).to_string(index=False))
    print(f"\n[6b] Games-Howell\n", gh.round(4).to_string(index=False))

    desc.round(4).to_csv(TAB_DIR / f"{label}_descriptives.csv")
    sw.to_csv(TAB_DIR / f"{label}_shapiro.csv", index=False)
    tuk.to_csv(TAB_DIR / f"{label}_tukey.csv", index=False)
    gh.to_csv(TAB_DIR / f"{label}_games_howell.csv", index=False)

    return dict(label=label, y=y, levels=levels, N=N, k=k, desc=desc, sw=sw, lev=lev,
                ss_b=ss_b, ss_w=ss_w, df_b=df_b, df_w=df_w, ms_b=ms_b, ms_w=ms_w, F=F, p=p_F,
                eta2=eta2, welch=welch, w_num=w_num, w_den=w_den, kw=kw, tuk=tuk, gh=gh)


# ---------------------------------------------------------------- LaTeX writers
def write_tables(df: pd.DataFrame, r: dict, ids: dict) -> pd.DataFrame:
    """Writes corridor, descriptive, ANOVA, post-hoc and slot tables. Returns the volatility table."""
    order = r["levels"]

    g = df.groupby("Corridor_Name")
    vol = pd.DataFrame({
        "dist": g["Distance_km"].median(),
        "n": g.size(),
        "mean_t": g["Estimated_Duration_Min"].mean(),
        "sd_t": g["Estimated_Duration_Min"].std(),
    })
    vol["cv_t"] = 100 * vol["sd_t"] / vol["mean_t"]
    vol = vol.reindex(order)
    vol.round(4).to_csv(TAB_DIR / "corridor_volatility.csv")

    # Table: monitored corridors (ID -> name)
    lines = [r"\begin{table}[!t]", r"\caption{Monitored corridors}", r"\label{tab:corridors}",
             r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrr}", r"\toprule",
             r"ID & Corridor & Dist. (km) & $n$ \\", r"\midrule"]
    for name in order:
        lines.append(f"{ids[name]} & {tex_escape(name)} & {vol.loc[name, 'dist']:.1f} & {int(vol.loc[name, 'n']):,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN_DIR / "corridor_table.tex").write_text("\n".join(lines) + "\n")

    # Table: descriptives of speed + CV of travel time
    d = r["desc"]
    lines = [r"\begin{table}[!t]", r"\caption{Speed (km/h) and travel-time volatility by corridor}",
             r"\label{tab:desc}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"ID & Mean & SD & Median & CV$_t$ (\%) \\", r"\midrule"]
    for name in order:
        row = d.loc[name]
        lines.append(f"{ids[name]} & {row['mean']:.2f} & {row['sd']:.2f} & {row['median']:.2f} & {vol.loc[name, 'cv_t']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN_DIR / "desc_table.tex").write_text("\n".join(lines) + "\n")

    # Table: ANOVA
    tot = r["ss_b"] + r["ss_w"]
    lines = [r"\begin{table}[!t]", r"\caption{One-way ANOVA of speed (km/h) by corridor}",
             r"\label{tab:anova}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Source & SS & df & MS & $F$ & $p$ \\", r"\midrule",
             f"Between corridors & {r['ss_b']:,.1f} & {r['df_b']} & {r['ms_b']:,.2f} & {r['F']:.2f} & {fmt_p_cell(r['p'])} \\\\",
             f"Within corridors & {r['ss_w']:,.1f} & {r['df_w']:,} & {r['ms_w']:,.2f} & & \\\\",
             f"Total & {tot:,.1f} & {r['df_b'] + r['df_w']:,} & & & \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN_DIR / "anova_table.tex").write_text("\n".join(lines) + "\n")

    # Table: post-hoc, the TOP_PAIRS largest absolute differences
    top = r["tuk"].reindex(r["tuk"]["meandiff"].abs().sort_values(ascending=False).index).head(TOP_PAIRS)
    which = f"all {len(top)} pairs" if len(top) == len(r["tuk"]) else f"the {len(top)} largest of {len(r['tuk'])} pairs"
    lines = [r"\begin{table}[!t]",
             rf"\caption{{Post-hoc comparisons of mean speed (km/h): {which}; "
             r"$\Delta\bar{x}$ = second $-$ first corridor}",
             r"\label{tab:tukey}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrcrr}", r"\toprule",
             r"Pair & $\Delta\bar{x}$ & 95\% CI (Tukey) & $p_{\mathrm{Tukey}}$ & $p_{\mathrm{GH}}$ \\", r"\midrule"]
    for _, row in top.iterrows():
        a, b = row["group1"], row["group2"]
        gh = get_pair(r["gh"], a, b)
        ghp = fmt_p_cell(gh["p"]) if gh else "--"
        lines.append(f"{ids[a]}--{ids[b]} & {'+' if row['meandiff'] >= 0 else ''}{n2(row['meandiff'])} & "
                     f"[{n2(row['lower'])}, {n2(row['upper'])}] & {fmt_p_cell(row['p_adj'])} & {ghp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN_DIR / "tukey_table.tex").write_text("\n".join(lines) + "\n")

    # Table: mean speed by corridor x time slot
    piv = df.pivot_table(index="Corridor_Name", columns="Time_Slot", values="Speed_kmh",
                         aggfunc="mean", observed=True).reindex(order)
    piv = piv[[s for s in SLOT_ORDER if s in piv.columns]]
    cols = "l" + "r" * len(piv.columns)
    lines = [r"\begin{table}[!t]", r"\caption{Mean speed (km/h) by corridor and time slot}",
             r"\label{tab:slots}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             "ID & " + " & ".join(s.replace(" ", "~") for s in piv.columns) + r" \\", r"\midrule"]
    for name, row in piv.iterrows():
        lines.append(f"{ids[name]} & " + " & ".join(f"{v:.2f}" for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN_DIR / "slot_table.tex").write_text("\n".join(lines) + "\n")
    return vol


def pair_sentence(a: str, b: str, r: dict, ids: dict) -> str:
    t, g = get_pair(r["tuk"], a, b), get_pair(r["gh"], a, b)
    if not (t and g):
        return "No pairwise comparison was possible."
    direction = "faster" if t["diff"] > 0 else "slower"
    verdict = "statistically significant" if t["p"] < ALPHA else "not statistically significant"
    return (f"Corridor {ids[b]} ({tex_escape(b)}) was {abs(t['diff']):.2f}~km/h {direction} than "
            f"{ids[a]} ({tex_escape(a)}) (Tukey HSD $p_{{\\mathrm{{adj}}}} {fmt_p(t['p'])}$, "
            f"95\\% CI [{n2(t['lo'])}, {n2(t['hi'])}]~km/h; Games--Howell $p {fmt_p(g['p'])}$), "
            f"which is {verdict} at $\\alpha = 0.05$.")


def write_macros(df: pd.DataFrame, r: dict, agg: dict, rho: float, ids: dict, vol: pd.DataFrame) -> None:
    ts = df["Timestamp"]
    means = r["desc"]["mean"]
    fast, slow = means.idxmax(), means.idxmin()
    n_eff = r["N"] * (1 - rho) / (1 + rho) if rho > -1 else r["N"]
    lab = lambda c: f"{ids[c]} ({tex_escape(c)})"          # noqa: E731
    slot_means = df.groupby("Time_Slot", observed=True)["Speed_kmh"].mean().dropna()

    m = ""
    m += macro("NObs", f"{r['N']:,}")
    m += macro("NDays", df["Date"].nunique())
    m += macro("NCorr", df["Corridor_Name"].nunique())
    m += macro("DateStart", ts.min().strftime("%d %B %Y"))
    m += macro("DateEnd", ts.max().strftime("%d %B %Y"))
    m += macro("AnovaF", f"{r['F']:.2f}")
    m += macro("AnovaDFB", r["df_b"])
    m += macro("AnovaDFW", r["df_w"])
    m += macro("AnovaP", fmt_p(r["p"]))
    m += macro("EtaSq", f"{r['eta2']:.3f}")
    m += macro("AnovaVerdict", "reject" if r["p"] < ALPHA else "fail to reject")
    m += macro("WelchF", f"{r['welch'].statistic:.2f}")
    m += macro("WelchDFnum", f"{r['w_num']:.0f}")
    m += macro("WelchDFden", f"{r['w_den']:.1f}")
    m += macro("WelchP", fmt_p(r["welch"].pvalue))
    m += macro("KWH", f"{r['kw'].statistic:.2f}")
    m += macro("KWP", fmt_p(r["kw"].pvalue))
    m += macro("LevW", f"{r['lev'].statistic:.2f}")
    m += macro("LevP", fmt_p(r["lev"].pvalue))
    m += macro("LevVerdict", "unequal" if r["lev"].pvalue < ALPHA else "homogeneous")
    m += macro("NormReject", f"{int((r['sw']['p'] < ALPHA).sum())} of {len(r['sw'])}")
    m += macro("FastestCorr", lab(fast))
    m += macro("FastestMean", f"{means[fast]:.2f}")
    m += macro("SlowestCorr", lab(slow))
    m += macro("SlowestMean", f"{means[slow]:.2f}")
    mv, lv = vol["cv_t"].idxmax(), vol["cv_t"].idxmin()
    m += macro("MostVolCorr", lab(mv))
    m += macro("MostVolCV", f"{vol.loc[mv, 'cv_t']:.1f}")
    m += macro("LeastVolCorr", lab(lv))
    m += macro("LeastVolCV", f"{vol.loc[lv, 'cv_t']:.1f}")
    m += macro("SlowSlot", slot_means.idxmin())
    m += macro("SlowSlotMean", f"{slot_means.min():.2f}")
    m += macro("FastSlot", slot_means.idxmax())
    m += macro("FastSlotMean", f"{slot_means.max():.2f}")
    m += macro("NPairs", len(r["tuk"]))
    m += macro("NSigTukey", int(r["tuk"]["reject"].sum()))
    m += macro("NSigGH", int(r["gh"]["reject"].sum()))
    m += macro("AggF", f"{agg['F']:.2f}")
    m += macro("AggDFB", agg["df_b"])
    m += macro("AggDFW", agg["df_w"])
    m += macro("AggP", fmt_p(agg["p"]))
    m += macro("RhoOne", f"{rho:.2f}")
    m += macro("NEff", f"{n_eff:,.0f}")
    m += macro("SentTop", pair_sentence(slow, fast, r, ids))
    (GEN_DIR / "results_macros.tex").write_text(m)


# ---------------------------------------------------------------- main
def main() -> None:
    df = load_processed()
    ids = corridor_ids(df)
    order = list(ids.keys())               # fastest -> slowest corridor
    pd.Series(ids, name="ID").rename_axis("Corridor").to_csv(TAB_DIR / "corridor_ids.csv")

    dates = df.groupby("Corridor_Name")["Date"].nunique()
    print(f"Calendar days per corridor: min {dates.min()}, max {dates.max()}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # constant series -> NaN autocorrelation
        rho = (df.sort_values("Timestamp").groupby("Corridor_Name")["Speed_kmh"]
                 .apply(lambda s: s.autocorr(1)).mean())
    print(f"Mean lag-1 autocorrelation of Speed_kmh within corridors: {rho:.3f}")
    print("   ANOVA assumes independent observations; effective N ~ N(1-rho)/(1+rho).")

    primary = run_suite(df, "Speed_kmh", "primary_speed_by_corridor", order)
    agg_df = df.groupby(["Date", "Corridor_Name"], observed=True)["Speed_kmh"].mean().reset_index()
    agg = run_suite(agg_df, "Speed_kmh", "robust_daily_corridor_means", order)

    vol = write_tables(df, primary, ids)
    write_macros(df, primary, agg, rho, ids, vol)
    print(f"\nLaTeX tables + macros written to {GEN_DIR}")


if __name__ == "__main__":
    main()
