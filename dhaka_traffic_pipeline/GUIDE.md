# Dhaka Traffic Volatility Paper — Operational Blueprint

Everything runs from the `dhaka_traffic_pipeline/` folder:

```
common.py                 shared settings + cleaning + time-slot logic
01_quality_check.py       Phase 1  health report + static-duration check
02_ingest_preprocess.py   Phase 2  fetch from GitHub, clean, assign time slots
03_inferential_stats.py   Phase 3  Shapiro, Levene, ANOVA, Welch, KW, Tukey, Games-Howell
04_figures.py             Phase 4  two IEEE-sized figures (PDF vector + 300 DPI PNG)
paper/main.tex            Phase 5  IEEEtran draft (numbers/tables auto-filled)
paper/references.bib
requirements.txt
```

---

## 0. Read this first: does OSRM actually give you *time-varying* durations?

**This decides whether the study is valid, so check it in the first 24 hours, not on day 14.**

The default OSRM car profile assigns a fixed speed to each road class (motorway, primary, residential, …). It does **not** ingest live traffic. A published test of an OSRM-based engine against Google Maps at 09:00 found its durations were only about 0.53–0.58 of Google's for Delhi and Bengaluru, because its speed table did not change with time of day (Ghosh et al., arXiv:2011.13556, cited in the paper).

If your logger calls a stock OSRM server, `Estimated_Duration_Min` for a corridor will be the **same number at 3 a.m. and 6 p.m.** Then every reading on a corridor is identical, there is no within-group variation, and an ANOVA (across days **or across corridors**) is undefined: the F-ratio divides by zero. `03_inferential_stats.py` detects this and stops with a message instead of printing nonsense.

**Test (after ~1 day of data):**

```bash
python 01_quality_check.py
```

Look at section 3. Per corridor you want `n_unique` in the hundreds, `cv_pct` well above 1, and `hourly_swing_pct` clearly above 0. The script prints `CRITICAL` if every corridor is static.

**If the data are static, switch the duration source** (the schema stays identical):

Traffic-aware routing APIs exist from Google (Routes / Distance Matrix, `duration_in_traffic`), TomTom, HERE and Mapbox (`driving-traffic` profile). Check each provider's current free-tier limits and terms; 4 corridors × 96 calls/day is small. Example replacement for the route call in `osrm_logger.py` using TomTom (verify parameters against the current TomTom docs):

```python
import os, requests

TOMTOM_KEY = os.environ["TOMTOM_API_KEY"]

def get_route(o_lat, o_lon, d_lat, d_lon):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
    r = requests.get(url, params={"key": TOMTOM_KEY, "traffic": "true",
                                  "travelMode": "car", "routeType": "fastest"}, timeout=30)
    r.raise_for_status()
    s = r.json()["routes"][0]["summary"]
    return s["lengthInMeters"] / 1000, s["travelTimeInSeconds"] / 60   # km, minutes
```

Store the key: GitHub repo → **Settings → Secrets and variables → Actions → New repository secret** → name `TOMTOM_API_KEY`. In `.github/workflows/traffic_logger.yml`, under the logging step add:

```yaml
        env:
          TOMTOM_API_KEY: ${{ secrets.TOMTOM_API_KEY }}
```

If you switch, edit Section III-B of `paper/main.tex` (say "TomTom Routing API" instead of OSRM) and keep the OSRM discussion only as related work.

---

## Setup (once)

```bash
cd dhaka_traffic_pipeline
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Open `common.py` and set `GITHUB_USER = "<your GitHub username>"`. Nothing else is required.
2. Timestamps: `LOGGER_TZ = "UTC"` is right if `osrm_logger.py` runs on GitHub's runners and writes `datetime.now()` or UTC. If your script writes Dhaka time, set `LOGGER_TZ = "Asia/Dhaka"`.
3. Private repo only: `export GITHUB_TOKEN=<personal access token with repo scope>` (PowerShell: `$env:GITHUB_TOKEN="..."`).
4. To work from a local copy instead: `export DATA_SOURCE=/path/to/dhaka_osrm_traffic_data.csv`.

Note: `raw.githubusercontent.com` caches for a few minutes, so a fresh commit may take ~5 min to appear.

---

## Phase 1 — Data monitoring & quality control (days 1–14)

### 1.1 Check the CSV on GitHub (click sequence)

1. Open `github.com/<you>/dhaka-traffic-research` → **Actions** tab.
2. In the left sidebar click your workflow's name (from `name:` in `traffic_logger.yml`). Green ticks = success. Expect up to 96 runs/day; GitHub cron is best-effort, so delayed or skipped runs are normal.
3. Click any red run → the job → expand the failing step → read the traceback (usually an HTTP timeout or a push conflict).
4. **Code** tab → click `dhaka_osrm_traffic_data.csv` → the header shows the line count. Check again 24 h later. Growth per day ≈ 96 × (number of corridors) × 60–95%.
5. **Code** tab → click the commit count → bot commits should appear roughly every 15 minutes.
6. **Settings → Actions → General → Workflow permissions** must be *Read and write permissions* (already done).

### 1.2 Check from a terminal

```bash
git clone https://github.com/<you>/dhaka-traffic-research.git     # first time only
cd dhaka-traffic-research && git pull
wc -l dhaka_osrm_traffic_data.csv                                  # total lines
tail -n 5 dhaka_osrm_traffic_data.csv                              # newest rows
cut -d, -f3 dhaka_osrm_traffic_data.csv | sort | uniq -c           # rows per corridor
awk -F, 'NR>1 && ($5==0 || $5=="")' dhaka_osrm_traffic_data.csv | wc -l   # zero/blank durations
```

(`cut`/`awk` assume corridor names contain no commas.)

### 1.3 Automated report

```bash
python 01_quality_check.py
```

It calls `common.clean()`, which applies these rules in order and reports how many rows each removes:

1. repeated header lines (logger appended the header again);
2. blank / `None` / `ERROR` API responses → missing → row dropped;
3. zero or negative duration or distance;
4. duplicate `(Timestamp, Corridor_Name)`;
5. conversion to `Asia/Dhaka` and **recomputation of `Day_of_Week` from local time** (a UTC weekday would put every evening after 18:00 Dhaka time on the wrong day);
6. distance more than 25% away from the corridor median (route changed);
7. `Speed_kmh = Distance_km / (Estimated_Duration_Min / 60)`;
8. speeds above 120 km/h.

Outliers in duration are deliberately **not** trimmed: volatility is the thing you are measuring.

**Go/no-go before analysis:**

| Check | Target |
|---|---|
| Section 3 static check | no `CRITICAL`, no static corridors |
| Missing/API-failure rate | under ~5% |
| Calendar days observed | ≥ 7 (14 is better) |
| Coverage per corridor-day | mostly ≥ 60%; check gaps > 45 min |

---

## Phase 2 — Ingestion & preprocessing

```bash
python 02_ingest_preprocess.py
```

Fetches the raw CSV from `raw.githubusercontent.com/<you>/dhaka-traffic-research/main/dhaka_osrm_traffic_data.csv`, cleans it, and writes `data/processed.csv`. Time slots use **local Dhaka hour** (`common.add_time_features`):

| Slot | Local time |
|---|---|
| Morning Peak | 07:00–09:59 |
| Midday | 10:00–15:59 |
| Evening Peak | 16:00–19:59 |
| Night | 20:00–06:59 |

These windows are my assumption; edit them in `add_time_features` if your advisor prefers different peaks, and state the windows in the paper (Table II already does).

Extra columns: `Speed_Rel` (speed ÷ corridor median, removes corridor-length/road-class effects) and `TTI` (duration ÷ corridor 5th-percentile duration, a free-flow proxy). `tables/volatility_by_slot.csv` gives mean, SD and coefficient of variation per corridor × weekday × slot.

---

## Phase 3 — Inferential statistics (ANOVA across corridors)

```bash
python 03_inferential_stats.py
```

**Question:** does mean speed differ between corridors? $H_0:\mu_1=\dots=\mu_k$ vs $H_1$: at least one corridor differs. Corridors get short IDs (C1 = fastest mean speed, C2, …) so tables and figures stay narrow; `tables/corridor_ids.csv` and Table I in the paper map IDs to names.

Two runs: **primary** (`Speed_kmh` by corridor, every 15-minute reading) and a **robustness check** (`Speed_kmh` averaged per date × corridor, which removes most autocorrelation).

### Formulas

$$F=\frac{MS_B}{MS_W}=\frac{\sum_j n_j(\bar y_j-\bar y)^2/(k-1)}{\sum_j\sum_i (y_{ij}-\bar y_j)^2/(N-k)},\qquad \eta^2=\frac{SS_B}{SS_T}$$

$$q=\frac{|\bar y_a-\bar y_b|}{\sqrt{\tfrac{MS_W}{2}\left(\tfrac1{n_a}+\tfrac1{n_b}\right)}}\quad\text{(Tukey–Kramer)},\qquad \mathrm{CV}_j=100\,\frac{s_{t,j}}{\bar t_j}\quad\text{(volatility)}$$

### How to read each block of output

| Block | What to look at | Decision rule |
|---|---|---|
| [1] Descriptives | `mean`, `sd`, `n` per corridor | Are group sizes roughly similar? |
| [2] Shapiro–Wilk | `W` near 1, `p`, `skew`, `excess_kurtosis` | `p < 0.05` = non-normal. With thousands of rows this almost always fires; judge by skew (absolute value below about 1 is tolerable) and lean on [5]. |
| [3] Levene | `W(df1, df2)`, `p` | `p < 0.05` = unequal variances, i.e. **speed volatility differs between corridors**, and Tukey is questionable, so use Games–Howell. |
| [4] ANOVA | `F(df_between, df_within)`, `p`, `eta^2` | `p < 0.05` → reject $H_0$ (some corridor mean differs; it does not say which). Effect size: $\eta^2\approx0.01$ small, $0.06$ medium, $0.14$ large. Report $\eta^2$ because with big $N$ tiny effects are "significant". |
| [5] Welch / Kruskal–Wallis | same `p` logic | If [3] rejects, quote Welch instead of classical $F$. Agreement with [4] means the conclusion is robust. |
| [6a] Tukey HSD | `meandiff` = mean(group2) − mean(group1); `p_adj`; `lower`/`upper` | `p_adj < 0.05` ⇔ CI excludes 0 ⇔ `reject = True`. |
| [6b] Games–Howell | same columns | Use when Levene rejects. |

Example: a row `Gulistan-Uttara  Motijheel-Dhanmondi  -2.92  0.000  -3.50  -2.35  True` reads "Motijheel-Dhanmondi is 2.92 km/h slower, CI excludes zero, significant after multiple-comparison adjustment".

### Things to state honestly in the paper

- **Serial correlation.** Successive 15-minute readings are not independent. The script prints lag-1 autocorrelation $\rho$ and $N_{\mathrm{eff}}\approx N(1-\rho)/(1+\rho)$. If $\rho>0.5$, treat p-values as optimistic and lead with the daily-means robustness result.
- **Corridors differ by design.** Corridors differ in length, road class and land use, so a significant ANOVA describes the corridors, not a cause. Say that.
- **Constant data stop the script.** If a corridor's speed is identical at every reading, the script exits with an explanation (section 0).

Outputs: CSVs in `tables/` (including `corridor_volatility.csv` with CV of travel time), LaTeX tables and `results_macros.tex` in `paper/generated/`.

---

## Phase 4 — Figures

```bash
python 04_figures.py
```

Creates in `figures/`:

- `fig1_speed_boxplot.pdf/.png` — `Speed_kmh` by corridor (C1…Ck).
- `fig2_timeslot_lines.pdf/.png` — mean speed by time slot, one line per corridor, 95% bootstrap CIs.

Both are 3.5 in wide (IEEE single column), Times-style serif at 7–8 pt, PDFs are vector with embedded TrueType fonts (what IEEE PDF eXpress needs), PNGs at 300 DPI. For a full-width figure change `COL_W` to `7.16` and use `figure*` with `width=\textwidth`. To plot travel time instead of speed in Fig. 2, set `FIG2_Y = "Estimated_Duration_Min"` and `FIG2_YLABEL` at the top of the script.

Check fonts: `pdffonts figures/fig1_speed_boxplot.pdf` → every row `emb = yes`.

---

## Phase 5 — Draft (IEEE mapping)

`paper/main.tex` contains, in IEEE order: Abstract and Keywords; I Introduction & Objectives; II Literature Review (Dhaka; Indian metropolitan regions; Bengaluru corridor study; static-routing gap); III Methodology & Pipeline; IV Results (descriptives and volatility, assumptions, ANOVA, post-hoc, time slots); V Conclusion, Limitations, Future Scope; IEEE references.

Numbers are **not typed in by hand**. `paper/generated/results_macros.tex` defines macros such as `\AnovaF`, `\AnovaP`, `\EtaSq`, `\FastestCorr`, `\MostVolCorr`, `\SentTop` from your actual data, and the tables are `\input` files. Re-run Phase 3 and recompile and the whole paper updates.

Before submitting you must:

1. Fill in the author block (full name, university, email) in `main.tex`.
2. Read Section IV's auto-generated sentences and add your own interpretation (why the fastest corridor is fast, why the most volatile one is volatile: road class, junctions, bus lanes, markets).
3. **Verify the references.** I confirmed the Dhaka, Indian-city and OSRM sources by web search, but check bibliographic details (NBER working-paper number, World Bank report year, MoveInSync and Citizen Matters entries, VEHITS page range) against the originals. The statistics and software references are standard citations written from memory; check their volumes and pages once.
4. If you switch away from OSRM (section 0), update Section III-B and the abstract wording.

---

## Phase 6 — Overleaf and compilation

### 6.1 Build the upload zip (after Phases 3–4)

```bash
cd dhaka_traffic_pipeline
mkdir -p paper/figures && cp figures/*.pdf paper/figures/
cd paper && zip -r ../paper_overleaf.zip main.tex references.bib generated figures
```

(Windows: select `main.tex`, `references.bib`, `generated`, `figures` → right-click → *Send to → Compressed (zipped) folder*.)

### 6.2 Create the project

**Option A (fastest):** Overleaf → **New Project → Upload Project** → choose `paper_overleaf.zip`. Overleaf's TeX Live already includes `IEEEtran.cls` and `IEEEtran.bst`, so `\documentclass[conference]{IEEEtran}` compiles as is.

**Option B (official template first):** download the IEEE conference LaTeX template from IEEE Author Center (Conference Template page) → Overleaf → **New Project → Upload Project** with that zip → then drag `main.tex`, `references.bib`, `generated/` and `figures/` into the project → **Menu → Main document → main.tex**.

### 6.3 Compiler settings

**Menu (top left) → Compiler: pdfLaTeX; TeX Live version: latest offered; Main document: main.tex.** Click **Recompile**. BibTeX runs automatically; if citations show `[?]`, recompile once more.

### 6.4 Figures and equations

- Figures must live at `figures/…` exactly as referenced: `\includegraphics[width=\columnwidth]{figures/fig1_speed_boxplot.pdf}`. Use the PDFs, not the PNGs.
- Equations: `\begin{equation} … \label{eq:x} \end{equation}`, cite with `Eq.~\eqref{eq:x}`. Do not use `$$…$$` (IEEEtran spacing breaks). Statistical symbols: `$F$`, `$p$`, `$\mu_j$`, `$\sigma^2$`, `$\bar{y}_j$`.
- In text mode escape `_ % & # $` as `\_ \% \& \# \$`.

### 6.5 Common compile errors

| Message | Cause and fix |
|---|---|
| `Undefined control sequence \NCorr` (or any macro in the results file) | `generated/results_macros.tex` missing or in the wrong folder. Upload the `generated` folder keeping its name. |
| `File 'generated/…tex' not found` | Same as above. |
| `File 'figures/….pdf' not found` | Upload the figure into a `figures` folder. |
| `Missing $ inserted` | An unescaped `_` or `^` in text. Escape it or put it in math mode. |
| `Citation … undefined` / `[?]` | Recompile; check the key matches `references.bib`; open **Logs and output files** for BibTeX errors. |
| `Paragraph ended before \newcommand was complete` | Corrupted `results_macros.tex`; re-run Phase 3 and re-upload. |
| `Overfull \hbox` in a table | Add `\setlength{\tabcolsep}{2pt}` or switch the table to `table*`. |

Turn on **Menu → Stop on first error** while debugging.

### 6.6 Final checks

1. **Menu → Download → PDF.** Run `pdffonts main.pdf` locally: all fonts `emb = yes`.
2. Check the page limit and margins required by your conference; IEEE PDF eXpress may re-validate the file.
3. Download the source zip (**Menu → Download → Source**) and commit it to the GitHub repo with `data/` and `tables/` for reproducibility.

---

## Verification status of this package

I ran `01`–`04` end to end (corridor version) on synthetic data (including malformed rows, duplicate rows, repeated headers, missing values and a UTC-vs-Dhaka weekday mismatch), inspected both figures and compiled `main.tex` with the generated tables. Two limits: my sandbox had no `statsmodels`, so the calls to `pairwise_tukeyhsd`, `anova_lm` and `anova_oneway` were exercised against a small stand-in built on SciPy (Games–Howell and every other statistic ran on the real libraries), and the LaTeX test used the `article` class because `IEEEtran.cls` was not installed. If either script throws an error on your machine, send me the traceback.
