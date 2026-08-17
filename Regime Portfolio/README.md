# Regime-Aware Portfolio Optimisation

Constituent-level regime detection and portfolio construction across five global equity
indices, 2009–2023.

---

## Requirements

- Python 3.10+
- R (optional — only for the MS-GARCH stage)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the five dataset files from
[qmfin/index_data](https://github.com/qmfin/index_data) (commit `e11cd8d`) and place
them in `data/raw/`:

```
RUSSELL1000.json   STOXX600.json   STOXX1800.json   SHANGHAI_A.json   TOPIX1000.json
```

Compressed (`.json.bz2`) files are also accepted.

## Usage

Run the full pipeline:

```bash
python -m src.run_all --jobs 6
```

Runtime is roughly 35–40 minutes. Output is written to `results/`.

### Options

| Flag | Effect |
|---|---|
| `--jobs N` | Run markets in parallel across N processes |
| `--fast` | Reduced settings for a quick smoke test (a few minutes) |
| `--with-msgarch` | Include the MS-GARCH stage (requires R) |
| `--full-sensitivity` | Include the lookback, cost and rebalance sweeps |
| `--only STAGE[,STAGE]` | Run only the named stages |
| `--skip STAGE[,STAGE]` | Skip the named stages |

Stages, in order: `preprocess`, `data_summary`, `eda`, `dependence`, `regimes`,
`msgarch`, `backtest`, `inference`, `sensitivity`, `figures`, `tables`, `validate`.

### Tests

```bash
python -m pytest -q
```

### Output validation

```bash
python -m src.validate_results
```

Checks the generated files against 783 correctness invariants, including recomputing
every reported metric from the underlying return series. Exits non-zero on failure.
This also runs automatically as the final pipeline stage.

### Viewing results

```bash
jupyter notebook notebooks/01_results_and_tables.ipynb
```

---

## Project structure

```
├── config.yaml                All parameters and hyper-parameter grids
├── requirements.txt
│
├── src/
│   ├── run_all.py               Pipeline orchestration (entry point)
│   ├── config.py                Configuration loader
│   ├── data_loader.py           Parse raw JSON into return panels
│   ├── preprocessing.py         Calendar alignment, survivorship, cleaning
│   ├── data_summary.py          Dataset and survivorship summaries
│   ├── eda.py                   Distributional and stationarity tests
│   ├── dependence.py            Correlations, Epps check, DCC-GARCH
│   ├── constituent_features.py  Cross-sectional feature computation
│   ├── regimes.py               Hidden Markov regime detection
│   ├── msgarch.py               Markov-switching GARCH via R (optional)
│   ├── portfolios.py            Estimation layer, MV / MinVar / HRP
│   ├── portfolios_regime.py     CVaR and equilibrium-anchored allocation
│   ├── metrics.py               Sharpe, Sortino, Calmar, drawdown, turnover
│   ├── backtest.py              Walk-forward backtest engine
│   ├── inference.py             Block bootstrap, Deflated Sharpe, FDR control
│   ├── sensitivity.py           Parameter sweeps and crisis breakdown
│   ├── figures.py               Figure generation
│   ├── tables.py                Formatted table generation
│   ├── validate_results.py      Output correctness checks
│   └── parallel.py              Concurrency helper
│
├── tests/                     18 modules, 108 tests
├── notebooks/                 Results notebook
├── data/raw/                  Input files (not included)
├── data/processed/            Cleaned panels (generated)
└── results/
    ├── tables/                  Result CSVs
    ├── paper_tables/            Formatted tables (Markdown, LaTeX, CSV)
    ├── figures/                 Figures (PDF, PNG)
    └── run_manifest.txt         Record of the last run
```

## Configuration

All parameters live in `config.yaml` rather than in the source, so a run is fully
described by that one file. Sections cover preprocessing, regime detection, portfolio
construction, backtesting, inference, sensitivity sweeps and figure output.

Runs are deterministic: all random operations are seeded (`seed: 42`), so re-running
reproduces identical results.

## MS-GARCH setup (optional)

This stage is not required and the pipeline runs without it. To enable it, install R,
then:

```r
install.packages("MSGARCH")
```

Verify the bridge:

```bash
python -c "from rpy2.robjects.packages import importr; importr('MSGARCH'); print('OK')"
```

Then pass `--with-msgarch`. If R is unavailable the stage fails cleanly and the
remaining stages still complete.

---

## Note on the absence of an executable

No standalone executable is provided. The software is a batch analysis pipeline with no
user interfacem it reads a dataset, runs a sequence of computations and writes files,
so an executable would offer nothing beyond the single command above. It also depends on
compiled scientific Python extensions (NumPy, SciPy, cvxpy, hmmlearn) that do not bundle
reliably across platforms, and optionally on R, which is a separate runtime that cannot
be embedded.

To run the code, follow **Installation**, **Data** and **Usage** above. The full analysis
is reproduced by:

```bash
python -m src.run_all --with-msgarch --full-sensitivity --jobs 6
```

## Troubleshooting

**A market is missing from the output.** The loader prints `[skip]` for any index it
cannot find — check the filenames in `data/raw/` match those listed above.

**`ModuleNotFoundError`.** The virtual environment is not active, or
`pip install -r requirements.txt` has not been run.

**MS-GARCH stage fails.** R or the `MSGARCH` package is not installed. Omit
`--with-msgarch`; nothing else depends on it.

**Out of memory with `--jobs`.** Each worker holds its own copy of the data. Reduce to
`--jobs 2` or omit the flag.
