# QuantCrate (.qcr)

**The SQLite of Quant Finance** — A zero-config, portable, high-performance binary container for financial time-series data.

[![Tests](https://img.shields.io/badge/tests-146%20passing-success)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What is QuantCrate?

QuantCrate is a specialized file format (`.qcr`) for financial OHLCV data that bundles:
- **Smart metadata** (ticker, exchange, timezone, timescale)
- **Automatic data quality audits** (5 built-in checks)
- **Corporate action tracking** (splits, dividends)
- **SQL querying** (DuckDB-powered, no database required)

All in a single, compressed, memory-mapped binary file.

### The Problem

Traditional workflows with CSV/Parquet:
- No standardized metadata — every vendor uses different column names
- No data quality guarantees — backtests fail on "ghost" spikes or gaps
- No audit trail — you don't know if data has been validated
- No SQL access — must load into a database or DataFrame first

### The QuantCrate Solution

```python
from qcr import save_sealed_qcr, load_qcr, run_audit
from qcr.schema import QcrMetadata

# Pack data with metadata and automatic audit
metadata = QcrMetadata(ticker="AAPL", asset_class="Equity", ...)
audit = run_audit(df, timescale="1d")
save_sealed_qcr(df, metadata, audit, "AAPL.qcr")

# Load instantly (memory-mapped)
df, meta = load_qcr("AAPL.qcr")

# Query with SQL (no database needed!)
from qcr.vtable import query_qcr
result = query_qcr("SELECT * FROM 'AAPL.qcr' WHERE volume > 1000000")
```

---

## Features

| Feature | Traditional (CSV/Parquet) | QuantCrate (.qcr) |
|:--------|:--------------------------|:------------------|
| Embedded metadata | None | Ticker, exchange, timezone, etc. |
| Data quality audit | Manual | Automatic (5 checks) |
| Audit trail | No | Sealed with timestamp |
| Corporate actions | Manual scripts | Stored in metadata |
| Split/dividend adjustments | Destructive | Non-destructive, on-the-fly |
| Instant metadata peek | Must load full file | Footer-only read |
| SQL queries | Load into DB first | Direct SQL via DuckDB |
| CLI tooling | Write your own | `qcr pack/info/ingest/sql` |
| File size | ~Same | ~Same (ZSTD compressed) |
| Read speed | ~Same | ~Same (memory-mapped) |

---

## Installation

```bash
# Core library
pip install quantcrate

# With CLI tools
pip install quantcrate[cli]

# With data adapters (Yahoo Finance, etc.)
pip install quantcrate[adapters]

# With SQL querying (DuckDB)
pip install quantcrate[sql]

# Everything
pip install quantcrate[cli,adapters,sql]
```

---

## Quick Start

### 1. Pack a CSV into a .qcr file

```bash
# From the command line
qcr pack aapl.csv --ticker AAPL --timescale 1d --output AAPL.qcr

# The auditor runs automatically and blocks sealing if errors are found
```

### 2. Inspect metadata

```bash
qcr info AAPL.qcr
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Field           ┃ Value            ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ ticker          │ AAPL             │
│ asset_class     │ Equity           │
│ currency        │ USD              │
│ exchange        │ XNAS             │
│ timezone        │ America/New_York │
│ timescale       │ 1d               │
│ audit_passed    │ ✓ True           │
│ data_gaps       │ 0                │
│ outliers_found  │ 0                │
└─────────────────┴──────────────────┘
```

### 3. Load and analyze

```python
from qcr import load_qcr

# Load the full file (instant, memory-mapped)
df, metadata = load_qcr("AAPL.qcr")

print(df.head())
print(metadata.identity.ticker)  # "AAPL"
print(metadata.audit.audit_passed)  # True
```

### 4. Query with SQL

```bash
# From the command line
qcr sql "SELECT * FROM 'AAPL.qcr' WHERE close > 150" --format table

# CSV output
qcr sql "SELECT timestamp, close, volume FROM 'AAPL.qcr'" --format csv --limit 100

# Cross-file JOIN
qcr sql "SELECT a.close, b.close FROM aapl JOIN msft ON a.timestamp = b.timestamp" \
    -r AAPL.qcr=aapl -r MSFT.qcr=msft
```

Or in Python:

```python
from qcr.vtable import query_qcr, query_qcr_df

# Get results as a structured object
result = query_qcr("SELECT avg(close) FROM 'AAPL.qcr'")
print(result.rows)  # [(150.23,)]

# Get results as a Polars DataFrame (zero-copy from DuckDB)
df = query_qcr_df("SELECT * FROM 'AAPL.qcr' WHERE volume > 1000000")
```

### 5. Fetch live data

```bash
# Fetch from Yahoo Finance, audit, and seal in one command
qcr ingest AAPL --start 2024-01-01 --end 2024-12-31 --timescale 1d
```

Or in Python:

```python
from qcr.ingest import ingest_to_qcr
from qcr.adapters import YahooFinanceAdapter
import asyncio

adapter = YahooFinanceAdapter()
result = await ingest_to_qcr(
    adapter=adapter,
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
    asset_class="Equity",
    timescale="1d",
)

if result.success:
    print(f"Sealed {result.rows} rows to {result.path}")
```

### 6. Apply corporate actions

```python
from qcr import load_qcr
from qcr.adjust import adjust_ohlcv
from qcr.schema import CorporateAction, ActionType
from datetime import datetime, timezone

# Load raw data
df, metadata = load_qcr("AAPL.qcr")

# Define a 4-for-1 split on 2022-08-31
actions = [
    CorporateAction(
        action_type=ActionType.SPLIT,
        effective_date=datetime(2022, 8, 31, tzinfo=timezone.utc),
        value=4.0,
    )
]

# Apply adjustments (non-destructive, returns new DataFrame)
adjusted_df = adjust_ohlcv(df, actions)

# Pre-split prices are divided by 4, volume is multiplied by 4
# Post-split rows are unchanged
```

---

## The Auditor — 5 Automatic Checks

Every `.qcr` file is sealed only after passing these checks:

| Check | Description | Severity |
|:------|:------------|:---------|
| **Logical Consistency** | `High >= Open`, `High >= Close`, `High >= Low`, `Low <= Open`, `Low <= Close`, `Volume >= 0` | ERROR |
| **Non-Zero Price** | All OHLC prices must be > 0 | ERROR |
| **Chronological Integrity** | Timestamps must be strictly increasing with no duplicates | ERROR |
| **Outlier Detection** | Flags price moves exceeding 5 standard deviations | WARNING |
| **Gap Detection** | Identifies missing candles based on the expected timescale | WARNING |

**Errors** block sealing. **Warnings** are recorded but don't block (use `--force` to override).

---

## Architecture

### File Anatomy

A `.qcr` file is a ZSTD-compressed Parquet file built on **Apache Arrow**:

1. **The Header (Metadata):** JSON stored under the `qcr_metadata` key in the Arrow schema metadata.
2. **The Audit Trail:** Quality-check results written by the `Auditor` before sealing.
3. **The Corporate Action Table:** Stores splits/dividends for dynamic adjustments.
4. **The Payload:** Columnar OHLCV data — `float32` prices, `uint64` volume, `timestamp[ns, UTC]`.

### Tech Stack

| Dependency | Purpose |
|:-----------|:--------|
| `pyarrow` | Arrow/Parquet I/O, schema metadata, ZSTD compression, memory-mapping |
| `polars` | DataFrame manipulation and validation |
| `pydantic` | Metadata validation (enforce required fields, types, enums) |
| `typer` + `rich` | CLI framework with beautiful terminal output |
| `yfinance` (optional) | Yahoo Finance data adapter for live OHLCV fetching |
| `duckdb` (optional) | SQL query engine with native Parquet support |
| `pytest` | Testing framework (146 tests passing) |

---

## CLI Commands

| Command | Description | Example |
|:--------|:------------|:--------|
| `qcr pack` | Ingest CSV, audit, seal to .qcr | `qcr pack aapl.csv --ticker AAPL --timescale 1d` |
| `qcr info` | Print metadata from .qcr file | `qcr info AAPL.qcr` |
| `qcr ingest` | Fetch live data, audit, seal | `qcr ingest AAPL --start 2024-01-01 --end 2024-12-31` |
| `qcr sql` | Query .qcr files with SQL | `qcr sql "SELECT * FROM 'AAPL.qcr' WHERE close > 150"` |

---

## Demo

Run the comprehensive demo to see all features in action:

```bash
python demo.py
```

This interactive script compares QuantCrate with traditional workflows step-by-step, including:
1. CSV ingestion
2. Data quality auditing
3. Metadata inspection
4. Corporate action adjustments
5. SQL querying
6. CLI usage
7. File size and load-time comparisons

---

## Project Structure

```
qcr/
├── qcr/                  # Core package
│   ├── __init__.py       # Package root, __version__
│   ├── schema.py         # Pydantic models, enums, Arrow PAYLOAD_SCHEMA
│   ├── storage.py        # save_qcr / load_qcr / read_qcr_metadata
│   ├── auditor.py        # Financial linter / validation rules
│   ├── adjust.py         # Split & dividend adjustments
│   ├── adapters.py       # BaseAdapter ABC + YahooFinanceAdapter
│   ├── ingest.py         # Async fetch → audit → seal pipeline
│   ├── vtable.py         # DuckDB-powered SQL engine for .qcr files
│   └── cli.py            # CLI entry point (typer)
├── tests/                # 146 tests across 8 modules
├── portfolio/            # Interactive web showcase (open index.html)
├── pyproject.toml        # hatchling build, deps, pytest config
├── demo.py               # Interactive demo script
└── README.md             # This file
```

---

## Portfolio Showcase

An interactive web-based showcase of QuantCrate is included in the `portfolio/` directory. Open `portfolio/index.html` in any browser to explore:

- **Architecture walkthrough** — data pipeline flow and file anatomy
- **Interactive demos** — simulated terminal output for sealing, auditing, adjustments, SQL queries, and CLI commands
- **Auditor deep dive** — click any of the 5 checks to see how it works under the hood
- **OHLCV chart** — toggle between close price, volume, and high-low range views
- **Feature comparison** — side-by-side table and radar chart vs traditional CSV/Parquet
- **Encoding strategy** — why each column uses a specific Parquet encoding

No build step or dependencies required — it's a single self-contained HTML file.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## License

MIT License — see `LICENSE` file for details.

---

## Why "QuantCrate"?

Because financial data deserves better than a CSV in a ZIP file. QuantCrate is:
- **Portable** — One file, no setup, works anywhere
- **Smart** — Knows what a stock split is, validates data quality
- **Fast** — Memory-mapped, compressed, columnar storage
- **Professional** — Audit trails, metadata, corporate actions built-in

**Think of it as SQLite for quant finance** — a single-file, zero-config database optimized for time-series data.

---

**Built with ❤️ for quantitative researchers, data scientists, and financial engineers.**
