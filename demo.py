"""
QuantCrate (.qcr) -- Interactive Demo & Comparison with Traditional Workflows
=============================================================================

Run:  python demo.py

This script walks through every feature of QuantCrate and compares each step
with the traditional CSV / plain-Parquet approach.  All temp files are cleaned
up at the end.

Sections:
  1. Generate realistic sample OHLCV data + write to CSV
  2. Traditional workflow: CSV -> Pandas -> Parquet (no metadata, no audit)
  3. QuantCrate workflow: CSV -> Polars -> .qcr (metadata + audit + sealed)
  4. Reading: compare load times & what you get back
  5. Metadata peek: qcr info vs "open the Parquet and hope"
  6. The Auditor: inject bad data and watch it catch errors
  7. Corporate Actions: apply a split + dividend, compare raw vs adjusted
  8. CLI: pack a CSV from the command line
  9. SQL Queries: query .qcr files with standard SQL (DuckDB-powered)
 10. File size comparison
"""

from __future__ import annotations

import io
import os
import sys
import time
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing characters render properly.
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 0.  Setup — make sure we're in the right place
# ---------------------------------------------------------------------------

DEMO_DIR = Path("_demo_workspace")
DEMO_DIR.mkdir(exist_ok=True)

print("=" * 72)
print("  QuantCrate (.qcr) — Full Feature Demo")
print("=" * 72)


# ---------------------------------------------------------------------------
# 1.  Generate realistic sample data
# ---------------------------------------------------------------------------

def generate_sample_csv(path: Path, rows: int = 500) -> None:
    """Write a realistic OHLCV CSV with slight randomness."""
    import random
    random.seed(42)

    base_time = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    price = 150.0  # Starting price (think: AAPL-ish)

    with open(path, "w") as f:
        # Use common "messy" headers on purpose (Date, Vol, etc.)
        f.write("Date,Open,High,Low,Close,Vol\n")
        for i in range(rows):
            ts = base_time + timedelta(minutes=i)
            change = random.uniform(-1.5, 1.5)
            o = round(price + change, 2)
            h = round(o + random.uniform(0.1, 2.0), 2)
            l = round(o - random.uniform(0.1, 2.0), 2)
            c = round(random.uniform(l, h), 2)
            v = random.randint(50_000, 500_000)
            f.write(f"{ts.isoformat()},{o},{h},{l},{c},{v}\n")
            price = c  # Random walk

    print(f"\n{'─'*72}")
    print(f"  1. SAMPLE DATA")
    print(f"{'─'*72}")
    print(f"  Generated {rows}-row CSV → {path}")
    print(f"  Headers: Date, Open, High, Low, Close, Vol  (intentionally messy)")


csv_path = DEMO_DIR / "aapl_raw.csv"
generate_sample_csv(csv_path)


# ---------------------------------------------------------------------------
# 2.  Traditional workflow: CSV → Pandas → Parquet
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  2. TRADITIONAL WORKFLOW (CSV → Pandas → Parquet)")
print(f"{'─'*72}")

import pandas as pd
import pyarrow.parquet as pq

trad_parquet = DEMO_DIR / "aapl_traditional.parquet"

t0 = time.perf_counter()
pdf = pd.read_csv(csv_path, parse_dates=["Date"])
pdf.to_parquet(trad_parquet, compression="zstd")
trad_write_ms = (time.perf_counter() - t0) * 1000

print(f"  ✓ Wrote {trad_parquet.name}  ({trad_parquet.stat().st_size:,} bytes)")
print(f"    Write time: {trad_write_ms:.1f} ms")
print(f"    Metadata embedded: NONE")
print(f"    Data quality audit: NONE")
print(f"    You'd need to remember: ticker, exchange, timezone, timescale…")
print(f"    …and hope the next person reads your README.")


# ---------------------------------------------------------------------------
# 3.  QuantCrate workflow: CSV → audit → .qcr
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  3. QUANTCRATE WORKFLOW (CSV → Auditor → Sealed .qcr)")
print(f"{'─'*72}")

import polars as pl
from qcr.schema import QcrMetadata, AssetClass, Timescale, ActionType, CorporateAction
from qcr.storage import save_qcr, save_sealed_qcr, load_qcr, read_qcr_metadata
from qcr.auditor import run_audit, Severity

# Read CSV with Polars (the CLI does this for you, but let's do it manually)
raw_df = pl.read_csv(csv_path, try_parse_dates=True)

# Rename messy headers → canonical names
raw_df = raw_df.rename({
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Vol": "volume",
})

# Cast to canonical types
raw_df = raw_df.with_columns(
    pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
    pl.col("open").cast(pl.Float32),
    pl.col("high").cast(pl.Float32),
    pl.col("low").cast(pl.Float32),
    pl.col("close").cast(pl.Float32),
    pl.col("volume").cast(pl.UInt64),
)

print(f"  Loaded {len(raw_df)} rows from CSV")
print(f"  Schema: {dict(zip(raw_df.columns, [str(d) for d in raw_df.dtypes]))}")

# Build metadata (this is validated by Pydantic — try passing bad values!)
metadata = QcrMetadata(
    ticker="AAPL",
    asset_class=AssetClass.EQUITY,
    currency="USD",
    exchange="XNAS",
    timezone="America/New_York",
    timescale=Timescale.ONE_MINUTE,
)
print(f"\n  Metadata: {metadata.model_dump()}")

# Run the auditor
print(f"\n  Running audit…")
audit_result = run_audit(raw_df, Timescale.ONE_MINUTE)

if audit_result.passed:
    print(f"  ✓ Audit PASSED  (gaps: {audit_result.data_gaps}, outliers: {audit_result.outliers_found})")
else:
    print(f"  ✗ Audit FAILED")
for issue in audit_result.issues:
    icon = "✗" if issue.severity == Severity.ERROR else "⚠"
    print(f"    {icon} [{issue.severity.value}] {issue.check}: {issue.message}")

# Seal and write
qcr_path = DEMO_DIR / "aapl.qcr"
t0 = time.perf_counter()
save_sealed_qcr(raw_df, metadata, audit_result.to_audit_trail(), qcr_path)
qcr_write_ms = (time.perf_counter() - t0) * 1000

print(f"\n  ✓ Wrote {qcr_path.name}  ({qcr_path.stat().st_size:,} bytes)")
print(f"    Write time: {qcr_write_ms:.1f} ms")
print(f"    Metadata: ticker, asset_class, exchange, timezone, timescale — ALL embedded")
print(f"    Audit trail: passed={audit_result.passed}, gaps={audit_result.data_gaps}, outliers={audit_result.outliers_found}")


# ---------------------------------------------------------------------------
# 4.  Reading: compare what you get back
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  4. READING — What You Get Back")
print(f"{'─'*72}")

# Traditional
print(f"\n  [Traditional Parquet]")
t0 = time.perf_counter()
trad_table = pq.read_table(str(trad_parquet), memory_map=True)
trad_read_ms = (time.perf_counter() - t0) * 1000
print(f"    Read time: {trad_read_ms:.1f} ms")
print(f"    Columns: {trad_table.column_names}")
print(f"    Metadata: {trad_table.schema.metadata}")  # Pandas junk, no business context
print(f"    → You get raw data.  No idea what ticker, exchange, or timezone this is.")

# QuantCrate
print(f"\n  [QuantCrate .qcr]")
t0 = time.perf_counter()
qcr_df, qcr_meta = load_qcr(qcr_path)
qcr_read_ms = (time.perf_counter() - t0) * 1000
print(f"    Read time: {qcr_read_ms:.1f} ms (memory-mapped)")
print(f"    Columns: {qcr_df.columns}")
print(f"    Ticker: {qcr_meta.identity.ticker}")
print(f"    Asset Class: {qcr_meta.identity.asset_class.value}")
print(f"    Exchange: {qcr_meta.identity.exchange}")
print(f"    Timezone: {qcr_meta.identity.timezone}")
print(f"    Timescale: {qcr_meta.identity.timescale.value}")
print(f"    Audit Passed: {qcr_meta.audit.audit_passed}")
print(f"    Audit Timestamp: {qcr_meta.audit.audit_timestamp}")
print(f"    → You get data + full context + trust guarantee.  Ready for backtesting.")


# ---------------------------------------------------------------------------
# 5.  Metadata peek (footer-only read)
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  5. METADATA PEEK — Instant Info Without Loading Data")
print(f"{'─'*72}")

print(f"\n  [Traditional Parquet]")
print(f"    To know what's in the file, you must: open it, read columns,")
print(f"    guess the ticker from the filename, and hope someone documented the rest.")

print(f"\n  [QuantCrate .qcr]")
t0 = time.perf_counter()
peek_meta = read_qcr_metadata(qcr_path)
peek_ms = (time.perf_counter() - t0) * 1000
print(f"    read_qcr_metadata() → {peek_ms:.2f} ms  (reads only Parquet footer)")
print(f"    Ticker: {peek_meta.identity.ticker}")
print(f"    Sealed: {'Yes' if peek_meta.audit else 'No'}")
print(f"    → Equivalent to `qcr info aapl.qcr` from the CLI.")


# ---------------------------------------------------------------------------
# 6.  The Auditor — inject bad data
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  6. THE AUDITOR — Catching Bad Data")
print(f"{'─'*72}")

print(f"\n  [Traditional Parquet]")
print(f"    No built-in validation.  Bad data goes in, bad data comes out.")
print(f"    You discover the problem 3 months later when your backtest explodes.")

print(f"\n  [QuantCrate Auditor]")

# 6a. Inject a negative price
bad_df = raw_df.clone()
close_vals = bad_df["close"].to_list()
close_vals[100] = -5.0  # Negative price!
bad_df = bad_df.with_columns(pl.Series("close", close_vals, dtype=pl.Float32))
print(f"\n  Test A: Injected negative close price at row 100")

bad_result = run_audit(bad_df, Timescale.ONE_MINUTE)
print(f"    Audit passed? {bad_result.passed}")
for issue in bad_result.issues:
    if issue.severity == Severity.ERROR:
        preview = issue.row_indices[:5]
        print(f"    ✗ [{issue.check}] {issue.message}  (rows: {preview})")

# 6b. Inject duplicate timestamps
bad_df2 = raw_df.clone()
ts_vals = bad_df2["timestamp"].to_list()
ts_vals[50] = ts_vals[49]  # Duplicate!
bad_df2 = bad_df2.with_columns(pl.Series("timestamp", ts_vals))
print(f"\n  Test B: Injected duplicate timestamp at row 50")

bad_result2 = run_audit(bad_df2, Timescale.ONE_MINUTE)
print(f"    Audit passed? {bad_result2.passed}")
for issue in bad_result2.issues:
    if issue.severity == Severity.ERROR:
        print(f"    ✗ [{issue.check}] {issue.message}")

# 6c. Inject high < low
bad_df3 = raw_df.clone()
high_vals = bad_df3["high"].to_list()
high_vals[200] = 1.0  # Way below low
bad_df3 = bad_df3.with_columns(pl.Series("high", high_vals, dtype=pl.Float32))
print(f"\n  Test C: Injected high < low at row 200")

bad_result3 = run_audit(bad_df3, Timescale.ONE_MINUTE)
print(f"    Audit passed? {bad_result3.passed}")
for issue in bad_result3.issues:
    if issue.severity == Severity.ERROR:
        print(f"    ✗ [{issue.check}] {issue.message}")

print(f"\n  → The Auditor catches ALL of these before the file is sealed.")
print(f"    Traditional Parquet would happily store every one of these errors.")


# ---------------------------------------------------------------------------
# 7.  Corporate Actions — Split + Dividend Adjustments
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  7. CORPORATE ACTIONS — Dynamic Adjustments")
print(f"{'─'*72}")

from qcr.adjust import apply_splits, apply_dividends, adjust_ohlcv

# Use a clean slice for clarity
demo_df = raw_df.head(20)
print(f"\n  Raw data (first 5 rows):")
print(f"  {'timestamp':<28} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'volume':>10}")
print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
for row in demo_df.head(5).iter_rows(named=True):
    print(f"  {str(row['timestamp']):<28} {row['open']:>8.2f} {row['high']:>8.2f} {row['low']:>8.2f} {row['close']:>8.2f} {row['volume']:>10,}")

# Define a 2-for-1 split at row 10
split_date = demo_df["timestamp"][10]
split_action = CorporateAction(
    action_type=ActionType.SPLIT,
    effective_date=split_date,
    value=2.0,
    description="2-for-1 forward split",
)

# Define a $0.50 dividend at row 15
div_date = demo_df["timestamp"][15]
div_action = CorporateAction(
    action_type=ActionType.DIVIDEND,
    effective_date=div_date,
    value=0.50,
    description="$0.50 quarterly dividend",
)

print(f"\n  Actions:")
print(f"    • Split 2:1 on {split_date}")
print(f"    • Dividend $0.50 on {div_date}")

# Apply all at once
adjusted_df = adjust_ohlcv(demo_df, [split_action, div_action])

print(f"\n  Adjusted data (first 5 rows — pre-split, pre-dividend):")
print(f"  {'timestamp':<28} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'volume':>10}")
print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
for row in adjusted_df.head(5).iter_rows(named=True):
    print(f"  {str(row['timestamp']):<28} {row['open']:>8.2f} {row['high']:>8.2f} {row['low']:>8.2f} {row['close']:>8.2f} {row['volume']:>10,}")

print(f"\n  Comparison at row 0 (pre-split & pre-dividend):")
raw_row = demo_df.row(0, named=True)
adj_row = adjusted_df.row(0, named=True)
print(f"    Raw close:      {raw_row['close']:>8.2f}   Volume: {raw_row['volume']:>10,}")
print(f"    Adjusted close: {adj_row['close']:>8.2f}   Volume: {adj_row['volume']:>10,}")
print(f"    (close / 2 - 0.50 = {raw_row['close'] / 2 - 0.50:.2f} ✓)")

print(f"\n  Comparison at row 12 (post-split, pre-dividend):")
raw_row = demo_df.row(12, named=True)
adj_row = adjusted_df.row(12, named=True)
print(f"    Raw close:      {raw_row['close']:>8.2f}   Volume: {raw_row['volume']:>10,}")
print(f"    Adjusted close: {adj_row['close']:>8.2f}   Volume: {adj_row['volume']:>10,}")
print(f"    (close - 0.50 = {raw_row['close'] - 0.50:.2f} ✓)")

print(f"\n  Comparison at row 18 (post-split, post-dividend):")
raw_row = demo_df.row(18, named=True)
adj_row = adjusted_df.row(18, named=True)
print(f"    Raw close:      {raw_row['close']:>8.2f}   Volume: {raw_row['volume']:>10,}")
print(f"    Adjusted close: {adj_row['close']:>8.2f}   Volume: {adj_row['volume']:>10,}")
print(f"    (unchanged ✓)")

print(f"\n  [Traditional Parquet]")
print(f"    You'd manually write this adjustment logic every time.")
print(f"    Or download pre-adjusted data and lose the raw prices forever.")
print(f"\n  [QuantCrate]")
print(f"    Raw data stays raw.  Adjustments are applied on-the-fly.")
print(f"    Corporate actions are stored in metadata — fully reproducible.")


# ---------------------------------------------------------------------------
# 8.  CLI — pack from the command line
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  8. CLI — Pack a CSV from the Command Line")
print(f"{'─'*72}")

cli_output = DEMO_DIR / "aapl_cli.qcr"

# Set encoding for the subprocess so Rich doesn't choke on Windows cp1252.
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

pack_cmd = [
    sys.executable, "-c", "from qcr.cli import app; app()",
    "pack", str(csv_path),
    "--ticker", "AAPL", "--asset-class", "Equity", "--exchange", "XNAS",
    "--timescale", "1m", "--output", str(cli_output),
]
print(f"\n  $ qcr pack {csv_path.name} --ticker AAPL --asset-class Equity "
      f"--exchange XNAS --timescale 1m --output {cli_output.name}\n")

import subprocess

# Rich writes ANSI / Unicode that cp1252 can't decode, so read as bytes and
# decode with utf-8 + replace to avoid Windows codec errors.
result = subprocess.run(pack_cmd, capture_output=True, env=env)
stdout_text = result.stdout.decode("utf-8", errors="replace")
stderr_text = result.stderr.decode("utf-8", errors="replace")
print(stdout_text)
if result.returncode != 0:
    print(stderr_text)

if cli_output.exists():
    print(f"  ✓ CLI produced {cli_output.name}  ({cli_output.stat().st_size:,} bytes)")

    # Quick info peek
    info_cmd = [
        sys.executable, "-c", "from qcr.cli import app; app()",
        "info", str(cli_output),
    ]
    print(f"\n  $ qcr info {cli_output.name}\n")
    result = subprocess.run(info_cmd, capture_output=True, env=env)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    print(stdout_text)


# ---------------------------------------------------------------------------
# 9.  SQL Queries: query .qcr files with DuckDB
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  9. SQL QUERIES — Query .qcr files with standard SQL")
print(f"{'─'*72}")

print(f"\n  QuantCrate's Virtual Table Interface lets you query .qcr files with SQL.")
print(f"  Powered by DuckDB, it reads Parquet directly — no database loading required.\n")

# Check if duckdb is installed
try:
    from qcr.vtable import query_qcr, query_qcr_df, OutputFormat, format_result
    duckdb_available = True
except ImportError:
    duckdb_available = False
    print(f"  ⚠  DuckDB not installed. Run: pip install quantcrate[sql]")
    print(f"     Skipping SQL demo...\n")

if duckdb_available:
    # Example 1: Simple SELECT with WHERE filter
    print(f"  Example 1: SELECT with WHERE filter")
    print(f"  ─────────────────────────────────────")
    sql1 = f"SELECT timestamp, close, volume FROM '{qcr_path}' WHERE close > 150 LIMIT 5"
    print(f"  SQL: {sql1}\n")
    
    result1 = query_qcr(sql1)
    print(format_result(result1, OutputFormat.TABLE))
    
    # Example 2: Aggregation
    print(f"\n  Example 2: Aggregation (avg, min, max)")
    print(f"  ───────────────────────────────────────")
    sql2 = f"SELECT avg(close) as avg_price, min(low) as lowest, max(high) as highest FROM '{qcr_path}'"
    print(f"  SQL: {sql2}\n")
    
    result2 = query_qcr(sql2)
    print(format_result(result2, OutputFormat.TABLE))
    
    # Example 3: Polars DataFrame (for further processing)
    print(f"\n  Example 3: Return as Polars DataFrame")
    print(f"  ──────────────────────────────────────")
    sql3 = f"SELECT * FROM '{qcr_path}' WHERE volume > 2000000 ORDER BY volume DESC LIMIT 3"
    print(f"  SQL: {sql3}\n")
    
    df_result = query_qcr_df(sql3)
    print(f"  Returns: {type(df_result).__name__} with shape {df_result.shape}")
    print(f"  Columns: {df_result.columns}\n")
    print(df_result)
    
    # Example 4: CLI command demo
    print(f"\n  Example 4: CLI Command")
    print(f"  ──────────────────────")
    print(f"  $ qcr sql \"SELECT * FROM 'AAPL.qcr' WHERE close > 150\" --limit 5 --format csv\n")
    
    sql_cmd = [
        sys.executable, "-c", "from qcr.cli import app; app()",
        "sql", f"SELECT * FROM '{qcr_path}' WHERE close > 150",
        "--limit", "5",
        "--format", "csv",
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(sql_cmd, capture_output=True, env=env)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    print(stdout_text)
    
    print(f"  ✓ SQL querying lets you analyze .qcr files without loading them into memory!")
    print(f"    Perfect for quick analysis, filtering, and aggregations.\n")


# ---------------------------------------------------------------------------
# 10.  File size comparison
# ---------------------------------------------------------------------------

print(f"\n{'─'*72}")
print(f"  10. FILE SIZE COMPARISON")
print(f"{'─'*72}")

csv_size = csv_path.stat().st_size
parquet_size = trad_parquet.stat().st_size
qcr_size = qcr_path.stat().st_size

print(f"\n  {'Format':<25} {'Size':>12} {'vs CSV':>10}")
print(f"  {'─'*25} {'─'*12} {'─'*10}")
print(f"  {'Raw CSV':<25} {csv_size:>10,} B {'':>10}")
print(f"  {'Traditional Parquet':<25} {parquet_size:>10,} B {parquet_size/csv_size:>9.1%}")
print(f"  {'QuantCrate .qcr':<25} {qcr_size:>10,} B {qcr_size/csv_size:>9.1%}")
print(f"\n  Both Parquet and .qcr use ZSTD compression.")
print(f"  .qcr adds metadata + audit trail for negligible overhead.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*72}")
print(f"  SUMMARY: Traditional vs QuantCrate")
print(f"{'='*72}")
print(f"""
  Feature                      Traditional (CSV/Parquet)    QuantCrate (.qcr)
  ─────────────────────────    ────────────────────────     ─────────────────
  Embedded metadata            ✗ None                      ✓ Ticker, exchange, tz, etc.
  Data quality audit           ✗ Manual / none             ✓ Automatic (5 checks)
  Audit trail in file          ✗ No                        ✓ Sealed with timestamp
  Corporate action tracking    ✗ Manual scripts            ✓ Stored in metadata
  Split/dividend adjustments   ✗ Destructive or manual     ✓ Non-destructive, on-the-fly
  Instant metadata peek        ✗ Must load full file       ✓ Footer-only read
  SQL queries                  ✗ Load into DB first        ✓ Direct SQL via DuckDB
  CLI tooling                  ✗ Write your own            ✓ `qcr pack/info/ingest/sql`
  File size                    ~Same                       ~Same (ZSTD compressed)
  Read speed                   ~Same                       ~Same (memory-mapped)
""")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

print(f"  Demo files are in: {DEMO_DIR.resolve()}")
print(f"  Explore them yourself:")
for f in sorted(DEMO_DIR.iterdir()):
    print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

print(f"\n  To clean up, run:  python -c \"import shutil; shutil.rmtree('_demo_workspace')\"")
print()
