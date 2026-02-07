"""QuantCrate CLI — the command-line interface for .qcr files.

Commands:
    qcr pack    — Ingest a CSV, audit it, and seal it into a .qcr file.
    qcr info    — Print metadata from a .qcr file without loading data.
    qcr ingest  — Fetch data from a live source (e.g. Yahoo Finance) and seal.
    qcr sql     — Query .qcr files with SQL (powered by DuckDB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="qcr",
    help="QuantCrate — the SQLite of Quant Finance.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# qcr pack
# ---------------------------------------------------------------------------

@app.command()
def pack(
    source: Path = typer.Argument(
        ...,
        help="Path to a CSV file to ingest.",
        exists=True,
        readable=True,
    ),
    ticker: str = typer.Option(
        ...,
        "--ticker", "-t",
        help="Asset ticker symbol (e.g. AAPL).",
    ),
    asset_class: str = typer.Option(
        ...,
        "--asset-class", "-a",
        help="Asset class: Equity, Crypto, FX, Futures, Options, Index.",
    ),
    currency: str = typer.Option(
        "USD",
        "--currency", "-c",
        help="ISO 4217 currency code (e.g. USD, EUR).",
    ),
    exchange: str = typer.Option(
        ...,
        "--exchange", "-e",
        help="Exchange MIC (e.g. XNAS, BINANCE).",
    ),
    timezone_str: str = typer.Option(
        "UTC",
        "--timezone", "-z",
        help="IANA timezone (e.g. America/New_York).",
    ),
    timescale: str = typer.Option(
        ...,
        "--timescale", "-s",
        help="Candle interval: tick, 1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output .qcr path. Defaults to <ticker>.qcr in the current directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Seal the file even if the audit has warnings (gaps/outliers).",
    ),
) -> None:
    """Ingest a CSV, run the financial auditor, and seal it into a .qcr file."""
    import polars as pl

    from qcr.auditor import AuditResult, Severity, run_audit
    from qcr.schema import AssetClass, QcrMetadata, Timescale
    from qcr.storage import save_qcr

    # --- 1. Parse enum values ---------------------------------------------------
    try:
        asset_cls = AssetClass(asset_class)
    except ValueError:
        allowed = ", ".join(e.value for e in AssetClass)
        err_console.print(f"[red]Error:[/red] Invalid asset class '{asset_class}'. Allowed: {allowed}")
        raise typer.Exit(code=1)

    try:
        ts = Timescale(timescale)
    except ValueError:
        allowed = ", ".join(e.value for e in Timescale)
        err_console.print(f"[red]Error:[/red] Invalid timescale '{timescale}'. Allowed: {allowed}")
        raise typer.Exit(code=1)

    # --- 2. Build metadata ------------------------------------------------------
    try:
        metadata = QcrMetadata(
            ticker=ticker,
            asset_class=asset_cls,
            currency=currency,
            exchange=exchange,
            timezone=timezone_str,
            timescale=ts,
        )
    except Exception as exc:
        err_console.print(f"[red]Metadata validation error:[/red] {exc}")
        raise typer.Exit(code=1)

    # --- 3. Read CSV ------------------------------------------------------------
    console.print(f"[bold]Reading[/bold] {source.name} …")
    try:
        df = _read_csv(source)
    except Exception as exc:
        err_console.print(f"[red]Failed to read CSV:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"  → {df.shape[0]:,} rows, {df.shape[1]} columns")

    # --- 4. Audit ---------------------------------------------------------------
    console.print("[bold]Auditing[/bold] …")
    result = run_audit(df, ts)

    _print_audit_result(result)

    if not result.passed:
        err_console.print("\n[red bold]✗ Audit FAILED[/red bold] — file not sealed.")
        err_console.print("  Fix the errors above and try again.")
        raise typer.Exit(code=1)

    # --- 5. Seal & write --------------------------------------------------------
    out_path = output or Path(f"{ticker}.qcr")
    console.print(f"\n[bold]Sealing[/bold] → {out_path}")

    from qcr.storage import save_sealed_qcr

    written = save_sealed_qcr(df, metadata, result.to_audit_trail(), out_path)

    console.print(
        Panel(
            f"[green bold]✓ Sealed[/green bold]  {written}\n"
            f"  Rows: {df.shape[0]:,}   Gaps: {result.data_gaps}   Outliers: {result.outliers_found}",
            title="QuantCrate",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# qcr info
# ---------------------------------------------------------------------------

@app.command()
def info(
    path: Path = typer.Argument(
        ...,
        help="Path to a .qcr file.",
        exists=True,
        readable=True,
    ),
) -> None:
    """Print metadata from a .qcr file without loading the full dataset."""
    from qcr.storage import read_encoding_stats, read_qcr_metadata

    try:
        meta = read_qcr_metadata(path)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    # --- Identity table ---------------------------------------------------------
    identity = meta.identity
    id_table = Table(title="Identity", show_header=True, header_style="bold cyan")
    id_table.add_column("Field", style="bold")
    id_table.add_column("Value")

    id_table.add_row("Ticker", identity.ticker)
    id_table.add_row("Asset Class", identity.asset_class.value)
    id_table.add_row("Currency", identity.currency)
    id_table.add_row("Exchange", identity.exchange)
    id_table.add_row("Timezone", identity.timezone)
    id_table.add_row("Timescale", identity.timescale.value)
    id_table.add_row("Adjusted", "Yes" if identity.is_adjusted else "No")
    id_table.add_row("Schema Version", str(identity.version))

    console.print()
    console.print(id_table)

    # --- Audit trail (if present) -----------------------------------------------
    if meta.audit is not None:
        audit = meta.audit
        a_table = Table(title="Audit Trail", show_header=True, header_style="bold cyan")
        a_table.add_column("Field", style="bold")
        a_table.add_column("Value")

        status = "[green]PASSED[/green]" if audit.audit_passed else "[red]FAILED[/red]"
        a_table.add_row("Status", status)
        a_table.add_row("Timestamp", audit.audit_timestamp)
        a_table.add_row("Data Gaps", str(audit.data_gaps))
        a_table.add_row("Outliers", str(audit.outliers_found))
        a_table.add_row("Auditor Version", audit.auditor_version)

        console.print()
        console.print(a_table)
    else:
        console.print("\n[yellow]⚠ No audit trail[/yellow] — this file has not been sealed by the Auditor.")

    # --- Encoding & compression stats -------------------------------------------
    try:
        stats = read_encoding_stats(path)
    except Exception:
        stats = None

    if stats:
        def _fmt_bytes(n: int) -> str:
            """Format bytes as a human-readable string."""
            if n >= 1_048_576:
                return f"{n / 1_048_576:.1f} MB"
            if n >= 1_024:
                return f"{n / 1_024:.1f} KB"
            return f"{n:,} B"

        # File-level summary
        console.print()
        summary_table = Table(
            title="Storage",
            show_header=True,
            header_style="bold cyan",
        )
        summary_table.add_column("Field", style="bold")
        summary_table.add_column("Value")
        summary_table.add_row("File Size", _fmt_bytes(stats["file_size_bytes"]))
        summary_table.add_row("Rows", f"{stats['num_rows']:,}")
        summary_table.add_row("Row Groups", str(stats["num_row_groups"]))
        summary_table.add_row("Compression", stats["compression"])
        console.print(summary_table)

        # Per-column encoding table
        enc_table = Table(
            title="Column Encodings",
            show_header=True,
            header_style="bold cyan",
        )
        enc_table.add_column("Column", style="bold")
        enc_table.add_column("Physical Type")
        enc_table.add_column("Encodings")
        enc_table.add_column("Compressed")
        enc_table.add_column("Uncompressed")
        enc_table.add_column("Ratio")

        for col in stats["columns"]:
            compressed = col["total_compressed_bytes"]
            uncompressed = col["total_uncompressed_bytes"]
            ratio = (
                f"{uncompressed / compressed:.1f}x"
                if compressed > 0
                else "N/A"
            )
            enc_table.add_row(
                col["name"],
                col["physical_type"],
                ", ".join(col["encodings"]),
                _fmt_bytes(compressed),
                _fmt_bytes(uncompressed),
                ratio,
            )

        console.print()
        console.print(enc_table)

    console.print()


# ---------------------------------------------------------------------------
# qcr ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    ticker: str = typer.Argument(
        ...,
        help="Asset ticker symbol (e.g. AAPL, BTC-USD).",
    ),
    start: str = typer.Option(
        ...,
        "--start", "-s",
        help="Start date (YYYY-MM-DD).",
    ),
    end: str = typer.Option(
        ...,
        "--end", "-e",
        help="End date (YYYY-MM-DD).",
    ),
    asset_class: str = typer.Option(
        "Equity",
        "--asset-class", "-a",
        help="Asset class: Equity, Crypto, FX, Futures, Options, Index.",
    ),
    currency: str = typer.Option(
        "USD",
        "--currency", "-c",
        help="ISO 4217 currency code (e.g. USD, EUR).",
    ),
    exchange: str = typer.Option(
        "XNAS",
        "--exchange", "-x",
        help="Exchange MIC (e.g. XNAS, BINANCE).",
    ),
    timezone_str: str = typer.Option(
        "UTC",
        "--timezone", "-z",
        help="IANA timezone (e.g. America/New_York).",
    ),
    timescale: str = typer.Option(
        "1d",
        "--timescale", "-t",
        help="Candle interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo.",
    ),
    source: str = typer.Option(
        "yahoo",
        "--source",
        help="Data source adapter (currently: yahoo).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output .qcr path. Defaults to <ticker>.qcr in the current directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Seal the file even if the audit has warnings (gaps/outliers).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume", "-R",
        help="Resume from an existing .qcr file — only download data after the last timestamp.",
    ),
) -> None:
    """Fetch OHLCV data from a live source, audit it, and seal into a .qcr file.

    \b
    With --resume, if the output file already exists, only the missing
    data (after the last timestamp in the file) is downloaded and merged.
    This makes incremental updates efficient and resilient to partial failures.
    """
    import asyncio

    from qcr.adapters import AdapterError, YahooFinanceAdapter
    from qcr.ingest import ingest_to_qcr
    from qcr.schema import AssetClass, Timescale

    # --- 1. Parse enum values -----------------------------------------------
    try:
        asset_cls = AssetClass(asset_class)
    except ValueError:
        allowed = ", ".join(e.value for e in AssetClass)
        err_console.print(f"[red]Error:[/red] Invalid asset class '{asset_class}'. Allowed: {allowed}")
        raise typer.Exit(code=1)

    try:
        ts = Timescale(timescale)
    except ValueError:
        allowed = ", ".join(e.value for e in Timescale)
        err_console.print(f"[red]Error:[/red] Invalid timescale '{timescale}'. Allowed: {allowed}")
        raise typer.Exit(code=1)

    # --- 2. Select adapter --------------------------------------------------
    if source.lower() == "yahoo":
        adapter = YahooFinanceAdapter()
    else:
        err_console.print(f"[red]Error:[/red] Unknown data source '{source}'. Available: yahoo")
        raise typer.Exit(code=1)

    # --- 3. Run the async ingest pipeline -----------------------------------
    out_path = output or Path(f"{ticker}.qcr")

    if resume and out_path.exists():
        console.print(
            f"[bold]Resuming[/bold] {ticker} from existing {out_path.name} ..."
        )
    else:
        console.print(
            f"[bold]Fetching[/bold] {ticker} from {adapter.source_name} "
            f"({start} to {end}, interval={ts.value}) ..."
        )

    result = asyncio.run(
        ingest_to_qcr(
            adapter=adapter,
            ticker=ticker,
            start=start,
            end=end,
            asset_class=asset_cls,
            currency=currency,
            exchange=exchange,
            timezone=timezone_str,
            timescale=ts,
            output=out_path,
            force=force,
            resume=resume,
        )
    )

    # --- 4. Report results --------------------------------------------------
    if result.audit:
        _print_audit_result(result.audit)

    if not result.success:
        err_console.print(f"\n[red bold]x Ingest FAILED[/red bold] -- {result.error}")
        raise typer.Exit(code=1)

    # Build a descriptive summary line.
    resume_info = ""
    if result.resumed:
        if result.new_rows == 0:
            resume_info = "  [cyan](already up to date — no new data)[/cyan]\n"
        else:
            resume_info = f"  [cyan](resumed — {result.new_rows:,} new rows merged)[/cyan]\n"

    console.print(
        Panel(
            f"[green bold]Sealed[/green bold]  {result.path}\n"
            f"{resume_info}"
            f"  Source: {adapter.source_name}   Rows: {result.rows:,}   "
            f"Gaps: {result.audit.data_gaps if result.audit else 0}   "
            f"Outliers: {result.audit.outliers_found if result.audit else 0}",
            title="QuantCrate",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# qcr sql
# ---------------------------------------------------------------------------

@app.command()
def sql(
    query: str = typer.Argument(
        ...,
        help="SQL query.  Reference .qcr files by path in the FROM clause, e.g. "
             "\"SELECT * FROM 'data.qcr' WHERE volume > 1000000\".",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, csv, json.",
    ),
    register: Optional[list[str]] = typer.Option(
        None,
        "--register", "-r",
        help="Register a .qcr file as a named table: FILE=NAME  (repeatable). "
             "e.g. -r AAPL.qcr=aapl  then query with 'SELECT * FROM aapl'.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit", "-l",
        help="Append LIMIT N to the query (convenience shortcut).",
    ),
    describe: Optional[Path] = typer.Option(
        None,
        "--describe", "-d",
        help="Describe a .qcr file (schema + metadata + row count) instead of running a query.",
    ),
) -> None:
    """Query .qcr files with SQL — powered by DuckDB.

    \b
    Examples:
      qcr sql "SELECT * FROM 'AAPL.qcr' WHERE volume > 1000000"
      qcr sql "SELECT avg(close) FROM 'AAPL.qcr'" --format json
      qcr sql "SELECT * FROM aapl LIMIT 5" -r AAPL.qcr=aapl
      qcr sql --describe AAPL.qcr ""
    """
    from qcr.vtable import (
        OutputFormat,
        VTableError,
        describe_qcr,
        execute_sql as vtable_execute,
        format_result,
        query_qcr,
        register_qcr,
    )

    # --- Describe mode -------------------------------------------------------
    if describe is not None:
        try:
            info_dict = describe_qcr(describe)
        except VTableError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

        _print_describe(info_dict)
        return

    # --- Parse output format --------------------------------------------------
    try:
        out_fmt = OutputFormat(format.lower())
    except ValueError:
        allowed = ", ".join(e.value for e in OutputFormat)
        err_console.print(
            f"[red]Error:[/red] Invalid format '{format}'. Allowed: {allowed}"
        )
        raise typer.Exit(code=1)

    # --- Append LIMIT if requested -------------------------------------------
    effective_query = query.rstrip().rstrip(";")
    if limit is not None:
        effective_query += f" LIMIT {limit}"

    # --- Register named tables (if any) --------------------------------------
    if register:
        try:
            con = None
            for mapping in register:
                if "=" not in mapping:
                    err_console.print(
                        f"[red]Error:[/red] --register expects FILE=NAME, got '{mapping}'"
                    )
                    raise typer.Exit(code=1)
                file_part, name_part = mapping.rsplit("=", 1)
                con = register_qcr(
                    file_part.strip(),
                    table_name=name_part.strip(),
                    connection=con,
                )

            result = vtable_execute(con, effective_query)
        except VTableError as exc:
            err_console.print(f"[red]SQL Error:[/red] {exc}")
            raise typer.Exit(code=1)
    else:
        # --- One-shot query (file paths in FROM clause) ----------------------
        try:
            result = query_qcr(effective_query)
        except VTableError as exc:
            err_console.print(f"[red]SQL Error:[/red] {exc}")
            raise typer.Exit(code=1)

    # --- Output ---------------------------------------------------------------
    output_str = format_result(result, out_fmt)
    console.print(output_str)


def _print_describe(info: dict) -> None:
    """Pretty-print the describe_qcr output."""
    # File info
    console.print(f"\n[bold]File:[/bold] {info['file']}")
    console.print(f"[bold]Rows:[/bold] {info['row_count']:,}")

    # Column schema
    col_table = Table(title="Columns", show_header=True, header_style="bold cyan")
    col_table.add_column("Name", style="bold")
    col_table.add_column("Type")
    for name, dtype in info["columns"].items():
        col_table.add_row(name, dtype)
    console.print()
    console.print(col_table)

    # QCR metadata
    meta = info.get("metadata", {})
    if meta:
        identity = meta.get("identity", {})
        if identity:
            id_table = Table(title="QCR Metadata", show_header=True, header_style="bold cyan")
            id_table.add_column("Field", style="bold")
            id_table.add_column("Value")
            for key, val in identity.items():
                id_table.add_row(key, str(val))
            console.print()
            console.print(id_table)

        audit = meta.get("audit")
        if audit:
            a_table = Table(title="Audit Trail", show_header=True, header_style="bold cyan")
            a_table.add_column("Field", style="bold")
            a_table.add_column("Value")
            for key, val in audit.items():
                a_table.add_row(key, str(val))
            console.print()
            console.print(a_table)

    console.print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> "pl.DataFrame":
    """Read a CSV into a Polars DataFrame with the canonical column names.

    Handles common header variations (e.g. "Date" → "timestamp",
    "Adj Close" → ignored, case-insensitive matching).
    """
    import polars as pl

    df = pl.read_csv(path, try_parse_dates=True)

    # Normalise column names: lowercase + strip whitespace.
    rename_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(rename_map)

    # Common CSV aliases → canonical names.
    _ALIASES: dict[str, str] = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "vol": "volume",
    }
    rename_aliases = {
        old: new for old, new in _ALIASES.items() if old in df.columns and new not in df.columns
    }
    if rename_aliases:
        df = df.rename(rename_aliases)

    # Ensure required columns exist.
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Keep only the required columns (drop extras like "adj close").
    df = df.select(sorted(required))

    # Cast types to match PAYLOAD_SCHEMA.
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.UInt64),
    )

    # Sort by timestamp (CSVs are not always ordered).
    df = df.sort("timestamp")

    return df


def _print_audit_result(result: "AuditResult") -> None:
    """Pretty-print audit issues to the console."""
    from qcr.auditor import Severity

    if not result.issues:
        console.print("  [green]✓ All checks passed[/green]")
        return

    for issue in result.issues:
        if issue.severity == Severity.ERROR:
            icon = "[red]✗[/red]"
            style = "red"
        else:
            icon = "[yellow]⚠[/yellow]"
            style = "yellow"

        row_info = ""
        if issue.row_indices:
            preview = issue.row_indices[:5]
            suffix = f" … (+{len(issue.row_indices) - 5} more)" if len(issue.row_indices) > 5 else ""
            row_info = f"  rows: {preview}{suffix}"

        console.print(f"  {icon} [{style}]{issue.check}[/{style}]: {issue.message}{row_info}")
