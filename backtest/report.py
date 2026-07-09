"""
backtest/report.py
------------------
Render the evaluation report: charts → a self-contained HTML file (base64-embedded
PNGs) plus a multi-page PDF. Matplotlib is imported lazily and uses the headless Agg
backend so this works in CI / containers with no display.

Charts: equity vs benchmarks, underwater drawdown, trade-PnL distribution,
monthly-returns heatmap, confusion matrix, calibration curve.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any


def _fig_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_figures(payload: dict[str, Any]) -> dict[str, Any]:
    """Build all matplotlib figures. Returns {name: figure}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figs: dict[str, Any] = {}
    strat_curve = payload["strategy"]["curve"]
    bh_curve = payload["buy_hold"]["curve"]
    ts = [p.timestamp for p in strat_curve]
    strat_eq = [p.equity for p in strat_curve]

    # 1) Equity vs benchmarks (normalised to 100).
    fig, ax = plt.subplots(figsize=(9, 4))
    base = strat_eq[0] if strat_eq else 1.0
    ax.plot(
        ts, [e / base * 100 for e in strat_eq], label="Baseline (rules)", color="#2563eb", lw=1.6
    )
    improved = payload.get("improved")
    if improved and improved.get("curve"):
        ic = improved["curve"]
        ib = ic[0].equity if ic else 1.0
        ax.plot(
            [p.timestamp for p in ic],
            [p.equity / ib * 100 for p in ic],
            label="Improved attempt (walk-forward ML)",
            color="#16a34a",
            lw=1.5,
        )
    if bh_curve:
        bh_base = bh_curve[0].equity
        ax.plot(
            [p.timestamp for p in bh_curve],
            [p.equity / bh_base * 100 for p in bh_curve],
            label="Buy & Hold",
            color="#9ca3af",
            lw=1.4,
        )
    ax.axhline(100, color="#d1d5db", lw=0.8, ls="--")
    ax.set_title("Equity curve vs. benchmark (normalised to 100)")
    ax.set_ylabel("Index")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    figs["equity"] = fig

    # 2) Underwater drawdown.
    fig, ax = plt.subplots(figsize=(9, 2.6))
    peak = -1e18
    dd = []
    for e in strat_eq:
        peak = max(peak, e)
        dd.append((e - peak) / peak * 100 if peak > 0 else 0.0)
    ax.fill_between(ts, dd, 0, color="#ef4444", alpha=0.4)
    ax.set_title("Underwater drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.25)
    figs["drawdown"] = fig

    # 3) Trade-PnL distribution.
    pnls = [t.pnl_usd for t in payload["strategy"]["trades"]]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    if pnls:
        ax.hist(pnls, bins=min(30, max(5, len(pnls))), color="#6366f1", alpha=0.8)
        ax.axvline(0, color="#111827", lw=1)
    ax.set_title("Trade PnL distribution ($)")
    ax.set_xlabel("PnL per trade ($)")
    ax.grid(alpha=0.25)
    figs["pnl_hist"] = fig

    # 4) Monthly-returns heatmap.
    monthly = payload["strategy"]["metrics"].get("monthly_returns", {})
    if monthly:
        years = sorted({m.split("-")[0] for m in monthly})
        grid = np.full((len(years), 12), np.nan)
        yi = {y: i for i, y in enumerate(years)}
        for ym, val in monthly.items():
            y, mo = ym.split("-")
            grid[yi[y], int(mo) - 1] = val
        fig, ax = plt.subplots(figsize=(8, 1.0 + 0.5 * len(years)))
        vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        for r in range(len(years)):
            for cc in range(12):
                if np.isfinite(grid[r, cc]):
                    ax.text(cc, r, f"{grid[r, cc]:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title("Monthly returns (%)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        figs["monthly"] = fig

    # 5) Confusion matrix + 6) calibration (ML).
    ml = payload.get("ml") or {}
    if ml.get("available"):
        cm = np.array(ml["confusion_matrix"]["matrix"])
        labels = ml["confusion_matrix"]["labels"]
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for r in range(cm.shape[0]):
            for cc in range(cm.shape[1]):
                ax.text(
                    cc,
                    r,
                    int(cm[r, cc]),
                    ha="center",
                    va="center",
                    color="white" if cm[r, cc] > cm.max() / 2 else "black",
                    fontsize=9,
                )
        ax.set_title(f"Confusion matrix (acc {ml['accuracy']:.3f})")
        figs["confusion"] = fig

        cal = ml.get("calibration", {})
        if cal.get("mean_predicted_confidence"):
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", label="perfect")
            ax.plot(
                cal["mean_predicted_confidence"],
                cal["observed_accuracy"],
                marker="o",
                color="#2563eb",
                label="model",
            )
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Observed accuracy")
            ax.set_title("Calibration")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            figs["calibration"] = fig

    return figs


def _metric_rows(m: dict) -> str:
    keys = [
        ("total_return_pct", "Total return %"),
        ("cagr_pct", "CAGR %"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("calmar", "Calmar"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("max_drawdown_duration_bars", "Max DD duration (bars)"),
        ("num_trades", "# trades"),
        ("win_rate_pct", "Win rate %"),
        ("profit_factor", "Profit factor"),
        ("expectancy_usd", "Expectancy $"),
        ("avg_win_usd", "Avg win $"),
        ("avg_loss_usd", "Avg loss $"),
        ("exposure_pct", "Exposure %"),
    ]
    return "".join(f"<tr><td>{label}</td><td>{m.get(k)}</td></tr>" for k, label in keys)


def generate_report(results_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    """Write EVALUATION_REPORT.html + .pdf into results_dir. Returns the paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figs = _build_figures(payload)

    cfg = payload["config"]
    strat = payload["strategy"]["metrics"]
    bh = payload["buy_hold"]["metrics"]
    rnd = payload.get("random") or {}
    ml = payload.get("ml") or {}
    improved = payload.get("improved")

    imgs = {name: _fig_to_base64(fig) for name, fig in figs.items()}

    def img(name: str, alt: str) -> str:
        if name not in imgs:
            return ""
        return f'<img alt="{alt}" src="data:image/png;base64,{imgs[name]}" />'

    ml_section = "<p><em>ML evaluation unavailable for this dataset.</em></p>"
    if ml.get("available"):
        pc = ml["per_class"]
        pc_rows = "".join(
            f"<tr><td>{c}</td><td>{pc[c]['precision']}</td><td>{pc[c]['recall']}</td>"
            f"<td>{pc[c]['f1']}</td><td>{pc[c]['support']}</td></tr>"
            for c in ml["confusion_matrix"]["labels"]
        )
        ml_section = f"""
        <p>Out-of-sample (chronological 80/20 split, no shuffle). Accuracy
        <b>{ml["accuracy"]:.4f}</b> vs. random baseline <b>{ml["random_baseline_accuracy"]}</b>
        on a 3-class problem (train {ml["train_size"]}, test {ml["test_size"]}).</p>
        <table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
        {pc_rows}</table>
        <div class="grid">{img("confusion", "confusion matrix")}{img("calibration", "calibration")}</div>
        <p class="muted">Top features by gain: {", ".join(f["feature"] for f in ml["top_features"][:6])}.</p>
        """

    rnd_row = ""
    if rnd:
        rnd_row = (
            f"<tr><td>Random entry (mean of {rnd.get('runs')})</td>"
            f"<td>{rnd.get('mean_total_return_pct')}</td><td>{rnd.get('mean_sharpe')}</td>"
            f"<td>{rnd.get('mean_win_rate_pct')}</td></tr>"
        )

    # Phase 5 before/after section.
    bna_section = ""
    if improved:
        im = improved["metrics"]
        vb = improved["vs_baseline"]
        verdict = (
            "✅ The walk-forward ML strategy <b>beat</b> the baseline out-of-sample — adopted."
            if improved["beats_baseline"]
            else "❌ The walk-forward ML strategy <b>did not beat</b> the rule baseline "
            "out-of-sample. We keep the simpler baseline (an honest, documented negative result)."
        )
        bna_section = f"""
<h2>Phase 5 — Before vs. after (retrain &amp; improve)</h2>
<p>We tried to improve the system by enabling the ML signal path with a <b>walk-forward</b>
methodology (retrain on past bars only, predict forward; isotonic-calibrated), so it is
evaluated out-of-sample with no in-sample bias.</p>
<table>
  <tr><th>Variant</th><th>Total return %</th><th>Sharpe</th><th>Win rate %</th><th># trades</th><th>Max DD %</th></tr>
  <tr><td>Baseline — rules (kept)</td><td>{strat["total_return_pct"]}</td><td>{strat["sharpe"]}</td>
      <td>{strat["win_rate_pct"]}</td><td>{strat["num_trades"]}</td><td>{strat["max_drawdown_pct"]}</td></tr>
  <tr><td>Improved attempt — walk-forward ML</td><td>{im["total_return_pct"]}</td><td>{im["sharpe"]}</td>
      <td>{im["win_rate_pct"]}</td><td>{im["num_trades"]}</td><td>{im["max_drawdown_pct"]}</td></tr>
</table>
<p class="muted">Δ return {vb["return_delta_pct"]} pp · Δ Sharpe {vb["sharpe_delta"]} ·
retrains {improved["walkforward_params"]["retrains"]}.</p>
<div class="warn">{verdict}</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 980px;
         margin: 2rem auto; padding: 0 1rem; color: #111827; line-height: 1.5; }}
  h1 {{ margin-bottom: 0.2rem; }} .muted {{ color: #6b7280; }}
  table {{ border-collapse: collapse; margin: 0.6rem 0; width: 100%; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 5px 9px; text-align: left; font-size: 14px; }}
  th {{ background: #f9fafb; }}
  img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; margin: 6px 0; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }} .grid img {{ max-width: 48%; }}
  .warn {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; }}
  .disc {{ background: #f3f4f6; border-radius: 6px; padding: 10px 14px; font-size: 13px; }}
</style></head><body>

<h1>Evaluation Report — Baseline</h1>
<p class="muted">{cfg["symbol"]} · {cfg["timeframe"]} · {cfg["since"]} → {cfg["until"]} ·
generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>

<div class="warn"><b>Read this first.</b> This is an <b>honest baseline</b> of the system's
rule-based pipeline (no ML, no lookahead, realistic fees + slippage). The strategy shows
<b>no demonstrable edge</b>: returns are tiny and statistically indistinguishable from
random entries. The ML classifier scores barely above chance. Educational/demo only —
not financial advice.</div>

<h2>Strategy vs. benchmarks</h2>
<table>
  <tr><th>Strategy</th><th>Total return %</th><th>Sharpe</th><th>Win rate %</th></tr>
  <tr><td>Rule-based strategy</td><td>{strat["total_return_pct"]}</td><td>{strat["sharpe"]}</td><td>{strat["win_rate_pct"]}</td></tr>
  <tr><td>Buy &amp; hold</td><td>{bh["total_return_pct"]}</td><td>{bh["sharpe"]}</td><td>—</td></tr>
  {rnd_row}
</table>
{img("equity", "equity curve")}
{img("drawdown", "drawdown")}
{bna_section}
<h2>Full strategy metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(strat)}</table>
<div class="grid">{img("pnl_hist", "pnl histogram")}{img("monthly", "monthly returns")}</div>

<h2>ML classifier evaluation (out-of-sample)</h2>
{ml_section}

<h2>Limitations &amp; caveats</h2>
<ul>
  <li><b>No edge demonstrated.</b> Strategy returns are near zero and indistinguishable
      from random entries with matched frequency. Beating buy &amp; hold here is mostly an
      artifact of low exposure during a falling market, not predictive skill.</li>
  <li><b>Conservative 2% sizing</b> caps both risk and reward, so absolute PnL is small by
      design — judge the strategy on Sharpe / win rate / vs-random, not on dollar return.</li>
  <li><b>Weak classifier.</b> ~33% is random on 3 classes; the model lands ~0.41–0.49 and
      the HOLD class is essentially unlearned. Directional precision is near coin-flip.</li>
  <li><b>Limited history.</b> Bundled sample data is modest; results are indicative, not
      a multi-year, multi-regime validation. Single asset, single timeframe.</li>
  <li><b>Exit model is explicit.</b> Paper mode has no native SL/TP, so the backtest applies
      a documented SL/TP bracket; live paper behaviour can differ.</li>
  <li><b>No survivorship/regime adjustments</b> and no slippage modelling beyond a flat
      per-side fraction. Real fills on thin books would be worse.</li>
</ul>

<div class="disc">Educational and demonstration purposes only. Not financial, investment,
or trading advice; not a solicitation. Simulated results do not indicate future
performance. No warranty; use at your own risk.</div>
</body></html>"""

    html_path = results_dir / "EVALUATION_REPORT.html"
    html_path.write_text(html, encoding="utf-8")

    # PDF: dump the figures (+ a summary page) into a multi-page PDF.
    pdf_path = results_dir / "EVALUATION_REPORT.pdf"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            cover = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            cover.text(0.1, 0.92, "Evaluation Report — Baseline", fontsize=18, weight="bold")
            cover.text(
                0.1,
                0.88,
                f"{cfg['symbol']} · {cfg['timeframe']} · {cfg['since']} → {cfg['until']}",
                fontsize=10,
                color="#555",
            )
            lines = [
                "HONEST BASELINE — rule-based pipeline, no lookahead, fees + slippage applied.",
                "",
                f"Strategy total return:   {strat['total_return_pct']} %",
                f"Buy & hold total return: {bh['total_return_pct']} %",
                f"Sharpe / Sortino / Calmar: {strat['sharpe']} / {strat['sortino']} / {strat['calmar']}",
                f"Max drawdown:            {strat['max_drawdown_pct']} %",
                f"Trades / win rate:       {strat['num_trades']} / {strat['win_rate_pct']} %",
                f"Profit factor / expectancy: {strat['profit_factor']} / {strat['expectancy_usd']} $",
            ]
            if rnd:
                lines.append(f"Random-entry mean return: {rnd.get('mean_total_return_pct')} %")
            if ml.get("available"):
                lines += [
                    "",
                    f"ML accuracy (test): {ml['accuracy']} vs random {ml['random_baseline_accuracy']}",
                ]
            lines += [
                "",
                "Finding: no demonstrable edge; results consistent with random entry.",
                "Educational/demo only. Not financial advice.",
            ]
            cover.text(0.1, 0.78, "\n".join(lines), fontsize=11, va="top", family="monospace")
            pdf.savefig(cover)
            plt.close(cover)
            for name in ("equity", "drawdown", "pnl_hist", "monthly", "confusion", "calibration"):
                if name in figs:
                    pdf.savefig(figs[name])
    except Exception:
        pdf_path = None  # PDF is best-effort; HTML is the primary artifact.

    # Close figures to free memory.
    try:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)
    except Exception:
        pass

    out = {"html": html_path}
    if pdf_path is not None:
        out["pdf"] = pdf_path
    return out
