from __future__ import annotations
from typing import List
import sqlite3
from datetime import datetime
from core.evaluator import EvalPolicy, evaluate_bot, EvalResult

def _md_header(bot: str) -> str:
    ts = datetime.utcnow().isoformat(timespec="seconds")
    return f"# Bot Review – {bot}\n_Gegenereerd: {ts} UTC_\n"

def _md_metrics(res: EvalResult) -> str:
    lines = [
        "## Samenvatting",
        f"- Periode: **{res.start_ts} → {res.end_ts}**",
        f"- Waarnemingen: **{res.periods}** equity-punten",
        f"- Trades: **{res.trades}** (fills)",
        f"- Rendement: **{res.ret_pct:.2f}%**",
        f"- Max Drawdown: **{res.max_dd_pct:.2f}%**",
        f"- Sharpe (≈dag): **{res.sharpe:.2f}**",
        "",
        "## Advies",
        f"- **Aanbeveling:** `{res.recommendation.upper()}`",
        f"- **Reden:** {res.reason}",
    ]
    return "\n".join(lines)

def render_markdown(conn: sqlite3.Connection, bot: str, policy: EvalPolicy) -> str:
    res = evaluate_bot(conn, bot, policy)
    return _md_header(bot) + "\n" + _md_metrics(res)

def evaluate_bots_markdown(db_path: str, bots: List[str], policy: EvalPolicy) -> str:
    with sqlite3.connect(db_path) as conn:
        blocks = [render_markdown(conn, b, policy) for b in bots]
    return "\n\n---\n\n".join(blocks)