"""Подбор top_k для агента по кривым метрик из results.json.

Usage:
    python pick_top_k.py
    python pick_top_k.py --target 0.95
    python pick_top_k.py --chunks-alpha 0.7 --titles-method vector_title

Перед запуском: run_eval.py + eval_title_search.py с нужным EVAL_KS в .env.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

RESULTS = config.RESULTS_DIR / "results.json"
TITLE_RESULTS = config.RESULTS_DIR / "title_search_results.json"


def _ks_from_row(row: dict) -> list[int]:
    ks = []
    for key in row:
        m = re.match(r"^(mrr|recall|precision|hit)@(\d+)$", key)
        if m:
            ks.append(int(m.group(2)))
    return sorted(set(ks))


def _load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Нет файла: {path}\nСначала: python run_eval.py --skip-chain")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _find_row(
    rows: list[dict],
    *,
    method: str,
    embed: str,
    alpha: float | None = None,
    model_key: str = "embed_model",
) -> dict:
    candidates = [r for r in rows if r.get("method") == method and r.get(model_key) == embed]
    if alpha is not None:
        candidates = [r for r in candidates if r.get("alpha") == alpha]
    if not candidates:
        avail = {(r.get("method"), r.get(model_key), r.get("alpha")) for r in rows}
        sys.exit(f"Нет строки method={method} embed={embed} alpha={alpha}\nПримеры: {list(avail)[:5]}")
    # если несколько (RRF и т.д.) — лучший по mrr@max(k)
    ks = _ks_from_row(candidates[0])
    pk = max(ks) if ks else 10
    return max(candidates, key=lambda r: r.get(f"mrr@{pk}", 0))


def _curve(row: dict, metric: str, ks: list[int]) -> list[tuple[int, float]]:
    return [(k, float(row.get(f"{metric}@{k}", 0))) for k in ks]


def _pick_k(
    curve: list[tuple[int, float]],
    *,
    metric: str,
    target_ratio: float,
    min_k: int = 1,
    max_k: int | None = None,
) -> tuple[int, str]:
    """Минимальный k, где metric >= target_ratio * metric@max_k."""
    if not curve:
        return min_k, "нет данных"
    if max_k is not None:
        curve = [(k, v) for k, v in curve if k <= max_k]
    if not curve:
        return min_k, "пусто после max_k"

    k_max, v_max = curve[-1]
    if v_max <= 0:
        return min_k, "метрика @max = 0"

    threshold = target_ratio * v_max
    for k, v in curve:
        if k >= min_k and v >= threshold:
            return (
                k,
                f"{metric}@{k}={v:.3f} >= {target_ratio:.0%}×{v_max:.3f}@{k_max}",
            )

    return curve[-1][0], f"порог {target_ratio:.0%} не достигнут, взят max k"


def _format_table(title: str, row: dict, ks: list[int], primary: str, secondary: str) -> list[str]:
    hdr = ["| k |"] + [f" {primary} |" for _ in ks] + [f" {secondary} |" for _ in ks]
    sep = ["|---|"] + ["------|" for _ in ks] + ["------|" for _ in ks]
    vals = ["| |"] + [f" {row.get(f'{primary}@{k}', 0):.3f} |" for k in ks]
    vals += [f" {row.get(f'{secondary}@{k}', 0):.3f} |" for k in ks]
    return [
        f"### {title}",
        f"config: `{row.get('method')}` embed=`{row.get('embed_model') or row.get('model')}` "
        f"α={row.get('alpha', '—')}",
        "",
        "".join(hdr),
        "".join(sep),
        "".join(vals),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Pick top_k for agent from eval JSON")
    p.add_argument("--embed", default="embeddinggemma")
    p.add_argument("--chunks-method", default="hybrid_linear")
    p.add_argument("--chunks-alpha", type=float, default=0.7)
    p.add_argument("--titles-method", default="vector_title")
    p.add_argument("--titles-alpha", type=float, default=None)
    p.add_argument("--target", type=float, default=0.95, help="Доля от метрики @max_k (0.95 = 95%%)")
    p.add_argument("--chunks-metric", default="recall", choices=("recall", "mrr"))
    p.add_argument("--titles-metric", default="hit", choices=("hit", "mrr", "recall"))
    p.add_argument("--chunks-max", type=int, default=15, help="Лимит агента search_by_chunks")
    p.add_argument("--titles-max", type=int, default=10, help="Лимит агента search_by_titles")
    args = p.parse_args()

    chunk_rows = _load(RESULTS)
    title_rows = _load(TITLE_RESULTS)

    chunk_row = _find_row(
        chunk_rows, method=args.chunks_method, embed=args.embed, alpha=args.chunks_alpha
    )
    title_row = _find_row(
        title_rows,
        method=args.titles_method,
        embed=args.embed,
        alpha=args.titles_alpha,
        model_key="model",
    )

    chunk_ks = _ks_from_row(chunk_row)
    title_ks = _ks_from_row(title_row)
    expected = set(config.EVAL_KS)

    lines: list[str] = [
        "# Подбор top_k для агента",
        "",
        f"EVAL_KS в .env: `{config.EVAL_KS}`",
        "",
    ]

    for name, ks in ("chunks", chunk_ks), ("titles", title_ks):
        missing = expected - set(ks)
        if missing:
            lines.append(
                f"⚠ **{name}**: в JSON нет метрик для k={sorted(missing)} — "
                f"перезапусти eval с `EVAL_KS={','.join(map(str, sorted(expected)))}`"
            )
        lines.append("")

    lines += _format_table(
        "Чанки (search_by_chunks)",
        chunk_row,
        chunk_ks,
        args.chunks_metric,
        "mrr" if args.chunks_metric != "mrr" else "recall",
    )
    lines.append("")

    lines += _format_table(
        "Заголовки (search_by_titles)",
        title_row,
        title_ks,
        args.titles_metric,
        "mrr" if args.titles_metric != "mrr" else "hit",
    )
    lines.append("")

    chunk_curve = _curve(chunk_row, args.chunks_metric, chunk_ks)
    title_curve = _curve(title_row, args.titles_metric, title_ks)

    k_chunks, why_c = _pick_k(
        chunk_curve,
        metric=args.chunks_metric,
        target_ratio=args.target,
        min_k=1,
        max_k=args.chunks_max,
    )
    k_titles, why_t = _pick_k(
        title_curve,
        metric=args.titles_metric,
        target_ratio=args.target,
        min_k=1,
        max_k=args.titles_max,
    )

    # альтернатива: max MRR@k в пределах лимита
    def _best_mrr_k(row: dict, ks: list[int], cap: int) -> int:
        ks2 = [k for k in ks if k <= cap]
        return max(ks2, key=lambda k: row.get(f"mrr@{k}", 0)) if ks2 else cap

    k_chunks_mrr = _best_mrr_k(chunk_row, chunk_ks, args.chunks_max)
    k_titles_mrr = _best_mrr_k(title_row, title_ks, args.titles_max)

    lines += [
        "## Рекомендация",
        "",
        f"| Инструмент | top_k ({args.chunks_metric}/{args.titles_metric} ≥ {args.target:.0%} @max) | top_k (max MRR в лимите) |",
        f"|------------|----------------------------------|---------------------------|",
        f"| `search_by_chunks` | **{k_chunks}** | {k_chunks_mrr} |",
        f"| `search_by_titles` | **{k_titles}** | {k_titles_mrr} |",
        "",
        f"- chunks: {why_c}",
        f"- titles: {why_t}",
        "",
        "Правка в агенте: `02_rag_agent/agent_service/rag_agent/agent.py` — defaults `top_k` и docstring.",
        "",
    ]

    report = "\n".join(lines)
    out = config.RESULTS_DIR / "pick_top_k.md"
    out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
