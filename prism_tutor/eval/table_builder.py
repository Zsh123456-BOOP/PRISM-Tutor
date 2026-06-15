"""Build CSV/LaTeX-ready result tables from metric rows."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def summarize_table(
    rows: list[dict[str, Any]],
    metrics: list[str],
    group_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    group_fields = group_fields or ["dataset", "method"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["n"] = len(group_rows)
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if isinstance(row.get(metric), (int, float, bool))]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
            summary[f"{metric}_coverage"] = len(values) / len(group_rows) if group_rows else 0.0
        output.append(summary)
    return output


def rows_to_latex(rows: list[dict[str, Any]], columns: list[str] | None = None, caption: str = "") -> str:
    if not rows:
        columns = columns or []
    else:
        columns = columns or list(rows[0].keys())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_escape(col) for col in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row.get(col)) for col in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    if caption:
        lines.append(f"\\caption{{{_escape(caption)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _format_cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return _escape(str(value))


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )
