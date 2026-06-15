"""Minimal figure builders.

The default PDF writer emits a valid simple PDF without depending on matplotlib.
Downstream scripts can replace the renderer while keeping the data contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pareto_points(rows: list[dict[str, Any]], quality_field: str, cost_field: str) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        quality = row.get(quality_field)
        cost = row.get(cost_field)
        if isinstance(quality, (int, float)) and isinstance(cost, (int, float)):
            points.append(
                {
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "quality": float(quality),
                    "cost": float(cost),
                }
            )
    return points


def risk_bucket_counts(rows: list[dict[str, Any]], risk_field: str = "risk_bucket") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get(risk_field) or "missing")
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def write_text_pdf(path: str | Path, title: str, lines: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content_lines = [title, *lines]
    stream = "BT /F1 12 Tf 72 760 Td\n"
    for idx, line in enumerate(content_lines[:42]):
        escaped = str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if idx:
            stream += "0 -16 Td\n"
        stream += f"({escaped}) Tj\n"
    stream += "ET"
    stream_bytes = stream.encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream_bytes)).encode() + b" >> stream\n" + stream_bytes + b"\nendstream endobj\n",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    output.write_bytes(pdf)
