#!/usr/bin/env python
"""Generate a radar/star plot from a summary metrics table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from _scene_metrics import load_label_map, load_method_order


MARKER_LIST = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a radar/star plot from a summary metrics table."
    )
    parser.add_argument("--input-table", required=True, help="Path to the summary table.")
    parser.add_argument(
        "--delimiter",
        choices=["auto", "tab", "comma"],
        default="auto",
        help="Input delimiter. Default: %(default)s",
    )
    parser.add_argument("--method-column", default="Method", help="Column containing method names.")
    parser.add_argument(
        "--metric-columns",
        nargs="+",
        default=None,
        help="Columns to plot, in order. Default: all columns except the method column.",
    )
    parser.add_argument(
        "--metric-labels",
        nargs="*",
        default=[],
        help="Optional display-label overrides in the form Column=Label.",
    )
    parser.add_argument(
        "--column-types",
        nargs="*",
        default=[],
        help=(
            "Optional column parser overrides in the form "
            "Column=type. Supported types: number, size_millions, "
            "duration_seconds, memory_gb, percent_0_100, fraction_0_1."
        ),
    )
    parser.add_argument(
        "--lower-is-better",
        nargs="*",
        default=None,
        help="Columns where smaller values are better. Default inferred from column types.",
    )
    parser.add_argument(
        "--log-scale-columns",
        nargs="*",
        default=None,
        help="Columns to normalize in log space. Default inferred from column types.",
    )
    parser.add_argument("--method-label-map", default=None, help="Optional JSON file mapping method names to labels.")
    parser.add_argument("--method-order-file", default=None, help="Optional method order text file.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output image path. Default: Visualization/plots/radar/radar_plot.png",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output file format. Default: %(default)s",
    )
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument(
        "--figure-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=[9.0, 9.0],
        help="Figure size in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI. Default: %(default)s")
    parser.add_argument("--legend-ncol", type=int, default=1,
                        help="1 = legend to the right (default); >1 = legend below the plot in that many columns.")
    return parser.parse_args()


def infer_delimiter(path: Path, mode: str) -> str:
    if mode == "tab":
        return "\t"
    if mode == "comma":
        return ","
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return "\t" if "\t" in first_line else ","


def parse_mapping_args(entries: list[str]) -> dict[str, str]:
    mapping = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected KEY=VALUE entry, got: {entry}")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def parse_size_millions(text: str) -> float | None:
    value = text.strip()
    if value in {"", "-", "?"}:
        return None
    if "+" in value:
        left, right = value.split("+", 1)
        left = left.replace("M", "").strip()
        right = right.replace("M", "").strip()
        if left in {"", "?"}:
            return float(right)
        return float(left) + float(right)
    value = value.replace("M", "").strip()
    try:
        return float(value)
    except ValueError:
        return float(value.split()[0])


def parse_duration_seconds(text: str) -> float | None:
    value = text.strip()
    if value in {"", "-", "?"} or "?" in value:
        return None
    value = value.replace(" ", "")
    if value.endswith("s"):
        value = value[:-1]
    if "+" in value:
        left, right = value.split("+", 1)
        return float(left) + float(right)
    if "-" in value and "e" not in value:
        left, _ = value.split("-", 1)
        return float(left)
    if "m" in value:
        minutes, seconds = value.split("m", 1)
        return float(minutes) * 60.0 + float(seconds)
    return float(value)


def parse_memory_gb(text: str) -> float | None:
    value = text.strip()
    if value in {"", "-", "?"} or "?" in value:
        return None
    if "+" in value:
        left, right = value.split("+", 1)
        left = left.strip()
        right = right.strip()
        base_str = left.replace("MB", "").replace("Mb", "").replace("M", "").replace("G", "").strip()
        base = float(base_str) if base_str else 0.0
        if any(unit in right for unit in ["MB", "Mb", "M"]):
            extra = right.replace("MB", "").replace("Mb", "").replace("M", "").strip()
            return base + (float(extra) / 1024.0 if extra else 0.0)
        if "G" in right:
            extra = right.replace("G", "").strip()
            return base + (float(extra) if extra else 0.0)
    if any(unit in value for unit in ["MB", "Mb", "M"]):
        return float(value.replace("MB", "").replace("Mb", "").replace("M", "").strip()) / 1024.0
    if "G" in value:
        return float(value.replace("G", "").strip())
    return float(value)


def parse_percent_0_100(text: str) -> float | None:
    value = text.strip().replace("%", "")
    if value in {"", "-", "?"} or "?" in value:
        return None
    numeric = float(value)
    if numeric < 0 or numeric > 100:
        raise ValueError(f"Expected percentage in [0, 100], got {numeric}")
    return numeric / 100.0


def parse_fraction_0_1(text: str) -> float | None:
    value = text.strip().replace("%", "")
    if value in {"", "-", "?"} or "?" in value:
        return None
    numeric = float(value)
    if numeric < 0 or numeric > 1:
        raise ValueError(f"Expected fraction in [0, 1], got {numeric}")
    return numeric


def parse_plain_number(text: str) -> float | None:
    value = text.strip()
    if value in {"", "-", "?"} or "?" in value:
        return None
    return float(value)


PARSERS = {
    "number": parse_plain_number,
    "size_millions": parse_size_millions,
    "duration_seconds": parse_duration_seconds,
    "memory_gb": parse_memory_gb,
    "percent_0_100": parse_percent_0_100,
    "fraction_0_1": parse_fraction_0_1,
}


def infer_column_type(column_name: str) -> str:
    name = column_name.casefold()
    if "size" in name or "params" in name:
        return "size_millions"
    if "speed" in name or "time" in name or "latency" in name:
        return "duration_seconds"
    if "vram" in name or "memory" in name or "mem" in name:
        return "memory_gb"
    if "map" in name or "recall" in name or "precision" in name or "iou" in name:
        return "percent_0_100"
    return "number"


def log_minmax(values: np.ndarray, invert: bool) -> np.ndarray:
    transformed = np.log10(values + 1e-9)
    minimum = transformed.min()
    maximum = transformed.max()
    if maximum == minimum:
        return np.ones_like(transformed)
    normalized = (transformed - minimum) / (maximum - minimum)
    return 1.0 - normalized if invert else normalized


def lin_minmax(values: np.ndarray, invert: bool) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return np.ones_like(values)
    normalized = (values - minimum) / (maximum - minimum)
    return 1.0 - normalized if invert else normalized


def resolve_output_path(output_path: str | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    return Path(__file__).resolve().parent / "plots" / "radar" / "radar_plot.png"


def save_figure(fig, output_path: Path, output_format: str, dpi: int) -> list[Path]:
    saved_paths = []
    formats = ["png", "pdf"] if output_format == "both" else [output_format]
    base_path = output_path if output_path.suffix else output_path.with_suffix(".png")
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def main() -> int:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1.")

    input_path = Path(args.input_table)
    delimiter = infer_delimiter(input_path, args.delimiter)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if args.method_column not in fieldnames:
        raise ValueError(f"Method column '{args.method_column}' not found in {input_path}")

    metric_columns = args.metric_columns or [name for name in fieldnames if name != args.method_column]
    if not metric_columns:
        raise ValueError("No metric columns were selected.")

    explicit_types = parse_mapping_args(args.column_types)
    metric_label_map = parse_mapping_args(args.metric_labels)
    column_types = {column: explicit_types.get(column, infer_column_type(column)) for column in metric_columns}
    for column, parser_name in column_types.items():
        if parser_name not in PARSERS:
            raise ValueError(f"Unsupported parser type '{parser_name}' for column '{column}'")

    lower_is_better = set(args.lower_is_better or [])
    if args.lower_is_better is None:
        lower_is_better = {
            column
            for column, parser_name in column_types.items()
            if parser_name in {"size_millions", "duration_seconds", "memory_gb"}
        }

    log_scale_columns = set(args.log_scale_columns or [])
    if args.log_scale_columns is None:
        log_scale_columns = {
            column
            for column, parser_name in column_types.items()
            if parser_name in {"size_millions", "duration_seconds", "memory_gb"}
        }

    label_map = load_label_map(args.method_label_map)
    method_order = load_method_order(args.method_order_file)

    parsed_rows = []
    for row in rows:
        method = row[args.method_column].strip()
        if not method:
            continue
        parsed_values = {}
        missing = False
        for column in metric_columns:
            parsed_value = PARSERS[column_types[column]](row[column])
            if parsed_value is None:
                missing = True
                break
            parsed_values[column] = parsed_value
        if missing:
            continue
        parsed_rows.append((method, parsed_values))

    if not parsed_rows:
        raise ValueError("No complete rows remained after parsing and filtering.")

    if method_order:
        rank = {entry: idx for idx, entry in enumerate(method_order)}
        parsed_rows.sort(key=lambda item: (rank.get(item[0], len(rank) + 1), item[0].casefold()))
    else:
        parsed_rows.sort(key=lambda item: item[0].casefold())

    methods = [label_map.get(method, method) for method, _ in parsed_rows]
    metric_axis_labels = [metric_label_map.get(column, column) for column in metric_columns]
    normalized_by_column = {}
    for column in metric_columns:
        values = np.array([metrics[column] for _, metrics in parsed_rows], dtype=float)
        invert = column in lower_is_better
        if column in log_scale_columns:
            normalized = log_minmax(values, invert=invert)
        else:
            normalized = lin_minmax(values, invert=invert)
        normalized_by_column[column] = normalized

    scores = {}
    for row_idx, (method, _) in enumerate(parsed_rows):
        scores[label_map.get(method, method)] = [
            float(normalized_by_column[column][row_idx]) for column in metric_columns
        ]

    average_scores = {method: float(np.mean(values)) for method, values in scores.items()}
    sorted_methods = sorted(average_scores, key=average_scores.get, reverse=True)
    best_method = sorted_methods[0]
    worst_method = sorted_methods[-1]

    n_methods = len(methods)
    cmap = plt.get_cmap("turbo", n_methods)
    colors = {method: cmap(idx) for idx, method in enumerate(methods)}
    markers = {method: MARKER_LIST[idx % len(MARKER_LIST)] for idx, method in enumerate(methods)}

    angles = np.linspace(0, 2 * np.pi, len(metric_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=tuple(args.figure_size))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for radius in [0.2, 0.4, 0.6, 0.8]:
        ax.text(np.deg2rad(270), radius, f"{radius:.1f}", ha="center", va="center", fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_axis_labels, fontsize=12)
    ax.tick_params(axis="x", which="major", pad=15)

    for method in methods:
        data = scores[method] + scores[method][:1]
        if method == best_method:
            linestyle, linewidth = "-.", 2.0
        elif method == worst_method:
            linestyle, linewidth = ":", 2.0
        else:
            linestyle, linewidth = "--", 1.8

        ax.plot(
            angles,
            data,
            linestyle=linestyle,
            linewidth=linewidth,
            color=colors[method],
            marker=markers[method],
            markersize=5,
            label=method,
        )

    if args.title:
        ax.set_title(args.title, fontsize=16, pad=20)

    handles, labels = ax.get_legend_handles_labels()
    below = args.legend_ncol > 1
    legend = ax.legend(
        handles,
        labels,
        loc="upper center" if below else "best",
        bbox_to_anchor=(0.5, -0.08) if below else (1.1, 1.0),
        ncol=args.legend_ncol,
        fontsize=9,
        borderaxespad=0.0,
        handlelength=3.0,
        handletextpad=0.8,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    saved_paths = save_figure(fig, output_path, args.output_format, args.dpi)
    plt.close(fig)
    print(f"Saved radar plot to {', '.join(str(path) for path in saved_paths)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
