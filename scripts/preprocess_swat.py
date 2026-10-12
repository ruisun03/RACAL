from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ParsedSplit:
    series: np.ndarray
    label_counts: dict[str, int]
    feature_names: list[str]
    first_timestamp: str
    last_timestamp: str
    binary_labels: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the raw SWaT CSV files into explicit train/test artifacts "
            "that match this repo's scripts/train.py and scripts/test.py interface."
        )
    )
    parser.add_argument(
        "--normal-path",
        type=Path,
        default=ROOT / "data" / "SWaT" / "SWaT_Dataset_Normal_v0.CSV",
        help="Path to the raw SWaT normal file.",
    )
    parser.add_argument(
        "--attack-path",
        type=Path,
        default=ROOT / "data" / "SWaT" / "SWaT_Dataset_Attack_v0.csv",
        help="Path to the raw SWaT attack file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "processed_swat",
        help="Destination directory for processed artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def clean_cell(value: str) -> str:
    return value.strip().replace("\ufeff", "")


def clean_header_cell(value: str) -> str:
    return clean_cell(value).strip(",")


def clean_label(value: str) -> str:
    return clean_cell(value).strip(",")


def resolve_cli_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root {path} already exists and is not empty. "
            "Pass --overwrite or choose a new destination."
        )
    path.mkdir(parents=True, exist_ok=True)


def iter_nonempty_rows(reader: Iterable[list[str]]) -> Iterable[list[str]]:
    for row in reader:
        if row and any(clean_cell(cell) != "" for cell in row):
            yield row


def count_data_rows(path: Path, delimiter: str) -> int:
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader, None)
        return sum(1 for _ in iter_nonempty_rows(reader))


def read_header(path: Path, delimiter: str) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader, None)
    if not header:
        raise ValueError(f"CSV file is empty: {path}")
    return [clean_header_cell(cell) for cell in header]


def extract_feature_names(header: list[str], source_name: str) -> list[str]:
    if len(header) < 3:
        raise ValueError(f"{source_name} must contain at least timestamp, one feature, and one label column.")
    if header[0] != "Timestamp":
        raise ValueError(f"{source_name} expected the first column to be Timestamp, got {header[0]!r}.")
    if header[-1] != "Normal/Attack":
        raise ValueError(
            f"{source_name} expected the last column to normalize to 'Normal/Attack', got {header[-1]!r}."
        )
    feature_names = header[1:-1]
    if not feature_names:
        raise ValueError(f"{source_name} does not contain any feature columns.")
    return feature_names


def load_split(
    path: Path,
    delimiter: str,
    decimal_comma: bool,
    split_name: str,
    expected_feature_names: list[str] | None = None,
    require_all_normal: bool = False,
    build_binary_labels: bool = False,
) -> ParsedSplit:
    header = read_header(path, delimiter=delimiter)
    feature_names = extract_feature_names(header, source_name=split_name)
    if expected_feature_names is not None and feature_names != expected_feature_names:
        raise ValueError(f"{split_name} feature columns do not match the reference feature ordering.")

    row_count = count_data_rows(path, delimiter=delimiter)
    num_features = len(feature_names)
    series = np.empty((row_count, num_features), dtype=np.float32)
    binary_labels = np.empty((row_count,), dtype=np.float32) if build_binary_labels else None
    label_counter: Counter[str] = Counter()
    first_timestamp = ""
    last_timestamp = ""

    converter = (
        (lambda token: float(clean_cell(token).replace(",", ".")))
        if decimal_comma
        else (lambda token: float(clean_cell(token)))
    )

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        raw_header = next(reader, None)
        if raw_header is None:
            raise ValueError(f"CSV file is empty: {path}")

        write_idx = 0
        expected_width = len(raw_header)
        for row in iter_nonempty_rows(reader):
            if len(row) != expected_width:
                raise ValueError(
                    f"{split_name} row {write_idx + 1} has {len(row)} columns, expected {expected_width}."
                )

            timestamp = clean_cell(row[0])
            label = clean_label(row[-1])
            if label not in {"Normal", "Attack"}:
                raise ValueError(f"{split_name} row {write_idx + 1} has unexpected label {label!r}.")
            if require_all_normal and label != "Normal":
                raise ValueError(f"{split_name} contains a non-normal label at row {write_idx + 1}: {label!r}.")

            if write_idx == 0:
                first_timestamp = timestamp
            last_timestamp = timestamp
            label_counter[label] += 1

            series[write_idx] = [converter(token) for token in row[1:-1]]
            if binary_labels is not None:
                binary_labels[write_idx] = 1.0 if label == "Attack" else 0.0
            write_idx += 1

    if write_idx != row_count:
        raise RuntimeError(f"{split_name} counted {row_count} rows but parsed {write_idx}.")

    return ParsedSplit(
        series=series,
        label_counts=dict(label_counter),
        feature_names=feature_names,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        binary_labels=binary_labels,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def write_manifest(path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "dataset",
        "train_path",
        "test_path",
        "test_labels_path",
        "num_train_steps",
        "num_test_steps",
        "num_nodes",
        "num_test_anomaly_points",
        "metadata_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    normal_path = args.normal_path.resolve()
    attack_path = args.attack_path.resolve()
    output_root = args.output_root.resolve()

    if not normal_path.exists():
        raise FileNotFoundError(f"Normal split not found: {normal_path}")
    if not attack_path.exists():
        raise FileNotFoundError(f"Attack split not found: {attack_path}")

    ensure_output_root(output_root, overwrite=bool(args.overwrite))

    print("Parsing SWaT normal split...")
    normal_split = load_split(
        path=normal_path,
        delimiter=",",
        decimal_comma=False,
        split_name="SWaT normal split",
        require_all_normal=True,
        build_binary_labels=False,
    )

    print("Parsing SWaT attack split...")
    attack_split = load_split(
        path=attack_path,
        delimiter=";",
        decimal_comma=True,
        split_name="SWaT attack split",
        expected_feature_names=normal_split.feature_names,
        require_all_normal=False,
        build_binary_labels=True,
    )

    if attack_split.binary_labels is None:
        raise RuntimeError("Expected binary labels for the SWaT attack split.")

    train_path = output_root / "train.npy"
    test_path = output_root / "test.npy"
    test_labels_path = output_root / "test_labels.npy"
    metadata_path = output_root / "metadata.json"
    summary_path = output_root / "summary.json"
    feature_names_path = output_root / "feature_names.json"
    manifest_path = output_root / "manifest.csv"

    np.save(train_path, normal_split.series.astype(np.float32, copy=False))
    np.save(test_path, attack_split.series.astype(np.float32, copy=False))
    np.save(test_labels_path, attack_split.binary_labels.astype(np.float32, copy=False))

    feature_names_payload = normal_split.feature_names
    write_json(feature_names_path, {"feature_names": feature_names_payload})

    num_test_anomaly_points = int(float(attack_split.binary_labels.sum()))
    num_nodes = int(normal_split.series.shape[1])

    metadata = {
        "dataset": "SWaT",
        "interface_mode": "explicit_splits",
        "repo_interface": {
            "train_arg": "--train-path",
            "test_arg": "--test-path",
            "test_labels_arg": "--test-labels-path",
        },
        "input_files": {
            "normal": resolve_cli_path(normal_path),
            "attack": resolve_cli_path(attack_path),
        },
        "output_files": {
            "train": resolve_cli_path(train_path),
            "test": resolve_cli_path(test_path),
            "test_labels": resolve_cli_path(test_labels_path),
            "feature_names": resolve_cli_path(feature_names_path),
        },
        "shapes": {
            "train": list(normal_split.series.shape),
            "test": list(attack_split.series.shape),
            "test_labels": list(attack_split.binary_labels.shape),
        },
        "num_nodes": num_nodes,
        "feature_names": feature_names_payload,
        "num_test_anomaly_points": num_test_anomaly_points,
        "normal_split": {
            "label_counts": normal_split.label_counts,
            "first_timestamp": normal_split.first_timestamp,
            "last_timestamp": normal_split.last_timestamp,
        },
        "attack_split": {
            "label_counts": attack_split.label_counts,
            "first_timestamp": attack_split.first_timestamp,
            "last_timestamp": attack_split.last_timestamp,
        },
    }
    write_json(metadata_path, metadata)

    summary = {
        "dataset": "SWaT",
        "num_train_steps": int(normal_split.series.shape[0]),
        "num_test_steps": int(attack_split.series.shape[0]),
        "num_nodes": num_nodes,
        "num_test_anomaly_points": num_test_anomaly_points,
        "train_path": resolve_cli_path(train_path),
        "test_path": resolve_cli_path(test_path),
        "test_labels_path": resolve_cli_path(test_labels_path),
        "metadata_path": resolve_cli_path(metadata_path),
    }
    write_json(summary_path, summary)

    write_manifest(
        manifest_path,
        {
            "dataset": "SWaT",
            "train_path": resolve_cli_path(train_path),
            "test_path": resolve_cli_path(test_path),
            "test_labels_path": resolve_cli_path(test_labels_path),
            "num_train_steps": int(normal_split.series.shape[0]),
            "num_test_steps": int(attack_split.series.shape[0]),
            "num_nodes": num_nodes,
            "num_test_anomaly_points": num_test_anomaly_points,
            "metadata_path": resolve_cli_path(metadata_path),
        },
    )

    print(f"Saved processed SWaT artifacts to {resolve_cli_path(output_root)}")
    print(f"Train shape: {normal_split.series.shape}")
    print(f"Test shape: {attack_split.series.shape}")
    print(f"Test anomalies: {num_test_anomaly_points}")


if __name__ == "__main__":
    main()
