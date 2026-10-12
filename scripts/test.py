from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    build_test_window_dataset_from_file,
    build_window_datasets_from_files,
    create_dataloaders,
)
from data.preprocessing import Standardizer
from models.contrastive_model import ContrastiveCausalModel
from training.trainer import Trainer
from utils.graph_utils import validate_adjacency_shape


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def safe_torch_load(path: str | Path, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the causal contrastive anomaly detector.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--test-labels-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_data_mode(config: Dict[str, Any]) -> str:
    data_cfg = config["data"]
    full_path = str(data_cfg.get("path", ""))
    test_path = str(data_cfg.get("test_path", ""))

    has_full_series = full_path != ""
    has_explicit_test = test_path != ""

    if has_full_series and has_explicit_test:
        raise ValueError("Specify either a full-series path or an explicit test-path, not both.")
    if has_explicit_test:
        return "explicit_test"
    if has_full_series:
        return "full_series"
    raise ValueError("No test input was provided. Use --data-path or --test-path.")


def save_scores_csv(output_path: Path, results: Dict[str, Any]) -> None:
    data = {
        "time_index": np.asarray(results["time_indices"]),
        "window_score": np.asarray(results["window_scores"]),
        "anomaly_score": np.asarray(results["anomaly_score"]),
        "label": np.asarray(results["labels"]),
    }
    if "causal_scores" in results:
        data["causal_score"] = np.asarray(results["causal_scores"])
    if "causal_zscores" in results:
        data["causal_zscore"] = np.asarray(results["causal_zscores"])
    if "A_embed" in results:
        data["window_embed_score"] = np.asarray(results["A_embed"])

    headers = list(data.keys())
    row_count = len(next(iter(data.values()))) if data else 0
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row_idx in range(row_count):
            writer.writerow([data[key][row_idx] for key in headers])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = safe_torch_load(args.checkpoint, map_location=device)

    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = checkpoint["config"]

    if args.data_path:
        config["data"]["path"] = args.data_path
        config["data"]["test_path"] = ""
    if args.labels_path is not None:
        config["data"]["labels_path"] = args.labels_path
    elif not args.config:
        # Avoid silently reusing the labels path stored in the training checkpoint.
        config["data"]["labels_path"] = ""
    if args.test_path:
        config["data"]["test_path"] = args.test_path
        config["data"]["path"] = ""
    if args.test_labels_path is not None:
        config["data"]["test_labels_path"] = args.test_labels_path
    elif not args.config:
        # Avoid silently reusing the split-test labels path stored in the training checkpoint.
        config["data"]["test_labels_path"] = ""

    data_mode = resolve_data_mode(config)
    if data_mode == "explicit_test":
        config["data"]["path"] = ""
        config["data"]["labels_path"] = ""
    else:
        config["data"]["train_path"] = ""
        config["data"]["test_path"] = ""
        config["data"]["test_labels_path"] = ""

    scaler_state = checkpoint.get("scaler_state")
    scaler = None if scaler_state is None else Standardizer.from_state_dict(scaler_state)

    num_nodes = int(checkpoint["num_nodes"])
    if data_mode == "explicit_test":
        test_dataset = build_test_window_dataset_from_file(
            data_path=config["data"]["test_path"],
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            labels_path=config["data"].get("test_labels_path", ""),
            standardize=bool(config["data"]["standardize"]),
            scaler=scaler,
            label_strategy=str(config["data"]["label_strategy"]),
        )
        data_num_nodes = int(test_dataset.windows.shape[2])
        loaders = {
            "test": DataLoader(
                test_dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["training"]["num_workers"]),
            )
        }
    else:
        bundle = build_window_datasets_from_files(
            data_path=config["data"]["path"],
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            train_ratio=float(config["data"]["train_ratio"]),
            val_ratio=float(config["data"]["val_ratio"]),
            labels_path=config["data"].get("labels_path", ""),
            standardize=bool(config["data"]["standardize"]),
            scaler=scaler,
            label_strategy=str(config["data"]["label_strategy"]),
        )
        data_num_nodes = int(bundle.raw_data["test"].shape[1])
        loaders = create_dataloaders(
            bundle,
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(config["training"]["num_workers"]),
            shuffle_train=False,
        )

    if data_num_nodes != num_nodes:
        raise ValueError(f"Checkpoint expects {num_nodes} variables, but the test data has {data_num_nodes}.")

    model = ContrastiveCausalModel.from_config(config, num_nodes=num_nodes)
    trainer = Trainer(model=model, config=config, device=device)
    trainer.load_checkpoint(args.checkpoint)

    adjacency = checkpoint["adjacency"].to(device=device, dtype=torch.float32)
    validate_adjacency_shape(adjacency, num_nodes=num_nodes, source_name="Checkpoint adjacency")
    results = trainer.evaluate(loaders["test"], adjacency)

    result_dir = Path(config["output"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_scores_csv(result_dir / "test_scores_from_checkpoint.csv", results)

    if results.get("metrics"):
        print("Test metrics:")
        for key, value in results["metrics"].items():
            print(f"  {key}: {value:.4f}")
    else:
        print("No valid labels were found. Saved anomaly scores to CSV only.")


if __name__ == "__main__":
    main()
