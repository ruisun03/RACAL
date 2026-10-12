from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_window_datasets_from_files, build_window_datasets_from_split_files, create_dataloaders
from models.contrastive_model import ContrastiveCausalModel
from training.trainer import Trainer
from utils.graph_utils import build_proxy_adjacency, load_adjacency, validate_adjacency_shape
from utils.seed import set_seed


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal contrastive anomaly detector.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--test-labels-path", type=str, default="")
    parser.add_argument("--adj-path", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def resolve_data_mode(config: Dict[str, Any]) -> str:
    """判断训练使用“整段时序模式”还是“显式 train/test 切分模式” 。"""

    data_cfg = config["data"]
    full_path = str(data_cfg.get("path", ""))
    train_path = str(data_cfg.get("train_path", ""))
    test_path = str(data_cfg.get("test_path", ""))

    has_full_series = full_path != ""
    has_explicit_splits = train_path != "" or test_path != ""

    if has_full_series and has_explicit_splits:
        raise ValueError("Specify either a full-series path or explicit train/test split paths, not both.")
    if has_explicit_splits:
        if train_path == "" or test_path == "":
            raise ValueError("Both train_path and test_path are required in explicit split mode.")
        return "explicit_splits"
    if has_full_series:
        return "full_series"
    raise ValueError("No data input was provided. Use --data-path or --train-path/--test-path.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # 命令行参数优先于 YAML 配置，这样实验时可以不改配置文件，直接切换
    # 数据来源或图先验。
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.labels_path:
        config["data"]["labels_path"] = args.labels_path
    if args.train_path:
        config["data"]["train_path"] = args.train_path
    if args.test_path:
        config["data"]["test_path"] = args.test_path
    if args.test_labels_path:
        config["data"]["test_labels_path"] = args.test_labels_path
    if args.adj_path:
        config["data"]["adjacency_path"] = args.adj_path
    if args.device:
        config["device"] = args.device

    data_mode = resolve_data_mode(config)
    if data_mode == "explicit_splits":
        # 显式切分模式下，主动清空整段时序字段，避免后面误混两种接口语义。
        config["data"]["path"] = ""
        config["data"]["labels_path"] = ""
    else:
        config["data"]["train_path"] = ""
        config["data"]["test_path"] = ""
        config["data"]["test_labels_path"] = ""

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    print(f"Using device: {device}")
    result_dir = Path(config["output"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    if data_mode == "explicit_splits":
        bundle = build_window_datasets_from_split_files(
            train_path=config["data"]["train_path"],
            test_path=config["data"]["test_path"],
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            val_ratio=float(config["data"]["val_ratio"]),
            test_labels_path=config["data"].get("test_labels_path", ""),
            standardize=bool(config["data"]["standardize"]),
            label_strategy=str(config["data"]["label_strategy"]),
        )
    else:
        bundle = build_window_datasets_from_files(
            data_path=config["data"]["path"],
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            train_ratio=float(config["data"]["train_ratio"]),
            val_ratio=float(config["data"]["val_ratio"]),
            labels_path=config["data"].get("labels_path", ""),
            standardize=bool(config["data"]["standardize"]),
            label_strategy=str(config["data"]["label_strategy"]),
        )
    loaders = create_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle_train=True,
    )

    num_nodes = bundle.raw_data["train"].shape[1]

    adj_path = str(config["data"].get("adjacency_path", ""))
    if adj_path:
        # 优先路径：直接使用用户提供或数据集自带的图结构。
        adjacency = load_adjacency(adj_path)
        validate_adjacency_shape(adjacency, num_nodes=num_nodes, source_name="External adjacency")
        print(f"Loaded external adjacency from {adj_path}")
    else:
        # 回退路径：从训练集学习一张可用的结构图。
        # 这里既支持轻量级 proxy 构图，也支持更重的神经因果发现实现。
        proxy_cfg = config["data"]["proxy_graph"]
        adjacency, proxy_scores = build_proxy_adjacency(
            bundle.raw_data["train"],
            method=str(proxy_cfg["method"]),
            threshold=float(proxy_cfg["threshold"]),
            lag=int(proxy_cfg["lag"]),
            max_parents=int(proxy_cfg["max_parents"]),
            extra_cfg={
                **proxy_cfg,
                "device": str(config["device"]),
                "seed": int(config["seed"]),
            },
        )
        result_dir = Path(config["output"]["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        # 同时保存二值图和阈值化前的原始分数，方便后续检查这个启发式
        # 到底找到了什么结构。
        np.savetxt(result_dir / "proxy_graph_scores.csv", proxy_scores, delimiter=",")
        np.savetxt(result_dir / "proxy_adjacency.csv", adjacency, delimiter=",")
        print("Built proxy adjacency from the training split.")

    validate_adjacency_shape(adjacency, num_nodes=num_nodes)
    adjacency_t = torch.tensor(adjacency, dtype=torch.float32, device=device)
    model = ContrastiveCausalModel.from_config(config, num_nodes=num_nodes)
    trainer = Trainer(model=model, config=config, device=device)

    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        adjacency=adjacency_t,
        scaler_state=bundle.scaler.state_dict(),
    )
    print(f"Best checkpoint saved to: {history['best_checkpoint_path']}")
    if bool(config["output"].get("export_final_matrices", True)):
        exported_paths = trainer.export_final_matrices(
            loader=loaders["train"],
            adjacency=adjacency_t,
            output_dir=result_dir,
            max_windows=int(config["output"].get("export_matrix_max_windows", 512)),
        )
        if exported_paths:
            print("Saved final matrices:")
            for name, path in exported_paths.items():
                print(f"  {name}: {path}")

    test_results = trainer.evaluate(loaders["test"], adjacency_t)
    result_dir = Path(config["output"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_scores_csv(result_dir / "test_scores.csv", test_results)

    if test_results.get("metrics"):
        print("Test metrics:")
        for key, value in test_results["metrics"].items():
            print(f"  {key}: {value:.4f}")
    else:
        print("No valid test labels were provided; only anomaly scores were saved.")


if __name__ == "__main__":
    main()
