# CAW

This repository contains the PyTorch implementation for a research framework for
multivariate time-series anomaly detection with graph-aware node representations,
causal residual scoring, confidence-aware positive weighting, and structured
negative sampling.

The primary model output is a node-level anomaly score for each window:

```text
x_i -> [A_{i,1}, A_{i,2}, ..., A_{i,N}]
```

where `x_i` is a window with shape `[T, N]` and `A_{i,j}` is the anomaly score of
node `j` in that window. When only window-level labels are available, this code
aggregates node scores into a window score for evaluation.

## Requirements

Python 3.10 or later is recommended.

Minimal dependencies:

```bash
pip install torch numpy pyyaml
```

Some graph construction methods and analysis workflows may require additional
packages depending on the selected configuration.

## Data

This repository does not include raw benchmark datasets. Download each dataset
from its original source and follow its license or access terms.

| Dataset | Source | Notes |
| --- | --- | --- |
| SWaT | https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/ | Requires iTrust dataset access approval. |
| WADI | https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/ | Requires iTrust dataset access approval. |
| PSM | https://github.com/eBay/RANSynCoders | The PSM sample data is released with the RANSynCoders repository. |
| SMD | https://github.com/NetManAIOps/OmniAnomaly | Use the `ServerMachineDataset` directory. |
| SMAP/MSL | https://github.com/khundman/telemanom | The telemetry data and `labeled_anomalies.csv` are used by the Telemanom benchmark. |

The training code supports two data interfaces.

### Explicit Train/Test Splits

This is the recommended interface for benchmark datasets:

```text
train.npy         # shape [num_train_steps, num_nodes]
test.npy          # shape [num_test_steps, num_nodes]
test_labels.npy   # shape [num_test_steps], 0 for normal and 1 for anomaly
adjacency.npy     # optional, shape [num_nodes, num_nodes]
```

Example layout:

```text
data/processed_swat/
  train.npy
  test.npy
  test_labels.npy
  default_swat.yaml
```

If `adjacency.npy` is not provided, `scripts/train.py` can build a proxy graph
from the training split according to the `data.proxy_graph` configuration.

### Full-Series Input

For custom datasets, a single full time series can also be used:

```text
series.npy        # shape [num_steps, num_nodes]
labels.npy        # optional, shape [num_steps]
adjacency.npy     # optional, shape [num_nodes, num_nodes]
```

The code will split the full series into contiguous train/validation/test
segments and fit standardization only on the training segment.

## Preprocessing

### SWaT

After receiving the SWaT dataset from iTrust, place the raw files locally:

```text
data/SWaT/
  SWaT_Dataset_Normal_v0.CSV
  SWaT_Dataset_Attack_v0.csv
```

Run:

```bash
python scripts/preprocess_swat.py \
  --normal-path data/SWaT/SWaT_Dataset_Normal_v0.CSV \
  --attack-path data/SWaT/SWaT_Dataset_Attack_v0.csv \
  --output-root data/processed_swat \
  --overwrite
```

This writes `train.npy`, `test.npy`, `test_labels.npy`, `feature_names.json`,
`metadata.json`, and manifest files under `data/processed_swat/`.

### PSM, SMD, and WADI

For datasets without a repository-specific preprocessing script, convert the raw
benchmark files into the explicit split format:

```text
data/processed_<dataset>/
  train.npy
  test.npy
  test_labels.npy
```

The provided dataset-specific YAML files show the expected default paths:

```text
data/processed_psm/default_psm.yaml
data/processed_wadi/default_wadi.yaml
```

## Training

Train with explicit split files:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --train-path data/processed_swat/train.npy \
  --test-path data/processed_swat/test.npy \
  --test-labels-path data/processed_swat/test_labels.npy \
  --device auto
```

Train with an external adjacency matrix:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --train-path data/processed_swat/train.npy \
  --test-path data/processed_swat/test.npy \
  --test-labels-path data/processed_swat/test_labels.npy \
  --adj-path data/processed_swat/adjacency.npy \
  --device auto
```

Train with a full time series:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data-path data/custom/series.npy \
  --labels-path data/custom/labels.npy \
  --device auto
```

Checkpoints and score files are written to the paths configured in:

```text
training.checkpoint_dir
output.result_dir
```

By default these paths are under `outputs/`.

## Testing

Evaluate a saved checkpoint on explicit test data:

```bash
python scripts/test.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --test-path data/processed_swat/test.npy \
  --test-labels-path data/processed_swat/test_labels.npy \
  --device auto
```

Evaluate with a custom config:

```bash
python scripts/test.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --config configs/default.yaml \
  --test-path data/processed_swat/test.npy \
  --test-labels-path data/processed_swat/test_labels.npy \
  --device auto
```

Test results are saved as `test_scores_from_checkpoint.csv` in the configured
output directory.

## Output Files

Training and testing save CSV files with columns such as:

```text
time_index
window_score
anomaly_score
label
causal_score
causal_zscore
window_embed_score
```

Depending on the model configuration, node-level score columns may also be
exported.

## Reproducibility Notes

- Set `seed` in the YAML config for deterministic experiment setup.
- Use explicit train/test splits for benchmark evaluation.
- Standardization is fitted on training data only.
- Do not commit raw datasets, processed arrays, checkpoints, or `outputs/`.
- Keep final paper tables and selected figures in a small separate directory if
  they are needed for the public repository.
