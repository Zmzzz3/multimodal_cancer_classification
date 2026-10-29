# Option 8: Multimodal tumour classification

## Introduction

This package provides the supplied multimodal baseline, including gated ABMIL for WSI aggregation, an RNA MLP, concatenation and a two-class linear head. Both encoders use width 768, giving 7,039,491 parameters.

## Getting Started

### Directory Structure
```text
COMP3419-A2-Opt8-Code/
  README.md                             # Setup and usage instructions
  pyproject.toml                        # Project dependencies
  uv.lock                               # Exact dependency versions
  .python-version                       # Required Python version
  configs/                              # Experiment configurations
    baseline.yaml                       # Baseline model and training settings
  train_subtyping.py                    # Training and validation
  scripts/train_launcher.sh             # Launch training with torchrun
  evaluation.py                         # Generate predictions and evaluate metrics
  models/baseline.py                    # ABMIL and RNA MLP model
  datasets/dataset_subtyping_sharded.py # Load paired WSI and RNA features
  utils/loader.py                       # Batch collation and device prefetching
  input/                                # Supplied features, splits and labels
  LICENSE                               # Code licence
```

### Setup Environment

On Linux or Windows, install [uv](https://docs.astral.sh/uv/getting-started/installation/) version 0.12.5 or newer. Run the commands from the package directory. uv installs Python 3.12 if needed and creates the project environment using `pyproject.toml` and `uv.lock`.

```bash
uv sync --locked
```

#### Google Colab

Select a GPU runtime and extract the package, including `input/`, under `/content/`. Run these lines in a notebook cell:

```python
%pip install "uv>=0.12.5"
%cd /content/COMP3419-A2-Opt8-Code
!uv sync --locked
```

For the training and evaluation commands below, add `!` before `uv`, for example:

```python
!uv run --locked python train_subtyping.py --config configs/baseline.yaml --experiment baseline
```

These commands use the project's Python environment independently of the notebook kernel.

### Prepare Data

The supplied data are in `input/`. There are 943 patients: 660 training, 141 validation and 142 test patients. Each patient has both modalities. Patient and slide identifiers are anonymised with test labels withheld.

```text
input/
  master_index.parquet        # IDs, shapes and scalar-element offsets
  slides/shard_0000.bin       # float32 WSI features
  rna.bin                    # float32 RNA vectors
  splits_0.csv               # fixed patient splits
  labels.csv                 # training and validation labels
  val_labels.csv             # validation labels for evaluation
```

#### Histopathology Data

Frozen Phikon features form a WSI bag of shape `[N,768]`, with 8 to 1,024 patches per patient. Training fills shorter bags to 1,024 patches by sampling existing patches. Validation and test inference use all supplied patches with batch size 1.

#### Transcriptomics Data

RNA vectors contain 4,538 gene features from the processed RSEM gene-level TPM data. The supplied vectors preserve the source values and feature order. No additional feature selection or normalisation is required. The loader reads the RNA width from the index and returns the modalities as `x=[rna,wsi]`.

### Training

Training uses the YAML-based configuration in `configs/baseline.yaml`. The default settings are 100 epochs and seed 42. The training loop uses Adam and batch size 16, starts from scratch, and selects the highest validation AUROC. Ties retain the earlier checkpoint.

After each epoch, the trainer predicts the validation patients to calculate metrics and select the best checkpoint. When validation AUROC improves, it saves `model_best.pth.tar`, `predictions.csv` (validation patients), and `validation_metrics.json` in `output/baseline/`.

```bash
uv run --locked python train_subtyping.py --config configs/baseline.yaml --experiment baseline
```

#### Using the launcher

On Linux or Colab, you can also use the Bash launcher to start training through `torchrun`. Run it from `COMP3419-A2-Opt8-Code/`. On Windows, use the direct Python command above.

| Launcher option | Meaning in the example |
| --- | --- |
| `-N 1` | Use one machine (node). |
| `-P 1` | Start one training process on one GPU. |
| `-t 4` | Set four OpenMP CPU threads per process. |

Keep launcher options **before** `train_subtyping.py`. Put `--config configs/baseline.yaml` **immediately after** the script name, followed by training options such as `--epochs` and `--experiment`. Training options on the command line override the corresponding YAML settings.

First, preview the command with `-d`. This prints the launch settings without starting training:

```bash
uv run --locked ./scripts/train_launcher.sh -N 1 -P 1 -t 4 -d \
  train_subtyping.py --config configs/baseline.yaml \
  --data-dir input --output output --experiment baseline
```

Then run one epoch to check the environment and data:

```bash
uv run --locked ./scripts/train_launcher.sh -N 1 -P 1 -t 4 \
  train_subtyping.py --config configs/baseline.yaml \
  --data-dir input --output output --experiment check --epochs 1
```

The check writes its results to `output/check/`. Remove `-d` from the preview command to run the configured 100 epochs and write results to `output/baseline/`. Choose a new `--experiment` name for each run. To see all launcher options, run `uv run --locked ./scripts/train_launcher.sh -h`.

For a CPU execution check, call the training script directly:

```bash
uv run --locked python train_subtyping.py --config configs/baseline.yaml --device cpu --epochs 1 --experiment cpu-check
```

### Evaluation

#### Generate final test predictions

After fixing all model choices, use the selected checkpoint to generate the test prediction file for submission:

```bash
uv run --locked python evaluation.py --data_dir input --weights output/baseline/model_best.pth.tar --out_dir output/test
```

This writes `output/test/predictions.csv` with the columns `patient_id,p_luad,p_lusc,predicted_label`. Test labels are withheld, so this command does not calculate test metrics.

#### Optional: re-evaluate validation

Training already saves validation results. Use this command only if you want to recompute them from a saved checkpoint:

```bash
uv run --locked python evaluation.py --data_dir input --weights output/baseline/model_best.pth.tar --out_dir output/val --split val --labels input/val_labels.csv
```

This writes `output/val/predictions.csv` and `output/val/metrics.json`. The metrics are AUROC, macro-F1, balanced accuracy and accuracy. Follow the assignment specification for evaluation and submission requirements.

## License

The code is released under the GNU General Public License v3.0. [LICENSE](LICENSE) also includes the upstream Apache 2.0 licence for the timm evaluation code.
