#!/usr/bin/env python3
""" Multimodal Validation Script

Adapted from timm v1.0.26 validate.py by Ross Wightman.
https://github.com/huggingface/pytorch-image-models/blob/v1.0.26/validate.py
The original timm source is licensed under the Apache License 2.0.
"""

import argparse
import csv
import json
import logging
import math
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging
from torch.utils.data import DataLoader

import models  # noqa: F401
from datasets import SlideOmicsShardedSubtypingDataset
from utils.loader import fast_collate


_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch Multimodal Validation')
parser.add_argument('--data_dir', type=Path, required=True, metavar='DIR',
                    help='path to the supplied feature data')
parser.add_argument('--weights', type=Path, required=True, metavar='PATH',
                    help='path to the training checkpoint')
parser.add_argument('--out_dir', type=Path, required=True, metavar='DIR',
                    help='directory for predictions and optional metrics')
parser.add_argument('--labels', type=Path,
                    help='Optional labels CSV for validation metrics.')
parser.add_argument('--split', choices=('val', 'test'), default='test',
                    help='dataset split (default: test)')
parser.add_argument('--device', choices=('cuda', 'cpu'), default=None,
                    help='device to use (default: CUDA when available, otherwise CPU)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')


def compute_metrics(labels, p_lusc):
    """Compute binary classification metrics with LUSC as the positive class.

    Args:
        labels: One-dimensional sequence of LUAD/LUSC names or 0/1 labels.
        p_lusc: Matching sequence of finite LUSC probabilities in [0, 1].

    Returns:
        A dict of Python floats: auroc, macro_f1, balanced_accuracy and
        accuracy. Threshold-based metrics predict LUSC at p_lusc >= 0.5.

    Raises:
        ValueError: If inputs are malformed, empty or lack either class.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(p_lusc, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be matching one-dimensional sequences.")
    if len(labels) == 0:
        raise ValueError("At least one prediction is required.")
    label_map = {"LUAD": 0, "LUSC": 1, 0: 0, 1: 1}
    try:
        truth = np.asarray([label_map[label] for label in labels.tolist()], dtype=int)
    except (KeyError, TypeError) as exc:
        raise ValueError("Labels must be LUAD/LUSC or 0/1.") from exc
    if set(truth.tolist()) != {0, 1}:
        raise ValueError("Both LUAD and LUSC labels are required to calculate AUROC.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(truth, probabilities)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }


def _read_csv(path, columns):
    """Read CSV rows keyed by unique patient IDs, preserving file order.

    Args:
        path: String or Path identifying the CSV file.
        columns: List of required column names in their exact order.

    Returns:
        A dict mapping patient IDs to lists of the remaining field strings.

    Raises:
        ValueError: If headers, row widths or IDs violate the file contract,
            or if no patient rows are present.
    """
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != columns:
            raise ValueError(f"{path}: columns must be exactly {','.join(columns)}.")
        for row in reader:
            if len(row) != len(columns):
                raise ValueError(f"{path}: row {reader.line_num} has the wrong number of fields.")
            patient_id = row[0]
            if not patient_id.strip():
                raise ValueError(f"{path}: empty patient_id at row {reader.line_num}.")
            if patient_id in rows:
                raise ValueError(f"{path}: duplicate patient_id {patient_id!r}.")
            rows[patient_id] = row[1:]
    if not rows:
        raise ValueError(f"{path}: no patient rows.")
    return rows


def score_predictions(predictions_path, labels_path):
    """Validate prediction and label CSVs, then score patients by matching IDs.

    Row order may differ, but the two files must contain identical patient sets.
    Probabilities must sum to one within 1e-5, and predicted_label must agree
    with the LUSC threshold of 0.5, including equality.

    Args:
        predictions_path: CSV with patient_id, p_luad, p_lusc, predicted_label.
        labels_path: CSV with patient_id and label.

    Returns:
        The metric dict defined by compute_metrics.

    Raises:
        ValueError: If CSV contents are invalid, patients differ, predictions
            are inconsistent, or the labels do not include both classes.
    """
    labels = _read_csv(labels_path, ['patient_id', 'label'])
    predictions = _read_csv(predictions_path, ['patient_id', 'p_luad', 'p_lusc', 'predicted_label'])
    if any(row[0] not in ('LUAD', 'LUSC') for row in labels.values()):
        raise ValueError("The labels CSV must use only LUAD and LUSC.")
    missing = labels.keys() - predictions.keys()
    extra = predictions.keys() - labels.keys()
    if missing or extra:
        raise ValueError(f"Patient IDs do not match: {len(missing)} missing, {len(extra)} unexpected.")
    probabilities = []
    for patient_id in labels:
        row = predictions[patient_id]
        try:
            p_luad, p_lusc = float(row[0]), float(row[1])
        except ValueError as exc:
            raise ValueError(f"{patient_id}: probabilities must be numbers.") from exc
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (p_luad, p_lusc)):
            raise ValueError(f"{patient_id}: probabilities must be finite and in [0, 1].")
        # Include the tolerance boundary despite floating-point addition error.
        if abs(p_luad + p_lusc - 1.0) > 1e-5 + math.ulp(1.0):
            raise ValueError(f"{patient_id}: probabilities must sum to 1 within 1e-5.")
        expected = "LUSC" if p_lusc >= 0.5 else "LUAD"
        if row[2] != expected:
            raise ValueError(f"{patient_id}: predicted_label must be {expected} at threshold 0.5.")
        probabilities.append(p_lusc)
    return compute_metrics([row[0] for row in labels.values()], probabilities)


def validate(args):
    """Write patient predictions and return metrics when labels are supplied."""
    torch.set_num_threads(4)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # CheckpointSaver stores the training configuration as a Namespace.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    config = checkpoint['args']
    if list(config.classes) != ['LUAD', 'LUSC'] or config.num_classes != 2:
        raise ValueError('Checkpoint class order must be LUAD, LUSC')

    # Resolve AMP from the checkpoint so inference matches training validation.
    amp_autocast = suppress
    if config.amp and device.type == 'cuda':
        amp_dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in torch.float32. AMP not enabled.')

    # create model
    model = create_model(config.model, num_classes=2, **config.model_kwargs)
    # Reload the same weights used to select this checkpoint during validation.
    use_ema = getattr(config, 'model_ema', False) and not getattr(config, 'model_ema_force_cpu', False)
    model.load_state_dict(checkpoint['state_dict_ema' if use_ema else 'state_dict'], strict=True)
    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (config.model, param_count))
    model = model.to(device=device)

    # create dataset and loader
    dataset = SlideOmicsShardedSubtypingDataset(args.data_dir, split=args.split)
    if config.model_kwargs['rna_embed_dim'] != dataset.rna_feature_dim:
        raise ValueError('Input RNA width differs from the training checkpoint')
    # Keep full, variable-length bags for validation and test inference.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=fast_collate,
    )

    batch_time = AverageMeter()
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / 'predictions.csv'
    with predictions_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patient_id', 'p_luad', 'p_lusc', 'predicted_label'])
        with torch.inference_mode():
            end = time.time()
            for batch_idx, batch in enumerate(loader):
                patient_id = batch['patient_id'][0]
                input = [tensor.to(device=device) for tensor in batch['x']]

                # compute output
                with amp_autocast():
                    output = model(input)

                # Compute probabilities in float32 even when inference uses AMP.
                probability = output.float().softmax(dim=1)[0, 1].item()
                if not 0 <= probability <= 1:
                    raise ValueError('Model produced an invalid probability')
                # Preserve probability values when the evaluator reads the CSV.
                writer.writerow([
                    patient_id, format(1 - probability, '.17g'), format(probability, '.17g'),
                    'LUSC' if probability >= 0.5 else 'LUAD',
                ])

                # measure elapsed time
                batch_size = output.shape[0]
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)'.format(
                            batch_idx,
                            len(loader),
                            batch_time=batch_time,
                            rate_avg=batch_size / batch_time.avg,
                        )
                    )

    results = OrderedDict()
    if args.labels is not None:
        results.update(score_predictions(predictions_path, args.labels))
        text = json.dumps(results, indent=2, allow_nan=False) + '\n'
        (args.out_dir / 'metrics.json').write_text(text, encoding='utf-8')
        _logger.info(
            ' * AUROC {auroc:.6f} Macro-F1 {macro_f1:.6f} '
            'Balanced accuracy {balanced_accuracy:.6f} Accuracy {accuracy:.6f}'.format(**results)
        )
    _logger.info('Predictions written to %s', predictions_path)

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    results = validate(args)
    if args.labels is not None:
        print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
