"""
Shared, dependency-light metric implementations used by every
model-specific evaluator in this package.

Nothing in here talks to a dataset or a model directly - these are
pure functions that take arrays / strings and return numbers, so
they can be unit-tested and reused across change detection,
optical-SAR fusion, and the VLM (VQA / captioning / grounding /
change-VQA) evaluators.
"""

import math
import re
from collections import Counter

import numpy as np


# ============================================================
# CLASSIFICATION METRICS
# (used for: VQA yes/no + multiple-choice, change-VQA)
# ============================================================


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0

    correct = sum(
        1
        for t, p in zip(y_true, y_pred)
        if _normalize_text(t) == _normalize_text(p)
    )

    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred, positive_label=None):
    """
    Binary or multi-class precision/recall/F1, macro-averaged
    across whatever labels appear in y_true when positive_label
    is not given.
    """

    labels = sorted(set(_normalize_text(t) for t in y_true))

    if positive_label is not None:
        labels = [_normalize_text(positive_label)]

    per_label = []

    for label in labels:
        tp = sum(
            1
            for t, p in zip(y_true, y_pred)
            if _normalize_text(t) == label and _normalize_text(p) == label
        )

        fp = sum(
            1
            for t, p in zip(y_true, y_pred)
            if _normalize_text(t) != label and _normalize_text(p) == label
        )

        fn = sum(
            1
            for t, p in zip(y_true, y_pred)
            if _normalize_text(t) == label and _normalize_text(p) != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_label.append((precision, recall, f1))

    if not per_label:
        return 0.0, 0.0, 0.0

    precisions, recalls, f1s = zip(*per_label)

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


def confusion_counts(y_true, y_pred, positive_label):
    """
    TP / FP / TN / FN counts for one label treated as "positive".
    Used for binary VQA (yes/no) style questions.
    """

    positive_label = _normalize_text(positive_label)

    tp = fp = tn = fn = 0

    for t, p in zip(y_true, y_pred):
        t, p = _normalize_text(t), _normalize_text(p)

        if t == positive_label and p == positive_label:
            tp += 1
        elif t != positive_label and p == positive_label:
            fp += 1
        elif t != positive_label and p != positive_label:
            tn += 1
        else:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def cohens_kappa(y_true, y_pred, labels):
    """
    Agreement metric that corrects for chance agreement -
    more informative than raw accuracy on imbalanced label sets
    (e.g. mostly "no change" pixels/answers).
    """

    n = len(y_true)

    if n == 0:
        return 0.0

    label_index = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)))

    for t, p in zip(y_true, y_pred):
        t, p = _normalize_text(t), _normalize_text(p)

        if t in label_index and p in label_index:
            matrix[label_index[t], label_index[p]] += 1

    observed_agreement = np.trace(matrix) / n

    row_marginals = matrix.sum(axis=1) / n
    col_marginals = matrix.sum(axis=0) / n

    expected_agreement = float(np.sum(row_marginals * col_marginals))

    if expected_agreement >= 1.0:
        return 1.0

    return float(
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
    )


# ============================================================
# SEGMENTATION / PIXEL-MAP METRICS
# (used for: change detection binary maps, grounding boxes)
# ============================================================


def iou_binary_masks(mask_true, mask_pred):
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    return float(intersection / union) if union else 1.0


def dice_binary_masks(mask_true, mask_pred):
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    denominator = mask_true.sum() + mask_pred.sum()

    return float(2 * intersection / denominator) if denominator else 1.0


def pixel_accuracy(mask_true, mask_pred):
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    return float(np.mean(mask_true == mask_pred))


def iou_boxes(box_true, box_pred):
    """
    IoU between two normalized [x1, y1, x2, y2] boxes, matching
    the "[x1 y1, x2 y2]" format produced by the grounding task in
    data/bigearthnet/vlm_training.json.
    """

    xa1, ya1, xa2, ya2 = box_true
    xb1, yb1, xb2, yb2 = box_pred

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area

    return float(inter_area / union) if union > 0 else 0.0


# ============================================================
# TEXT GENERATION METRICS
# (used for: image captioning)
# ============================================================


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def bleu_n(reference, hypothesis, n=4):
    """
    Standard corpus-free, single-reference BLEU-N with a brevity
    penalty. Reimplemented directly (no nltk dependency) so this
    package has no extra requirements beyond numpy.
    """

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    precisions = []

    for order in range(1, n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + order])
            for i in range(len(ref_tokens) - order + 1)
        )

        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i + order])
            for i in range(len(hyp_tokens) - order + 1)
        )

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        overlap = sum(
            min(count, ref_ngrams.get(gram, 0))
            for gram, count in hyp_ngrams.items()
        )

        precisions.append(overlap / sum(hyp_ngrams.values()))

    if min(precisions) == 0:
        geometric_mean = 0.0
    else:
        geometric_mean = math.exp(
            sum(math.log(p) for p in precisions) / len(precisions)
        )

    brevity_penalty = (
        1.0
        if len(hyp_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    )

    return geometric_mean * brevity_penalty


def rouge_l(reference, hypothesis):
    """
    ROUGE-L F1 based on the longest common subsequence.
    """

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _longest_common_subsequence(ref_tokens, hyp_tokens)

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _longest_common_subsequence(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[-1][-1]


# ============================================================
# IMAGE QUALITY METRICS
# (used for: optical-SAR fusion output)
# ============================================================


def psnr(reference, comparison, data_range=1.0):
    reference = reference.astype(np.float64)
    comparison = comparison.astype(np.float64)

    mse = np.mean((reference - comparison) ** 2)

    if mse == 0:
        return float("inf")

    return float(20 * math.log10(data_range) - 10 * math.log10(mse))


def pearson_correlation(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def ssim(reference, comparison, data_range=1.0):
    """
    Simplified single-scale SSIM (global, not windowed) - no
    scikit-image dependency required. Good enough as a relative
    quality signal between fusion runs; for a windowed/official
    SSIM, swap this for skimage.metrics.structural_similarity if
    that dependency is available in the deployment environment.
    """

    reference = reference.astype(np.float64)
    comparison = comparison.astype(np.float64)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mean_ref = reference.mean()
    mean_comp = comparison.mean()

    var_ref = reference.var()
    var_comp = comparison.var()

    covariance = np.mean(
        (reference - mean_ref) * (comparison - mean_comp)
    )

    numerator = (2 * mean_ref * mean_comp + c1) * (2 * covariance + c2)
    denominator = (
        (mean_ref ** 2 + mean_comp ** 2 + c1)
        * (var_ref + var_comp + c2)
    )

    return float(numerator / denominator) if denominator else 1.0


def shannon_entropy(image):
    """
    Entropy of a normalized [0, 1] image, in bits. Higher entropy
    in a fused output (relative to either input alone) generally
    indicates more information was retained/combined.
    """

    histogram, _ = np.histogram(image, bins=256, range=(0, 1), density=False)
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def _normalize_text(value):
    return str(value).strip().lower()
