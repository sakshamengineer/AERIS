// Human-friendly label + formatting overrides for evaluation metric
// keys returned by the backend's /models/metrics endpoint. Falls
// back to a generic "snake_case -> Title Case" prettifier for any
// key we haven't explicitly labeled, so new metrics added on the
// backend still render sensibly without a frontend change.

const METRIC_LABELS = {
    pixel_iou: "Pixel IoU",
    pixel_dice: "Pixel Dice",
    pixel_accuracy: "Pixel Accuracy",
    scene_level_precision: "Scene Precision",
    scene_level_recall: "Scene Recall",
    scene_level_f1: "Scene F1",
    cohens_kappa: "Cohen's Kappa",
    mean_model_confidence: "Mean Confidence",
    mean_correlation_with_optical: "Correlation vs Optical",
    mean_correlation_with_sar: "Correlation vs SAR",
    mean_ssim_with_optical: "SSIM vs Optical",
    mean_entropy_gain_bits: "Entropy Gain",
    mean_inference_time_sec: "Avg Inference Time",
    vqa_accuracy: "VQA Accuracy",
    vqa_macro_f1: "VQA Macro F1",
    grounding_mean_iou: "Grounding Mean IoU",
    captioning_bleu4: "Caption BLEU-4",
    captioning_rouge_l: "Caption ROUGE-L"
}

const RATIO_KEYS = new Set([
    "pixel_iou", "pixel_dice", "pixel_accuracy",
    "scene_level_precision", "scene_level_recall", "scene_level_f1",
    "cohens_kappa", "mean_model_confidence",
    "mean_correlation_with_optical", "mean_correlation_with_sar",
    "mean_ssim_with_optical",
    "vqa_accuracy", "vqa_macro_f1", "grounding_mean_iou",
    "captioning_bleu4", "captioning_rouge_l"
])

const TIME_KEYS = new Set(["mean_inference_time_sec"])

export function formatMetricLabel(key) {
    if (METRIC_LABELS[key]) {
        return METRIC_LABELS[key]
    }

    return key
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
}

export function formatMetricValue(key, value) {
    if (value === null || value === undefined) {
        return "—"
    }

    if (TIME_KEYS.has(key)) {
        return `${value.toFixed(2)}s`
    }

    if (RATIO_KEYS.has(key)) {
        // Cohen's kappa and entropy-style signed metrics can go
        // negative/above 1 in edge cases - only render as a percent
        // when it's a clean 0-1 ratio, otherwise show the raw number.
        if (value >= 0 && value <= 1) {
            return `${(value * 100).toFixed(1)}%`
        }

        return value.toFixed(3)
    }

    if (typeof value === "number") {
        return Number.isInteger(value) ? value.toString() : value.toFixed(3)
    }

    return String(value)
}
