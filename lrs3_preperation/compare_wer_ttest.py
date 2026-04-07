#!/usr/bin/env python3
"""
Dependency list:
- Python 3.8+
- scipy
- matplotlib (optional, only needed if --plot-diff-hist is used)

Compare ASR decode JSON outputs from a baseline and an improved model.
The expected JSON structure is a dictionary of parallel lists:
{
  "utt_id": ["utt1", "utt2", ...],
  "ref":    ["reference text 1", ...],
  "hypo":   ["hypothesis text 1", ...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "scipy is required for statistical testing. Install with: pip install scipy"
    ) from exc


@dataclass
class UtteranceScore:
    utt_id: str
    ref: str
    hypo: str
    n_ref_words: int
    subs: int
    dels: int
    ins: int
    wer: float


def normalize_text(text: str, remove_punct: bool = False) -> str:
    """Apply configurable text normalization in one place."""
    if text is None:
        text = ""
    text = str(text).lower().strip()
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
    text = re.sub(r"\s+", " ", text)
    return text


def validate_and_convert_decode_json(path: str, remove_punct: bool) -> Dict[str, Tuple[str, str]]:
    """
    Load a decode JSON and return mapping: utt_id -> (normalized_ref, normalized_hypo).

    Raises ValueError with a helpful message if the JSON structure is invalid.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(data).__name__}")

    required_keys = ["utt_id", "ref", "hypo"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing keys in {path}: {missing}")

    utt_ids = data["utt_id"]
    refs = data["ref"]
    hypos = data["hypo"]

    if not (isinstance(utt_ids, list) and isinstance(refs, list) and isinstance(hypos, list)):
        raise ValueError(
            f"In {path}, 'utt_id', 'ref', and 'hypo' must all be lists (parallel arrays)."
        )

    if not (len(utt_ids) == len(refs) == len(hypos)):
        raise ValueError(
            f"List lengths mismatch in {path}: "
            f"len(utt_id)={len(utt_ids)}, len(ref)={len(refs)}, len(hypo)={len(hypos)}"
        )

    out: Dict[str, Tuple[str, str]] = {}
    dup_count = 0
    for i, (utt_id, ref, hypo) in enumerate(zip(utt_ids, refs, hypos)):
        if utt_id is None or str(utt_id).strip() == "":
            raise ValueError(f"Empty utt_id at index {i} in {path}")

        utt = str(utt_id)
        if utt in out:
            dup_count += 1
        out[utt] = (normalize_text(ref, remove_punct), normalize_text(hypo, remove_punct))

    if dup_count > 0:
        print(
            f"Warning: {dup_count} duplicate utt_id entries detected in {path}. "
            "Last occurrence was kept."
        )

    return out


def levenshtein_alignment_counts(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int, int]:
    """
    Compute word-level Levenshtein alignment and return (subs, dels, ins).

    Uses dynamic programming + backtrace to recover operation counts.
    """
    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        rw = ref_words[i - 1]
        for j in range(1, m + 1):
            hw = hyp_words[j - 1]
            if rw == hw:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                dp[i][j] = min(sub, dele, ins)

    i, j = n, m
    subs = dels = ins = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue

        moved = False
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
            moved = True
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
            moved = True
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
            moved = True

        if not moved:
            if i > 0 and j > 0:
                i -= 1
                j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1

    return subs, dels, ins


def compute_scores(utt_map: Dict[str, Tuple[str, str]]) -> Dict[str, UtteranceScore]:
    """Compute per-utterance WER stats for one system."""
    scores: Dict[str, UtteranceScore] = {}

    for utt_id, (ref, hypo) in utt_map.items():
        ref_words = ref.split() if ref else []
        hyp_words = hypo.split() if hypo else []
        subs, dels, ins = levenshtein_alignment_counts(ref_words, hyp_words)
        n = len(ref_words)

        if n == 0:
            wer = 0.0 if len(hyp_words) == 0 else 1.0
        else:
            wer = (subs + dels + ins) / n

        scores[utt_id] = UtteranceScore(
            utt_id=utt_id,
            ref=ref,
            hypo=hypo,
            n_ref_words=n,
            subs=subs,
            dels=dels,
            ins=ins,
            wer=wer,
        )

    return scores


def write_system_csv(path: str, scores: Dict[str, UtteranceScore]) -> None:
    """Write per-utterance metrics for one system."""
    rows = sorted(scores.values(), key=lambda x: x.utt_id)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "ref", "hypo", "n_ref_words", "subs", "dels", "ins", "wer"])
        for r in rows:
            writer.writerow([r.utt_id, r.ref, r.hypo, r.n_ref_words, r.subs, r.dels, r.ins, f"{r.wer:.6f}"])


def write_missing_report_csv(
    path: str, only_in_baseline: List[str], only_in_model: List[str]
) -> None:
    """Write missing-utterance report for both directions."""
    max_len = max(len(only_in_baseline), len(only_in_model))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["missing_in_model_only_in_baseline", "missing_in_baseline_only_in_model"])
        for i in range(max_len):
            b = only_in_baseline[i] if i < len(only_in_baseline) else ""
            m = only_in_model[i] if i < len(only_in_model) else ""
            writer.writerow([b, m])


def write_comparison_csv(
    path: str,
    baseline_scores: Dict[str, UtteranceScore],
    model_scores: Dict[str, UtteranceScore],
    paired_ids: List[str],
) -> None:
    """Write merged baseline-vs-model utterance comparison CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "utt_id",
                "baseline_wer",
                "model_wer",
                "wer_diff",
                "baseline_subs",
                "baseline_dels",
                "baseline_ins",
                "model_subs",
                "model_dels",
                "model_ins",
            ]
        )

        for utt_id in paired_ids:
            b = baseline_scores[utt_id]
            m = model_scores[utt_id]
            diff = b.wer - m.wer
            writer.writerow(
                [
                    utt_id,
                    f"{b.wer:.6f}",
                    f"{m.wer:.6f}",
                    f"{diff:.6f}",
                    b.subs,
                    b.dels,
                    b.ins,
                    m.subs,
                    m.dels,
                    m.ins,
                ]
            )


def corpus_wer(scores: Dict[str, UtteranceScore]) -> Tuple[int, int, int, int, float]:
    """Compute corpus-level WER = total(S+D+I)/total_ref_words."""
    total_s = sum(x.subs for x in scores.values())
    total_d = sum(x.dels for x in scores.values())
    total_i = sum(x.ins for x in scores.values())
    total_n = sum(x.n_ref_words for x in scores.values())

    if total_n == 0:
        wer = math.nan
    else:
        wer = (total_s + total_d + total_i) / total_n

    return total_s, total_d, total_i, total_n, wer


def run_paired_ttest(baseline_wers: List[float], model_wers: List[float]) -> Tuple[float, float, float]:
    """
    Paired t-test on utterance WER with one-sided alternative: model < baseline.

    Returns: (t_statistic, p_one_sided, p_two_sided)
    """
    if len(baseline_wers) != len(model_wers):
        raise ValueError("Paired t-test requires equal-length vectors")
    if len(baseline_wers) < 2:
        raise ValueError("Need at least 2 paired utterances for a t-test")

    try:
        res_one = stats.ttest_rel(baseline_wers, model_wers, alternative="greater")
        t_stat = float(res_one.statistic)
        p_one = float(res_one.pvalue)

        res_two = stats.ttest_rel(baseline_wers, model_wers, alternative="two-sided")
        p_two = float(res_two.pvalue)
    except TypeError:
        # Fallback for older SciPy: derive one-sided p-value from two-sided output.
        res_two = stats.ttest_rel(baseline_wers, model_wers)
        t_stat = float(res_two.statistic)
        p_two = float(res_two.pvalue)
        if t_stat > 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - (p_two / 2.0)

    return t_stat, p_one, p_two


def run_wilcoxon_if_requested(
    baseline_wers: List[float], model_wers: List[float], enabled: bool
) -> Tuple[float, float] | None:
    """Optional Wilcoxon signed-rank test (one-sided: baseline > model)."""
    if not enabled:
        return None

    try:
        res = stats.wilcoxon(baseline_wers, model_wers, alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except Exception as exc:
        print(f"Warning: Wilcoxon test failed: {exc}")
        return None


def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute percentile with linear interpolation; p is in [0, 1]."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if p <= 0.0:
        return sorted_values[0]
    if p >= 1.0:
        return sorted_values[-1]

    pos = (len(sorted_values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def run_bootstrap_mean_ci_if_requested(
    diffs: List[float],
    enabled: bool,
    n_samples: int,
    ci: float,
    seed: int | None,
) -> Tuple[float, float, float] | None:
    """
    Optional paired bootstrap CI for mean( baseline_wer - model_wer ).

    Returns: (bootstrap_mean_of_means, ci_low, ci_high)
    """
    if not enabled:
        return None
    if len(diffs) == 0:
        print("Warning: No paired utterances available; bootstrap CI not computed.")
        return None

    rng = random.Random(seed)
    n = len(diffs)
    means: List[float] = []

    for _ in range(n_samples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - ci
    low = _percentile(means, alpha / 2.0)
    high = _percentile(means, 1.0 - alpha / 2.0)
    mean_of_means = sum(means) / len(means)
    return mean_of_means, low, high


def save_histogram_if_requested(diffs: List[float], out_png: str, enabled: bool) -> None:
    """Optional histogram of baseline_wer - model_wer per utterance."""
    if not enabled:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: Could not import matplotlib; histogram not saved: {exc}")
        return

    if not diffs:
        print("Warning: No paired diffs available; histogram not saved.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(diffs, bins=40)
    plt.axvline(0.0, linestyle="--")
    plt.title("Per-Utterance WER Difference (baseline - model)")
    plt.xlabel("WER difference")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance-level WER between baseline and improved ASR decode JSON files."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline decode JSON")
    parser.add_argument("--model", required=True, help="Path to improved-model decode JSON")
    parser.add_argument("--outdir", required=True, help="Directory to store CSV outputs/reports")

    parser.add_argument(
        "--remove-punct",
        action="store_true",
        help="Remove punctuation during normalization (default: keep punctuation).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for one-sided paired t-test (default: 0.05).",
    )
    parser.add_argument(
        "--wilcoxon",
        action="store_true",
        help="Also run a one-sided Wilcoxon signed-rank test.",
    )
    parser.add_argument(
        "--plot-diff-hist",
        action="store_true",
        help="Save histogram of per-utterance WER differences (baseline - model).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Also compute paired bootstrap CI for mean WER difference (baseline - model).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of paired bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence level in (0,1), e.g. 0.95 (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Optional random seed for bootstrap reproducibility.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.alpha <= 0.0 or args.alpha >= 1.0:
        print("Error: --alpha must be in (0, 1).")
        return 2
    if args.bootstrap_samples < 100:
        print("Error: --bootstrap-samples must be >= 100.")
        return 2
    if args.bootstrap_ci <= 0.0 or args.bootstrap_ci >= 1.0:
        print("Error: --bootstrap-ci must be in (0, 1).")
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    try:
        baseline_map = validate_and_convert_decode_json(args.baseline, args.remove_punct)
        model_map = validate_and_convert_decode_json(args.model, args.remove_punct)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading inputs: {exc}")
        return 2

    baseline_scores = compute_scores(baseline_map)
    model_scores = compute_scores(model_map)

    baseline_ids = set(baseline_scores.keys())
    model_ids = set(model_scores.keys())

    paired_ids = sorted(baseline_ids & model_ids)
    only_in_baseline = sorted(baseline_ids - model_ids)
    only_in_model = sorted(model_ids - baseline_ids)

    baseline_csv = os.path.join(args.outdir, "baseline_per_utt_wer.csv")
    model_csv = os.path.join(args.outdir, "model_per_utt_wer.csv")
    comp_csv = os.path.join(args.outdir, "baseline_vs_model_comparison.csv")
    missing_csv = os.path.join(args.outdir, "missing_utterances_report.csv")
    hist_png = os.path.join(args.outdir, "wer_diff_histogram.png")

    write_system_csv(baseline_csv, baseline_scores)
    write_system_csv(model_csv, model_scores)
    write_comparison_csv(comp_csv, baseline_scores, model_scores, paired_ids)
    write_missing_report_csv(missing_csv, only_in_baseline, only_in_model)

    print("=== ASR WER Comparison ===")
    print(f"Baseline file: {args.baseline}")
    print(f"Model file:    {args.model}")
    print(f"Output dir:    {args.outdir}")
    print(f"Normalization: lowercase + trim + collapse spaces + remove_punct={args.remove_punct}")
    print()

    print(f"Utterances in baseline: {len(baseline_ids)}")
    print(f"Utterances in model:    {len(model_ids)}")
    print(f"Paired utterances used: {len(paired_ids)}")
    print(f"Missing in model:       {len(only_in_baseline)}")
    print(f"Missing in baseline:    {len(only_in_model)}")

    if only_in_baseline:
        print("Sample missing in model:", ", ".join(only_in_baseline[:10]))
    if only_in_model:
        print("Sample missing in baseline:", ", ".join(only_in_model[:10]))
    print()

    b_s, b_d, b_i, b_n, b_cwer = corpus_wer(baseline_scores)
    m_s, m_d, m_i, m_n, m_cwer = corpus_wer(model_scores)

    print("Corpus-level WER:")
    print(f"Baseline: S={b_s}, D={b_d}, I={b_i}, N={b_n}, WER={b_cwer:.6f}" if not math.isnan(b_cwer) else
          f"Baseline: S={b_s}, D={b_d}, I={b_i}, N={b_n}, WER=nan")
    print(f"Model:    S={m_s}, D={m_d}, I={m_i}, N={m_n}, WER={m_cwer:.6f}" if not math.isnan(m_cwer) else
          f"Model:    S={m_s}, D={m_d}, I={m_i}, N={m_n}, WER=nan")
    print()

    if len(paired_ids) < 2:
        print("Not enough paired utterances for paired significance testing (need at least 2).")
        print("Per-utterance CSV files were still saved.")
        return 0

    baseline_wers = [baseline_scores[u].wer for u in paired_ids]
    model_wers = [model_scores[u].wer for u in paired_ids]
    diffs = [b - m for b, m in zip(baseline_wers, model_wers)]

    mean_b = sum(baseline_wers) / len(baseline_wers)
    mean_m = sum(model_wers) / len(model_wers)
    mean_diff = sum(diffs) / len(diffs)

    try:
        t_stat, p_one, p_two = run_paired_ttest(baseline_wers, model_wers)
    except ValueError as exc:
        print(f"Could not run paired t-test: {exc}")
        return 1

    print("Paired utterance-level WER statistics:")
    print(f"Mean baseline utterance WER: {mean_b:.6f}")
    print(f"Mean model utterance WER:    {mean_m:.6f}")
    print(f"Mean difference (B - M):     {mean_diff:.6f}")
    print(f"Paired t-test t-statistic:   {t_stat:.6f}")
    print(f"One-sided p-value:           {p_one:.6g}")
    print(f"Two-sided p-value:           {p_two:.6g}")

    if p_one < args.alpha:
        print(
            f"Result: Significant at alpha={args.alpha:.3f}. "
            "Reject H0 in favor of model better than baseline."
        )
    else:
        print(
            f"Result: Not significant at alpha={args.alpha:.3f}. "
            "Do not reject H0 for model improvement."
        )

    wilcoxon_res = run_wilcoxon_if_requested(baseline_wers, model_wers, args.wilcoxon)
    if wilcoxon_res is not None:
        w_stat, w_p = wilcoxon_res
        print()
        print("Wilcoxon signed-rank test (one-sided, baseline > model):")
        print(f"W-statistic: {w_stat:.6f}")
        print(f"p-value:     {w_p:.6g}")

    bootstrap_res = run_bootstrap_mean_ci_if_requested(
        diffs,
        enabled=args.bootstrap,
        n_samples=args.bootstrap_samples,
        ci=args.bootstrap_ci,
        seed=args.bootstrap_seed,
    )
    if bootstrap_res is not None:
        b_mean, ci_low, ci_high = bootstrap_res
        print()
        print("Paired bootstrap CI for mean difference (B - M):")
        print(f"Bootstrap samples: {args.bootstrap_samples}")
        print(f"Bootstrap mean:    {b_mean:.6f}")
        print(f"{args.bootstrap_ci * 100:.1f}% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    save_histogram_if_requested(diffs, hist_png, args.plot_diff_hist)

    print()
    print("Saved files:")
    print(f"- {baseline_csv}")
    print(f"- {model_csv}")
    print(f"- {comp_csv}")
    print(f"- {missing_csv}")
    if args.plot_diff_hist:
        print(f"- {hist_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
