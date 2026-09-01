#!/usr/bin/env python3
"""Run the real traffic evaluation from the CSV files.

The 26 files in ``data/runtime_aces/`` hold real IoT traffic flows, one row
per flow, already converted to compact ACE text (the ``runtime_ace`` column).
This script matches those flows against the 28 reference MUD profiles with
four methods and runs two evaluations:

  cumulative   Identification as flows accumulate in temporal order,
               including the rank distribution over the first 10000 flows.
  windows      Identification from disjoint 50-flow windows (9023 windows
               across 25 devices), binned by exact-overlap score.

Both experiments need per-flow embeddings. Build them once with:

  embed        Embed every distinct runtime ACE text with BGE-M3, whiten
               with the transform fitted on the raw reference bank, and save
               a local runtime embedding bank.

The input handling lives in ``realtraffic/data.py`` and the matching methods
in ``realtraffic/scoring.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from realtraffic.data import (
    DEFAULT_COMPACT_DIR,
    DEFAULT_RUNTIME_DIR,
    read_traces,
    seed_index,
)
from realtraffic.scoring import (
    CUMULATIVE_TIE_EPSILON,
    DEFAULT_RAW_NPZ,
    DEFAULT_RUNTIME_NPZ,
    METHOD_ORDER,
    RANK_TIE_EPSILON,
    WHITEN_K,
    WINDOW_TIE_EPSILON,
    apply_whitening,
    build_score_tables,
    conservative_rank,
    cumulative_scores,
    l2_normalise,
    load_reference_bank,
    per_flow_matrix,
    prepare,
    unique_leader_correct,
    window_score_vectors,
    window_starts,
)


def run_embed(args: argparse.Namespace) -> None:
    from gen_emb import build_model, encode_texts

    traces = read_traces(args.runtime_dir)
    distinct = sorted({text for trace in traces for text in trace.texts})
    whitened_ref, _, ref_texts, transform = load_reference_bank(args.raw_npz)

    vector_by_text: dict[str, np.ndarray] = {}
    for idx, text in enumerate(ref_texts):
        vector_by_text.setdefault(text, whitened_ref[idx])
    reused = [text for text in distinct if text in vector_by_text]
    new_texts = [text for text in distinct if text not in vector_by_text]

    print(f"{len(distinct)} distinct runtime ACE texts from {len(traces)} traces.")
    print(f"{len(reused)} match a reference ACE; embedding {len(new_texts)} new texts...")
    model = build_model(args.model_name, args.device)
    raw = encode_texts(model, new_texts, args.batch_size)
    whitened_new = l2_normalise(
        apply_whitening(
            raw,
            transform["mean"],
            transform["components"],
            transform["singular_values"],
            WHITEN_K,
            transform["n_reference"],
        )
    )
    vector_by_text.update(zip(new_texts, whitened_new))

    embeddings = np.vstack([vector_by_text[text] for text in distinct])
    args.runtime_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.runtime_npz,
        texts=np.asarray(distinct, dtype=object),
        embeddings=embeddings.astype(np.float32),
        model_name=np.asarray(args.model_name),
        whiten_k=np.asarray(WHITEN_K),
    )
    print(f"Saved {args.runtime_npz} ({embeddings.shape[0]}x{embeddings.shape[1]}).")


def run_cumulative(args: argparse.Namespace) -> None:
    traces, devices, ref_features, ref_rows, ref_signatures, vectors = prepare(args)
    checkpoints = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 50000]
    correct_at = {m: {} for m in METHOD_ORDER}
    final_correct = {m: 0 for m in METHOD_ORDER}
    active_at = {}
    rank_counts = {m: np.zeros(len(devices) + 1, dtype=np.int64) for m in METHOD_ORDER}
    rank_queries = 0
    convergence_rows = []

    for trace in traces:
        tables = build_score_tables(
            trace, devices, ref_features, ref_rows, ref_signatures, vectors
        )
        gt_col = devices.index(trace.gt_mud)
        cumulative = cumulative_scores(trace, tables)
        n = len(trace.texts)
        cap = min(n, 10000)
        rank_queries += cap
        for method, scores in cumulative.items():
            correct = unique_leader_correct(scores, gt_col, CUMULATIVE_TIE_EPSILON)
            ranks = conservative_rank(scores[:cap], gt_col, RANK_TIE_EPSILON)
            counts = np.bincount(
                np.minimum(ranks, len(devices)), minlength=len(devices) + 1
            )
            rank_counts[method] += counts
            final_correct[method] += int(correct[n - 1])
            for k in checkpoints:
                if k <= n:
                    correct_at[method].setdefault(k, 0)
                    correct_at[method][k] += int(correct[k - 1])
            if args.full_curve_output:
                for i in range(n):
                    convergence_rows.append(
                        (method, trace.device_name, i + 1, int(correct[i]))
                    )
        for k in checkpoints:
            if k <= n:
                active_at[k] = active_at.get(k, 0) + 1

    print("\nTop-1 correct traces as flows accumulate (correct/active):")
    header = "flows".rjust(7) + "".join(m.rjust(18) for m in METHOD_ORDER)
    print(header)
    for k in checkpoints:
        if k not in active_at:
            continue
        row = f"{k:7d}"
        for method in METHOD_ORDER:
            row += f"{correct_at[method][k]:>13d}/{active_at[k]:<4d}"
        print(row)
    row = "  final"
    for method in METHOD_ORDER:
        row += f"{final_correct[method]:>13d}/{len(traces):<4d}"
    print(row + "   (each trace at its last flow)")

    print("\nRank of the correct device over the first 10000 flows per trace:")
    print("method             top1    top3    top5")
    summary_rank = {}
    for method in METHOD_ORDER:
        counts = rank_counts[method]
        top1 = counts[1] / rank_queries
        top3 = counts[1:4].sum() / rank_queries
        top5 = counts[1:6].sum() / rank_queries
        summary_rank[method] = {"top1": top1, "top3": top3, "top5": top5}
        print(f"{method:<17s}{top1:8.4f}{top3:8.4f}{top5:8.4f}")

    if args.output:
        payload = {
            "checkpoints": {
                str(k): {
                    "active": active_at[k],
                    **{m: correct_at[m][k] for m in METHOD_ORDER},
                }
                for k in checkpoints
                if k in active_at
            },
            "final": {
                "active": len(traces),
                **{m: final_correct[m] for m in METHOD_ORDER},
            },
            "rank_distribution": {
                "queries": rank_queries,
                **{m: summary_rank[m] for m in METHOD_ORDER},
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Saved {args.output}.")
    if args.full_curve_output:
        args.full_curve_output.parent.mkdir(parents=True, exist_ok=True)
        with args.full_curve_output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "device_name", "flow_idx", "correct"])
            writer.writerows(convergence_rows)
        print(f"Saved {args.full_curve_output}.")


def run_windows(args: argparse.Namespace) -> None:
    traces, devices, ref_features, ref_rows, ref_signatures, vectors = prepare(args)
    seed_by_device = seed_index(traces)
    rows = []
    for trace in traces:
        tables = build_score_tables(
            trace, devices, ref_features, ref_rows, ref_signatures, vectors
        )
        gt_col = devices.index(trace.gt_mud)
        starts = window_starts(
            len(trace.texts),
            args.window_size,
            args.windows_per_device,
            args.seed + seed_by_device[trace.device_name] * 9973,
        )
        if len(starts) == 0:
            print(
                f"Skipping {trace.device_name}: fewer than "
                f"{args.window_size} flows."
            )
            continue
        exact_flow = per_flow_matrix(trace, tables.exact, tables.text_index)
        maxsim_flow = per_flow_matrix(trace, tables.maxsim, tables.text_index)
        vec_flow = per_flow_matrix(trace, tables.vectors, tables.text_index)
        for start in starts:
            end = int(start + args.window_size)
            window_scores = window_score_vectors(
                trace, tables, exact_flow, maxsim_flow, vec_flow, int(start), end
            )
            record = {"device": trace.device_name}
            for method, scores in window_scores.items():
                record[f"{method}_correct"] = bool(
                    unique_leader_correct(
                        scores[None, :], gt_col, WINDOW_TIE_EPSILON
                    )[0]
                )
            record["exact_top_score"] = float(window_scores["exact_hit_count"].max())
            record["exact_gt_score"] = float(window_scores["exact_hit_count"][gt_col])
            rows.append(record)

    subsets = [
        ("All windows", lambda r: True),
        ("Exact-overlap < 0.50", lambda r: r["exact_top_score"] < 0.50),
        ("Exact-overlap < 0.10", lambda r: r["exact_gt_score"] < 0.10),
        ("Exact-overlap = 0", lambda r: r["exact_gt_score"] <= WINDOW_TIE_EPSILON),
    ]
    print(f"\n{len(rows)} windows across {len({r['device'] for r in rows})} devices.")
    print(f"{'Window set':<22s}{'Windows':>9s}{'Devices':>9s}"
          f"{'Exact':>9s}{'MeanPool':>10s}{'MaxSim':>9s}")
    table_payload = []
    for label, keep in subsets:
        subset = [r for r in rows if keep(r)]
        n_dev = len({r["device"] for r in subset})
        counts = {
            m: sum(r[f"{m}_correct"] for r in subset)
            for m in ("exact_hit_count", "mean_pool", "maxsim")
        }
        print(
            f"{label:<22s}{len(subset):>9d}{n_dev:>9d}"
            f"{counts['exact_hit_count']:>9d}{counts['mean_pool']:>10d}"
            f"{counts['maxsim']:>9d}"
        )
        table_payload.append(
            {"window_set": label, "windows": len(subset), "devices": n_dev, **counts}
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table_payload, indent=2))
        print(f"Saved {args.output}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--compact-dir", type=Path, default=DEFAULT_COMPACT_DIR)
    parser.add_argument("--raw-npz", type=Path, default=DEFAULT_RAW_NPZ,
                        help="Raw per-ACE reference bank used for whitening.")
    parser.add_argument("--runtime-npz", type=Path, default=DEFAULT_RUNTIME_NPZ,
                        help="Runtime embedding bank (written by 'embed').")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="Embed the distinct runtime ACE texts.")
    embed.add_argument("--model-name", default="BAAI/bge-m3")
    embed.add_argument("--device", default=None)
    embed.add_argument("--batch-size", type=int, default=32)

    cumulative = sub.add_parser("cumulative", help="Evidence-accumulation evaluation.")
    cumulative.add_argument("--output", type=Path, default=None)
    cumulative.add_argument("--full-curve-output", type=Path, default=None,
                            help="Optional per-flow correctness CSV (large).")

    windows = sub.add_parser("windows", help="Short-window evaluation.")
    windows.add_argument("--window-size", type=int, default=50)
    windows.add_argument("--windows-per-device", type=int, default=500)
    windows.add_argument("--seed", type=int, default=1729)
    windows.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "embed":
        run_embed(args)
    elif args.command == "cumulative":
        run_cumulative(args)
    else:
        run_windows(args)


if __name__ == "__main__":
    main()
