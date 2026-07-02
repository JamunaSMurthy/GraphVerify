"""
Fits a temperature calibrator on a validation split's predictions
and saves the calibrator for use during inference.

Usage:
  python calibrate.py \\
      --pred_dir output/predictions/hotpotqa \\
      --dataset hotpotqa \\
      --split validation \\
      --seed 0 \\
      --output_dir output/calibrators
"""
from __future__ import annotations

import argparse
import os

from graphverify.calibrator import TemperatureCalibrator
from evaluate import load_predictions, aggregate_claim_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",    type=str, required=True)
    p.add_argument("--dataset",     type=str, required=True)
    p.add_argument("--split",       type=str, default="validation")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--output_dir",  type=str, default="output/calibrators")
    p.add_argument("--n_bins",      type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()

    preds = load_predictions(args.pred_dir, args.dataset, args.split, args.seed)
    pv, gv, _, _, rel_scores = aggregate_claim_labels(preds)

    if not rel_scores:
        print("No reliability scores found.")
        return

    labels     = [1 if p == g else 0 for p, g in zip(pv, gv)]
    calibrator = TemperatureCalibrator(n_bins=args.n_bins)
    result     = calibrator.fit(rel_scores, labels)

    print(f"Temperature: {result.temperature:.4f}")
    print(f"ECE before:  {result.ece_before:.4f}")
    print(f"ECE after:   {result.ece_after:.4f}")
    print(f"N samples:   {result.n_samples}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_seed{args.seed}.json")
    calibrator.save(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
