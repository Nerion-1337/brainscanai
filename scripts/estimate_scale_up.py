"""Étape 5 : faisabilité du passage à l'échelle (4 M d'images, 5 000 EUR)."""
from __future__ import annotations

import argparse

import pandas as pd

from brainscanai.config import load_config, resolve_path
from brainscanai.scale_up.cost import (
    ScaleUpAssumptions,
    estimate,
    sensitivity,
    validation_sample_size,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ass = ScaleUpAssumptions.from_config(cfg)
    res = estimate(ass)
    print(f"Budget cible par image : {res['budget_per_image']:.5f} EUR "
          f"(pilote : {res['pilot_cost_per_image']:.3f} EUR/image)")
    print(f"Extrapolation naïve du pilote : {res['naive_pilot_extrapolation_eur']:,.0f} EUR -> infaisable")
    print(pd.Series(res["items"]).round(1).to_string())
    print(f"TOTAL : {res['total_eur']:.0f} EUR / {res['budget_eur']:.0f} EUR | faisable : {res['feasible']}")
    print(f"Revue experte : {res['expert_images']} images (min. {res['expert_images_needed_for_validation']} "
          f"pour un IC ±5 %) = {res['expert_share_of_dataset_pct']:.3f} % du jeu")
    print(validation_sample_size())
    tdir = resolve_path(cfg, "tables", create=True)
    pd.Series(res["items"], name="EUR").round(2).to_csv(tdir / "scale_up_budget.csv")
    sensitivity(ass).to_csv(tdir / "scale_up_sensitivity.csv")
    print(sensitivity(ass))


if __name__ == "__main__":
    main()
