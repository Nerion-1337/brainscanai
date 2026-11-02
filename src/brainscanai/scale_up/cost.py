"""Estimation de faisabilité : étiqueter 4 M d'images avec 5 000 EUR.

TOUS les tarifs/débits sont des HYPOTHÈSES paramétrables (configs/config.yaml > scale_up) :
à remplacer par des devis cloud et la grille tarifaire des radiologues avant décision.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class ScaleUpAssumptions:
    n_images: int = 4_000_000
    budget_eur: float = 5000.0
    pilot_images: int = 1500
    pilot_budget_eur: float = 300.0
    gpu_eur_per_hour: float = 0.80
    throughput_img_s: float = 400.0
    avg_image_mb: float = 0.12
    storage_eur_per_gb_month: float = 0.02
    storage_months: float = 2.0
    cpu_preprocessing_eur_per_hour: float = 0.40
    cpu_throughput_img_s: float = 300.0
    expert_review_eur_per_image: float = 3.0
    active_learning_images: int = 1000   # cas à faible confiance envoyés aux radiologues
    retraining_rounds: int = 3
    engineering_overhead_pct: float = 0.15
    contingency_pct: float = 0.20

    @classmethod
    def from_config(cls, cfg: dict) -> ScaleUpAssumptions:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in cfg["scale_up"].items() if k in keys})


def validation_sample_size(expected_recall: float = 0.95, margin: float = 0.05,
                           prevalence: float = 0.5, z: float = 1.96) -> dict:
    """Taille d'échantillon expert pour estimer le rappel avec une marge ± ``margin`` (IC 95 %)."""
    n_pos = math.ceil(z**2 * expected_recall * (1 - expected_recall) / margin**2)
    return dict(n_positives=n_pos, n_images=math.ceil(n_pos / prevalence))


def estimate(a: ScaleUpAssumptions, expert_images: int | None = None) -> dict:
    """Décompose le coût d'un pipeline 100 % automatique + revue experte ciblée.

    Étapes : prétraitement CPU -> embeddings (GPU) -> clustering/pseudo-labels ->
    ``retraining_rounds`` fine-tunings + ré-inférences -> stockage -> revue experte
    (échantillon de validation + cas à faible confiance, dans la limite du budget).
    """
    n = a.n_images
    cpu_h = n / a.cpu_throughput_img_s / 3600
    gpu_passes = 1 + a.retraining_rounds  # embeddings + ré-inférence CNN à chaque tour
    gpu_h = gpu_passes * n / a.throughput_img_s / 3600
    storage_gb = n * a.avg_image_mb / 1024
    items = {
        "Prétraitement CPU (décodage/resize)": cpu_h * a.cpu_preprocessing_eur_per_hour,
        f"Inférence GPU ({gpu_passes} passes)": gpu_h * a.gpu_eur_per_hour,
        "Stockage": storage_gb * a.storage_eur_per_gb_month * a.storage_months,
    }
    infra = sum(items.values())
    items["Ingénierie / orchestration"] = infra * a.engineering_overhead_pct
    tech = sum(items.values())
    items["Aléas"] = tech * a.contingency_pct
    tech_total = sum(items.values())

    remaining = max(a.budget_eur - tech_total, 0.0)
    max_expert = int(remaining // a.expert_review_eur_per_image)
    needed = validation_sample_size()["n_images"]
    if expert_images is None:  # validation statistique + apprentissage actif, dans la limite du budget
        expert_images = min(max_expert, needed + a.active_learning_images)
    expert_cost = expert_images * a.expert_review_eur_per_image
    items["Revue experte"] = expert_cost
    total = tech_total + expert_cost

    naive = a.pilot_budget_eur / a.pilot_images * n
    return dict(
        items=items, total_eur=total, budget_eur=a.budget_eur, margin_eur=a.budget_eur - total,
        feasible=bool(total <= a.budget_eur and expert_images >= needed),
        gpu_hours=gpu_h, cpu_hours=cpu_h, storage_gb=storage_gb,
        budget_per_image=a.budget_eur / n, pilot_cost_per_image=a.pilot_budget_eur / a.pilot_images,
        naive_pilot_extrapolation_eur=naive, expert_images=expert_images,
        expert_images_needed_for_validation=needed,
        expert_share_of_dataset_pct=100 * expert_images / n,
    )


def sensitivity(a: ScaleUpAssumptions, throughputs=(100, 200, 400, 800), gpu_prices=(0.4, 0.8, 1.6, 3.0)) -> pd.DataFrame:
    """Coût technique total selon débit GPU et prix horaire (pire cas inclus)."""
    rows = {}
    for p in gpu_prices:
        rows[f"{p:.1f} EUR/h"] = {}
        for t in throughputs:
            d = asdict(a) | dict(throughput_img_s=t, gpu_eur_per_hour=p)
            res = estimate(ScaleUpAssumptions(**d), expert_images=0)
            rows[f"{p:.1f} EUR/h"][f"{t} img/s"] = round(res["total_eur"], 1)
    return pd.DataFrame(rows).T
