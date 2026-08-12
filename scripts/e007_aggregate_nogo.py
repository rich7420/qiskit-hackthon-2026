"""Aggregate the e007 directional NO-GO evidence into a summary JSON + CSV.

Combines: (1) global-QFI and CFI diagonal effective ranks at a trained MNIST operating
point, (2) the 5-seed decisive per-seed directional coefficients, (3) the actual-Bures
rescue, (4) the R-vs-Bures validation. Reads results/e007_decisive_seed*.json,
results/e007_bures_rescue.json, results/e007_bures.json.

Run:
    python scripts/e007_aggregate_nogo.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.e005_softmax import bce_loss, classical_fisher_diag, make_softmax_qnode  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"


def _eff_rank(v) -> float:
    v = np.clip(np.asarray(v, float), 0, None)
    return float((v.sum() ** 2) / np.sum(v**2))


def spectrum(seed: int = 42) -> dict:
    t1, _ = load_two_tasks(n_features=16, n_train=400, n_test=200, seed=seed)
    clf, ws = make_softmax_qnode(4, 20)
    qfi_q, _ = make_qnode(4, 20)
    w = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(ws), requires_grad=True)
    X = pnp.array(t1.X_train, requires_grad=False)
    y = pnp.array(t1.y_train, requires_grad=False)
    opt = qml.AdamOptimizer(0.02)
    for _ in range(20):
        w = opt.step(lambda W: bce_loss(clf, W, X, y), w)
    qfi = quantum_fisher_diag(qfi_q, w, t1.X_train, n_samples=48, seed=seed)
    cfi = classical_fisher_diag(clf, w, t1.X_train, t1.y_train)
    return {"n_weights": int(qfi.size),
            "global_qfi_effective_rank": round(_eff_rank(qfi), 1),
            "global_qfi_max_over_min": round(float(qfi.max() / max(qfi.min(), 1e-12)), 2),
            "cfi_effective_rank": round(_eff_rank(cfi), 1),
            "cfi_max_over_min": float(f"{cfi.max()/max(cfi.min(),1e-30):.2e}")}


def main() -> None:
    files = sorted(glob.glob(str(RESULTS / "e007_decisive_seed*.json")))
    seeds = [json.load(open(f)) for f in files]
    keys = ("Q_global", "Q_readout", "Q_cfi")
    per_seed = []
    agg = {k: {"beta2": [], "dR2": []} for k in keys}
    for d in seeds:
        row = {"seed": d["seed"], "R2_step": d["R2_forgetting_on_stepsize"]}
        for k in keys:
            b = d["incremental_over_stepsize"][k]["beta_directional"]
            r = d["incremental_over_stepsize"][k]["delta_R2_over_stepsize"]
            row[f"{k}_beta2"] = b
            row[f"{k}_dR2"] = r
            agg[k]["beta2"].append(b)
            agg[k]["dR2"].append(r)
        per_seed.append(row)

    def sign_summary(vals):
        v = np.array(vals)
        return {"n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum()),
                "consistent": bool((v > 0).all() or (v < 0).all())}

    directional = {k: {"beta2_signs": sign_summary(agg[k]["beta2"]),
                       "mean_dR2": round(float(np.mean(agg[k]["dR2"])), 4),
                       "std_dR2": round(float(np.std(agg[k]["dR2"])), 4)} for k in keys}

    bures = json.load(open(RESULTS / "e007_bures_rescue.json"))
    rval = json.load(open(RESULTS / "e007_bures.json"))

    summary = {
        "experiment": "e007_directional_nogo",
        "verdict": "NO-GO: forgetting is step-magnitude (radial), not quantum-geometric (directional)",
        "evidence_1_isotropy": spectrum(),
        "evidence_2_directional_multiseed": {"n_seeds": len(seeds), "per_task": directional,
                                             "note": "mixed-sign beta2 across seeds => noise, not signal"},
        "evidence_3_exact_bures_vs_euclid": {
            "corr_euclid": bures["corr_with_forgetting"]["euclid_step"],
            "corr_bures_global": bures["corr_with_forgetting"]["bures_global"],
            "corr_bures_readout": bures["corr_with_forgetting"]["bures_readout"],
            "R2_on_euclid": bures["R2_forgetting_on_euclid"],
            "incremental_R2": bures["incremental_R2_over_euclid"],
            "note": "exact state distance does not beat Euclidean parameter movement"},
        "sanity_R_measures_state_drift": {"pearson_pred_vs_bures": rval["pearson_pred_vs_actual"],
                                          "slope": rval["slope_overall"]},
        "per_seed": per_seed,
    }
    (RESULTS / "e007_nogo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with open(RESULTS / "e007_decisive_multiseed.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        wr.writeheader()
        wr.writerows(per_seed)

    print("Wrote results/e007_nogo_summary.json and results/e007_decisive_multiseed.csv")
    print(f"  global QFI eff-rank: {summary['evidence_1_isotropy']['global_qfi_effective_rank']}"
          f"/{summary['evidence_1_isotropy']['n_weights']} (isotropic)")
    for k in keys:
        print(f"  {k}: beta2 consistent={directional[k]['beta2_signs']['consistent']} "
              f"({directional[k]['beta2_signs']['n_pos']}+/{directional[k]['beta2_signs']['n_neg']}-) "
              f"mean_dR2={directional[k]['mean_dR2']}")


if __name__ == "__main__":
    main()
