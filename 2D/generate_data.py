"""Data generation: coefficient patches and their local labels.

For a patch kappa (16 x 16 cells), the exact discrete kappa-harmonic extensions H of the 64
nodal boundary traces are split into the Coons lift E and a zero-trace correction,
W64 = H - E (225 x 64 interior values). The constant trace is the only null mode of W64, so
Range(W64) has dimension 63. The label is an energy-orthonormal basis Q63
(225 x 63) of Range(W64). Each label costs one Cholesky factorization on the patch; no global
solve is involved. Each patch also stores its 4 high-flux traces (the boundary traces of
largest flux energy relative to their boundary L2 norm), which are used to monitor the block
during training.

    python generate_data.py --split train --per-group 3940 --seed 1 --out data/train.npz
    python generate_data.py --split val   --per-group 67   --seed 2 --out data/val.npz --exclude data/train.npz
    python generate_data.py --split test  --per-group 67   --seed 3 --out data/test.npz --exclude data/train.npz data/val.npz --no-labels

--per-group N gives N patches for each of the 11 families x 3 contrasts (33 N patches):
about 1.3e5 training, 2,200 validation and 2,200 test patches with the commands above
(patches that fail the numerical quality checks or repeat an earlier patch are dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.linalg

from coefficients import FAMILIES, CONTRASTS, fresh_coefficient, plan_members
from fem import (
    boundary_ids,
    boundary_metric,
    exact_local_data,
    full_boundary_lift,
    high_flux_traces,
    interior_ids,
)

CELLS = 16
Q = 63


def response_diagnostics(coefficient: np.ndarray) -> dict:
    local = exact_local_data(coefficient, CELLS, need_map=False)
    dense_aii = local["aii"].toarray()
    chol = scipy.linalg.cholesky(dense_aii, lower=True, check_finite=False)
    local["interior_map"] = scipy.linalg.cho_solve((chol, True), -local["aib"], check_finite=False)
    local["schur"] = 0.5 * (
        (local["abb"] + local["aib"].T @ local["interior_map"])
        + (local["abb"] + local["aib"].T @ local["interior_map"]).T
    )
    lift = full_boundary_lift(CELLS)[interior_ids(CELLS)]
    w = local["interior_map"] - lift
    weighted = chol.T @ w
    g = 0.5 * (weighted.T @ weighted + (weighted.T @ weighted).T)
    values, vectors = scipy.linalg.eigh(g, check_finite=False)
    singular = scipy.linalg.svdvals(weighted, check_finite=False)
    smax = max(float(singular[0]), np.finfo(float).tiny)
    svd_tolerance = max(weighted.shape) * np.finfo(np.float64).eps * smax
    svd_rank = int(np.count_nonzero(singular > svd_tolerance))
    maximum = max(float(values[-1]), np.finfo(float).tiny)
    g_backward_tolerance = max(g.shape) * np.finfo(np.float64).eps * maximum
    numerical_null = np.abs(values) <= g_backward_tolerance
    constant = np.ones(64, dtype=np.float64)
    constant /= np.linalg.norm(constant)
    overlaps = np.abs(vectors.T @ constant)
    constant_index = int(np.argmax(overlaps))
    residual = local["aii"] @ local["interior_map"] + local["aib"]
    pde = float(np.linalg.norm(residual) / max(np.linalg.norm(local["aib"]), np.finfo(float).tiny))
    boundary = boundary_ids(CELLS)
    physical_lift = full_boundary_lift(CELLS)
    trace_defect = float(np.max(np.abs(physical_lift[boundary] - np.eye(64))))
    return {
        "local": local,
        "lift": lift,
        "W64": w,
        "weighted_response": weighted,
        "G64": g,
        "G64_eigenvalues_ascending": values,
        "G64_eigenvectors": vectors,
        "energy_singular_values_descending": singular,
        "svd_tolerance": svd_tolerance,
        "svd_rank": svd_rank,
        "g_backward_tolerance": g_backward_tolerance,
        "g_null_count": int(np.count_nonzero(numerical_null)),
        "g_null_mask": numerical_null,
        "constant": constant,
        "constant_eigen_index": constant_index,
        "constant_null_alignment": float(overlaps[constant_index]),
        "constant_eigenvalue": float(values[constant_index]),
        "constant_residual": float(
            np.linalg.norm(weighted @ constant)
            / max(np.linalg.norm(chol.T @ local["interior_map"] @ constant), np.finfo(float).tiny)
        ),
        "PDE_residual": pde,
        "boundary_trace_defect": trace_defect,
        "minimum_kappa": float(np.min(coefficient)),
        "maximum_kappa": float(np.max(coefficient)),
        "actual_contrast": float(np.max(coefficient) / np.min(coefficient)),
        "finite": bool(
            all(np.isfinite(x).all() for x in (coefficient, w, g, values, singular, local["interior_map"]))
        ),
    }


def q63_objects(coefficient: np.ndarray, diagnostics: dict | None = None) -> dict:
    diagnostics = response_diagnostics(coefficient) if diagnostics is None else diagnostics
    local = diagnostics["local"]
    lift = diagnostics["lift"]
    w = diagnostics["W64"]
    values = diagnostics["G64_eigenvalues_ascending"]
    vectors = diagnostics["G64_eigenvectors"]
    rank = diagnostics["svd_rank"]
    null_count = diagnostics["g_null_count"]
    if rank != 63 or null_count != 1:
        raise RuntimeError(
            f"response rank/null failure energy_rank={rank} G_nulls={null_count} svd_tol={diagnostics['svd_tolerance']:.3e} G_tol={diagnostics['g_backward_tolerance']:.3e}"
        )
    null_index = diagnostics["constant_eigen_index"]
    if not diagnostics["g_null_mask"][null_index] or diagnostics["constant_null_alignment"] < 1 - 1e-8:
        raise RuntimeError("constant trace is not the unique numerical G64 null mode")
    positive = np.ones(64, dtype=bool)
    positive[null_index] = False
    lp = values[positive]
    vp = vectors[:, positive]
    if np.any(lp <= diagnostics["g_backward_tolerance"]):
        raise RuntimeError(f"non-positive physical G64 eigenvalue {float(lp.min()):.3e}")
    q = w @ (vp / np.sqrt(lp)[None, :])
    pre_ortho = float(np.max(np.abs(q.T @ (local["aii"] @ q) - np.eye(63))))
    # symmetric re-orthonormalization in the energy inner product (same 63-dimensional space)
    qgram = 0.5 * (q.T @ (local["aii"] @ q) + (q.T @ (local["aii"] @ q)).T)
    qvalues, qvectors = scipy.linalg.eigh(qgram, check_finite=False)
    if np.any(qvalues <= 0):
        raise RuntimeError("Q63 metric reorthonormalization is not SPD")
    q = q @ qvectors @ (qvectors.T / np.sqrt(qvalues)[:, None])
    ortho = float(np.max(np.abs(q.T @ (local["aii"] @ q) - np.eye(63))))
    pde = diagnostics["PDE_residual"]
    forcing = local["aii"] @ lift + local["aib"]
    reduced = 0.5 * (q.T @ (local["aii"] @ q) + (q.T @ (local["aii"] @ q)).T)
    coefficients = scipy.linalg.solve(reduced, -q.T @ forcing, assume_a="pos", check_finite=False)
    qmap = lift + q @ coefficients
    response_rel = float(
        np.linalg.norm(qmap - local["interior_map"])
        / max(np.linalg.norm(local["interior_map"]), np.finfo(float).tiny)
    )
    flux = high_flux_traces(local["schur"], boundary_metric(CELLS), 4)
    return {
        "local": local,
        "W64": w,
        "Q63": q,
        "G64_eigenvalues_ascending": values,
        "positive_eigenvalues_descending": np.sort(lp)[::-1],
        "rank": rank,
        "null_count": null_count,
        "svd_tolerance": diagnostics["svd_tolerance"],
        "g_backward_tolerance": diagnostics["g_backward_tolerance"],
        "energy_singular_values_descending": diagnostics["energy_singular_values_descending"],
        "constant": diagnostics["constant"],
        "constant_residual": diagnostics["constant_residual"],
        "null_alignment": diagnostics["constant_null_alignment"],
        "pre_reorthonormalization_defect": pre_ortho,
        "ortho": ortho,
        "pde": pde,
        "qmap": qmap,
        "response_rel": response_rel,
        "high_flux": flux,
    }


def label_patch(coefficient: np.ndarray, contrast: float) -> dict:
    """Label space Q63 and high-flux probes of one patch, after numerical quality checks."""
    diagnostics = response_diagnostics(coefficient)
    obj = q63_objects(coefficient, diagnostics)
    boundary = boundary_ids(CELLS)
    lift = full_boundary_lift(CELLS)
    trace_defect = float(np.max(np.abs(lift[boundary] - np.eye(64))))
    stored = (coefficient, obj["W64"], obj["Q63"], obj["high_flux"], obj["qmap"])
    finite = all(np.isfinite(x).all() for x in stored)
    gates = {
        "shape_positive_finite": coefficient.shape == (16, 16) and bool(np.all(coefficient > 0)) and finite,
        "declared_actual_contrast": bool(
            np.isclose(coefficient.max() / coefficient.min(), contrast, rtol=2e-12)
        ),
        "PDE_residual_le_1e-10": obj["pde"] <= 1e-10,
        "trace_defect_le_1e-13": trace_defect <= 1e-13,
        "constant_trace_correction_le_1e-10": obj["constant_residual"] <= 1e-10,
        "one_null_mode": obj["null_count"] == 1,
        "rank_63": obj["rank"] == 63,
        "null_vector_constant_alignment": obj["null_alignment"] >= 1 - 1e-8,
        "Q63_A_orthonormality_le_2e-8": obj["ortho"] <= 2e-8,
        "Q63_full_response_map_le_1e-8": obj["response_rel"] <= 1e-8,
    }
    if not all(gates.values()):
        raise RuntimeError(f"quality check failed: {[k for k, v in gates.items() if not v]}")
    return {
        "teacher": obj["Q63"],
        "high_flux": obj["high_flux"],
        "response_rel": obj["response_rel"],
        "ortho": obj["ortho"],
    }


def kappa_hash(kappa: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(kappa, np.float64)).tobytes()).hexdigest()


def _one(task):
    member, labels = task
    kappa, meta = fresh_coefficient(member)
    out = {"kappa": kappa, "attempt": int(meta["construction_attempt"])}
    if labels:
        try:
            out.update(label_patch(kappa, member["contrast"]))
        except Exception as exc:  # rejected patches are reported and skipped
            out["error"] = repr(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .npz file")
    ap.add_argument("--split", default="train", help="name stored with the data (train / val / test)")
    ap.add_argument("--per-group", type=int, required=True, help="patches per (family, contrast) group")
    ap.add_argument("--seed", type=int, required=True, help="seed of the generator-seed stream")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--contrasts", nargs="*", type=int, default=list(CONTRASTS))
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="other .npz files whose coefficients must not be repeated"
    )
    ap.add_argument("--no-labels", action="store_true", help="coefficients only (e.g. for test mosaics)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    members = plan_members(
        args.per_group, args.seed, tuple(args.families), tuple(args.contrasts), args.split.upper()
    )
    seen = set()
    for path in args.exclude:
        with np.load(path, allow_pickle=False) as z:
            seen.update(kappa_hash(k) for k in z["kappa"])
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_one, [(m, not args.no_labels) for m in members], chunksize=4))
    keep, rejected, duplicates = [], 0, 0
    for member, res in zip(members, results):
        h = kappa_hash(res["kappa"])
        if "error" in res:
            rejected += 1
            continue
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        keep.append((member, res))
    arrays = {
        "kappa": np.stack([r["kappa"] for _, r in keep]),
        "family": np.array([m["family"] for m, _ in keep]),
        "contrast": np.array([m["contrast"] for m, _ in keep], dtype=np.int64),
        "oscillation_target": np.array([m["oscillation_target"] for m, _ in keep]),
        "generator_seed": np.array([m["generator_seed"] for m, _ in keep], dtype=np.int64),
        "construction_attempt": np.array([r["attempt"] for _, r in keep], dtype=np.int64),
    }
    if not args.no_labels:
        arrays["teacher"] = np.stack([r["teacher"] for _, r in keep])
        arrays["high_flux"] = np.stack([r["high_flux"] for _, r in keep])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, split=np.array(args.split), **arrays)
    print(
        json.dumps(
            {
                "out": args.out,
                "split": args.split,
                "planned": len(members),
                "kept": len(keep),
                "rejected_by_quality_gates": rejected,
                "duplicates_dropped": duplicates,
                "labels": not args.no_labels,
                "seconds": round(time.time() - started, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
