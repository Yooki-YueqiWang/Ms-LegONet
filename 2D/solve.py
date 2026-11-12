"""Online solver: Ms-LegONet on a mosaic of primitives.

The domain (0, 1)^2 is a side x side mosaic of primitives, each with 16 x 16 Q1 cells and its
own coefficient patch; global node id = x + (16 side + 1) y. On every primitive i,
    u|_{omega_i} = p_i + Phi_i lambda_i,   Phi_i = [I on the boundary; interior map],
where p_i is the zero-trace particular solution of the source on the primitive and the
interior map is the Coons lift plus the local Galerkin projection onto the learned space
(or the exact discrete-harmonic map for exact blocks). Each primitive is condensed to its 64
boundary unknowns; `depth` levels of 2 x 2 static condensation follow, and one sparse solve on
the remaining skeleton gives the interface traces, which are propagated back to the leaves.
The result is the Galerkin solution in the global multiscale space.

    python solve.py --checkpoint runs/msl2d/best.pt --data data/test.npz --side 16 --problem grf --exact

For checking, the script also computes the monolithic Q1 solution on the full fine grid and
prints the relative energy and L2 differences.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fem import (
    assemble_q1,
    boundary_ids,
    exact_local_data,
    full_boundary_lift,
    interior_ids,
    mosaic_coefficient,
    q1_mass_sparse,
)


def metric_basis(raw: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    local = exact_local_data(kappa, 16, need_map=False)
    gram = raw.T @ (local["aii"] @ raw)
    chol = np.linalg.cholesky(0.5 * (gram + gram.T))
    return np.linalg.solve(chol, raw.T).T


def coons_candidate(local: dict, basis: np.ndarray) -> dict:
    """Local Galerkin map: Coons lift plus the energy projection onto the learned space."""
    lift_i = full_boundary_lift(16)[interior_ids(16)]
    forcing = local["aii"] @ lift_i + local["aib"]
    reduced = basis.T @ (local["aii"] @ basis)
    reduced = 0.5 * (reduced + reduced.T)
    chol = scipy.linalg.cho_factor(reduced, lower=True, check_finite=False)
    coefficients = scipy.linalg.cho_solve(chol, -basis.T @ forcing, check_finite=False)
    interior = lift_i + basis @ coefficients
    schur = (
        local["abb"]
        + local["aib"].T @ interior
        + interior.T @ local["aib"]
        + interior.T @ (local["aii"] @ interior)
    )
    return {
        "coefficients": coefficients,
        "interior_map": interior,
        "schur": 0.5 * (schur + schur.T),
    }


N = 16


LB, LI = boundary_ids(N), interior_ids(N)


_ML = q1_mass_sparse(N)


def local_xy(ids):
    return ids % (N + 1), ids // (N + 1)


@dataclass
class Leaf:
    gids: np.ndarray  # global ids of all 289 local nodes
    bg: np.ndarray  # global ids of the 64 boundary nodes (LB order)
    phi_i: np.ndarray  # 225 x 64 interior map
    p: np.ndarray  # 289 particular solution (zero trace)
    S: np.ndarray  # 64 x 64 condensed stiffness
    r: np.ndarray  # 64 condensed load
    A: sp.csr_matrix  # local stiffness
    basis: np.ndarray | None = None  # energy-orthonormal learned basis (225 x 63)
    lift_i: np.ndarray | None = None  # Coons lift interior rows (225 x 64)


def leaf_gids(ix, iy, side):
    G = N * side
    lx, ly = local_xy(np.arange((N + 1) ** 2))
    return (ix * N + lx) + (G + 1) * (iy * N + ly)


def build_leaf(kappa16, f_local, gids, side, raw=None, lift="coons"):
    loc = exact_local_data(kappa16, N, need_map=True)
    A = loc["matrix"].tocsr()
    F = (_ML @ f_local) / side**2  # q1_mass_sparse(16) is for a unit patch; global domain is (0,1)^2
    p = np.zeros((N + 1) ** 2)
    p[LI] = loc["factor"].solve(F[LI])
    basis = lift_i = None
    if raw is None:
        phi_i = loc["interior_map"]
    else:
        basis = metric_basis(raw, kappa16)
        if lift == "coons":
            cand = coons_candidate(loc, basis)
            lift_i = full_boundary_lift(N)[LI]
        else:
            raise ValueError(f"unknown lift {lift}; the 2D block uses the Coons lift")
        phi_i = cand["interior_map"]
    Phi = np.zeros(((N + 1) ** 2, 64))
    Phi[LB] = np.eye(64)
    Phi[LI] = phi_i
    S = Phi.T @ (A @ Phi)
    S = 0.5 * (S + S.T)
    r = Phi.T @ (F - A @ p)
    return Leaf(gids, gids[LB], phi_i, p, S, r, A, basis, lift_i)


class Block:
    """Condensed block: boundary global ids, Schur S, load r, back-substitution data."""

    def __init__(self, bg, S, r, children=None, inner=None, solve=None, coupling=None):
        self.bg, self.S, self.r = bg, S, r
        self.children, self.inner, self.solve, self.coupling = children, inner, solve, coupling


def compose(children, on_parent_boundary):
    allids = np.unique(np.concatenate([c.bg for c in children]))
    outer = allids[on_parent_boundary(allids)]
    inner = np.setdiff1d(allids, outer)
    ids = np.concatenate([outer, inner])
    pos = {g: k for k, g in enumerate(ids)}
    K = np.zeros((len(ids), len(ids)))
    R = np.zeros(len(ids))
    for c in children:
        m = np.array([pos[g] for g in c.bg])
        K[np.ix_(m, m)] += c.S
        R[m] += c.r
    no = len(outer)
    Kii, Kio = K[no:, no:], K[no:, :no]
    cf = scipy.linalg.cho_factor(Kii, lower=True, check_finite=False)
    X = scipy.linalg.cho_solve(cf, Kio, check_finite=False)
    y = scipy.linalg.cho_solve(cf, R[no:], check_finite=False)
    S = K[:no, :no] - Kio.T @ X
    S = 0.5 * (S + S.T)
    r = R[:no] - Kio.T @ y
    return Block(outer, S, r, children, inner, (X, y), None)


def back_substitute(block, values):
    """values: dict global id -> trace value for block.bg; fills leaf traces recursively."""
    if block.children is None:
        block.trace = np.array([values[g] for g in block.bg])
        return
    X, y = block.solve
    out = np.array([values[g] for g in block.bg])
    inner = y - X @ out
    v = dict(values)
    v.update(zip(block.inner.tolist(), inner))
    for c in block.children:
        back_substitute(c, v)


def solve_mosaic(kappas, side, f_nodes, g_boundary, raws=None, depth=0, lift="coons"):
    """kappas: list (row-major y*side+x) of 16x16; f_nodes, g_boundary: global nodal arrays."""
    G = N * side
    t = {}
    t0 = time.perf_counter()
    leaves = []
    for iy in range(side):
        for ix in range(side):
            k = iy * side + ix
            gids = leaf_gids(ix, iy, side)
            leaves.append(
                build_leaf(kappas[k], f_nodes[gids], gids, side, None if raws is None else raws[k], lift)
            )
    t["leaves"] = time.perf_counter() - t0
    for lf in leaves:
        lf.bgS = lf.bg
    blocks = [[Block(lf.bg, lf.S, lf.r) for lf in leaves[iy * side : (iy + 1) * side]] for iy in range(side)]
    for b, lf in zip([b for row in blocks for b in row], leaves):
        b.leaf = lf
    t0 = time.perf_counter()
    width, cells = side, N
    for lev in range(depth):
        if width == 1:
            break
        nb = []
        for by in range(0, width, 2):
            row = []
            for bx in range(0, width, 2):
                x0, y0, w = bx * cells, by * cells, 2 * cells

                def onb(ids, x0=x0, y0=y0, w=w):
                    x, y = ids % (G + 1), ids // (G + 1)
                    return (x == x0) | (x == x0 + w) | (y == y0) | (y == y0 + w)

                row.append(
                    compose(
                        [blocks[by][bx], blocks[by][bx + 1], blocks[by + 1][bx], blocks[by + 1][bx + 1]], onb
                    )
                )
            nb.append(row)
        blocks, width, cells = nb, width // 2, cells * 2
    t["recursion"] = time.perf_counter() - t0
    # flat solve on the remaining skeleton
    t0 = time.perf_counter()
    flat = [b for row in blocks for b in row]
    allids = np.unique(np.concatenate([b.bg for b in flat]))
    x, y = allids % (G + 1), allids // (G + 1)
    dirichlet = (x == 0) | (x == G) | (y == 0) | (y == G)
    free, fixed = allids[~dirichlet], allids[dirichlet]
    pos = {g: k for k, g in enumerate(np.concatenate([free, fixed]))}
    rows, cols, vals = [], [], []
    R = np.zeros(len(allids))
    for b in flat:
        m = np.array([pos[g] for g in b.bg])
        rows.append(np.repeat(m, len(m)))
        cols.append(np.tile(m, len(m)))
        vals.append(b.S.ravel())
        R[m] += b.r
    K = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(allids), len(allids))
    )
    nf = len(free)
    gfix = g_boundary[fixed]
    lam = spla.spsolve(K[:nf, :nf].tocsc(), R[:nf] - K[:nf, nf:] @ gfix) if nf else np.zeros(0)
    values = dict(zip(free.tolist(), lam))
    values.update(zip(fixed.tolist(), gfix))
    for b in flat:
        back_substitute(b, values)
    t["skeleton"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    u = np.zeros((G + 1) ** 2)
    cnt = np.zeros((G + 1) ** 2)

    def collect(b):
        if b.children is None:
            lf = b.leaf
            loc = lf.p.copy()
            loc[LB] += b.trace
            loc[LI] += lf.phi_i @ b.trace
            np.add.at(u, lf.gids, loc)
            np.add.at(cnt, lf.gids, 1)
            lf.trace = b.trace
            return
        for c in b.children:
            collect(c)

    for b in flat:
        collect(b)
    u /= cnt
    t["reconstruct"] = time.perf_counter() - t0
    return u, leaves, t


def monolithic(kappas, side, f_nodes, g_boundary):
    G = N * side
    coeff = mosaic_coefficient(kappas, side, N)
    A = assemble_q1(coeff).tocsr()
    M = q1_mass_sparse(G).tocsr()
    F = M @ f_nodes
    ids = np.arange((G + 1) ** 2)
    x, y = ids % (G + 1), ids // (G + 1)
    bd = (x == 0) | (x == G) | (y == 0) | (y == G)
    u = np.zeros((G + 1) ** 2)
    u[bd] = g_boundary[bd]
    fr = ~bd
    u[fr] = spla.splu(A[fr][:, fr].tocsc()).solve(F[fr] - A[fr][:, bd] @ u[bd])
    return u, A, M, F


def errors(u, ref, A, M):
    d = u - ref
    return (
        float(np.sqrt(max(d @ (A @ d), 0) / (ref @ (A @ ref)))),
        float(np.sqrt(max(d @ (M @ d), 0) / (ref @ (M @ ref)))),
    )


def grf(side, rng, alpha=2.0, tau=3.0):
    """Gaussian random field on the global nodes: N(0, (tau^2 - Laplacian)^(-alpha)), unit RMS.

    Lengths (and tau) are measured in units of one primitive."""
    G = N * side
    n = G + 1
    k = np.fft.fftfreq(2 * n, d=1.0 / (2 * n)) * np.pi / side  # wave numbers on (0, side)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    spec = (tau**2 + kx**2 + ky**2) ** (-alpha / 2)
    z = rng.standard_normal((2 * n, 2 * n)) + 1j * rng.standard_normal((2 * n, 2 * n))
    field = np.real(np.fft.ifft2(spec * z))[:n, :n]
    field = field / np.sqrt(np.mean(field**2))
    return field.ravel()


def problems(side, seed):
    rng = np.random.default_rng(seed)
    G = 16 * side
    z = np.zeros((G + 1) ** 2)
    return {
        "f1": (np.ones((G + 1) ** 2), z),
        "grf": (grf(side, rng), z),
        "dirichlet": (z, grf(side, rng, alpha=3.0)),
    }


def predict_raw(model, kappas: np.ndarray, device: str = "cpu", batch: int = 64) -> np.ndarray:
    """One network evaluation per primitive: (n, 16, 16) coefficients -> (n, 225, 63) raw fields (float64)."""
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(kappas), batch):
            value = torch.as_tensor(np.asarray(kappas[i : i + batch]), device=device, dtype=torch.float32)
            out.append(model(value).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def full_depth(side: int) -> int:
    depth = 0
    while side > 1 and side % 2 == 0:
        side //= 2
        depth += 1
    return depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="trained block (train.py); omit to solve with exact blocks only")
    ap.add_argument("--data", required=True, help="coefficient patches (.npz from generate_data.py)")
    ap.add_argument("--side", type=int, default=16, help="mosaic of side x side primitives")
    ap.add_argument(
        "--select-seed", type=int, default=0, help="seed for drawing the side^2 patches of the mosaic"
    )
    ap.add_argument(
        "--problem",
        default="grf",
        choices=("f1", "grf", "dirichlet"),
        help="f1: f = 1, g = 0; grf: random source, g = 0; dirichlet: f = 0, random Dirichlet data",
    )
    ap.add_argument("--problem-seed", type=int, default=0)
    ap.add_argument(
        "--depth",
        type=int,
        default=-1,
        help="levels of 2x2 static condensation before the skeleton solve (-1: as many as side allows)",
    )
    ap.add_argument("--exact", action="store_true", help="also solve with exact local blocks")
    ap.add_argument("--device", default="cpu", help="device for the network evaluation")
    ap.add_argument("--save", help="write the solutions to this .npz file")
    args = ap.parse_args()
    if args.checkpoint is None and not args.exact:
        ap.error("give --checkpoint and/or --exact")
    side = args.side
    depth = full_depth(side) if args.depth < 0 else args.depth
    with np.load(args.data, allow_pickle=False) as z:
        pool = np.asarray(z["kappa"], np.float64)
    n = side * side
    rows = np.random.default_rng(args.select_seed).choice(len(pool), n, replace=len(pool) < n)
    ks = list(pool[rows])
    f, g = problems(side, args.problem_seed)[args.problem]
    t0 = time.perf_counter()
    ref, A, M, F = monolithic(ks, side, f, g)
    t_ref = time.perf_counter() - t0
    report = {
        "side": side,
        "primitives": n,
        "fine_grid_nodes": int((16 * side + 1) ** 2),
        "problem": args.problem,
        "depth": depth,
        "monolithic_seconds": round(t_ref, 3),
    }
    saved = {"reference": ref, "kappa": mosaic_coefficient(ks, side, 16)}
    if args.checkpoint:
        from model import load_model

        model = load_model(args.checkpoint, args.device)
        t0 = time.perf_counter()
        raws = predict_raw(model, np.stack(ks), args.device)
        t_net = time.perf_counter() - t0
        u, leaves, tm = solve_mosaic(ks, side, f, g, list(raws), depth, lift="coons")
        e, l2 = errors(u, ref, A, M)
        report["ms_legonet"] = {
            "energy_error": e,
            "L2_error": l2,
            "network_seconds": round(t_net, 3),
            **{f"{k}_seconds": round(v, 3) for k, v in tm.items()},
        }
        saved["ms_legonet"] = u
    if args.exact:
        u, _, tm = solve_mosaic(ks, side, f, g, None, depth)
        e, l2 = errors(u, ref, A, M)
        report["exact_blocks"] = {
            "energy_error": e,
            "L2_error": l2,
            **{f"{k}_seconds": round(v, 3) for k, v in tm.items()},
        }
        saved["exact_blocks"] = u
    print(json.dumps(report, indent=2), flush=True)
    if args.save:
        np.savez(args.save, **saved)


if __name__ == "__main__":
    main()
