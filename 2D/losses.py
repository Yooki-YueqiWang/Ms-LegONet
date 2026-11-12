"""Q1 forms in torch (float64), energy orthonormalization and the training loss.

For a patch with coefficient kappa, `learned` and `teacher` are (225, 63) interior nodal fields
that are orthonormal in the energy inner product a_kappa (`orthonormalize`). The loss
`training_loss` combines
  - the subspace terms: defects 1 - sigma_r^2 of the singular values sigma_r of the energy overlap
    teacher^T A learned (mean over all r: space_mean; mean of the 8 largest: space_tail), and
  - the physical term: relative errors of the local Galerkin responses (Coons lift + energy
    projection onto the space) on the M probe traces of the patch,
        mean(energy) + CVaR_rho(energy) + mean(L2_kappa),
    with CVaR_rho the mean of the ceil(rho M) largest energy errors (rho = 0.2) and L2_kappa
    the error in the kappa-weighted L2 norm.
With beta = (beta_space, beta_tail, beta_phys) = (1 - 0.5 w, 1, w), the weights move from
(1, 1, 0) to (0.5, 1, 1) as `physical_weight` w goes from 0 to 1.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fem import full_boundary_lift, interior_ids

CELLS = 16

KE = torch.tensor(
    [
        [2 / 3, -1 / 6, -1 / 3, -1 / 6],
        [-1 / 6, 2 / 3, -1 / 6, -1 / 3],
        [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
        [-1 / 6, -1 / 3, -1 / 6, 2 / 3],
    ],
    dtype=torch.float64,
)


ME = torch.tensor([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=torch.float64) / 36


def vertices(field: torch.Tensor, cells: int) -> torch.Tensor:
    grid = field.reshape(cells + 1, cells + 1, -1)
    return torch.stack((grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), 2).reshape(
        -1, 4, field.shape[-1]
    )


def full_field(interior: torch.Tensor, cells: int) -> torch.Tensor:
    output = torch.zeros(((cells + 1) ** 2, interior.shape[-1]), device=interior.device, dtype=interior.dtype)
    output[torch.as_tensor(interior_ids(cells), device=interior.device)] = interior
    return output


def bilinear(
    first: torch.Tensor, second: torch.Tensor, coefficient: torch.Tensor, cells: int, mass: bool = False
) -> torch.Tensor:
    left, right = vertices(first, cells), vertices(second, cells)
    metric = ME.to(first.device) * (1 / cells**2) if mass else KE.to(first.device)
    weights = coefficient.reshape(-1)
    return torch.einsum("eia,ij,ejb,e->ab", left, metric, right, weights)


def orthonormalize(
    interior: torch.Tensor, coefficient: torch.Tensor, cells: int
) -> tuple[torch.Tensor, float]:
    full = full_field(interior.to(torch.float64), cells)
    gram = bilinear(full, full, coefficient, cells)
    gram = 0.5 * (gram + gram.T)
    chol, info = torch.linalg.cholesky_ex(gram)
    if int(info.item()):
        raise RuntimeError("basis rank/SPD failure")
    basis = torch.linalg.solve_triangular(chol, interior.to(torch.float64).T, upper=False).T
    audit = bilinear(full_field(basis, cells), full_field(basis, cells), coefficient, cells)
    return basis, float(torch.max(torch.abs(audit - torch.eye(basis.shape[1], device=basis.device))).detach())


_FULL_LIFT_NP: np.ndarray | None = None


def response(
    basis: torch.Tensor, traces: torch.Tensor, coefficient: torch.Tensor, cells: int = CELLS
) -> torch.Tensor:
    global _FULL_LIFT_NP
    if cells != CELLS:
        raise ValueError("the response is defined for 16x16 patches")
    if _FULL_LIFT_NP is None:
        _FULL_LIFT_NP = full_boundary_lift(CELLS)
    lift = torch.as_tensor(_FULL_LIFT_NP, device=basis.device, dtype=torch.float64) @ traces
    full = full_field(basis, cells)
    gram = bilinear(full, full, coefficient, cells)
    coupling = bilinear(full, lift, coefficient, cells)
    weights = torch.linalg.solve(0.5 * (gram + gram.T), -coupling)
    return lift + full @ weights


def response_errors(
    learned: torch.Tensor, teacher: torch.Tensor, coefficient: torch.Tensor, traces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    learned_response = response(learned, traces, coefficient, CELLS)
    teacher_response = response(teacher, traces, coefficient, CELLS)
    delta = learned_response - teacher_response
    en = torch.diagonal(bilinear(delta, delta, coefficient, CELLS)).clamp_min(0)
    ed = torch.diagonal(bilinear(teacher_response, teacher_response, coefficient, CELLS)).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    ln = torch.diagonal(bilinear(delta, delta, coefficient, CELLS, True)).clamp_min(0)
    ld = torch.diagonal(bilinear(teacher_response, teacher_response, coefficient, CELLS, True)).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return torch.sqrt(en / ed), torch.sqrt(ln / ld)


def subspace_losses(
    learned: torch.Tensor, teacher: torch.Tensor, coefficient: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    overlap = bilinear(full_field(teacher, CELLS), full_field(learned, CELLS), coefficient, CELLS)
    singular = torch.linalg.svdvals(overlap)
    defects = torch.clamp(1.0 - singular.square(), min=0.0, max=1.0)
    mean = defects.mean()
    tail = torch.topk(defects, 8).values.mean()
    return mean, tail, singular


def physical_errors(
    learned: torch.Tensor, teacher: torch.Tensor, coefficient: torch.Tensor, traces: dict[str, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor]:
    energies, l2s = [], []
    for values in traces.values():
        trace = torch.as_tensor(values, device=learned.device, dtype=torch.float64)
        energy, l2 = response_errors(learned, teacher, coefficient, trace)
        energies.append(energy)
        l2s.append(l2)
    return torch.cat(energies), torch.cat(l2s)


RHO = 0.2


def cvar(values: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    flat = values.reshape(-1)
    return torch.topk(flat, max(1, math.ceil(rho * flat.numel()))).values.mean()


def phase_blend(update: int, start: int, end: int) -> float:
    if update <= start:
        return 0.0
    if update >= end:
        return 1.0
    return float(update - start) / float(end - start)


def training_loss(
    learned: torch.Tensor,
    teacher: torch.Tensor,
    coefficient: torch.Tensor,
    traces: dict[str, np.ndarray],
    physical_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    space_mean, space_tail, singular = subspace_losses(learned, teacher, coefficient)
    if physical_weight > 0:
        energy, l2 = physical_errors(learned, teacher, coefficient, traces)
        physical = energy.mean() + cvar(energy) + l2.mean()
    else:
        energy = torch.zeros(1, device=learned.device, dtype=torch.float64)
        l2 = torch.zeros(1, device=learned.device, dtype=torch.float64)
        physical = torch.zeros((), device=learned.device, dtype=torch.float64)
    beta_space, beta_tail, beta_phys = 1.0 - 0.5 * physical_weight, 1.0, physical_weight
    total = beta_space * space_mean + beta_tail * space_tail + beta_phys * physical
    return total, {
        "space_mean": float(space_mean.detach()),
        "space_tail": float(space_tail.detach()),
        "sigma_min": float(singular[-1].detach()),
        "energy_mean": float(energy.mean().detach()),
        "energy_cvar": float(cvar(energy).detach()),
        "L2_mean": float(l2.mean().detach()),
    }
