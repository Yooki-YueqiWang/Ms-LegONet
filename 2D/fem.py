"""Q1 finite-element primitives and exact local solves on the reference primitive.

A patch with `cells` x `cells` bilinear Q1 cells has (cells + 1)^2 nodes, numbered
node = x + (cells + 1) * y. `boundary_ids` lists the 4 * cells boundary nodes
counter-clockwise from the corner (0, 0); `interior_ids` lists the interior nodes row by row.
The coefficient is cellwise constant on the 16 x 16 cells of the primitive.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TINY = np.finfo(np.float64).tiny


CONSTANT_KE = np.asarray(
    [
        [2 / 3, -1 / 6, -1 / 3, -1 / 6],
        [-1 / 6, 2 / 3, -1 / 6, -1 / 3],
        [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
        [-1 / 6, -1 / 3, -1 / 6, 2 / 3],
    ],
    dtype=np.float64,
)


def boundary_ids(cells: int) -> np.ndarray:
    n = cells + 1
    return np.concatenate(
        (
            np.arange(n),
            np.arange(1, n) * n + cells,
            cells * n + np.arange(cells - 1, -1, -1),
            np.arange(cells - 1, 0, -1) * n,
        )
    ).astype(np.int64)


def interior_ids(cells: int) -> np.ndarray:
    n = cells + 1
    return np.asarray([x + n * y for y in range(1, cells) for x in range(1, cells)], dtype=np.int64)


def edge_node_ids(cells: int, edge: int) -> np.ndarray:
    n = cells + 1
    if edge == 0:
        return np.arange(n, dtype=np.int64)
    if edge == 1:
        return np.arange(n, dtype=np.int64) * n + cells
    if edge == 2:
        return cells * n + np.arange(cells, -1, -1, dtype=np.int64)
    if edge == 3:
        return np.arange(cells, -1, -1, dtype=np.int64) * n
    raise ValueError(edge)


def faber_schauder(t: np.ndarray, level: int, cell: int) -> np.ndarray:
    left, mid, right = cell / 2**level, (2 * cell + 1) / 2 ** (level + 1), (cell + 1) / 2**level
    return np.maximum(0.0, np.minimum((t - left) / (mid - left), (right - t) / (right - mid)))


def h32_trace_basis(cells: int) -> np.ndarray:
    if cells % 8:
        raise ValueError("H:32 requires cells divisible by eight")
    total = (cells + 1) ** 2
    values: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, cells + 1)
    corners = ((0, 0), (cells, 0), (cells, cells), (0, cells))
    for corner in range(4):
        column = np.zeros(total)
        for edge in range(4):
            ids = edge_node_ids(cells, edge)
            if corner == edge:
                column[ids] = 1 - t
            elif corner == (edge + 1) % 4:
                column[ids] = t
        x, y = corners[corner]
        column[x + (cells + 1) * y] = 1.0
        values.append(column)
    for edge in range(4):
        ids = edge_node_ids(cells, edge)
        for level in range(3):
            for cell in range(2**level):
                column = np.zeros(total)
                column[ids] = faber_schauder(t, level, cell)
                values.append(column)
    answer = np.column_stack(values)
    if answer.shape != (total, 32):
        raise RuntimeError("H:32 shape failure")
    return answer


def full_boundary_lift(cells: int) -> np.ndarray:
    n = cells + 1
    bids = boundary_ids(cells)
    boundary = np.zeros((n * n, len(bids)))
    boundary[bids, np.arange(len(bids))] = 1.0
    boundary = boundary.reshape(n, n, len(bids))
    result = np.zeros((n * n, len(bids)))
    for iy, y in enumerate(np.linspace(0.0, 1.0, n)):
        for ix, x in enumerate(np.linspace(0.0, 1.0, n)):
            left, right = boundary[iy, 0], boundary[iy, cells]
            bottom, top = boundary[0, ix], boundary[cells, ix]
            g00, g10, g01, g11 = (
                boundary[0, 0],
                boundary[0, cells],
                boundary[cells, 0],
                boundary[cells, cells],
            )
            result[ix + n * iy] = (
                (1 - x) * left
                + x * right
                + (1 - y) * bottom
                + y * top
                - (1 - x) * (1 - y) * g00
                - x * (1 - y) * g10
                - (1 - x) * y * g01
                - x * y * g11
            )
    if np.max(np.abs(result[bids] - np.eye(len(bids)))) > 1e-14:
        raise RuntimeError("Coons trace defect")
    return result


def q1_mass_sparse(cells: int) -> sp.csr_matrix:
    h = 1.0 / cells
    one = sp.diags(
        ([h / 6] * (cells), [2 * h / 3] * (cells + 1), [h / 6] * (cells)), (-1, 0, 1), format="csr"
    )
    one[0, 0] = h / 3
    one[-1, -1] = h / 3
    return sp.kron(one, one, format="csr")


def boundary_mass(cells: int) -> np.ndarray:
    bids = boundary_ids(cells)
    lookup = {int(v): i for i, v in enumerate(bids)}
    out = np.zeros((len(bids), len(bids)))
    h = 1.0 / cells
    local = h * np.asarray([[2, 1], [1, 2]]) / 6
    for edge in range(4):
        ids = edge_node_ids(cells, edge)
        for a, b in zip(ids[:-1], ids[1:]):
            ij = [lookup[int(a)], lookup[int(b)]]
            out[np.ix_(ij, ij)] += local
    return out


def assemble_q1(coefficients: np.ndarray) -> sp.csr_matrix:
    values = np.asarray(coefficients, dtype=np.float64)
    cells = values.shape[0]
    if values.shape != (cells, cells):
        raise ValueError("square coefficients required")
    n = cells + 1
    rows = []
    cols = []
    data = []
    for ey in range(cells):
        for ex in range(cells):
            ids = (ex + n * ey, ex + 1 + n * ey, ex + 1 + n * (ey + 1), ex + n * (ey + 1))
            ke = values[ey, ex] * CONSTANT_KE
            for a in range(4):
                for b in range(4):
                    rows.append(ids[a])
                    cols.append(ids[b])
                    data.append(ke[a, b])
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n * n, n * n)).tocsr()
    return 0.5 * (matrix + matrix.T)


def refined_coefficient(coefficient: np.ndarray, cells: int) -> np.ndarray:
    factor = cells // 16
    if cells % 16 or factor < 1:
        raise ValueError("dyadic refinement of the 16x16 coefficient required")
    return np.repeat(np.repeat(np.asarray(coefficient, dtype=np.float64), factor, axis=0), factor, axis=1)


def mass_normalize(columns: np.ndarray, metric: np.ndarray) -> np.ndarray:
    values = np.asarray(columns, dtype=np.float64)
    norms = np.sqrt(np.maximum(np.einsum("ik,ij,jk->k", values, metric, values), TINY))
    return values / norms[None, :]


@lru_cache(maxsize=None)
def boundary_metric(cells: int) -> np.ndarray:
    return boundary_mass(cells)


@lru_cache(maxsize=None)
def h32_boundary(cells: int) -> np.ndarray:
    return mass_normalize(h32_trace_basis(cells)[boundary_ids(cells)], boundary_metric(cells))


def exact_local_data(coefficient16: np.ndarray, cells: int, need_map: bool = True) -> dict:
    coefficient = refined_coefficient(coefficient16, cells)
    matrix = assemble_q1(coefficient)
    bids, ii = boundary_ids(cells), interior_ids(cells)
    aii = matrix[ii][:, ii].tocsc()
    aib = matrix[ii][:, bids].toarray()
    abb = matrix[bids][:, bids].toarray()
    answer = {"matrix": matrix, "aii": aii, "aib": aib, "abb": abb, "coefficient": coefficient}
    if need_map:
        factor = spla.splu(aii, options={"Equil": True, "IterRefine": "DOUBLE"})
        interior_map = factor.solve(-aib)
        schur = abb + aib.T @ interior_map
        schur = 0.5 * (schur + schur.T)
        answer["factor"] = factor
        answer["schur"] = schur
        answer["interior_map"] = interior_map
    return answer


def high_flux_traces(schur: np.ndarray, mass: np.ndarray, count: int = 4) -> np.ndarray:
    inverse_action = scipy.linalg.solve(mass, schur, assume_a="pos", check_finite=False)
    numerator = schur.T @ inverse_action
    values, vectors = scipy.linalg.eigh(0.5 * (numerator + numerator.T), mass, check_finite=False)
    return mass_normalize(vectors[:, -count:], mass)


def mosaic_coefficient(coefficients16: list[np.ndarray], side: int, cells: int) -> np.ndarray:
    refined = [refined_coefficient(value, cells) for value in coefficients16]
    return np.block([[refined[y * side + x] for x in range(side)] for y in range(side)])
