"""The eleven 2D coefficient families.

Each patch is a cellwise-constant coefficient on 16 x 16 cells with contrast
chi = max(kappa) / min(kappa) in {1e2, 1e3, 1e4}. Every patch is binary, kappa in {1, chi};
in `clipped_lognormal_structures` the high-conductivity phase also contains the upper tail of a
smooth lognormal field, clipped to the two values. A member is a dict with keys family, contrast, oscillation_target ("HIGH_OSCILLATION" or
"MODERATE_OSCILLATION") and generator_seed; `fresh_coefficient(member)` is deterministic
in the member.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

CONTRASTS = (100, 1000, 10000)


FAMILIES = (
    "long_connected_channels",
    "branching_channel_networks",
    "one_cell_bottlenecks",
    "near_percolating_narrow_gaps",
    "correlated_fragmented_inclusions",
    "multi_component_boundary_touching",
    "multiscale_inclusions",
    "mixed_channel_inclusion",
    "mixed_fragmented_near_percolating",
    "clipped_lognormal_structures",
    "controlled_diagonal_connectivity",
)


def motif(name, rng):
    z = np.zeros((16, 16), bool)
    if name == "long_connected_channels":
        y = int(rng.integers(3, 13))
        for x in range(16):
            y = int(np.clip(y + rng.choice((-1, 0, 0, 1)), 1, 14))
            z[y, x] = 1
    elif name == "branching_channel_networks":
        z[7, :] = 1
        z[:, 7] = 1
        z[3, 3:13] = 1
        z[12, 2:14] = 1
    elif name == "one_cell_bottlenecks":
        z[4:8, :7] = 1
        z[8:12, 9:] = 1
        z[7:9, 6:10] = 1
        z[7, 7:9] = 0
        z[8, 7:9] = 1
    elif name == "near_percolating_narrow_gaps":
        z[5:11, :7] = 1
        z[5:11, 8:] = 1
        z[8, 7] = bool(rng.integers(2))
    elif name == "correlated_fragmented_inclusions":
        for _ in range(10):
            y, x = rng.integers(1, 15, 2)
            z[max(0, y - 1) : min(16, y + 2), max(0, x - 1) : min(16, x + 2)] = 1
    elif name == "multi_component_boundary_touching":
        z[1:5, :3] = 1
        z[10:15, :3] = 1
        z[:3, 10:14] = 1
        z[12:, 11:15] = 1
    elif name == "multiscale_inclusions":
        z[2, 2] = 1
        z[5:7, 10:12] = 1
        z[10:14, 3:7] = 1
        z[8, 13] = 1
    elif name == "mixed_channel_inclusion":
        z[7, :] = 1
        z[2:5, 3:6] = 1
        z[11:14, 10:13] = 1
    elif name == "mixed_fragmented_near_percolating":
        z[5:11, :7] = 1
        z[5:11, 8:] = 1
        for _ in range(6):
            y, x = rng.integers(0, 16, 2)
            z[y, x] = 1
    elif name == "clipped_lognormal_structures":
        z[7:9, :] = 1
        for _ in range(8):
            y, x = rng.integers(0, 16, 2)
            z[y, x] = 1
    else:
        for k in range(16):
            z[k, k] = 1
        z[4:12, 5] = 1
        z[10, 3:13] = 1
    return z


def remove_alternating_crosspoints(z):
    z = z.copy()
    for _ in range(512):
        changed = False
        for j in range(15):
            for i in range(15):
                a, b, c, d = z[j, i], z[j, i + 1], z[j + 1, i + 1], z[j + 1, i]
                if a == c and b == d and a != b:
                    # break a diagonal-only contact between cells of the same phase
                    z[j + 1, i + 1] = b
                    changed = True
        if not changed:
            return z
    raise RuntimeError("cross-point cleanup did not terminate")


def physical_metadata(cells):
    from scipy.ndimage import label

    a = np.asarray(cells, float)
    z = np.log(a)
    logchi = float(z.max() - z.min())
    jumps = np.r_[np.abs(z[1:] - z[:-1]).ravel(), np.abs(z[:, 1:] - z[:, :-1]).ravel()]
    rho = float(np.mean(jumps >= 0.5 * logchi - 1e-12))
    osc = "HIGH_OSCILLATION" if rho >= 0.35 else "MODERATE_OSCILLATION" if rho >= 0.15 else "LOW_OSCILLATION"
    high = z >= 0.5 * (float(z.min()) + float(z.max()))
    severe = 0
    for j in range(1, 16):
        for i in range(1, 16):
            v = np.asarray([z[j - 1, i - 1], z[j - 1, i], z[j, i], z[j, i - 1]])
            p = np.asarray([high[j - 1, i - 1], high[j - 1, i], high[j, i], high[j, i - 1]])
            if (
                p[0] == p[2]
                and p[1] == p[3]
                and p[0] != p[1]
                and np.all(np.abs(v - np.roll(v, -1)) >= 0.5 * logchi - 1e-12)
            ):
                severe += 1
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], int)
    _, nh = label(high, structure)
    _, nl = label(~high, structure)
    return {
        "contrast": float(a.max() / a.min()),
        "rho_jump": rho,
        "TV_log": float(np.sum(jumps) / (480 * logchi)),
        "oscillation_class": osc,
        "N_severe_xpt": severe,
        "rho_xpt": severe / 225,
        "high_phase_connected_component_count": int(nh),
        "low_phase_connected_component_count": int(nl),
    }


def fresh_coefficient(member: dict) -> tuple[np.ndarray, dict]:
    family = member["family"]
    chi = float(member["contrast"])
    seed = int(member["generator_seed"])
    target = member["oscillation_target"]
    for attempt in range(512):
        rng = np.random.default_rng(seed + attempt)
        base = motif(family, rng)
        yy, xx = np.indices((16, 16))
        coord = xx if attempt % 2 == 0 else yy
        background = (
            ((coord + attempt // 2) % 2 == 0)
            if target == "HIGH_OSCILLATION"
            else (((coord + attempt // 2) // 2) % 2 == 0)
        )
        raw = (background | base) if attempt % 4 < 2 else (background ^ base)
        for _ in range(10):
            py, px = rng.integers(0, 15, 2)
            raw[py : py + 2, px : px + 2] = bool(rng.integers(2))
        raw[0, 0] = True
        raw[0, 1] = True
        raw[1, 0] = True
        mask = remove_alternating_crosspoints(raw)
        if family == "clipped_lognormal_structures":
            field = gaussian_filter(rng.standard_normal((16, 16)), sigma=1.25, mode="reflect")
            mask = remove_alternating_crosspoints(mask | (field > np.quantile(field, 0.8)))
            mask[15, 15] = False
        cells = np.where(mask, chi, 1.0).astype(float)
        meta = physical_metadata(cells)
        loga = np.log(cells)
        sep = float(np.max(np.abs(loga - loga[:, [0]] - loga[[0], :] + loga[0, 0])))
        valid = (
            meta["oscillation_class"] == target
            and meta["N_severe_xpt"] == 0
            and sep > 1e-12
            and np.isclose(cells.max() / cells.min(), chi, rtol=2e-12)
        )
        if valid:
            return cells, {
                **meta,
                "target_oscillation": target,
                "construction_attempt": attempt,
                "topology": family,
                "connectivity_hard": family
                in set(FAMILIES)
                - {
                    "correlated_fragmented_inclusions",
                    "multiscale_inclusions",
                    "clipped_lognormal_structures",
                },
                "separability_log_cross_residual": sep,
            }
    raise RuntimeError("no admissible coefficient found within 512 attempts")


def plan_members(
    per_group: int, seed: int, families=FAMILIES, contrasts=CONTRASTS, split: str = "TRAIN"
) -> list[dict]:
    """Members for `per_group` patches of every (family, contrast) group.

    Within a group the oscillation targets alternate between HIGH_OSCILLATION and
    MODERATE_OSCILLATION. Generator seeds are drawn from one numpy Generator(seed)
    without repetition.
    """
    rng = np.random.default_rng(seed)
    used, out = set(), []
    for family in families:
        for contrast in contrasts:
            for j in range(per_group):
                s = int(rng.integers(0, 2**63 - 1))
                while s in used:
                    s = int(rng.integers(0, 2**63 - 1))
                used.add(s)
                out.append(
                    {
                        "family": family,
                        "contrast": int(contrast),
                        "oscillation_target": "HIGH_OSCILLATION" if j % 2 == 0 else "MODERATE_OSCILLATION",
                        "generator_seed": s,
                        "split": split,
                        "index": len(out),
                    }
                )
    return out
