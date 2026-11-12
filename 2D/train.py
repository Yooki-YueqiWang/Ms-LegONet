"""Training of the 2D Ms-LegONet block.

    python train.py --train data/train.npz --val data/val.npz --out runs/msl2d

Defaults are the settings of the paper: Adam, batch 96 drawn in rotation over the 33
(family, contrast) groups, gradient-norm clipping at 1 and 600k updates. The
learning rate is warmed up linearly to 2e-4 over 5k updates, then 7e-5 after 120k, 2e-5
after 300k and 5e-6 after 480k updates. The loss weights start at
beta = (beta_space, beta_tail, beta_phys) = (1, 1, 0); between updates 100k and 140k
beta_space falls linearly to 0.5 and beta_phys rises linearly to 1 (`losses.training_loss`).

Every patch carries M = 24 fixed probe traces for the physical-response loss: random
combinations of the 64 nodal port traces, drawn once from a seed and the patch index and
normalized in the boundary L2 norm.

The validation patches are evaluated every `--eval-every` updates with the full loss
(beta = (0.5, 1, 1)); the checkpoint with the smallest validation loss is kept as
OUT/best.pt. OUT/latest.pt holds the most recent state; rerunning the same command
continues from it.

--schedule-scale s shrinks every update count of the schedule by the factor s
(s = 0.0005 gives a complete 300-update schedule for a quick test).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from fem import boundary_metric, h32_boundary
from losses import orthonormalize, phase_blend, response_errors, subspace_losses, training_loss
from model import MultiscaleFPNTrialSpace, parameter_breakdown

Q = 63
CELLS = 16
PROBES = 24


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe_traces(seed: int, index: int, count: int = PROBES) -> np.ndarray:
    """The fixed probe traces of one patch: `count` random combinations of the nodal port basis."""
    values = np.random.default_rng([seed, index]).standard_normal((64, count))
    metric = boundary_metric(CELLS)
    norms = np.sqrt(np.maximum(np.einsum("ik,ij,jk->k", values, metric, values), np.finfo(np.float64).tiny))
    return values / norms[None, :]


@dataclass
class Case:
    member: dict
    kappa: np.ndarray
    teacher: np.ndarray
    high_flux: np.ndarray
    probe_seed: int

    def probes(self) -> np.ndarray:
        return probe_traces(self.probe_seed, int(self.member["index"]))


class StratifiedCoefficientSampler:
    """Family/contrast sampler: each batch cycles through the (family, contrast) groups."""

    def __init__(self, members: list[dict], seed: int, batch_size: int):
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        grouped: dict[tuple[str, int], list[int]] = {}
        for index, member in enumerate(members):
            grouped.setdefault((str(member["family"]), int(member["contrast"])), []).append(index)
        self.keys = sorted(grouped)
        self.orders = {key: np.asarray(grouped[key], dtype=np.int64) for key in self.keys}
        self.cursors = {key: len(values) for key, values in self.orders.items()}
        self.rotation = 0

    def _draw(self, key: tuple[str, int]) -> int:
        order = self.orders[key]
        cursor = self.cursors[key]
        if cursor >= len(order):
            self.rng.shuffle(order)
            cursor = 0
        answer = int(order[cursor])
        self.cursors[key] = cursor + 1
        return answer

    def next(self) -> np.ndarray:
        indices = []
        for offset in range(self.batch_size):
            key = self.keys[(self.rotation + offset) % len(self.keys)]
            indices.append(self._draw(key))
        self.rotation = (self.rotation + self.batch_size) % len(self.keys)
        return np.asarray(indices, dtype=np.int64)

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.bit_generator.state,
            "batch_size": self.batch_size,
            "keys": [[family, contrast] for family, contrast in self.keys],
            "orders": {f"{family}|{contrast}": values for (family, contrast), values in self.orders.items()},
            "cursors": {f"{family}|{contrast}": value for (family, contrast), value in self.cursors.items()},
            "rotation": self.rotation,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["batch_size"]) != self.batch_size:
            raise RuntimeError("coefficient batch changed across resume")
        expected = [[family, contrast] for family, contrast in self.keys]
        if state["keys"] != expected:
            raise RuntimeError("coefficient groups changed across resume")
        self.rng.bit_generator.state = state["rng_state"]
        for family, contrast in self.keys:
            label = f"{family}|{contrast}"
            self.orders[(family, contrast)] = np.asarray(state["orders"][label], dtype=np.int64)
            self.cursors[(family, contrast)] = int(state["cursors"][label])
        self.rotation = int(state["rotation"])


def gradient_norms(model: nn.Module) -> dict[str, float]:
    encoder = 0.0
    decoder = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().double().square().sum())
        if name.startswith("decoder"):
            decoder += value
        else:
            encoder += value
    return {"encoder_gradient_norm": math.sqrt(encoder), "decoder_gradient_norm": math.sqrt(decoder)}


def qstats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        name: float(np.quantile(array, q)) for name, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95))
    } | {"mean": float(array.mean()), "maximum": float(array.max()), "count": int(array.size)}


def evaluate_cases(model: nn.Module, cases: list[Case], device: torch.device, inference_batch: int = 16) -> dict:
    """Full loss, subspace defects and local response errors of the block on a set of patches."""
    groups = {
        "probe": None,
        "H32": h32_boundary(CELLS),
        "nodal": np.eye(64, dtype=np.float64),
        "high_flux": None,
    }
    distributions = {name: [] for name in groups}
    objective, space_mean, space_tail, angles = [], [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), inference_batch):
            block = cases[start : start + inference_batch]
            raw_block = model(
                torch.as_tensor(np.stack([case.kappa for case in block]), device=device, dtype=torch.float32)
            )
            for case, raw in zip(block, raw_block):
                coefficient = torch.as_tensor(case.kappa, device=device, dtype=torch.float64)
                teacher = torch.as_tensor(case.teacher, device=device, dtype=torch.float64)
                learned, _ = orthonormalize(raw, coefficient, CELLS)
                probes = case.probes()
                loss, _ = training_loss(learned, teacher, coefficient, {"probe": probes}, 1.0)
                objective.append(float(loss))
                mean, tail, singular = subspace_losses(learned, teacher, coefficient)
                space_mean.append(float(mean))
                space_tail.append(float(tail))
                angles.append(float(math.acos(float(torch.clamp(singular[-1], 0, 1)))))
                for name, values in groups.items():
                    if values is None:
                        values = probes if name == "probe" else case.high_flux
                    traces = torch.as_tensor(values, device=device, dtype=torch.float64)
                    energy, _ = response_errors(learned, teacher, coefficient, traces)
                    distributions[name].extend(energy.cpu().tolist())
    if was_training:
        model.train()
    result = {
        "patches": len(cases),
        "loss": float(np.mean(objective)),
        "space_mean": qstats(space_mean),
        "space_tail": qstats(space_tail),
        "maximum_principal_angle_radians": qstats(angles),
    }
    for name, values in distributions.items():
        result[f"{name}_energy_error"] = qstats(values)
    return result


def schedule(scale: float = 1.0) -> dict:
    return {
        "maximum": max(1, round(600000 * scale)),
        "warmup": max(1, round(5000 * scale)),
        "steps": (round(120000 * scale), round(300000 * scale), round(480000 * scale)),
        "blend": (round(100000 * scale), round(140000 * scale)),
    }


def learning_rate(config: dict, update: int) -> float:
    first, second, third = config["steps"]
    if update <= config["warmup"]:
        return 2e-4 * update / config["warmup"]
    if update <= first:
        return 2e-4
    if update <= second:
        return 7e-5
    if update <= third:
        return 2e-5
    return 5e-6


def flatten_metrics(metrics: dict | None, prefix: str = "") -> dict:
    if metrics is None:
        return {}
    output = {}
    for key, value in metrics.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            output.update(flatten_metrics(value, label + "."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[label] = value
    return output


def write_history_csv(path: Path, history: list[dict]) -> None:
    rows = []
    for record in history:
        row = {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
        row.update(flatten_metrics(record.get("val"), "val."))
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(
        {
            "format": "ms_legonet_2d_checkpoint_v1",
            "model": state["model"].state_dict(),
            "optimizer": state["optimizer"].state_dict(),
            "architecture": state["architecture"],
            "seed": state["seed"],
            "schedule_scale": state["schedule_scale"],
            "update": state["update"],
            "samples_seen": state["samples_seen"],
            "best": state["best"],
            "coefficient_sampler_state": state["sampler"].state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "history": state["history"],
        },
        temporary,
    )
    os.replace(temporary, path)


def restore(path: Path, model, optimizer, sampler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["coefficient_sampler_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and state["torch_cuda_rng_states"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_rng_states"]])
    return int(state["update"]), int(state["samples_seen"]), list(state["history"]), state["best"]


def load_cases(path, probe_seed: int) -> list[Case]:
    """Patches written by generate_data.py (with labels)."""
    with np.load(path, allow_pickle=False) as z:
        if "teacher" not in z.files:
            raise SystemExit(f"{path} holds no labels (it was generated with --no-labels)")
        rows = zip(z["family"], z["contrast"], z["kappa"], z["teacher"], z["high_flux"])
        return [
            Case(
                member={"family": str(f), "contrast": int(c), "index": i},
                kappa=np.asarray(k, np.float64),
                teacher=np.asarray(t, np.float64),
                high_flux=np.asarray(h, np.float64),
                probe_seed=probe_seed,
            )
            for i, (f, c, k, t, h) in enumerate(rows)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="training patches (.npz from generate_data.py)")
    parser.add_argument("--val", required=True, help="validation patches (.npz from generate_data.py)")
    parser.add_argument("--out", required=True, help="run directory (checkpoints, history.csv, status.json)")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--architecture", default="LARGE", choices=("LARGE", "MEDIUM"))
    parser.add_argument("--schedule-scale", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=None, help="stop this invocation at this update")
    parser.add_argument("--log-every", type=int, default=5000, help="updates between log lines and checkpoints")
    parser.add_argument("--eval-every", type=int, default=20000, help="updates between validation evaluations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-wall-seconds", type=float, default=float("inf"))
    args = parser.parse_args()

    config = schedule(args.schedule_scale)
    train_cases = load_cases(args.train, args.seed + 63)
    val_cases = load_cases(args.val, args.seed + 64)
    size = len(train_cases)
    batch = min(args.batch, size)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)
    print(
        json.dumps(
            {
                "run": str(directory),
                "train": size,
                "val": len(val_cases),
                "batch": batch,
                "maximum_updates": config["maximum"],
                "device": str(device),
            }
        ),
        flush=True,
    )
    model = MultiscaleFPNTrialSpace(args.architecture).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate(config, 1), betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    sampler = StratifiedCoefficientSampler([case.member for case in train_cases], args.seed + 17, batch)
    latest, best_path = directory / "latest.pt", directory / "best.pt"
    update, samples_seen, history, best = 0, 0, [], None

    def checkpoint(path: Path) -> None:
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            architecture=args.architecture,
            seed=args.seed,
            schedule_scale=args.schedule_scale,
            update=update,
            samples_seen=samples_seen,
            best=best,
            sampler=sampler,
            history=history,
        )

    def validate() -> dict:
        nonlocal best
        metrics = evaluate_cases(model, val_cases, device, inference_batch=32)
        if best is None or metrics["loss"] < best["loss"]:
            best = {"update": update, "loss": metrics["loss"]}
            checkpoint(best_path)
        return metrics

    if latest.exists():
        update, samples_seen, history, best = restore(latest, model, optimizer, sampler, device)
        print(json.dumps({"resumed_at_update": update}), flush=True)
    else:
        history.append({"kind": "evaluation", "update": 0, "val": validate()})
        checkpoint(latest)

    target = config["maximum"] if args.max_updates is None else min(config["maximum"], args.max_updates)
    started = time.time()
    last_gradients: dict[str, float] = {}
    model.train()
    while update < target:
        indices = sampler.next()
        block = [train_cases[int(index)] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        raw_block = model(
            torch.as_tensor(np.stack([case.kappa for case in block]), device=device, dtype=torch.float32)
        )
        total = torch.zeros((), device=device, dtype=torch.float64)
        blend = phase_blend(update + 1, *config["blend"])
        for case, raw in zip(block, raw_block):
            coefficient = torch.as_tensor(case.kappa, device=device, dtype=torch.float64)
            teacher = torch.as_tensor(case.teacher, device=device, dtype=torch.float64)
            learned, defect = orthonormalize(raw, coefficient, CELLS)
            if defect > 1e-9:
                raise RuntimeError(f"energy orthonormalization defect {defect} exceeds tolerance")
            loss, _ = training_loss(learned, teacher, coefficient, {"probe": case.probes()}, blend)
            total = total + loss / len(block)
        if not torch.isfinite(total):
            raise RuntimeError("non-finite training loss")
        total.backward()
        last_gradients = gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        update += 1
        samples_seen += len(block)
        lr = learning_rate(config, update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if update % args.log_every == 0 or update == target:
            record = {
                "kind": "optimization",
                "update": update,
                "samples_seen": samples_seen,
                "effective_epochs": samples_seen / size,
                "learning_rate": lr,
                "physical_loss_blend": blend,
                "loss": float(total.detach()),
                **last_gradients,
            }
            history.append(record)
            if update % args.eval_every == 0 or update == config["maximum"]:
                metrics = validate()
                history.append({"kind": "evaluation", "update": update, "val": metrics})
                record = record | {
                    "val_loss": metrics["loss"],
                    "val_H32_energy_p50_p95": [
                        metrics["H32_energy_error"]["p50"],
                        metrics["H32_energy_error"]["p95"],
                    ],
                    "best_update": best["update"],
                }
            checkpoint(latest)
            write_history_csv(directory / "history.csv", history)
            print(json.dumps(record, sort_keys=True), flush=True)
            model.train()
        if time.time() - started >= args.max_wall_seconds:
            break

    checkpoint(latest)
    complete = update >= config["maximum"]
    status = {
        "update": update,
        "maximum_updates": config["maximum"],
        "complete": complete,
        "best": best,
        "checkpoint": str(best_path),
        "parameters": parameter_breakdown(model),
        "seconds": time.time() - started,
    }
    (directory / "status.json").write_text(json.dumps(status, indent=2, default=str) + "\n")
    print(json.dumps({"update": update, "complete": complete, "best": best}), flush=True)


if __name__ == "__main__":
    main()
