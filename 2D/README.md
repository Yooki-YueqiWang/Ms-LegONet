# Ms-LegONet in 2D

This folder contains the running code for the two-dimensional experiments of the paper: data generation, training of the local block and the online multiscale solver.

**Not included:** pretrained checkpoints, plotting scripts and error-analysis scripts. To reproduce the method, generate the training data and train the block with the code in this folder.

## Setting

| Item | Value |
|---|---|
| Reference primitive | $`\widehat\omega=(0,1)^2`$, $`16\times16`$ bilinear $`Q_1`$ cells |
| Nodes per primitive | 225 interior, 64 boundary |
| Coefficient | cellwise constant, binary $`\kappa\in\{1,\chi\}`$, $`\chi\in\{10^2,10^3,10^4\}`$ |
| Coefficient families | 11 (connected and branching channels, one-cell bottlenecks, near-percolating gaps, fragmented / multiscale / boundary-touching inclusions, clipped lognormal, channels + inclusions, fragmented + percolating, diagonally connected) |
| Port space | complete nodal trace space, $`m_\Gamma=64`$ |
| Lifting $`E_i`$ | Coons lift |
| Learned space dimension | $`q=63`$ |
| Network input | normalized $`\log\kappa`$ on the $`16\times16`$ cells plus 2 coordinate channels |
| Network output | $`q=63`$ nodal zero-trace fields, energy-orthonormalized |
| Data split | $`1.3\times10^5`$ train, $`2{,}200`$ validation, $`2{,}200`$ test patches |

## Workflow

The code follows the three stages of the method.

**1. Data generation (offline).** Sample patches from the 11 coefficient families and three contrasts. For each patch, compute the 64 zero-trace corrections $`\mathcal{C}^{\kappa}_{h}b_\alpha=\mathcal{H}^{\kappa}_{h}b_\alpha-E\,b_\alpha`$ by local $`Q_1`$ solves. These are the labels; no global solve is needed.

**2. Training (offline, once).** Train the block $`\mathcal{N}^{\mathrm{MsL}}_\theta`$ with the subspace losses $`\mathcal{L}_{\mathrm{space}}`$, $`\mathcal{L}_{\mathrm{tail}}`$ and the physical-response loss $`\mathcal{L}_{\mathrm{phys}}`$. Settings used in the paper:

| Setting | Value |
|---|---|
| $`q_{\mathrm{tail}}`$, $`\rho`$, probe traces $`M`$ | 8, 0.2, 24 fixed random combinations of the port basis per patch |
| Loss weights $`\beta=(\beta_{\mathrm{space}},\beta_{\mathrm{tail}},\beta_{\mathrm{phys}})`$ | start at $`(1,1,0)`$; $`\beta_{\mathrm{space}}\to0.5`$ and $`\beta_{\mathrm{phys}}\to1`$ linearly between updates 100k and 140k, then fixed |
| Optimizer | Adam, gradient-norm clipping at 1 |
| Batch size, peak learning rate, updates | 96, $`2\times10^{-4}`$, 600k |
| Input scale $`s_\kappa`$ | $`\log 10^4`$ |
| Checkpoint | smallest validation loss (`best.pt`) |

**3. Online solve.** For a new coefficient on an $`S\times S`$ mosaic of primitives, with source $`f`$ and Dirichlet data $`g_\Omega`$:

1. evaluate the trained block once per primitive and energy-orthonormalize the output;
2. compute the 64 port responses of each primitive by the local $`q\times q`$ Galerkin solve;
3. compute the local particular fields $`p_i(f)`$;
4. compose the primitives by recursive static condensation and solve the global skeleton system;
5. return $`u_{\mathrm{ms},\theta}=\mathcal{H}_{\theta,\Omega}g_\Omega+p_f+w^{f}_{\mathrm{ms},\theta}`$.

## Code

| File | Purpose |
|---|---|
| `coefficients.py` | the 11 coefficient families: deterministic generator of $`16\times16`$ patches for a family, a contrast and a seed |
| `generate_data.py` | samples patches and computes the labels: an energy-orthonormal basis of the 63-dimensional space of zero-trace corrections (one local $`Q_1`$ factorization per patch), plus the 4 high-flux traces of each patch for monitoring |
| `fem.py` | $`Q_1`$ assembly, Coons lift, boundary probe bases, exact local solves |
| `model.py` | network of the local block (18.5M parameters) |
| `losses.py` | energy orthonormalization, local Galerkin responses and the training loss ($`\mathcal{L}_{\mathrm{space}}`$, $`\mathcal{L}_{\mathrm{tail}}`$, $`\mathcal{L}_{\mathrm{phys}}`$) |
| `train.py` | training loop: balanced patch sampler, fixed probe traces, $`\beta`$ and learning-rate schedules, validation and checkpoint selection, resumable runs |
| `solve.py` | online solver: network evaluation, local Galerkin maps, particular fields, recursive static condensation, skeleton solve; comparison with the fine-grid $`Q_1`$ solution |
| `run_example.sh` | runs the whole pipeline at a small size |
| `requirements.txt` | Python dependencies |

### Requirements

Python 3.10 or newer with NumPy, SciPy and PyTorch (tested with Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1 and PyTorch 2.4.0 with CUDA 12.4). Training needs a GPU in practice; data generation and the solver run on the CPU.

```bash
pip install -r requirements.txt
```

### Quick start

```bash
cd 2D
bash run_example.sh
```

The script generates 264 training, 66 validation and 33 test patches, trains the block for 300 updates and solves one $`4\times4`$ mosaic with a random source. It prints the relative energy and $`L^2`$ errors of Ms-LegONet and of exact local blocks against the fine-grid $`Q_1`$ solution. It takes a few minutes on one GPU. After 300 updates the block is far from converged (about 50% energy error in our test run), so its error is much larger than in the paper; the example only shows that every stage runs. With exact blocks the error is at round-off level, because the nodal port space is complete.

### Full run

```bash
# 1. data (CPU)
python generate_data.py --split train --per-group 3940 --seed 1 --out data/train.npz
python generate_data.py --split val   --per-group 67   --seed 2 --out data/val.npz  --exclude data/train.npz
python generate_data.py --split test  --per-group 67   --seed 3 --out data/test.npz --exclude data/train.npz data/val.npz --no-labels

# 2. training with the settings listed above (600k updates; resumable: rerun the same command to continue)
python train.py --train data/train.npz --val data/val.npz --out runs/msl2d

# 3. solve on a 16 x 16 mosaic of test patches (random source; also: --problem f1 | dirichlet)
python solve.py --checkpoint runs/msl2d/best.pt --data data/test.npz --side 16 --problem grf --exact
```

`--per-group N` draws N patches for each of the 11 families and 3 contrasts. With 3,940 and 67 this gives about $`1.3\times10^5`$ training and about 2,200 validation and test patches. The training file holds the labels in double precision (about 15 GB), and `train.py` keeps them in memory.

In `solve.py`, `--depth` sets the number of $`2\times2`$ static-condensation levels before the skeleton solve; by default the composition goes up to the whole domain. All depths give the same solution up to round-off.

`train.py` writes `latest.pt` (most recent state, used to resume) and `best.pt` (smallest validation loss) to the run directory; use `best.pt` for solving.
