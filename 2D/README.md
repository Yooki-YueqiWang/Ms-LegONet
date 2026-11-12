# Ms-LegONet in 2D

This folder contains the running code for the two-dimensional experiments of the paper: data generation, training of the local block and the online multiscale solver.

**Not included:** pretrained checkpoints, plotting scripts and error-analysis scripts. To reproduce the method, generate the training data and train the block with the code in this folder.

## Setting

| Item | Value |
|---|---|
| Reference primitive | $\widehat\omega=(0,1)^2$, $16\times16$ bilinear $Q_1$ cells |
| Nodes per primitive | 225 interior, 64 boundary |
| Coefficient | cellwise constant, binary $\kappa\in\{1,\chi\}$, $\chi\in\{10^2,10^3,10^4\}$ |
| Coefficient families | 11 (connected and branching channels, one-cell bottlenecks, near-percolating gaps, fragmented / multiscale / boundary-touching inclusions, clipped lognormal, channels + inclusions, fragmented + percolating, diagonally connected) |
| Port space | complete nodal trace space, $m_\Gamma=64$ |
| Lifting $E_i$ | Coons lift |
| Learned space dimension | $q=63$ |
| Network input | normalized $\log\kappa$ on the $16\times16$ cells plus 2 coordinate channels |
| Network output | $q=63$ nodal zero-trace fields, energy-orthonormalized |
| Data split | $1.3\times10^5$ train, $2{,}200$ validation, $2{,}200$ test patches |

## Workflow

The code follows the three stages of the method.

**1. Data generation (offline).** Sample patches from the 11 coefficient families and three contrasts. For each patch, compute the 64 zero-trace corrections $\mathcal{C}^{\kappa}_{h}b_\alpha=\mathcal{H}^{\kappa}_{h}b_\alpha-E\,b_\alpha$ by local $Q_1$ solves. These are the labels; no global solve is needed.

**2. Training (offline, once).** Train the block $\mathcal{N}^{\mathrm{MsL}}_\theta$ with the subspace losses $\mathcal{L}_{\mathrm{space}}$, $\mathcal{L}_{\mathrm{tail}}$ and the physical-response loss $\mathcal{L}_{\mathrm{phys}}$. Settings used in the paper:

| Setting | Value |
|---|---|
| $q_{\mathrm{tail}}$, $\rho$, probe traces $M$ | 8, 0.2, 24 |
| Loss weights $\beta=(\beta_{\mathrm{space}},\beta_{\mathrm{tail}},\beta_{\mathrm{phys}})$ | start at $(1,1,0)$; $\beta_{\mathrm{space}}\to0.5$ and $\beta_{\mathrm{phys}}\to1$ linearly early in training, then fixed |
| Optimizer | Adam, gradient-norm clipping at 1 |
| Batch size, peak learning rate, updates | 96, $2\times10^{-4}$, 600k |
| Input scale $s_\kappa$ | $\log 10^4$ |

**3. Online solve.** For a new coefficient on an $S\times S$ mosaic of primitives, with source $f$ and Dirichlet data $g_\Omega$:

1. evaluate the trained block once per primitive and energy-orthonormalize the output;
2. compute the 64 port responses of each primitive by the local $q\times q$ Galerkin solve;
3. compute the local particular fields $p_i(f)$;
4. compose the primitives by recursive static condensation and solve the global skeleton system;
5. return $u_{\mathrm{ms},\theta}=\mathcal{H}_{\theta,\Omega}g_\Omega+p_f+w^{f}_{\mathrm{ms},\theta}$.

## Code

<!--
TODO (authors): add the .py files here and describe them in the table below,
for example:

| File | Purpose |
|---|---|
| `generate_data.py` | sample coefficient patches and compute the zero-trace correction labels |
| `model.py` | network architecture of the local block |
| `train.py` | training loop and losses |
| `solve.py` | online solver: local responses, static condensation, global solve |

Also list the dependencies (Python version, PyTorch, NumPy, SciPy, ...)
and the commands to run each stage.
-->

The source files and run commands will be listed here.
