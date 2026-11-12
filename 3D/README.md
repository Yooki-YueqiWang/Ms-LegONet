# Ms-LegONet in 3D

**The 3D code will be released here after the paper is accepted.**

The three-dimensional study in the paper uses layered media on chains $`\Omega_N=(0,N)\times(0,1)^2`$ of unit primitives with $`16^3`$ cells. Each $`x`$-face carries the reduced sine port space

```math
\mathcal{B}^{x}_{\Gamma}=\mathrm{span}\{\sin(m\pi y)\sin(n\pi z):\ 1\le m,n\le 4\},
```

so each primitive has $`m_\Gamma=32`$ ports and the learned space has dimension $`q=32`$. Each predicted function is a sum of four separated terms $`X(x)\,Y(y)\,Z(z)`$ with piecewise quintic factors, so the energy Gram matrices reduce to one-dimensional integrals.
