# $\varepsilon$-Insensitive Loss in the Inverse Solver

## Motivation

When a user supplies a measured composite property together with its experimental
standard deviation $\sigma_k$, the solver should tolerate predictions that land anywhere
within the measurement uncertainty without penalising them. The $\varepsilon$-insensitive
(dead-zone) loss achieves this: residuals smaller than the dead-zone width
contribute zero to the objective, so the optimiser is not pulled toward the noise
floor of the measurement.

---

## Loss Function

For each target output $k$, let

| Symbol | Meaning |
|--------|---------|
| $\hat{y}_k$ | surrogate prediction of property $k$ |
| $t_k$ | user-entered target value |
| $\varepsilon_k$ | dead-zone half-width for property $k$ |
| $s_k$ | output standard deviation from surrogate training (normalisation factor) |

The per-channel contribution to the objective is

$$
L_k =
\begin{cases}
0 & \text{if } |\hat{y}_k - t_k| \leq \varepsilon_k \\[6pt]
\left(\dfrac{|\hat{y}_k - t_k| - \varepsilon_k}{s_k}\right)^{\!2} & \text{otherwise}
\end{cases}
$$

The total loss summed over all $N$ target channels is

$$
L = \sum_{k=1}^{N} L_k
$$

---

## Dead-Zone Width $\varepsilon_k$

The dead-zone half-width for channel $k$ is

$$
\varepsilon_k = \sigma_k \cdot \alpha
$$

where

- $\sigma_k$ is the standard deviation entered by the user in the $\sigma$ column of the
  Targets panel (physical units, same as the target value).
- $\alpha$ is the **$\varepsilon$ scale** multiplier set in the solver bar (default **0.5**).

Setting $\alpha = 0.5$ means the dead-zone extends $\pm\tfrac{1}{2}\sigma$ around the target:
predictions within half a standard deviation of the measurement are treated as exact matches.
Setting $\alpha = 1.0$ extends the dead-zone to the full $\pm 1\sigma$ range.

---

## Normalisation

Predictions and residuals are divided by $s_k$ (the per-output standard deviation
recorded during surrogate training) before squaring. This puts all output channels
on a dimensionless, comparable scale so that a stiff modulus (order GPa) and a
thermal expansion coefficient (order $10^{-6}\ \text{K}^{-1}$) contribute equally to the objective.

Without the $\varepsilon$ tube, the normalised squared-residual is simply

$$
L_k = \left(\frac{\hat{y}_k - t_k}{s_k}\right)^2
$$

which is the standard mean-squared error in normalised output space.

---

## Implementation Reference

`core/inverse.py`, function `_loss_fn` (lines 68–90):

```python
if use_epsilon_loss and sigmas is not None and sigmas[i] > 0.0:
    raw    = jnp.abs(pred - t)
    excess = jnp.maximum(0.0, raw - sigmas[i])   # zero inside dead-zone
    errs.append((excess / scale) ** 2)
else:
    errs.append(((pred - t) / scale) ** 2)
```

`sigmas[i]` here is already $\sigma_k \cdot \alpha$ (the scaling by $\alpha$ is applied in
`gui_inverse.py` before the list is passed to the solver).

---

## GUI Controls

| Control | Default | Effect |
|---------|---------|--------|
| "Use $\varepsilon$-insensitive loss" checkbox | **checked** | Enables the dead-zone formulation |
| "$\varepsilon$ scale" entry ($\alpha$) | **0.5** | Multiplies every $\sigma_k$ to set $\varepsilon_k$ |

If the checkbox is unchecked, or if $\sigma_k = 0$ for a channel, the solver falls back
to plain normalised MSE for that channel.
