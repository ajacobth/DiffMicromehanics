"""fim.py – Fisher Information Matrix utilities for identifiability analysis.

Normalised FIM approach:
    J_norm[i,j] = J[i,j] * (hi_j - lo_j) / sigma_i

Eigenvalue thresholds (dimensionless):
    lambda < 1        -> POOR     (cannot identify within parameter range)
    1 <= lambda < 100 -> MARGINAL
    lambda >= 100     -> WELL
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_jacobian(predict_array, x, free_indices, target_out_indices):
    """Jacobian of selected outputs w.r.t. selected inputs at x.

    Parameters
    ----------
    predict_array      : callable jnp.ndarray -> jnp.ndarray
    x                  : jnp.ndarray  full input vector
    free_indices       : list[int]
    target_out_indices : list[int]

    Returns
    -------
    J : np.ndarray  shape (n_targets, n_free)
    """
    free_idx_arr   = jnp.array(free_indices,        dtype=jnp.int32)
    target_idx_arr = jnp.array(target_out_indices,   dtype=jnp.int32)

    def f(theta):
        x_new = x.at[free_idx_arr].set(theta.astype(x.dtype))
        return predict_array(x_new)[target_idx_arr]

    theta = x[free_idx_arr]
    J = jax.jacobian(f)(theta)
    return np.array(J, dtype=float)


def sample_parameter_space(free_inputs, bounds, N=200, seed=42):
    """Latin-hypercube sampling over the free-parameter box.

    Parameters
    ----------
    free_inputs : Sequence[str]   parameter names (defines column order)
    bounds      : dict {name: (lo, hi)}
    N           : int             number of samples
    seed        : int

    Returns
    -------
    samples : list of np.ndarray, each length n_free
    """
    keys = list(free_inputs)
    n    = len(keys)
    rng  = np.random.default_rng(seed)

    # LHS: divide [0,1] into N strata per dimension, shuffle independently
    samples_unit = np.zeros((N, n))
    for j in range(n):
        perm             = rng.permutation(N)
        u                = rng.uniform(size=N)
        samples_unit[:, j] = (perm + u) / N

    result = []
    for row in samples_unit:
        theta = np.array([
            bounds[k][0] + row[j] * (bounds[k][1] - bounds[k][0])
            for j, k in enumerate(keys)
        ])
        result.append(theta)
    return result


def _safe_fim_inverse(F):
    """Pseudo-inverse of F with correct treatment of near-zero eigenvalues.

    Directions with near-zero eigenvalue get very large variance (not zero),
    so parameters in the null space are correctly classified as POOR.
    """
    eigvals, eigvecs = np.linalg.eigh(F)
    eigvals = np.maximum(eigvals, 0.0)
    inv_eigvals = np.where(eigvals > 1e-10, 1.0 / np.maximum(eigvals, 1e-30), 1e15)
    return eigvecs @ np.diag(inv_eigvals) @ eigvecs.T


# ── public API (per spec) ─────────────────────────────────────────────────────

def compute_normalised_fim(predict_array, x_template, free_inputs,
                            free_indices, target_outputs, out_idx,
                            sigmas, bounds, N_samples=200):
    """
    Normalised FIM where eigenvalues are dimensionless.

    J_norm[i,j] = J[i,j] * param_range[j] / sigma_i

    eigenvalue < 1   -> POOR   (cannot identify within param range)
    eigenvalue < 100 -> MARGINAL
    eigenvalue >= 100 -> WELL

    Parameters
    ----------
    sigmas : dict {output_name: float} measurement noise in physical units.
             If a key is missing or <= 0, falls back to 2% of the predicted
             output value — physically meaningful at any unit scale.
    bounds : dict {param_name: (lo, hi)}
    """
    target_out_indices = [out_idx[k] for k in target_outputs.keys()]
    n_free       = len(list(free_inputs))
    F            = np.zeros((n_free, n_free))
    samples      = sample_parameter_space(free_inputs, bounds, N=N_samples)
    free_idx_arr  = jnp.array(free_indices)
    target_idx_arr = jnp.array(target_out_indices)

    param_ranges = np.array([bounds[k][1] - bounds[k][0] for k in free_inputs])

    # JIT-compile jacobian once for speed
    def _f(theta):
        x = x_template.at[free_idx_arr].set(theta.astype(x_template.dtype))
        return predict_array(x)[target_idx_arr]

    J_fn = jax.jit(jax.jacobian(_f))
    _theta0 = jnp.array(
        [(bounds[k][0] + bounds[k][1]) / 2.0 for k in free_inputs],
        dtype=x_template.dtype,
    )
    J_fn(_theta0).block_until_ready()

    for theta in samples:
        theta_jax = jnp.array(theta, dtype=x_template.dtype)
        J = np.array(J_fn(theta_jax))  # (n_targets, n_free)

        for i, k in enumerate(target_outputs.keys()):
            sigma_i = float(sigmas.get(k, 0.0))
            if sigma_i <= 0:
                # Fallback: 2% of predicted output value at this sample point.
                # This is physically meaningful regardless of unit scale (MPa, GPa, etc.)
                # and avoids the sigma=1.0 bug that inflates identifiability for large-unit outputs.
                y_pred = float(predict_array(
                    x_template.at[free_idx_arr].set(
                        jnp.array(theta, dtype=x_template.dtype)
                    )
                )[out_idx[k]])
                sigma_i = max(abs(y_pred) * 0.02, 1e-12)
            j_norm = J[i, :] * param_ranges / sigma_i
            F     += np.outer(j_norm, j_norm)

    return F / max(len(samples), 1)


def compute_relative_sensitivity(predict_array, x_nominal, free_indices,
                                  all_out_indices, param_names, output_names):
    """
    S[i,j] = |dyi/dthj| * |thj / yi|

    Returns S as np.ndarray shape (n_outputs, n_free).
    Used for the heatmap showing which outputs are sensitive to which params.
    """
    J     = compute_jacobian(predict_array, x_nominal,
                             free_indices, all_out_indices)
    theta = np.array(x_nominal)[free_indices]
    y     = np.array(predict_array(x_nominal))[all_out_indices]
    S     = np.abs(J) * np.abs(theta[np.newaxis, :]) / (np.abs(y[:, np.newaxis]) + 1e-12)
    return S


def _precompute_all_output_contributions(predict_array, x_template, free_inputs,
                                          free_indices, out_idx, all_output_names,
                                          sigmas, bounds, N_samples=200):
    """Efficient: compute per-output FIM contributions in a single pass.

    Returns
    -------
    F_per_output : dict {output_name: np.ndarray (n_free, n_free)}
    valid_names  : list[str]  output names present in out_idx
    """
    keys         = list(free_inputs)
    n_free       = len(keys)
    valid_names  = [n for n in all_output_names if n in out_idx]
    free_idx_arr = jnp.array(free_indices)
    all_out_idx  = jnp.array([out_idx[n] for n in valid_names])
    param_ranges = np.array([bounds[k][1] - bounds[k][0] for k in keys])

    # JIT the full jacobian (all outputs at once)
    def _f_all(theta):
        x = x_template.at[free_idx_arr].set(theta.astype(x_template.dtype))
        return predict_array(x)[all_out_idx]

    J_fn = jax.jit(jax.jacobian(_f_all))
    _theta0 = jnp.array(
        [(bounds[k][0] + bounds[k][1]) / 2.0 for k in keys],
        dtype=x_template.dtype,
    )
    J_fn(_theta0).block_until_ready()

    samples = sample_parameter_space(free_inputs, bounds, N=N_samples)
    F_per_output = {name: np.zeros((n_free, n_free)) for name in valid_names}

    for theta in samples:
        theta_jax = jnp.array(theta, dtype=x_template.dtype)
        J_all = np.array(J_fn(theta_jax))  # (n_valid_outputs, n_free)

        for i, name in enumerate(valid_names):
            sigma_i = float(sigmas.get(name, 0.0))
            if sigma_i <= 0:
                # Fallback: 2% of predicted output — physically meaningful at any unit scale.
                y_pred = float(predict_array(
                    x_template.at[free_idx_arr].set(
                        jnp.array(theta, dtype=x_template.dtype)
                    )
                )[out_idx[name]])
                sigma_i = max(abs(y_pred) * 0.02, 1e-12)
            j_norm = J_all[i, :] * param_ranges / sigma_i
            F_per_output[name] += np.outer(j_norm, j_norm)

    n = max(len(samples), 1)
    for name in valid_names:
        F_per_output[name] /= n

    return F_per_output, valid_names


def run_identifiability_check(predict_array, x_template, free_inputs,
                               free_indices, target_outputs, out_idx,
                               sigmas, bounds, all_output_names, sig_out,
                               N_samples=200):
    """
    Single entry point called by the GUI. Returns a structured dict.

    Parameters
    ----------
    predict_array    : callable jnp.ndarray -> jnp.ndarray
    x_template       : jnp.ndarray  full input vector (nominal values)
    free_inputs      : list[str]    names of free parameters
    free_indices     : list[int]    corresponding positions in x
    target_outputs   : dict {name: float}  selected outputs for FIM
    out_idx          : dict {name: int}    output name -> index
    sigmas           : dict {name: float}  measurement noise per output (model units)
    bounds           : dict {name: (lo, hi)}  parameter bounds (model units)
    all_output_names : list[str]   all model output names (for candidate ranking)
    sig_out          : jnp.ndarray  model output std (kept for compatibility)
    N_samples        : int          number of LHS samples
    """
    param_list = list(free_inputs)
    n_free     = len(param_list)

    # ── efficient: compute all-output FIM contributions in one pass ───────────
    F_per_output, valid_names = _precompute_all_output_contributions(
        predict_array, x_template, free_inputs, free_indices,
        out_idx, all_output_names, sigmas, bounds, N_samples=N_samples,
    )

    # ── FIM for selected targets ──────────────────────────────────────────────
    target_keys = [k for k in target_outputs.keys() if k in F_per_output]
    if target_keys:
        F = sum(F_per_output[k] for k in target_keys)
    else:
        F = np.zeros((n_free, n_free))

    # ── eigendecomposition ────────────────────────────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(F)
    eigvals = np.maximum(eigvals, 0.0)

    cond = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-10 else np.inf

    dominant_params = [
        param_list[int(np.argmax(np.abs(eigvecs[:, i])))]
        for i in range(n_free)
    ]

    # ── per-parameter CR bound ────────────────────────────────────────────────
    F_inv = _safe_fim_inverse(F)

    # ── per-parameter status from eigenvalues ────────────────────────────────
    # Use eigenvalue-based thresholds (dimensionless, same scale as normalised FIM):
    #   lambda >= 100  -> WELL
    #   lambda >= 1    -> MARGINAL
    #   lambda <  1    -> POOR
    # The dominant eigenvector for each eigenvalue tells us which parameter
    # that direction is most aligned with.
    eigenvalue_status = []
    for lam in eigvals:
        if lam >= 100:
            eigenvalue_status.append("WELL")
        elif lam >= 1:
            eigenvalue_status.append("MARGINAL")
        else:
            eigenvalue_status.append("POOR")

    # Map eigenvalue status back to individual parameters via dominant eigenvector.
    # Each parameter inherits the status of the eigenvector it dominates.
    # If a parameter dominates multiple eigenvectors, it takes the worst status.
    _status_rank = {"WELL": 0, "MARGINAL": 1, "POOR": 2}
    status_dict = {k: "WELL" for k in param_list}
    for ev_idx, (status, dom) in enumerate(zip(eigenvalue_status, dominant_params)):
        if _status_rank[status] > _status_rank[status_dict[dom]]:
            status_dict[dom] = status

    # CR std in physical units (for WELL/MARGINAL only; None for POOR)
    F_inv = _safe_fim_inverse(F)
    cr_std_dict = {}
    for j, k in enumerate(param_list):
        param_range = float(bounds[k][1] - bounds[k][0])
        var_norm    = float(max(F_inv[j, j], 0.0))
        cr_norm     = float(np.sqrt(var_norm))   # fraction of param range
        if status_dict[k] == "POOR":
            cr_std_dict[k] = None
        else:
            cr_std_dict[k] = cr_norm * param_range

    # ── sensitivity matrix (selected targets vs free params) ─────────────────
    target_out_indices = [out_idx[k] for k in target_keys]
    S = compute_relative_sensitivity(
        predict_array, x_template, free_indices,
        target_out_indices, param_list, target_keys,
    )

    # ── recommendations via precomputed contributions (fast) ──────────────────
    base_eigvals    = np.linalg.eigvalsh(F)
    lambda_min_base = max(float(np.min(base_eigvals)), 1e-12)

    # When the baseline FIM is near-singular (any POOR parameter exists),
    # ranking by improvement_factor = new/base is numerically unstable:
    # dividing by ~1e-12 amplifies noise and produces meaningless huge ratios.
    # Instead rank by new_min_eigenvalue directly (absolute E-optimality).
    # This correctly identifies which experiment most improves the worst direction.
    baseline_is_singular = lambda_min_base < 1.0

    current_keys = set(target_outputs.keys())
    candidates   = [n for n in valid_names if n not in current_keys]

    recs = []
    for cand in candidates:
        F_new          = F + F_per_output[cand]
        new_eigvals    = np.linalg.eigvalsh(F_new)
        lambda_min_new = max(float(np.min(new_eigvals)), 1e-12)
        improvement    = lambda_min_new / lambda_min_base
        recs.append({
            "name":               cand,
            "improvement_factor": improvement,
            "new_min_eigenvalue": lambda_min_new,
        })

    if baseline_is_singular:
        # Rank by absolute new_min_eigenvalue — more stable and physically correct
        recs.sort(key=lambda d: d["new_min_eigenvalue"], reverse=True)
    else:
        recs.sort(key=lambda d: d["improvement_factor"], reverse=True)

    return {
        "status":                   status_dict,
        "cramer_rao_std":           cr_std_dict,
        "eigenvalues":              eigvals,
        "eigenvectors":             eigvecs,
        "dominant_params":          dominant_params,
        "condition_number":         cond,
        "recommendations":          recs[:5],
        "sensitivity_matrix":       S,
        "sensitivity_output_names": target_keys,
        "sensitivity_param_names":  param_list,
        "F":                        F,
    }


def rank_candidate_experiments(predict_array, x_template, free_inputs,
                                free_indices, current_targets, out_idx,
                                sigmas, bounds, all_output_names, N_samples=200):
    """
    Rank candidate experiments by E-optimality improvement.

    Returns list of dicts sorted descending by improvement_factor:
        [{"name": str, "improvement_factor": float, "new_min_eigenvalue": float}, ...]
    """
    F_base       = compute_normalised_fim(
        predict_array, x_template, free_inputs, free_indices,
        current_targets, out_idx, sigmas, bounds, N_samples=N_samples,
    )
    base_eigvals    = np.linalg.eigvalsh(F_base)
    lambda_min_base = max(float(np.min(base_eigvals)), 1e-12)
    baseline_is_singular = lambda_min_base < 1.0

    current_keys = set(current_targets.keys())
    candidates   = [n for n in all_output_names if n not in current_keys and n in out_idx]

    results = []
    for cand in candidates:
        new_targets       = dict(current_targets)
        new_targets[cand] = 0.0
        new_sigmas        = dict(sigmas)
        if cand not in new_sigmas or new_sigmas[cand] <= 0:
            new_sigmas[cand] = 1.0

        F_new          = compute_normalised_fim(
            predict_array, x_template, free_inputs, free_indices,
            new_targets, out_idx, new_sigmas, bounds, N_samples=N_samples,
        )
        new_eigvals    = np.linalg.eigvalsh(F_new)
        lambda_min_new = max(float(np.min(new_eigvals)), 1e-12)
        improvement    = lambda_min_new / lambda_min_base

        results.append({
            "name":               cand,
            "improvement_factor": improvement,
            "new_min_eigenvalue": lambda_min_new,
        })

    if baseline_is_singular:
        results.sort(key=lambda d: d["new_min_eigenvalue"], reverse=True)
    else:
        results.sort(key=lambda d: d["improvement_factor"], reverse=True)
    return results
