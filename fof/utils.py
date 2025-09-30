# --- JIT core: returns padded results of length N (static shape) ---
import jax
import jax.numpy as jnp

@jax.jit
def cluster_centers_masses_padded(pos, compact_labels, weights=None):
    """
    pos: (N, D) float32/float64
    compact_labels: (N,) int32 in [0..C-1] or -1 for invalid/pad
    weights: (N,) float or None (defaults to 1.0 per point)
    Returns:
      centers_padded: (N, D) cluster centers; valid in [:n_clusters]
      masses_padded:  (N,)   cluster masses;  valid in [:n_clusters]
    """
    pos = jnp.asarray(pos)
    N, D = pos.shape

    if weights is None:
        w = jnp.ones((N,), dtype=pos.dtype)
    else:
        w = jnp.asarray(weights, dtype=pos.dtype)

    valid = (compact_labels >= 0)
    safe_labels = jnp.where(valid, compact_labels, 0)           # avoid -1 index
    w_masked = jnp.where(valid, w, jnp.array(0, pos.dtype))

    # Mass per cluster (padded to length N; zeros beyond real clusters)
    masses_padded = jnp.bincount(safe_labels, weights=w_masked, length=N)  # (N,)

    # Weighted position sums per cluster (padded)
    weighted_pos = pos * w_masked[:, None]                                   # (N,D)
    sums = jnp.zeros((N, D), dtype=pos.dtype).at[safe_labels].add(weighted_pos)

    # Safe divide (avoids NaNs for padded entries with mass==0)
    eps = jnp.finfo(pos.dtype).tiny
    centers_padded = sums / jnp.maximum(masses_padded, eps)[:, None]         # (N,D)

    return centers_padded, masses_padded

# --- Host-side helper: trim to the actual number of clusters ---
def cluster_centers_masses(pos, compact_labels, n_clusters, weights=None):
    """
    Thin wrapper that trims the padded outputs to [:n_clusters].
    """
    centers_pad, masses_pad = cluster_centers_masses_padded(pos, compact_labels, weights)
    C = int(n_clusters)
    return centers_pad[:C], masses_pad[:C]