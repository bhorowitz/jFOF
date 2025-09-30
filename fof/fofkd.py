from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import jaxkd as jk 

def _ensure_xyz32(pos):
    """Ensure pos is (N,3) float32 without copying more than needed."""
    pos = jnp.asarray(pos)
    # Promote to float32 for robust distance thresholds
    pos = pos.astype(jnp.float32, copy=False)
    assert pos.ndim == 2, "pos must be (N,D)"
    N, D = pos.shape
    if D == 3:
        return pos
    elif D == 2:
        # pad a zero z-coordinate
        z = jnp.zeros((N,1), dtype=pos.dtype)
        return jnp.concatenate([pos, z], axis=1)
    else:
        raise ValueError(f"Expected D in {2,3}, got D={D}")

def _cc_label_prop_dense_masked_old(pos_xyz, nbr_idx, b, max_iters):
    N, k = nbr_idx.shape
    labels0 = jnp.arange(N, dtype=jnp.int32)

    i   = jnp.arange(N, dtype=jnp.int32)
    src = jnp.repeat(i, k)                  # (N*k,)
    dst = nbr_idx.reshape(-1)               # (N*k,)

    dst_ge0   = (dst >= 0)
    non_self  = (src != jnp.where(dst_ge0, dst, 0))
    valid_idx = dst_ge0 & non_self
    dst_safe  = jnp.where(dst_ge0, dst, 0)

    d  = pos_xyz[src] - pos_xyz[dst_safe]
    b2 = jnp.asarray(b, jnp.float32) ** 2
    within = jnp.sum(d * d, axis=1) <= b2
    valid = valid_idx & within

    big = jnp.int32(N)

    def one_iter(labels):
        incoming = jnp.full((N,), big, dtype=jnp.int32)
        incoming = incoming.at[dst_safe].min(jnp.where(valid, labels[src], big))
        incoming = incoming.at[src].min(jnp.where(valid, labels[dst_safe], big))
        return jnp.minimum(labels, incoming)

    # Use while_loop so max_iters can be dynamic
    def cond_fn(carry):
        labels, it = carry
        return it < max_iters

    def body_fn(carry):
        labels, it = carry
        return (one_iter(labels), it + 1)

    labels = lax.while_loop(cond_fn, body_fn, (labels0, jnp.int32(0)))[0]
    return labels


def _cc_label_prop_dense_masked(pos64_xyz, nbr_idx, b, max_iters):
    """
    Connected components via fixed-iter min-label propagation.
    Static shapes; masked edges; distance test in float64.
    """
    N, k = nbr_idx.shape
    labels0 = jnp.arange(N, dtype=jnp.int32)

    i   = jnp.arange(N, dtype=jnp.int32)
    src = jnp.repeat(i, k)                  # (N*k,)
    dst = nbr_idx.reshape(-1)               # (N*k,)

    # Mask of potentially valid neighbor slots
    dst_ge0   = (dst >= 0)
    # Avoid self edges too
    non_self  = (src != jnp.where(dst_ge0, dst, 0))
    valid_idx = dst_ge0 & non_self

    # Replace invalid dst with 0 for safe gathering; will be masked out later
    dst_safe = jnp.where(dst_ge0, dst, 0)

    # Distance mask (float64)
    d  = pos64_xyz[src] - pos64_xyz[dst_safe]             # (N*k,3)
    b2 = jnp.asarray(b, jnp.float32) ** 2
    within = jnp.sum(d * d, axis=1) <= b2

    valid = valid_idx & within

    big = jnp.int32(N)  # sentinel "no-edge" larger than any label

    def one_iter(labels):
        incoming = jnp.full((N,), big, dtype=jnp.int32)
        # u -> v
        incoming = incoming.at[dst_safe].min(jnp.where(valid, labels[src], big))
        # v -> u (undirected)
        incoming = incoming.at[src].min(jnp.where(valid, labels[dst_safe], big))
        # keep own label
        return jnp.minimum(labels, incoming)

    return lax.fori_loop(0, max_iters, lambda _, lab: one_iter(lab), labels0)

@partial(jax.jit, static_argnames=('k','max_iters','cuda'))  # b is NOT static
def fof_clusters_jit(pos, b: float = 1.0, k: int = 32, max_iters: int = 50, cuda=True):
    k = int(k)
    max_iters = int(max_iters)
    pos_xyz = _ensure_xyz32(pos)
    tree = jk.build_tree(pos_xyz.astype(jnp.float32), cuda=cuda)
    nbr_idx, distances = jk.query_neighbors(tree, pos_xyz.astype(jnp.float32), k=k, cuda=cuda)
    labels = _cc_label_prop_dense_masked(pos_xyz, nbr_idx, b, max_iters)
    N = labels.shape[0]
    uniq, inv = jnp.unique(labels, size=N, fill_value=-1, return_inverse=True)
    valid = (uniq != -1)
    sizes_all = jnp.bincount(inv, length=N)
    sizes_padded = jnp.where(valid, sizes_all, 0)
    compact_map = (jnp.cumsum(valid.astype(jnp.int32)) - 1).astype(jnp.int32)
    compact_labels = jnp.where(valid[inv], compact_map[inv], -1)
    n_clusters = compact_map[-1] + 1
    return labels, compact_labels, sizes_padded, n_clusters


@partial(jax.jit, static_argnames=('k','max_iters','cuda'))
def fof_clusters_jit_old(pos, b: float = 1.0, k: int = 32, max_iters: int = 50,cuda=True):
    """
    Fully-jitted FoF using the KD-tree primitive from kdtree.
    Returns:
      labels: (N,)       raw component labels (min-id per component)
      compact_labels: (N,) labels remapped to 0..C-1 (or -1 for pad if ever used)
      sizes_padded: (N,) cluster sizes padded with zeros
      n_clusters: ()     number of valid clusters
    """
    # 0) Make positions well-defined for distance checks
    pos_xyz = _ensure_xyz32(pos)  # (N,3) 

    # 1) KD-tree kNN (k must be static for your primitive)
    #nbr_idx = jaxkdtree.kNN(pos64_xyz.astype(jnp.float32), k=k, max_radius=float(b))  # (N,k), -1 padded
    #not sure if it is best to build tree within the jit function? test more...
    #cuda should be able to be turned on for more fun/speed... pure cuda bindings!
    tree = jk.build_tree(pos_xyz.astype(jnp.float32),cuda=cuda)
    nbr_idx, distances = jk.query_neighbors(tree, pos_xyz.astype(jnp.float32), k=k,cuda=cuda)
    # 2) Connected components via label propagation
    labels = _cc_label_prop_dense_masked(pos_xyz, nbr_idx, b, max_iters)  # (N,)

    # 3) Static-shape unique + compacting (no boolean indexing)
    N = labels.shape[0]
    uniq, inv = jnp.unique(labels, size=N, fill_value=-1, return_inverse=True)  # (N,), (N,)
    valid = (uniq != -1)                                # (N,) bool
    sizes_all = jnp.bincount(inv, length=N)             # count by position in uniq
    sizes_padded = jnp.where(valid, sizes_all, 0)       # zero padded entries

    compact_map = (jnp.cumsum(valid.astype(jnp.int32)) - 1).astype(jnp.int32)  # 0..C-1, padded -1
    compact_labels = jnp.where(valid[inv], compact_map[inv], -1)
    n_clusters = compact_map[-1] + 1

    return labels, compact_labels, sizes_padded, n_clusters