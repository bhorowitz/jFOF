from functools import lru_cache
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import jaxkd as jk 

from dataclasses import dataclass

from typing import NamedTuple
from functools import lru_cache

class FrozenLabels(NamedTuple):
    labels: jnp.ndarray        # [N] compact labels 0..C-1 or -1
    n_clusters: jnp.ndarray    # scalar int32
    valid_mask: jnp.ndarray    # [N] bool (labels >= 0)

# ---------- helpers (unchanged) ----------
def _ensure_xyz32(pos):
    pos = jnp.asarray(pos).astype(jnp.float32, copy=False)
    assert pos.ndim == 2 and pos.shape[1] in (2, 3)
    if pos.shape[1] == 3:
        return pos
    z = jnp.zeros((pos.shape[0], 1), dtype=pos.dtype)
    return jnp.concatenate([pos, z], axis=1)

def _cc_label_prop_dense_masked(pos_xyz, nbr_idx, b, max_iters):
    N, k = nbr_idx.shape
    labels0 = jnp.arange(N, dtype=jnp.int32)

    i   = jnp.arange(N, dtype=jnp.int32)
    src = jnp.repeat(i, k)           # (N*k,)
    dst = nbr_idx.reshape(-1)        # (N*k,)

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

    # while_loop allows dynamic max_iters
    def cond_fn(carry): labels, it = carry; return it < max_iters
    def body_fn(carry): labels, it = carry; return (one_iter(labels), it + 1)

    labels, _ = lax.while_loop(cond_fn, body_fn, (labels0, jnp.int32(0)))
    return labels

# ---------- cached kernel factory ----------


def _to_py_int(x, name):
    x = jax.device_get(x)  # handle DeviceArray / ArrayImpl
    if isinstance(x, (int, np.integer)):
        return int(x)
    xnp = np.asarray(x)
    if xnp.shape == ():
        return int(xnp.item())
    if xnp.size == 1:       # covers shape (1,) or (1,1), etc.
        return int(xnp.reshape(()).item())
    raise TypeError(f"{name} must be a scalar, got array with shape {xnp.shape}")

def _to_py_bool(x, name):
    x = jax.device_get(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    xnp = np.asarray(x)
    if xnp.shape == ():
        return bool(xnp.item())
    if xnp.size == 1:
        return bool(xnp.reshape(()).item())
    raise TypeError(f"{name} must be a scalar bool, got array with shape {xnp.shape}")

def _normalize_statics(k, max_iters, cuda):
    return _to_py_int(k, "k"), _to_py_int(max_iters, "max_iters"), _to_py_bool(cuda, "cuda")
    
def get_fof_clusters_kernel(k, max_iters, cuda):
    k_n, mi_n, cu_n = _normalize_statics(k, max_iters, cuda)
    return _get_fof_clusters_kernel_cached(k_n, mi_n, cu_n)

@lru_cache(maxsize=None)
def _get_fof_clusters_kernel_cached(k: int, max_iters: int, cuda: bool):
    # capture normalized Python constants in the closure
    k_const, max_iters_const, cuda_const = k, max_iters, cuda

    @jax.jit  # no static_argnames needed; constants are closed-over
    def kernel(pos, b):
        pos_xyz = _ensure_xyz32(pos)
        tree = jk.build_tree(pos_xyz.astype(jnp.float32), cuda=cuda_const)
        nbr_idx, _ = jk.query_neighbors(
            tree, pos_xyz.astype(jnp.float32), k=k_const, cuda=cuda_const
        )
        labels = _cc_label_prop_dense_masked(pos_xyz, nbr_idx, b, max_iters_const)
        if False: #chatgpt suggested faster routine, but wrong?

            N = labels.shape[0]
            i = jnp.arange(N, dtype=jnp.int32)
            
            # Identify component roots (= their own min id)
            roots = (labels == i)                                  # (N,) bool
            
            # Cluster sizes at root indices (root domain)
            sizes_at_root = jnp.zeros((N,), jnp.int32).at[labels].add(1)  # (N,)
            
            # Map root indices -> compact ids [0..C-1] (root → compact domain)
            root_to_compact = (jnp.cumsum(roots.astype(jnp.int32)) - 1).astype(jnp.int32)  # (N,)
            # Per-particle compact id [0..C-1]
            compact_labels = root_to_compact[labels]               # (N,)
            # Number of clusters
            n_clusters = root_to_compact[-1] + 1                   # scalar int32
            
            # -------- min_size filtering done CONSISTENTLY in compact domain --------
            # Decide which roots survive, in root domain:
            keep_root = roots & (sizes_at_root >= min_size)        # (N,) bool
            
            # Lift keep flags to compact domain via scatter (length-N static buffer):
            keep_compact = jnp.zeros((N,), bool).at[root_to_compact[roots]].set(keep_root[roots])
            
            # Build compact-domain remap 0..C'-1 and apply to per-particle labels:
            remap_compact = jnp.where(keep_compact,
                                      jnp.cumsum(keep_compact.astype(jnp.int32)) - 1,
                                      -1).astype(jnp.int32)        # (N,)
            lab2 = jnp.where(keep_compact[compact_labels], remap_compact[compact_labels], -1)  # (N,)
            valid = (lab2 >= 0)
            nC2 = jnp.sum(keep_compact.astype(jnp.int32))          # scalar int32
        else: #this still seems to be a hair off?
            N = labels.shape[0]
            uniq, inv = jnp.unique(labels, size=N, fill_value=-1, return_inverse=True)
            valid = (uniq != -1)
            sizes_all = jnp.bincount(inv, length=N)
            sizes_padded = jnp.where(valid, sizes_all, 0)
            compact_map = (jnp.cumsum(valid.astype(jnp.int32)) - 1).astype(jnp.int32)
            compact_labels = jnp.where(valid[inv], compact_map[inv], -1)
            n_clusters = compact_map[-1] + 1
        return labels, compact_labels, sizes_padded, n_clusters

    return kernel
# ---------- core impl (no int(...) casts!) ----------
def _fof_impl(pos, b: float, k: int, max_iters: int, min_size: int = 5, cuda: bool = True):
    kernel = get_fof_clusters_kernel(k, max_iters, cuda)  # k/max_iters/cuda are Python here
    labels, compact_labels, sizes_padded, nC = kernel(pos, b)

    keep  = sizes_padded >= min_size
    remap = jnp.where(keep, jnp.cumsum(keep.astype(jnp.int32)) - 1, -1)
    lab2  = jnp.where(compact_labels >= 0, remap[compact_labels], -1)
    valid = lab2 >= 0
    nC2   = jnp.max(jnp.where(valid, lab2, -1)) + 1
    return FrozenLabels(labels=lab2, n_clusters=nC2, valid_mask=valid)

# ---------- custom_vjp with nondiff_argnums ----------
# k, max_iters, min_size, cuda are STATIC/NON-DIFF by API
@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5)) 
def fof_frozen(pos, b: float, k: int, max_iters: int, min_size: int = 5, cuda: bool = True):
    return _fof_impl(pos, b, k, max_iters, min_size, cuda)

def _fof_fwd(k, max_iters, min_size, cuda, pos, b):
    out = _fof_impl(pos, b, k, max_iters, min_size, cuda)
    return out, ((pos.shape, pos.dtype),)

def _fof_bwd(k, max_iters, min_size, cuda, res, ct):
    (pos_shape, pos_dtype), = res
    dpos = jnp.zeros(pos_shape, dtype=pos_dtype)  # zero grad wrt pos
    # No grads for b/k/max_iters/min_size/cuda
    return (dpos, None)

fof_frozen.defvjp(_fof_fwd, _fof_bwd)

# -------- builder with b baked in --------
@lru_cache(maxsize=None)
def make_fof_frozen(k: int, max_iters: int, min_size: int = 5, cuda: bool = True, b: float = 0.19):
    """
    Returns fof_fn(pos) -> FrozenLabels with a custom VJP.
    All statics (k, max_iters, min_size, cuda, b) are captured in the closure.
    """
    kernel = _get_fof_clusters_kernel_cached(int(k), int(max_iters), bool(cuda))
    min_size_const = int(min_size)
    # Normalize b ONCE here to a plain Python float / JAX scalar
    b_const = float(np.asarray(b).reshape(()).item())  # handles python, 0-d, length-1
    b_scalar = jnp.asarray(b_const, dtype=jnp.float32)

    def _impl(pos):
        labels, compact_labels, sizes_padded, _ = kernel(pos, b_scalar)
        keep  = sizes_padded >= min_size_const
        remap = jnp.where(keep, jnp.cumsum(keep.astype(jnp.int32)) - 1, -1)
        lab2  = jnp.where(compact_labels >= 0, remap[compact_labels], -1)
        valid = lab2 >= 0
        nC2   = jnp.max(jnp.where(valid, lab2, -1)) + 1
        return FrozenLabels(labels=lab2, n_clusters=nC2, valid_mask=valid)

    @jax.custom_vjp
    def fof(pos):
        return _impl(pos)
    
    def fwd(pos):
        out = _impl(pos)
        # Save shape as plain Python ints + a zero-sized scalar token to carry dtype
        shape_token = tuple(pos.shape)  # OK to stash Python ints
        dtype_token = jnp.zeros((), dtype=pos.dtype)  # JAX array, carries dtype safely
        return out, (shape_token, dtype_token)
    
    def bwd(res, ct):
        shape_token, dtype_token = res
        dpos = jnp.zeros(shape_token, dtype=dtype_token.dtype)  # build zeros with correct dtype
        return (dpos,)
    
    fof.defvjp(fwd, bwd)
    return fof
    
fof_fn = make_fof_frozen(k=16, max_iters=300, min_size=5, b=0.20*0.95, cuda=True)

def _is_dtype_object(x) -> bool:
    # True for dtype instances (e.g., np.dtype('float32'))
    if isinstance(x, np.dtype):
        return True
    # True for dtype classes (e.g., np.float32, jnp.float32)
    if isinstance(x, type) and np.issctype(x):
        return True
    # True for NumPy scalar instances (e.g., np.float32(1.0)) — these are ok if numeric,
    # but if someone passes the scalar itself instead of an array, it would reshape to 0-D.
    # We don't want to accept that as "positions".
    if isinstance(x, np.generic):
        return True
    return False

def wrap_pos_arg(fn):
    def _call(pos):
        # Reject dtype objects / scalar dtypes
        if _is_dtype_object(pos):
            raise TypeError(
                "fof_fn(pos): 'pos' must be an array, not a dtype or numpy scalar "
                "(e.g., np.float32/jnp.float32 or np.float32(1.0))."
            )
        # Enforce array shape (N,2) or (N,3), float32
        pos = jnp.asarray(pos)
        if pos.ndim != 2 or pos.shape[1] not in (2, 3):
            raise TypeError(f"fof_fn(pos): expected pos shape (N,2) or (N,3), got {pos.shape}")
        pos = pos.astype(jnp.float32, copy=False)
        return fn(pos)
    return _call

class FrozenMeans(NamedTuple):
    means_full: jnp.ndarray     # (N, D)
    n_clusters: jnp.int32       # ()
    valid_clusters: jnp.ndarray # (N,) bool
def _seg_add(vals, labels, size):  # size is static (use N), labels may be traced
    return jnp.zeros((size, vals.shape[-1]), vals.dtype).at[labels].add(vals)

@jax.custom_vjp
def frozen_groupby_mean(x, fl: FrozenLabels):
    """
    Returns per-cluster means packed into a fixed shape (N, D).
    Only the first fl.n_clusters rows are valid cluster centers.
    """
    N, D = x.shape
    labels = jnp.where(fl.valid_mask, fl.labels, 0)  # [N], 0..K-1 for valid, 0 for invalid
    # counts: scatter into length-N (static) vector
    cnt = jnp.zeros((N,), x.dtype).at[labels].add(fl.valid_mask.astype(x.dtype))
    # sums: scatter into (N, D) (static)
    sums = _seg_add(jnp.where(fl.valid_mask[:, None], x, 0), labels, N)
    means_full = sums / jnp.clip(cnt[:, None], 1., None)  # (N, D)
    valid_clusters = jnp.arange(N) < fl.n_clusters       # (N,) bool
    return FrozenMeans(means_full=means_full, n_clusters=fl.n_clusters, valid_clusters=valid_clusters)

def _fgm_fwd(x, fl):
    N, D = x.shape
    labels = jnp.where(fl.valid_mask, fl.labels, 0)
    cnt = jnp.zeros((N,), x.dtype).at[labels].add(fl.valid_mask.astype(x.dtype))
    sums = _seg_add(jnp.where(fl.valid_mask[:, None], x, 0), labels, N)
    means_full = sums / jnp.clip(cnt[:, None], 1., None)
    valid_clusters = jnp.arange(N) < fl.n_clusters
    out = FrozenMeans(means_full=means_full, n_clusters=fl.n_clusters, valid_clusters=valid_clusters)
    # Save only what we need for backward
    return out, (labels, cnt, fl.valid_mask)

def _fgm_bwd(res, g_out: FrozenMeans):
    labels, cnt, valid_pts = res
    # Backprop only through x; no grads w.r.t. fl fields
    inv = (1.0 / jnp.clip(cnt, 1., None))[:, None]  # (N,1)
    # g_out.means_full has shape (N,D). Every point i contributes gradient equal to
    # the gradient of its cluster mean divided by cluster count, if that point was valid.
    per_i = g_out.means_full[labels] * inv[labels]            # [N,D]
    return (jnp.where(valid_pts[:, None], per_i, 0), None)

frozen_groupby_mean.defvjp(_fgm_fwd, _fgm_bwd)

#remove eventually...
fof_fn = make_fof_frozen(k=16, max_iters=300, min_size=5, b=0.20*0.95, cuda=True)

fof_fn = wrap_pos_arg(fof_fn)   # use this everywhere
