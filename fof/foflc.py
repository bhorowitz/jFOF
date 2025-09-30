# Periodic Friends-of-Friends via linked-cell grid (JAX)
# mode="precompute"  -> builds nbr_idx (faster, more memory)
# mode="streaming"   -> no nbr_idx (slower, much lower memory, tunable batch)
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

# ---------- utils ----------
def _as_L3(L, dtype):
    L = jnp.asarray(L, dtype=dtype)
    if L.ndim == 0:
        L = jnp.stack([L, L, L])
    assert L.shape == (3,)
    return L

def _wrap_pos(pos, L3):
    return jnp.mod(pos, L3)

def _min_image(d, L3):
    return d - L3 * jnp.round(d / L3)

def _grid_setup(L3, b):
    L3 = jnp.asarray(L3, dtype=jnp.float32)
    b  = jnp.asarray(b,  dtype=jnp.float32)
    b  = jnp.maximum(b, jnp.finfo(b.dtype).tiny)
    n  = jnp.maximum(1, jnp.ceil(L3 / b).astype(jnp.int32))   # (3,)
    cell_size = L3 / n.astype(L3.dtype)
    return n, cell_size

def _to_cell_idx(pos, L3, n):
    frac = pos / L3
    ijk  = jnp.minimum((frac * n).astype(jnp.int32), n - 1)
    lin  = (ijk[:,2] * (n[0]*n[1]) + ijk[:,1] * n[0] + ijk[:,0]).astype(jnp.int32)
    return ijk, lin

def _to_cell_idx_single(x, L3, n):
    frac = x / L3
    ijk  = jnp.minimum((frac * n).astype(jnp.int32), n - 1)
    lin  = (ijk[2] * (n[0]*n[1]) + ijk[1] * n[0] + ijk[0]).astype(jnp.int32)
    return ijk, lin

def _neighbor_cells_of(ijk, n):
    offs = jnp.array(
        [[-1,-1,-1],[-1,-1,0],[-1,-1,1],
         [-1,0,-1],[-1,0,0],[-1,0,1],
         [-1,1,-1],[-1,1,0],[-1,1,1],
         [0,-1,-1],[0,-1,0],[0,-1,1],
         [0,0,-1],[0,0,0],[0,0,1],
         [0,1,-1],[0,1,0],[0,1,1],
         [1,-1,-1],[1,-1,0],[1,-1,1],
         [1,0,-1],[1,0,0],[1,0,1],
         [1,1,-1],[1,1,0],[1,1,1]], dtype=jnp.int32)
    return (ijk[None, :] + offs) % n  # (27,3)

def _cell_lin_ids(cells_ijk, n):
    return (cells_ijk[:,2] * (n[0]*n[1]) + cells_ijk[:,1] * n[0] + cells_ijk[:,0]).astype(jnp.int32)

def _lookup_cell_ranges_sorted(nbr_lin, sorted_cell_id):
    left  = jnp.searchsorted(sorted_cell_id, nbr_lin, side='left').astype(jnp.int32)
    right = jnp.searchsorted(sorted_cell_id, nbr_lin, side='right').astype(jnp.int32)
    count = (right - left).astype(jnp.int32)
    return left, count

# ---------- gather fixed-size candidates for one particle ----------
def _gather_candidates_for_particle(pid, pos, L3, n, order,
                                    sorted_cell_id, max_per_cell):
    pos_pid = pos[pid]                                   # (3,)
    ijk, _   = _to_cell_idx_single(pos_pid, L3, n)       # (3,)
    nbr_cells = _neighbor_cells_of(ijk, n)               # (27,3)
    nbr_lin   = _cell_lin_ids(nbr_cells, n)              # (27,)
    c_starts, c_counts = _lookup_cell_ranges_sorted(nbr_lin, sorted_cell_id)
    rng = jnp.arange(max_per_cell, dtype=jnp.int32)
    def take_from_cell(s, cnt):
        slice_ = lax.dynamic_slice_in_dim(order, s, max_per_cell)
        valid  = rng < jnp.minimum(cnt, max_per_cell)
        return jnp.where(valid, slice_, -jnp.ones_like(slice_))
    cand = jax.vmap(take_from_cell)(c_starts, c_counts)  # (27, max_per_cell)
    return cand.reshape(-1)                               # (27*max_per_cell,)

# ---------- PRECOMPUTE: neighbor list (N, k_max) with stable-partition ----------
def _build_neighbor_list(pos, L3, b, n, order, sorted_cell_id,
                         max_per_cell, k_max):
    N   = pos.shape[0]
    L3f = jnp.asarray(L3, jnp.float32)
    b2  = jnp.asarray(b,  jnp.float32)**2
    K   = 27 * max_per_cell

    def one(pid):
        cand = _gather_candidates_for_particle(pid, pos, L3f, n, order, sorted_cell_id, max_per_cell)  # (K,)
        is_valid  = (cand >= 0)
        cand_safe = jnp.where(is_valid, cand, pid)

        d   = _min_image(pos[cand_safe] - pos[pid], L3f)              # (K,3)
        keep = is_valid & (cand_safe != pid) & (jnp.sum(d*d, axis=1) <= b2)

        key = (~keep).astype(jnp.int32)
        key_sorted, cand_sorted = lax.sort((key, cand_safe), num_keys=1)

        top = cand_sorted[:k_max]
        kept = (key_sorted[:k_max] == 0)
        out = jnp.where(kept, top, -jnp.int32(1))
        return out

    return jax.vmap(one)(jnp.arange(N, dtype=jnp.int32))   # (N,k_max)

def _label_propagation_periodic(pos, L3, nbr_idx, b, max_iters):
    N, k = nbr_idx.shape
    labels = jnp.arange(N, dtype=jnp.int32)

    i   = jnp.arange(N, dtype=jnp.int32)
    src = jnp.repeat(i, k)
    dst = nbr_idx.reshape(-1)
    dst_ge0  = (dst >= 0)
    dst_safe = jnp.where(dst_ge0, dst, 0)

    d = _min_image(pos[src] - pos[dst_safe], jnp.asarray(L3, jnp.float32))
    within = (jnp.sum(d*d, axis=1) <= (jnp.asarray(b, jnp.float32)**2))
    valid = dst_ge0 & (src != dst_safe) & within

    big = jnp.int32(N)
    def step(lab):
        incoming = jnp.full((N,), big, dtype=jnp.int32)
        incoming = incoming.at[dst_safe].min(jnp.where(valid, lab[src], big))
        incoming = incoming.at[src].min(jnp.where(valid, lab[dst_safe], big))
        return jnp.minimum(lab, incoming)

    return lax.fori_loop(0, max_iters, lambda _, Lc: step(Lc), labels)

# ---------- STREAMING: no nbr_idx; relax in batches, k_iter neighbors per pass ----------
def _relax_once_streaming(pos, L3, b, n, order, sorted_cell_id,
                          max_per_cell, batch_size, k_iter, labels):
    """
    One relaxation pass (labels -> labels') without constructing nbr_idx.
    Uses fixed batch_size and masks out final partial batch to avoid
    concretization errors from dynamic jnp.arange.
    """
    N   = pos.shape[0]
    L3f = jnp.asarray(L3, jnp.float32)
    b2  = jnp.asarray(b,  jnp.float32)**2
    K   = 27 * max_per_cell
    k_iter = int(k_iter)  # static slice for JIT

    iota_B = lax.iota(jnp.int32, batch_size)  # (batch_size,)

    def batch_step(lab, batch_start):
        # Fixed-length indices for this batch, with mask for tail
        idx_full = batch_start + iota_B                         # (B,)
        valid_p  = idx_full < N                                 # (B,) bool
        idx      = jnp.where(valid_p, idx_full, jnp.int32(0))   # clamp for gathers

        # Gather fixed-size candidate list per particle
        def gather_one(p):
            return _gather_candidates_for_particle(
                p, pos, L3f, n, order, sorted_cell_id, max_per_cell
            )  # (K,)
        C = jax.vmap(gather_one)(idx)                           # (B,K)

        # Build masks (no boolean indexing)
        is_valid  = (C >= 0)                                    # (B,K)
        C_safe    = jnp.where(is_valid, C, idx[:,None])         # safe gather ids

        dp    = _min_image(pos[C_safe] - pos[idx][:,None,:], L3f)   # (B,K,3)
        keep  = is_valid & (C_safe != idx[:,None]) & (jnp.sum(dp*dp, axis=2) <= b2)

        # Stable partition: keeps first
        key = (~keep).astype(jnp.int32)                         # (B,K)
        key_sorted, C_sorted = lax.sort((key, C_safe), num_keys=1, dimension=-1)
        top    = C_sorted[:, :k_iter]                           # (B,k_iter)
        kept   = (key_sorted[:, :k_iter] == 0)
        q      = jnp.where(kept, top, -jnp.int32(1))            # (B,k_iter)

        # Scatter-min both directions with masks
        p_lab   = lab[idx]                                      # (B,)
        q_valid = (q >= 0) & valid_p[:,None]                    # also mask tail rows
        q_idx   = jnp.where(q_valid, q, jnp.int32(0))
        q_lab   = lab[q_idx]                                    # (B,k_iter)

        # p <- min(p, min q_lab) only for valid rows
        min_q = jnp.where(q_valid, q_lab, jnp.iinfo(jnp.int32).max).min(axis=1)
        new_p = jnp.minimum(p_lab, min_q)
        lab   = lab.at[idx].min(jnp.where(valid_p, new_p, p_lab))

        # q <- min(q, p_lab) only where valid edges
        lab = lab.at[q_idx].min(jnp.where(q_valid, p_lab[:,None], lab[q_idx]))
        return lab, None

    # Fixed step starts (concrete): 0, B, 2B, ...
    num_steps = (N + batch_size - 1) // batch_size
    steps = jnp.arange(num_steps, dtype=jnp.int32) * jnp.int32(batch_size)
    labels, _ = lax.scan(batch_step, labels, steps)
    return labels



def _build_neighbor_list_batched(pos, L3, b, n, order, sorted_cell_id,
                                 max_per_cell, k_max, row_block_build: int = 500_000):
    """
    Memory-safe neighbor construction:
      processes 'row_block_build' particles at a time, writes into nbr_idx_out.
    pos: (N,3) float32
    L3 : (3,)  float32
    Returns: (N, k_max) int32, -1 padded
    """
    N   = pos.shape[0]
    L3f = jnp.asarray(L3, jnp.float32)
    b2  = jnp.asarray(b,  jnp.float32)**2
    K   = 27 * max_per_cell

    # Preallocate output
    nbr_idx_out = -jnp.ones((N, k_max), dtype=jnp.int32)

    iota_K = lax.iota(jnp.int32, max_per_cell)
    iota_B = lax.iota(jnp.int32, row_block_build)

    def gather_candidates_for_particle(pid):
        # single-particle candidate gather (length K)
        pos_pid = pos[pid]
        # cell + neighbor cells
        frac = pos_pid / L3f
        ijk  = jnp.minimum((frac * n).astype(jnp.int32), n - 1)   # (3,)
        offs = jnp.array(
            [[-1,-1,-1],[-1,-1,0],[-1,-1,1],
             [-1,0,-1],[-1,0,0],[-1,0,1],
             [-1,1,-1],[-1,1,0],[-1,1,1],
             [0,-1,-1],[0,-1,0],[0,-1,1],
             [0,0,-1],[0,0,0],[0,0,1],
             [0,1,-1],[0,1,0],[0,1,1],
             [1,-1,-1],[1,-1,0],[1,-1,1],
             [1,0,-1],[1,0,0],[1,0,1],
             [1,1,-1],[1,1,0],[1,1,1]], dtype=jnp.int32)
        nbr_cells = (ijk[None, :] + offs) % n                     # (27,3)
        nbr_lin = (nbr_cells[:,2] * (n[0]*n[1]) + nbr_cells[:,1] * n[0] + nbr_cells[:,0]).astype(jnp.int32)  # (27,)

        # ranges via searchsorted on the global sorted cell ids
        s = jnp.searchsorted(sorted_cell_id, nbr_lin, side='left').astype(jnp.int32)   # (27,)
        e = jnp.searchsorted(sorted_cell_id, nbr_lin, side='right').astype(jnp.int32)  # (27,)
        cnt = (e - s).astype(jnp.int32)                                                # (27,)

        def take_from_cell(s_i, cnt_i):
            slice_ = lax.dynamic_slice_in_dim(order, s_i, max_per_cell)  # (max_per_cell,)
            valid  = iota_K < jnp.minimum(cnt_i, max_per_cell)
            return jnp.where(valid, slice_, -jnp.ones_like(slice_))      # (max_per_cell,)

        cand = jax.vmap(take_from_cell)(s, cnt)        # (27, max_per_cell)
        return cand.reshape(-1)                        # (K,)

    def block_step(state, start):
        nbr_idx_cur = state
        start = jnp.int32(start)
        end   = jnp.minimum(start + jnp.int32(row_block_build), jnp.int32(N))
        B     = end - start

        rows_full = start + iota_B                     # (row_block_build,)
        valid_row = rows_full < end                    # (row_block_build,)
        rows      = jnp.where(valid_row, rows_full, jnp.int32(0))  # clamp

        # gather candidates per row (B, K)
        C = jax.vmap(gather_candidates_for_particle)(rows)

        # filter & keep up to k_max per row (fixed shapes; no boolean indexing)
        is_valid  = (C >= 0)
        C_safe    = jnp.where(is_valid, C, rows[:,None])

        pi = pos[rows]                                 # (B,3)
        pj = pos[C_safe]                               # (B,K,3)
        d  = pj - pi[:,None,:]
        d  = d - L3f * jnp.round(d / L3f)
        within = (jnp.sum(d*d, axis=2) <= b2)
        keep = is_valid & (C_safe != rows[:,None]) & within  # (B,K)

        key = (~keep).astype(jnp.int32)
        key_sorted, C_sorted = lax.sort((key, C_safe), num_keys=1, dimension=-1)
        top = C_sorted[:, :k_max]                      # (B,k_max)
        kept = (key_sorted[:, :k_max] == 0)
        out_rows = jnp.where(kept, top, -jnp.int32(1)) # (B,k_max)

        # write into global output (masked on valid_row)
        nbr_idx_cur = nbr_idx_cur.at[rows].set(out_rows)
        return nbr_idx_cur, None

    n_steps = (N + row_block_build - 1) // row_block_build
    starts = jnp.arange(n_steps, dtype=jnp.int32) * jnp.int32(row_block_build)
    nbr_idx_out, _ = lax.scan(block_step, nbr_idx_out, starts)
    return nbr_idx_out

def _label_propagation_periodic_batched(pos, L3, nbr_idx, b, max_iters, row_block: int = 10_000):
    """
    Memory-safe FoF label propagation using row blocks.
    pos: (N,3) float32
    L3 : (3,)  float32
    nbr_idx: (N,k) int32  (-1 padded)
    """
    N, k = nbr_idx.shape
    L3   = jnp.asarray(L3, pos.dtype)
    b2   = jnp.asarray(b,  pos.dtype)**2
    labels = jnp.arange(N, dtype=jnp.int32)

    # fixed column index helper
    col_iota = lax.iota(jnp.int32, k)

    def block_step(lab, start):
        start = jnp.int32(start)
        end   = jnp.minimum(start + jnp.int32(row_block), jnp.int32(N))
        B     = end - start
        # build row indices for this block (fixed-length with mask)
        row = start + lax.iota(jnp.int32, row_block)          # (row_block,)
        row_valid = row < end
        row = jnp.where(row_valid, row, jnp.int32(0))

        # slice neighbors for this block
        nbr = nbr_idx[row]                                     # (row_block, k)
        valid = (nbr >= 0) & row_valid[:, None]                # mask out padded rows/cols

        # safe gather ids (avoid OOB)
        nbr_safe = jnp.where(valid, nbr, row[:, None])         # fallback to self

        # compute min-image distances for this block
        pi  = pos[row]                                         # (row_block, 3)
        pj  = pos[nbr_safe]                                    # (row_block, k, 3)
        d   = pj - pi[:, None, :]                              # (row_block, k, 3)
        d   = d - L3 * jnp.round(d / L3)
        within = (jnp.sum(d*d, axis=2) <= b2)
        keep = valid & (nbr_safe != row[:, None]) & within     # (row_block, k)

        # gather labels
        li = lab[row]                                          # (row_block,)
        lj = lab[nbr_safe]                                     # (row_block, k)

        # p <- min(p, min_j lj) over kept edges
        lj_masked = jnp.where(keep, lj, jnp.iinfo(jnp.int32).max)
        min_lj = jnp.min(lj_masked, axis=1)
        new_li = jnp.minimum(li, min_lj)
        lab = lab.at[row].min(jnp.where(row_valid, new_li, li))

        # q <- min(q, li) for kept edges
        # scatter in blocks: flatten kept edges
        q_idx = jnp.where(keep, nbr_safe, 0)                   # (row_block, k)
        li_rep = jnp.broadcast_to(li[:, None], (row_block, k))
        lab = lab.at[q_idx].min(jnp.where(keep, li_rep, lab[q_idx]))

        return lab, None

    # concrete starts: 0, row_block, 2*row_block, ...
    n_steps = (N + row_block - 1) // row_block
    starts = jnp.arange(n_steps, dtype=jnp.int32) * jnp.int32(row_block)

    def iter_body(_, lab0):
        lab1, _ = lax.scan(block_step, lab0, starts)
        return lab1

    labels = lax.fori_loop(0, max_iters, lambda _, L: iter_body(_, L), labels)
    return labels
    
def _label_propagation_streaming(pos, L3, b, n, order, sorted_cell_id,
                                 max_per_cell, max_iters, batch_size, k_iter):
    N = pos.shape[0]
    labels = jnp.arange(N, dtype=jnp.int32)
    def body(_, lab):
        return _relax_once_streaming(pos, L3, b, n, order, sorted_cell_id,
                                     max_per_cell, batch_size, k_iter, lab)
    return lax.fori_loop(0, max_iters, lambda _, Lc: body(_, Lc), labels)

# ---------- top-level ----------
@partial(jax.jit, static_argnames=("mode","b","max_per_cell","k_max","max_iters","batch_size","k_iter"))
def fof_clusters_grid_periodic(pos,
                               L,
                               *,
                               mode: str = "precompute",  # "precompute" | "streaming"
                               b: float = 1.0,
                               max_per_cell: int = 64,
                               k_max: int = 64,          # used in precompute mode
                               max_iters: int = 50,
                               batch_size: int = 8192,   # used in streaming mode
                               k_iter: int = 4           # neighbors relaxed per iter (streaming)
                               ):
    """
    Returns:
      labels:          (N,) int32
      compact_labels:  (N,) int32
      sizes_padded:    (N,) int32
      n_clusters:      scalar int32
    """
    # Use float32 to cut memory; keep indices int32
    pos = pos.astype(jnp.float32)
    L3  = _as_L3(L, pos.dtype)
    pos = _wrap_pos(pos, L3)

    n, _ = _grid_setup(L3, b)

    # cell ids & sort permutation
    _, cell_id = _to_cell_idx(pos, L3, n)                          # (N,), int32
    order = jnp.argsort(cell_id, stable=True).astype(jnp.int32)    # (N,)
    sorted_cell_id = cell_id[order]                                 # (N,)

    if mode == "precompute":
        nbr_idx = _build_neighbor_list_batched(
            pos, L3, b, n, order, sorted_cell_id,
            max_per_cell=max_per_cell, k_max=k_max,
            row_block_build=1_000   # tune to your GPU RAM
        )
        labels  = _label_propagation_periodic_batched(
            pos, L3, nbr_idx, b, max_iters, row_block=1_000
        )
    else:
        labels  = _label_propagation_streaming(pos, L3, b, n, order, sorted_cell_id,
                                               max_per_cell, max_iters, batch_size, k_iter)

    # compact cluster ids + sizes (stable; small memory)
    N = labels.shape[0]
    uniq, inv = jnp.unique(labels, size=N, fill_value=-1, return_inverse=True)
    valid = (uniq != -1)
    sizes_all = jnp.bincount(inv, length=N)
    sizes_padded = jnp.where(valid, sizes_all, 0)
    compact_map = (jnp.cumsum(valid.astype(jnp.int32)) - 1).astype(jnp.int32)
    compact_labels = jnp.where(valid[inv], compact_map[inv], -1)
    n_clusters = compact_map[-1] + 1
    return labels, compact_labels, sizes_padded, n_clusters