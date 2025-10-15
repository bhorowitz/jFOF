"""
Slabbed FoF on GPU (precompute per slab + global fusion), based on foflc.py

Usage
-----
from fof.foflc import fof_clusters_grid_periodic
from foflc_slab_precompute import fof_slabbed_precompute

labels, compact_labels, sizes, n_halos = fof_slabbed_precompute(
    pos,                 # (N,3) float32 positions
    L=box_size,          # float or (3,) box lengths
    b=linking_len,
    n_slabs=16,          # number of x-slabs (tune for VRAM)
    max_per_cell=64,
    k_max=64,
    max_iters=50,
)

Notes
-----
- Each slab runs fof_clusters_grid_periodic(mode="precompute") on its working set (center + ±b ghosts).
- We then *fuse* components across slabs by unifying labels for duplicate particles that appear in the ghost overlaps.
- Final labels are returned for *all* particles (global order), then compacted and sized.

This scaffold emphasizes clarity and determinism over extreme micro-optimizations... Basically I'm bad at optimizing :D 
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import jax
import jax.numpy as jnp

# We rely on the existing, kernels in foflc.py
from fof.foflc import fof_clusters_grid_periodic


@dataclass
class SlabResult:
    slab_id: int
    center_gids: np.ndarray        # (Nc,) global particle ids owned by this slab (center region only)
    center_local_labels: np.ndarray  # (Nc,) local labels from this slab's run
    work_gids: np.ndarray          # (Nw,) global particle ids for the working set (center+ghost)
    work_local_labels: np.ndarray  # (Nw,) local labels for the working set


def _as_L3_np(L) -> np.ndarray:
    L = np.asarray(L, dtype=np.float32)
    if L.ndim == 0:
        L = np.array([L, L, L], dtype=np.float32)
    assert L.shape == (3,), "L must be scalar or length-3"
    return L


def _wrap_pos_np(pos: np.ndarray, L3: np.ndarray) -> np.ndarray:
    return np.mod(pos, L3)


def _slab_ranges_x(Lx: float, n_slabs: int) -> List[Tuple[float, float]]:
    # half-open slabs [x_i, x_{i+1})
    edges = np.linspace(0.0, float(Lx), n_slabs + 1, dtype=np.float32)
    return [(float(edges[i]), float(edges[i+1])) for i in range(n_slabs)]


def _indices_in_x_band(pos_x: np.ndarray, Lx: float, x0: float, x1: float) -> np.ndarray:
    """Return indices of particles in the periodic x-interval [x0, x1).
    Supports x0 <= x1 and wrap-around intervals where x1 < x0.
    """
    if x1 >= x0:
        return np.nonzero((pos_x >= x0) & (pos_x < x1))[0]
    else:
        # wrap: [x0, L) U [0, x1)
        return np.nonzero((pos_x >= x0) | (pos_x < x1))[0]


def _build_slab_workset(
    pos: np.ndarray, L3: np.ndarray, slab: Tuple[float, float], b: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute working-set indices and masks for a slab.

    Returns
    -------
    work_idx : (Nw,) indices into global pos for this slab's working set (center + ghosts)
    center_mask : (Nw,) bool mask indicating which rows are in the slab "center" (owned)
    center_idx_global : (Nc,) indices (subset of work_idx) that belong to center
    """
    Lx = float(L3[0])
    x0, x1 = slab  # center interval

    # Thickness epsilon to be safe against rounding
    eps = float(max(1e-6 * b, np.finfo(np.float32).eps))

    # Working set extends by b in both directions (periodic)
    work_left = (x0 - b - eps) % Lx
    work_right = (x1 + b + eps) % Lx

    x = pos[:, 0]
    work_idx = _indices_in_x_band(x, Lx, work_left, work_right)
    center_idx = _indices_in_x_band(x, Lx, x0, x1)

    # Build mask over work_idx to mark center rows
    center_set = set(center_idx.tolist())
    center_mask = np.fromiter((i in center_set for i in work_idx), dtype=bool)

    return work_idx.astype(np.int64), center_mask, center_idx.astype(np.int64)


def _union_find(n: int):
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union, parent


def fof_slabbed_precompute(
    pos: np.ndarray | jnp.ndarray,
    *,
    L: float | Tuple[float, float, float],
    b: float,
    n_slabs: int = 16,
    max_per_cell: int = 64,
    k_max: int = 64,
    max_iters: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Friends-of-Friends via slabbed precompute + cross-slab fusion.

    Parameters
    ----------
    pos : (N,3) float32 array-like
        Particle positions. Will be wrapped into [0,L) periodically.
    L : float or (3,) float
        Box size. If scalar, will be expanded to (L,L,L).
    b : float
        Linking length.
    n_slabs : int
        Number of slabs along x. Tune for VRAM; 8–64 is typical on 80GB GPUs.
    Other params mirror foflc.fof_clusters_grid_periodic for precompute mode.

    Returns
    -------
    labels : (N,) int32
        Global (uncompacted) labels (min-representative ids). Deterministic.
    compact_labels : (N,) int32
        Dense 0..(n_halos-1) labels.
    sizes : (N,) int32
        Sizes per compact label, padded to N (0 where label is invalid).
    n_halos : int
        Number of compact halos.
    """
    # Host numpy for orchestration
    pos_np = np.asarray(pos, dtype=np.float32)
    assert pos_np.ndim == 2 and pos_np.shape[1] == 3

    L3 = _as_L3_np(L)
    pos_np = _wrap_pos_np(pos_np, L3)

    N = pos_np.shape[0]
    L3 = _as_L3_np(L)
    x_ranges = _slab_ranges_x(float(L3[0]), n_slabs)

    # Global ids are 0..N-1
    gids = np.arange(N, dtype=np.int64)

    slab_results: List[SlabResult] = []

    # Per-slab processing (can be parallelized at the driver level)
    for s_id, slab in enumerate(x_ranges):
        work_idx, center_mask, center_idx = _build_slab_workset(pos_np, L3, slab, float(b))
        pos_work = pos_np[work_idx]

        # Run precompute FoF on GPU for this working set
        labels, _, _, _ = fof_clusters_grid_periodic(
            jnp.asarray(pos_work, jnp.float32),
            L=jnp.asarray(L3, jnp.float32),
            mode="precompute",
            b=float(b),
            max_per_cell=int(max_per_cell),
            k_max=int(k_max),
            max_iters=int(max_iters),
            # batched builders inside foflc keep memory bounded
            batch_size=0,   # unused in precompute
            k_iter=0,       # unused in precompute
        )
        labels_np = np.asarray(labels, dtype=np.int32)

        work_gids = gids[work_idx]
        center_gids = gids[work_idx[center_mask]]
        slab_results.append(
            SlabResult(
                slab_id=s_id,
                center_gids=center_gids,
                center_local_labels=labels_np[center_mask],
                work_gids=work_gids,
                work_local_labels=labels_np,
            )
        )

    # Build mapping from (slab_id, local_label) -> node id in union-find graph
    node_id: Dict[Tuple[int, int], int] = {}
    def get_node(sid: int, lab: int) -> int:
        key = (sid, int(lab))
        nid = node_id.get(key)
        if nid is None:
            nid = len(node_id)
            node_id[key] = nid
        return nid

    # Gather edges by matching duplicate particles across slabs (ghost overlaps)
    # For each global id present in multiple slabs' *working sets*, union the corresponding component labels.
    gid_to_occurs: Dict[int, List[Tuple[int, int]]] = {}
    for sr in slab_results:
        for g, lab in zip(sr.work_gids.tolist(), sr.work_local_labels.tolist()):
            gid_to_occurs.setdefault(int(g), []).append((sr.slab_id, int(lab)))

    # Initialize union-find over all possible nodes (lazy: create on the fly)
    # First, count an upper bound of nodes
    # We will add nodes during unioning; to do this with a fixed UF, we enumerate all keys first.
    for occurs in gid_to_occurs.values():
        # sort & unique by (slab, label) to avoid self-unions
        uniq = sorted(set(occurs))
        for sid, lab in uniq:
            get_node(sid, lab)
    find, union, parent = _union_find(len(node_id))

    # Add union edges for any gid that appears in >= 2 slabs
    for occurs in gid_to_occurs.values():
        uniq = sorted(set(occurs))
        if len(uniq) < 2:
            continue
        base = get_node(*uniq[0])
        for sid, lab in uniq[1:]:
            union(base, get_node(sid, lab))

    # Finalize representative id per node (choose min node id as canonical)
    # Path compression done in find(); build map rep -> compact representative
    reps = {i: find(i) for i in range(len(parent))}
    # For determinism across runs, rebase representatives to the minimal (slab,label) by sorting keys
    rep_to_canon: Dict[int, int] = {}
    # Canonical key order is by minimal (slab,label) tuple among members
    members: Dict[int, List[Tuple[int, int, int]]] = {}
    for (sid, lab), nid in node_id.items():
        r = reps[nid]
        members.setdefault(r, []).append((sid, lab, nid))
    for r, items in members.items():
        items.sort()  # sort by (slab, label)
        rep_to_canon[r] = items[0][2]  # choose nid of lexicographically smallest (sid,lab)

    # Build lookup: (slab,label) -> canonical node id
    canon_of: Dict[Tuple[int, int], int] = {}
    for (sid, lab), nid in node_id.items():
        r = reps[nid]
        canon_of[(sid, lab)] = rep_to_canon[r]

    # Produce global labels for *owned* (center) particles only
    global_labels = np.full(N, -1, dtype=np.int64)
    for sr in slab_results:
        canon_ids = np.array([canon_of[(sr.slab_id, int(l))] for l in sr.center_local_labels], dtype=np.int64)
        global_labels[sr.center_gids] = canon_ids

    # Safety: all particles should be assigned exactly once (each gid is owned by exactly one center slab)
    assert np.all(global_labels >= 0), "Some particles were not assigned a global label (check slab ownership and overlap)."

    # Optionally, rebase to a more physical deterministic id: min global gid in each canonical set
    # Compute min gid per canonical id using a single pass
    uniq_canon, inv = np.unique(global_labels, return_inverse=True)
    # Map canon -> min gid
    min_gid_per_canon = np.full(uniq_canon.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
    # Scatter-min via numpy
    for gid, cid in enumerate(inv):
        if gid == 0 and False:
            pass
        if global_labels[gid] >= 0:
            if gid < min_gid_per_canon[cid]:
                min_gid_per_canon[cid] = gid
    # Replace labels by min-gid canonical ids for human-friendly determinism
    canonical_min = min_gid_per_canon[inv]
    labels_out = canonical_min.astype(np.int64)

    # Compact labels and sizes (mirror foflc.py semantics)
    uniq, inv = np.unique(labels_out, return_inverse=True)
    n_halos = uniq.shape[0]
    compact_labels = inv.astype(np.int32)
    sizes = np.bincount(compact_labels, minlength=n_halos).astype(np.int32)
    sizes_padded = sizes[compact_labels]

    return labels_out.astype(np.int64), compact_labels, sizes_padded, int(n_halos)


# --- Multi-GPU parallel version ---
# Run slabs concurrently across available GPUs using JAX device contexts.
# This keeps the original fusion logic (duplicate-particle union) unchanged.

from concurrent.futures import ThreadPoolExecutor, as_completed


def _per_slab_precompute_jitted_frozen(b: float, max_per_cell: int, k_max: int, max_iters: int):
    """Return a cached jitted function with *static* hyperparameters closed over.
    This avoids passing Tracers into static args inside foflc.
    """
    b_f = float(b)
    mpc_i = int(max_per_cell)
    kmax_i = int(k_max)
    iters_i = int(max_iters)

    @jax.jit
    def _run(pos_work_jnp, L3_jnp):
        labels, _, _, _ = fof_clusters_grid_periodic(
            pos_work_jnp,
            L=L3_jnp,
            mode="precompute",
            b=b_f,
            max_per_cell=mpc_i,
            k_max=kmax_i,
            max_iters=iters_i,
            batch_size=0,
            k_iter=0,
        )
        return labels
    return _run


def fof_slabbed_precompute_multi(
    pos: np.ndarray | jnp.ndarray,
    *,
    L: float | Tuple[float, float, float],
    b: float,
    n_slabs: int = 16,
    max_per_cell: int = 64,
    k_max: int = 64,
    max_iters: int = 50,
    row_block_build: int = 1_000,  # kept for API symmetry, not used here
    row_block_label: int = 1_000,  # kept for API symmetry, not used here
    devices: List[jax.Device] | None = None,
    max_workers: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Multi-GPU slabbed FoF: dispatch each slab to a device in parallel threads.

    Each worker picks a device via ``with jax.default_device(device):`` and runs the
    jitted per-slab precompute kernel. After all slabs finish, we fuse components by
    unifying duplicate particles in the ±b overlaps.
    """
    # Inputs -> host numpy
    pos_np = np.asarray(pos, dtype=np.float32)
    assert pos_np.ndim == 2 and pos_np.shape[1] == 3

    L3 = _as_L3_np(L)
    pos_np = _wrap_pos_np(pos_np, L3)

    N = pos_np.shape[0]
    gids = np.arange(N, dtype=np.int64)
    x_ranges = _slab_ranges_x(float(L3[0]), n_slabs)

    # Devices & workers
    if devices is None:
        devices = [d for d in jax.devices() if d.platform in ("gpu", "tpu")]
        if not devices:
            devices = [jax.devices()[0]]  # CPU fallback
    if max_workers is None:
        max_workers = len(devices)

    run_jit = _per_slab_precompute_jitted_frozen(b, max_per_cell, k_max, max_iters)

    def _do_slab(s_id: int, slab: Tuple[float, float], device: jax.Device) -> SlabResult:
        work_idx, center_mask, _center_idx = _build_slab_workset(pos_np, L3, slab, float(b))
        pos_work = pos_np[work_idx]
        with jax.default_device(device):
            labels = run_jit(
                jnp.asarray(pos_work, jnp.float32),
                jnp.asarray(L3, jnp.float32),
            )
        labels_np = np.asarray(labels, dtype=np.int32)
        work_gids = gids[work_idx]
        center_gids = gids[work_idx[center_mask]]
        return SlabResult(
            slab_id=s_id,
            center_gids=center_gids,
            center_local_labels=labels_np[center_mask],
            work_gids=work_gids,
            work_local_labels=labels_np,
        )

    # Run slabs in parallel
    slab_results: List[SlabResult] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, slab in enumerate(x_ranges):
            dev = devices[i % len(devices)]
            futures.append(ex.submit(_do_slab, i, slab, dev))
        for fut in as_completed(futures):
            slab_results.append(fut.result())

    # ===== Cross-slab fusion (duplicate-particle union) =====
    node_id: Dict[Tuple[int, int], int] = {}
    def get_node(sid: int, lab: int) -> int:
        key = (sid, int(lab))
        nid = node_id.get(key)
        if nid is None:
            nid = len(node_id)
            node_id[key] = nid
        return nid

    gid_to_occurs: Dict[int, List[Tuple[int, int]]] = {}
    for sr in slab_results:
        for g, lab in zip(sr.work_gids.tolist(), sr.work_local_labels.tolist()):
            gid_to_occurs.setdefault(int(g), []).append((sr.slab_id, int(lab)))

    # Materialize nodes
    for occurs in gid_to_occurs.values():
        for sid, lab in sorted(set(occurs)):
            get_node(sid, lab)
    find, union, parent = _union_find(len(node_id))

    # Union labels that co-occur for the same global particle across slabs
    for occurs in gid_to_occurs.values():
        uniq = sorted(set(occurs))
        if len(uniq) < 2:
            continue
        base = get_node(*uniq[0])
        for sid, lab in uniq[1:]:
            union(base, get_node(sid, lab))

    reps = {i: find(i) for i in range(len(parent))}
    # Canonical representative per set: lexicographically smallest (slab,label)
    rep_to_canon: Dict[int, int] = {}
    members: Dict[int, List[Tuple[int, int, int]]] = {}
    for (sid, lab), nid in node_id.items():
        r = reps[nid]
        members.setdefault(r, []).append((sid, lab, nid))
    for r, items in members.items():
        items.sort()
        rep_to_canon[r] = items[0][2]

    canon_of: Dict[Tuple[int, int], int] = {}
    for (sid, lab), nid in node_id.items():
        canon_of[(sid, lab)] = rep_to_canon[reps[nid]]

    # Final labels for owned (center) particles only
    global_labels = np.full(N, -1, dtype=np.int64)
    for sr in slab_results:
        canon_ids = np.array([canon_of[(sr.slab_id, int(l))] for l in sr.center_local_labels], dtype=np.int64)
        global_labels[sr.center_gids] = canon_ids
    assert np.all(global_labels >= 0)

    # Compact + sizes
    uniq, inv = np.unique(global_labels, return_inverse=True)
    n_halos = uniq.shape[0]
    compact_labels = inv.astype(np.int32)
    sizes = np.bincount(compact_labels, minlength=n_halos).astype(np.int32)
    sizes_padded = sizes[compact_labels]

    # Rebase to min global gid per component (deterministic human-friendly id)
    min_gid_per_canon = np.full(uniq.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
    for gid, cid in enumerate(compact_labels):
        if gid < min_gid_per_canon[cid]:
            min_gid_per_canon[cid] = gid
    labels_out = min_gid_per_canon[compact_labels]

    return labels_out.astype(np.int64), compact_labels, sizes_padded, int(n_halos)
