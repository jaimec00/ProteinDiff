import torch
from unittest.mock import MagicMock

from proteus.types import Bool, Float, Int, T

if torch.cuda.is_available():
    import triton
    import triton.language as tl
else:
    triton = MagicMock()
    tl = MagicMock()


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=w)
        for w in [4, 8, 16]
    ],
    key=["BLOCK_A"],
    restore_value=["per_token_loss_ptr", "d_pred_coords_ptr"],
)
@triton.jit
def distance_loss_fwd_bwd(
    # inputs
    pred_coords_ptr, gt_coords_ptr, atom_mask_ptr, cu_seqlens_ptr,
    # outputs
    per_token_loss_ptr, d_pred_coords_ptr,
    # shapes
    BL, B, num_atoms,
    # strides - pred_coords (BL, num_atoms, 3)
    stride_pred_bl, stride_pred_a, stride_pred_xyz,
    # strides - gt_coords (BL, num_atoms, 3)
    stride_gt_bl, stride_gt_a, stride_gt_xyz,
    # strides - atom_mask (BL, num_atoms)
    stride_mask_bl, stride_mask_a,
    # strides - d_pred_coords (BL, num_atoms, 3)
    stride_dpred_bl, stride_dpred_a, stride_dpred_xyz,
    # clamp threshold
    clamp_max: tl.constexpr,
    # block size for atoms (power of 2, >= num_atoms)
    BLOCK_A: tl.constexpr,
):
    """Fused distance loss forward and backward.

    For each residue pair (i, j) in same sequence (including i == j):
      For each valid atom pair (a in i, b in j) where atom_mask[i,a] and atom_mask[j,b]:
        pred_d = ||pred_coords[i,a] - pred_coords[j,b]||
        gt_d = ||gt_coords[i,a] - gt_coords[j,b]||
        loss_pair = min((pred_d - gt_d)², clamp_max)

    Normalize: divide by valid_atom_pairs per residue i to get mean loss per atom pair
    Output: sum(per_token_loss) computed outside kernel

    Backward: compute gradients w.r.t. pred_coords only.
    Each pair (i,j) contributes to loss at both residue i (norm_i) and j (norm_j).
    Gradient: (1/va_i + 1/va_j) / total_valid_atoms
    """
    pid = tl.program_id(0)  # residue index i

    # find sequence bounds via linear search
    seq_start = 0
    seq_end = 0
    for s in range(B):
        start_s = tl.load(cu_seqlens_ptr + s)
        end_s = tl.load(cu_seqlens_ptr + s + 1)
        if start_s <= pid and pid < end_s:
            seq_start = start_s
            seq_end = end_s

    seq_len = seq_end - seq_start

    # atom offsets with masking for block bounds
    offs_a = tl.arange(0, BLOCK_A)
    block_mask = offs_a < num_atoms

    # Load atom_mask[i, :] - shape (BLOCK_A,) bool
    atom_mask_i = tl.load(
        atom_mask_ptr + pid * stride_mask_bl + offs_a * stride_mask_a,
        mask=block_mask, other=False
    )

    # Load pred_coords[i, :, :] and gt_coords[i, :, :] - shape (BLOCK_A, 3)
    pred_i_x = tl.load(
        pred_coords_ptr + pid * stride_pred_bl + offs_a * stride_pred_a + 0,
        mask=block_mask, other=0.0
    )
    pred_i_y = tl.load(
        pred_coords_ptr + pid * stride_pred_bl + offs_a * stride_pred_a + 1,
        mask=block_mask, other=0.0
    )
    pred_i_z = tl.load(
        pred_coords_ptr + pid * stride_pred_bl + offs_a * stride_pred_a + 2,
        mask=block_mask, other=0.0
    )
    gt_i_x = tl.load(
        gt_coords_ptr + pid * stride_gt_bl + offs_a * stride_gt_a + 0,
        mask=block_mask, other=0.0
    )
    gt_i_y = tl.load(
        gt_coords_ptr + pid * stride_gt_bl + offs_a * stride_gt_a + 1,
        mask=block_mask, other=0.0
    )
    gt_i_z = tl.load(
        gt_coords_ptr + pid * stride_gt_bl + offs_a * stride_gt_a + 2,
        mask=block_mask, other=0.0
    )

    # Count valid atoms at residue i
    valid_atoms_i = tl.sum(atom_mask_i)

    # accumulators for loss and valid pair count
    loss_acc = 0.0
    valid_pairs_acc = 0.0
    total_valid_atoms = 0.0  # sum of va_j for all j in sequence

    # gradient accumulators for (1/norm_i + 1/norm_j) scaling
    # A = raw gradient (will divide by va_i * total_valid_atoms after loop)
    # C = gradient pre-divided by va_j (will divide by total_valid_atoms after loop)
    A_x = tl.zeros([BLOCK_A], dtype=tl.float32)
    A_y = tl.zeros([BLOCK_A], dtype=tl.float32)
    A_z = tl.zeros([BLOCK_A], dtype=tl.float32)
    C_x = tl.zeros([BLOCK_A], dtype=tl.float32)
    C_y = tl.zeros([BLOCK_A], dtype=tl.float32)
    C_z = tl.zeros([BLOCK_A], dtype=tl.float32)

    # Loop over residues j in sequence (including self i == j)
    for j in range(seq_start, seq_end):
        # Load atom_mask[j, :] - shape (BLOCK_A,) bool
        atom_mask_j = tl.load(
            atom_mask_ptr + j * stride_mask_bl + offs_a * stride_mask_a,
            mask=block_mask, other=False
        )

        # Count valid atoms at residue j
        valid_atoms_j = tl.sum(atom_mask_j)
        total_valid_atoms += valid_atoms_j

        # Load pred_coords[j, :, :] and gt_coords[j, :, :] - shape (BLOCK_A,)
        pred_j_x = tl.load(
            pred_coords_ptr + j * stride_pred_bl + offs_a * stride_pred_a + 0,
            mask=block_mask, other=0.0
        )
        pred_j_y = tl.load(
            pred_coords_ptr + j * stride_pred_bl + offs_a * stride_pred_a + 1,
            mask=block_mask, other=0.0
        )
        pred_j_z = tl.load(
            pred_coords_ptr + j * stride_pred_bl + offs_a * stride_pred_a + 2,
            mask=block_mask, other=0.0
        )
        gt_j_x = tl.load(
            gt_coords_ptr + j * stride_gt_bl + offs_a * stride_gt_a + 0,
            mask=block_mask, other=0.0
        )
        gt_j_y = tl.load(
            gt_coords_ptr + j * stride_gt_bl + offs_a * stride_gt_a + 1,
            mask=block_mask, other=0.0
        )
        gt_j_z = tl.load(
            gt_coords_ptr + j * stride_gt_bl + offs_a * stride_gt_a + 2,
            mask=block_mask, other=0.0
        )

        # Compute pairwise distances: (BLOCK_A,) x (BLOCK_A,) -> (BLOCK_A, BLOCK_A)
        # pred_diff[a, b] = pred_i[a] - pred_j[b]
        pred_dx = pred_i_x[:, None] - pred_j_x[None, :]  # (BLOCK_A, BLOCK_A)
        pred_dy = pred_i_y[:, None] - pred_j_y[None, :]
        pred_dz = pred_i_z[:, None] - pred_j_z[None, :]
        pred_dist_sq = pred_dx * pred_dx + pred_dy * pred_dy + pred_dz * pred_dz
        pred_dist = tl.sqrt(pred_dist_sq + 1e-8)  # (BLOCK_A, BLOCK_A)

        gt_dx = gt_i_x[:, None] - gt_j_x[None, :]
        gt_dy = gt_i_y[:, None] - gt_j_y[None, :]
        gt_dz = gt_i_z[:, None] - gt_j_z[None, :]
        gt_dist_sq = gt_dx * gt_dx + gt_dy * gt_dy + gt_dz * gt_dz
        gt_dist = tl.sqrt(gt_dist_sq + 1e-8)  # (BLOCK_A, BLOCK_A)

        # Squared error and clamping
        diff = pred_dist - gt_dist  # (BLOCK_A, BLOCK_A)
        sq_err = diff * diff
        in_clamp = sq_err < clamp_max
        clamped_loss = tl.where(in_clamp, sq_err, clamp_max)

        # Mask for valid atom pairs: atom_mask_i[a] & atom_mask_j[b]
        pair_mask = atom_mask_i[:, None] & atom_mask_j[None, :]  # (BLOCK_A, BLOCK_A)
        clamped_loss = tl.where(pair_mask, clamped_loss, 0.0)

        # Accumulate loss and valid pair count
        loss_acc += tl.sum(clamped_loss)
        valid_pairs_acc += tl.sum(pair_mask.to(tl.float32))

        # === Backward ===
        # Raw gradient: d(sq_err)/d_pred_coords_ia = 2 * diff * (pred_ia - pred_jb) / pred_dist
        # grad_scale[a, b] = 2 * diff[a, b] / pred_dist[a, b] if in_clamp else 0
        raw_grad_scale = tl.where(in_clamp & pair_mask, 2.0 * diff / pred_dist, 0.0)

        # Partial gradients for this j
        partial_x = tl.sum(raw_grad_scale * pred_dx, axis=1)
        partial_y = tl.sum(raw_grad_scale * pred_dy, axis=1)
        partial_z = tl.sum(raw_grad_scale * pred_dz, axis=1)

        # Accumulate A (raw) and C (pre-divided by va_j)
        A_x += partial_x
        A_y += partial_y
        A_z += partial_z
        inv_va_j = 1.0 / tl.maximum(valid_atoms_j, 1.0)
        C_x += partial_x * inv_va_j
        C_y += partial_y * inv_va_j
        C_z += partial_z * inv_va_j

    # Loss normalization: divide by valid_pairs to get mean loss per atom pair
    norm_factor = tl.maximum(valid_pairs_acc, 1.0)
    loss_acc = loss_acc / norm_factor

    # Gradient normalization:
    # grad = (A / va_i + C) / total_valid_atoms
    inv_va_i = 1.0 / tl.maximum(valid_atoms_i, 1.0)
    grad_denom = tl.maximum(total_valid_atoms, 1.0)
    d_pred_i_x = (A_x * inv_va_i + C_x) / grad_denom
    d_pred_i_y = (A_y * inv_va_i + C_y) / grad_denom
    d_pred_i_z = (A_z * inv_va_i + C_z) / grad_denom

    # Zero out gradients for masked atoms
    d_pred_i_x = tl.where(atom_mask_i, d_pred_i_x, 0.0)
    d_pred_i_y = tl.where(atom_mask_i, d_pred_i_y, 0.0)
    d_pred_i_z = tl.where(atom_mask_i, d_pred_i_z, 0.0)

    # Store per-token loss
    tl.store(per_token_loss_ptr + pid, loss_acc)

    # Store gradients for d_pred_coords[i, :, :]
    tl.store(
        d_pred_coords_ptr + pid * stride_dpred_bl + offs_a * stride_dpred_a + 0,
        d_pred_i_x, mask=block_mask
    )
    tl.store(
        d_pred_coords_ptr + pid * stride_dpred_bl + offs_a * stride_dpred_a + 1,
        d_pred_i_y, mask=block_mask
    )
    tl.store(
        d_pred_coords_ptr + pid * stride_dpred_bl + offs_a * stride_dpred_a + 2,
        d_pred_i_z, mask=block_mask
    )


def distance_loss(
    pred_coords: Float[T, "BL num_atoms 3"],
    gt_coords: Float[T, "BL num_atoms 3"],
    atom_mask: Bool[T, "BL num_atoms"],
    cu_seqlens: Int[T, "B+1"],
    clamp_max: float = 25.0,
) -> Float[T, "1"]:
    return DistanceLoss.apply(pred_coords, gt_coords, atom_mask, cu_seqlens, clamp_max)


class DistanceLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        pred_coords: Float[T, "BL num_atoms 3"],
        gt_coords: Float[T, "BL num_atoms 3"],
        atom_mask: Bool[T, "BL num_atoms"],
        cu_seqlens: Int[T, "B+1"],
        clamp_max: float,
    ) -> Float[T, "1"]:
        # shapes
        BL, num_atoms, _ = pred_coords.shape
        B = cu_seqlens.shape[0] - 1

        # assertions
        assert pred_coords.shape == gt_coords.shape, (
            f"pred_coords {pred_coords.shape} != gt_coords {gt_coords.shape}"
        )
        assert pred_coords.shape[2] == 3, f"pred_coords last dim {pred_coords.shape[2]} != 3"
        assert atom_mask.shape == (BL, num_atoms), (
            f"atom_mask {atom_mask.shape} != ({BL}, {num_atoms})"
        )
        assert cu_seqlens.shape == (B + 1,), f"cu_seqlens {cu_seqlens.shape} != ({B + 1},)"
        assert pred_coords.is_cuda and gt_coords.is_cuda and cu_seqlens.is_cuda and atom_mask.is_cuda

        # get orig dtype
        pred_dtype = pred_coords.dtype

        # all computation in fp32
        pred_coords = pred_coords.to(torch.float32).contiguous()
        gt_coords = gt_coords.to(torch.float32).contiguous()
        atom_mask = atom_mask.bool().contiguous()
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()

        # allocate outputs (fp32)
        per_token_loss = torch.zeros(BL, device=pred_coords.device, dtype=torch.float32)
        d_pred_coords = torch.zeros(BL, num_atoms, 3, device=pred_coords.device, dtype=torch.float32)

        # block size for atoms (power of 2, >= num_atoms)
        BLOCK_A = max(16, triton.next_power_of_2(num_atoms))
        grid = (BL,)

        # launch kernel
        distance_loss_fwd_bwd[grid](
            # inputs
            pred_coords, gt_coords, atom_mask, cu_seqlens,
            # outputs
            per_token_loss, d_pred_coords,
            # shapes
            BL, B, num_atoms,
            # strides - pred_coords
            pred_coords.stride(0), pred_coords.stride(1), pred_coords.stride(2),
            # strides - gt_coords
            gt_coords.stride(0), gt_coords.stride(1), gt_coords.stride(2),
            # strides - atom_mask
            atom_mask.stride(0), atom_mask.stride(1),
            # strides - d_pred_coords
            d_pred_coords.stride(0), d_pred_coords.stride(1), d_pred_coords.stride(2),
            # clamp threshold
            clamp_max,
            # block size
            BLOCK_A,
        )

        # save grads for backward
        ctx.save_for_backward(d_pred_coords)
        ctx.pred_dtype = pred_dtype

        return per_token_loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        d_pred_coords, = ctx.saved_tensors
        pred_dtype = ctx.pred_dtype

        # scale by upstream grad and cast back to orig dtype
        d_pred_coords = (d_pred_coords * grad_output).to(pred_dtype)

        # None for gt_coords, atom_mask, cu_seqlens, clamp_max
        return d_pred_coords, None, None, None, None
