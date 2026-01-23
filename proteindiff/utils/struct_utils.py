import torch
import torch.nn.functional as F

from proteindiff.types import Tuple, Float, Int, Bool, T


def normalize_vec(vec: Float[T, "..."]) -> Float[T, "..."]:
    """normalize vectors along last dim"""
    return F.normalize(vec, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def get_backbone(C: Float[T, "ZN 14 3"]) -> Float[T, "ZN 4 3"]:
    """extract N, CA, C and compute virtual CB from full coords"""
    n = C[:, 0, :]
    ca = C[:, 1, :]
    c = C[:, 2, :]

    b1 = ca - n
    b2 = c - ca
    b3 = torch.linalg.cross(b1, b2, dim=-1)

    cb = ca - 0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

    return torch.stack([n, ca, c, cb], dim=1)


@torch.no_grad()
def compute_frames(C_backbone: Float[T, "ZN 4 3"]) -> Tuple[Float[T, "ZN 3"], Float[T, "ZN 3 3"]]:
    """compute local reference frames from backbone coords. returns (origin, frames)"""
    n, ca, c, cb = torch.chunk(C_backbone, dim=1, chunks=4)

    # y points from ca to cb
    y = normalize_vec(cb - ca)

    # x is c-n projected onto plane normal to y
    cn = c - n
    x = normalize_vec(cn - y * torch.linalg.vecdot(cn, y, dim=-1).unsqueeze(-1))

    # z is cross product
    z = normalize_vec(torch.linalg.cross(x, y, dim=-1))

    frames = torch.cat([x, y, z], dim=1)  # ZN,3,3
    origin = cb.squeeze(1)  # ZN,3

    return origin, frames


def get_bb_vecs(C: Float[T, "ZN 14 3"]) -> Tuple[Float[T, "ZN 3"], Float[T, "ZN 3 3"]]:
    """compute Cb position and Ca->X unit vectors from full coords. returns (Cb, unit_vecs) where unit_vecs stacks [CaN, CaCb, CaC]"""
    n, ca, c = C[:, 0, :], C[:, 1, :], C[:, 2, :]

    # compute virtual Cb
    b1 = ca - n
    b2 = c - ca
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    cb = ca - 0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

    # unit vectors from Ca: [CaN, CaCb, CaC]
    CaN = normalize_vec(n - ca)
    CaCb = normalize_vec(cb - ca)
    CaC = normalize_vec(c - ca)

    unit_vecs = torch.stack([CaN, CaCb, CaC], dim=1)  # (ZN, 3, 3)

    return cb, unit_vecs


# --- Helper functions for coordinate reconstruction ---


def gram_schmidt(x: Float[T, "ZN 3"], y: Float[T, "ZN 3"]) -> Float[T, "ZN 3 3"]:
    """
    Orthonormalize two vectors using Gram-Schmidt to produce a rotation matrix.

    Args:
        x: First vector (will become first column after normalization)
        y: Second vector (will be orthogonalized against x)

    Returns:
        R: (ZN, 3, 3) orthonormal rotation matrix where columns are [e0, e1, e2]
    """
    e0 = F.normalize(x, p=2, dim=-1, eps=1e-8)  # (ZN, 3)

    # Orthogonalize y against e0
    y_proj = torch.sum(y * e0, dim=-1, keepdim=True) * e0
    e1 = F.normalize(y - y_proj, p=2, dim=-1, eps=1e-8)  # (ZN, 3)

    # Cross product for third axis
    e2 = torch.linalg.cross(e0, e1, dim=-1)  # (ZN, 3)

    # Stack as columns: R @ [1,0,0] = e0, R @ [0,1,0] = e1, etc.
    R = torch.stack([e0, e1, e2], dim=-1)  # (ZN, 3, 3)
    return R


def normalize_torsions(
    sin: Float[T, "ZN K"],
    cos: Float[T, "ZN K"],
    eps: float = 1e-8
) -> Tuple[Float[T, "ZN K"], Float[T, "ZN K"]]:
    """
    Normalize sin/cos pairs to lie on the unit circle.

    Args:
        sin: Sine values
        cos: Cosine values
        eps: Small value for numerical stability

    Returns:
        (sin_normalized, cos_normalized) on unit circle
    """
    norm = torch.sqrt(sin**2 + cos**2 + eps)
    return sin / norm, cos / norm


def torsion_to_frames(
    sin: Float[T, "ZN 4"],
    cos: Float[T, "ZN 4"]
) -> Float[T, "ZN 4 3 3"]:
    """
    Convert chi angle sin/cos to rotation matrices around the x-axis.

    The rotation matrix Rx(theta) rotates around the x-axis:
    [[1,    0,     0   ],
     [0,  cos, -sin],
     [0,  sin,  cos]]

    Args:
        sin: Sine of chi1-chi4 angles
        cos: Cosine of chi1-chi4 angles

    Returns:
        R: (ZN, 4, 3, 3) rotation matrices for each chi angle
    """
    ZN = sin.shape[0]
    device, dtype = sin.device, sin.dtype

    # Build rotation matrices for each of the 4 chi angles
    zeros = torch.zeros(ZN, 4, device=device, dtype=dtype)
    ones = torch.ones(ZN, 4, device=device, dtype=dtype)

    # Row 0: [1, 0, 0]
    row0 = torch.stack([ones, zeros, zeros], dim=-1)  # (ZN, 4, 3)
    # Row 1: [0, cos, -sin]
    row1 = torch.stack([zeros, cos, -sin], dim=-1)  # (ZN, 4, 3)
    # Row 2: [0, sin, cos]
    row2 = torch.stack([zeros, sin, cos], dim=-1)  # (ZN, 4, 3)

    R = torch.stack([row0, row1, row2], dim=-2)  # (ZN, 4, 3, 3)
    return R


def make_se3(R: Float[T, "ZN 3 3"], t: Float[T, "ZN 3"]) -> Float[T, "ZN 4 4"]:
    """
    Build a 4x4 SE(3) transformation matrix from rotation and translation.

    The matrix is:
    [[R, t],
     [0, 1]]

    Args:
        R: (ZN, 3, 3) rotation matrix
        t: (ZN, 3) translation vector

    Returns:
        T: (ZN, 4, 4) homogeneous transformation matrix
    """
    ZN = R.shape[0]
    device, dtype = R.device, R.dtype

    T = torch.zeros(ZN, 4, 4, device=device, dtype=dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def torsion_angles_to_frames(
    backbone_frame: Float[T, "ZN 4 4"],
    chi_rot: Float[T, "ZN 4 3 3"],
    default_frames: Float[T, "ZN 8 4 4"],
    chi_mask: Float[T, "ZN 4"],
) -> Float[T, "ZN 8 4 4"]:
    """
    Compose all 8 rigid group frames from backbone frame and chi rotations.

    Rigid groups:
    - 0: backbone (N, CA, C, CB) - identity relative to backbone
    - 1: pre-omega (unused, identity)
    - 2: phi (unused for atoms, only H)
    - 3: psi (O atom)
    - 4: chi1 group
    - 5: chi2 group
    - 6: chi3 group
    - 7: chi4 group

    Composition (following AF2 convention):
    - group 0: backbone_frame
    - group 1: backbone_frame @ default_frames[1] (identity in practice)
    - group 2: backbone_frame @ default_frames[2]
    - group 3: backbone_frame @ default_frames[3]
    - group 4 (chi1): backbone_frame @ default_frames[4] @ Rx(chi1)
    - group 5 (chi2): group4_frame @ default_frames[5] @ Rx(chi2)
    - group 6 (chi3): group5_frame @ default_frames[6] @ Rx(chi3)
    - group 7 (chi4): group6_frame @ default_frames[7] @ Rx(chi4)

    Args:
        backbone_frame: (ZN, 4, 4) backbone SE(3) transform
        chi_rot: (ZN, 4, 3, 3) chi rotation matrices around x-axis
        default_frames: (ZN, 8, 4, 4) default frames from AF2 constants
        chi_mask: (ZN, 4) mask for which chi angles are valid

    Returns:
        all_frames: (ZN, 8, 4, 4) SE(3) transform for each rigid group
    """
    ZN = backbone_frame.shape[0]
    device, dtype = backbone_frame.device, backbone_frame.dtype

    all_frames = torch.zeros(ZN, 8, 4, 4, device=device, dtype=dtype)

    # Group 0 (backbone): just the backbone frame
    all_frames[:, 0] = backbone_frame

    # Groups 1-3 (pre-omega, phi, psi): backbone @ default
    for i in range(1, 4):
        all_frames[:, i] = torch.matmul(backbone_frame, default_frames[:, i])

    # Groups 4-7 (chi1-chi4): chained composition
    # Convert chi rotations to 4x4 matrices
    chi_se3 = torch.zeros(ZN, 4, 4, 4, device=device, dtype=dtype)
    chi_se3[:, :, :3, :3] = chi_rot
    chi_se3[:, :, 3, 3] = 1.0

    # Chi1 frame: backbone @ default[4] @ Rx(chi1)
    prev_frame = torch.matmul(backbone_frame, default_frames[:, 4])
    rotated_frame = torch.matmul(prev_frame, chi_se3[:, 0])
    mask_4 = (chi_mask[:, 0] > 0.5).view(ZN, 1, 1).expand(-1, 4, 4)
    all_frames[:, 4] = torch.where(mask_4, rotated_frame, prev_frame)

    # Chi2 frame: chi1_frame @ default[5] @ Rx(chi2)
    prev_frame = torch.matmul(all_frames[:, 4], default_frames[:, 5])
    rotated_frame = torch.matmul(prev_frame, chi_se3[:, 1])
    mask_5 = (chi_mask[:, 1] > 0.5).view(ZN, 1, 1).expand(-1, 4, 4)
    all_frames[:, 5] = torch.where(mask_5, rotated_frame, prev_frame)

    # Chi3 frame: chi2_frame @ default[6] @ Rx(chi3)
    prev_frame = torch.matmul(all_frames[:, 5], default_frames[:, 6])
    rotated_frame = torch.matmul(prev_frame, chi_se3[:, 2])
    mask_6 = (chi_mask[:, 2] > 0.5).view(ZN, 1, 1).expand(-1, 4, 4)
    all_frames[:, 6] = torch.where(mask_6, rotated_frame, prev_frame)

    # Chi4 frame: chi3_frame @ default[7] @ Rx(chi4)
    prev_frame = torch.matmul(all_frames[:, 6], default_frames[:, 7])
    rotated_frame = torch.matmul(prev_frame, chi_se3[:, 3])
    mask_7 = (chi_mask[:, 3] > 0.5).view(ZN, 1, 1).expand(-1, 4, 4)
    all_frames[:, 7] = torch.where(mask_7, rotated_frame, prev_frame)

    return all_frames


def frames_to_atom14_pos(
    all_frames: Float[T, "ZN 8 4 4"],
    rigid_group_positions: Float[T, "ZN 14 3"],
    atom_to_group: Int[T, "ZN 14"],
) -> Float[T, "ZN 14 3"]:
    """
    Apply rigid group frames to reference atom positions.

    For each atom, look up which rigid group it belongs to, get that group's
    frame, and transform the reference position.

    Args:
        all_frames: (ZN, 8, 4, 4) SE(3) transform for each rigid group
        rigid_group_positions: (ZN, 14, 3) reference positions in local frame
        atom_to_group: (ZN, 14) which rigid group each atom belongs to

    Returns:
        atom14_pos: (ZN, 14, 3) transformed atom positions
    """
    ZN = all_frames.shape[0]
    device, dtype = all_frames.device, all_frames.dtype

    # Gather the frame for each atom based on its rigid group
    # atom_to_group: (ZN, 14) -> indices into all_frames[:, :, ...]
    # We need to gather frames: (ZN, 8, 4, 4) -> (ZN, 14, 4, 4)

    # Expand atom_to_group for gathering: (ZN, 14) -> (ZN, 14, 1, 1, 1)
    group_idx = atom_to_group.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (ZN, 14, 1, 1, 1)
    group_idx = group_idx.expand(-1, -1, 4, 4, 1).squeeze(-1)  # (ZN, 14, 4, 4)

    # Actually we need to use advanced indexing
    # Create batch indices
    batch_idx = torch.arange(ZN, device=device).unsqueeze(1).expand(-1, 14)  # (ZN, 14)

    # Gather frames for each atom
    atom_frames = all_frames[batch_idx, atom_to_group]  # (ZN, 14, 4, 4)

    # Convert reference positions to homogeneous coordinates
    ones = torch.ones(ZN, 14, 1, device=device, dtype=dtype)
    pos_homo = torch.cat([rigid_group_positions, ones], dim=-1)  # (ZN, 14, 4)

    # Apply transformation: frame @ pos (for each atom)
    # atom_frames: (ZN, 14, 4, 4), pos_homo: (ZN, 14, 4)
    # Result: (ZN, 14, 4), take first 3 components
    transformed = torch.einsum('znij,znj->zni', atom_frames, pos_homo)
    atom14_pos = transformed[:, :, :3]  # (ZN, 14, 3)

    return atom14_pos


def coords_from_txy_sincos(
    t: Float[T, "ZN 3"],
    x: Float[T, "ZN 3"],
    y: Float[T, "ZN 3"],
    sin: Float[T, "ZN 4"],
    cos: Float[T, "ZN 4"],
    labels: Int[T, "ZN"],
    restype_rigid_group_default_frame: Float[T, "21 8 4 4"],
    restype_atom14_rigid_group_positions: Float[T, "21 14 3"],
    restype_atom14_to_rigid_group: Int[T, "21 14"],
    restype_atom14_mask: Float[T, "21 14"],
    chi_angles_mask: Float[T, "21 4"],
) -> Tuple[Float[T, "ZN 14 3"], Bool[T, "ZN 14"]]:
    """
    Reconstruct atom14 coordinates from predicted frame vectors and torsion angles.

    This implements ESM3-style structure decoding using:
    - Backbone frame from translation (t) and Gram-Schmidt orthonormalized axes (x, y)
    - Chi angle rotations from sin/cos predictions
    - AF2 rigid group constants for atom placement

    Args:
        t: CA position (translation vector)
        x: Frame x-axis direction vector
        y: Frame y-axis direction vector (will be orthogonalized)
        sin: Sine of chi1-chi4 angles
        cos: Cosine of chi1-chi4 angles
        labels: Amino acid labels (0-19, or 20 for unknown)
        restype_rigid_group_default_frame: AF2 constant (21, 8, 4, 4)
        restype_atom14_rigid_group_positions: AF2 constant (21, 14, 3)
        restype_atom14_to_rigid_group: AF2 constant (21, 14)
        restype_atom14_mask: AF2 constant (21, 14)
        chi_angles_mask: AF2 constant (21, 4)

    Returns:
        atom14_coords: (ZN, 14, 3) reconstructed atom positions
        atom_mask: (ZN, 14) boolean mask of valid atoms
    """
    ZN = t.shape[0]

    # 1. Build backbone frame from gram_schmidt
    R = gram_schmidt(x, y)  # (ZN, 3, 3)
    backbone_frame = make_se3(R, t)  # (ZN, 4, 4)

    # 2. Normalize torsions to unit circle
    sin_n, cos_n = normalize_torsions(sin, cos)

    # 3. Build chi rotation matrices
    chi_frames = torsion_to_frames(sin_n, cos_n)  # (ZN, 4, 3, 3)

    # 4. Look up AF2 constants for this sequence
    default_frames = restype_rigid_group_default_frame[labels]  # (ZN, 8, 4, 4)
    rigid_pos = restype_atom14_rigid_group_positions[labels]  # (ZN, 14, 3)
    atom_to_group = restype_atom14_to_rigid_group[labels]  # (ZN, 14)
    atom_mask_float = restype_atom14_mask[labels]  # (ZN, 14)
    chi_mask = chi_angles_mask[labels]  # (ZN, 4)

    # 5. Compose all rigid group frames
    all_frames = torsion_angles_to_frames(backbone_frame, chi_frames, default_frames, chi_mask)  # (ZN, 8, 4, 4)

    # 6. Apply frames to reference positions
    atom14_pos = frames_to_atom14_pos(all_frames, rigid_pos, atom_to_group)  # (ZN, 14, 3)

    # Convert mask to boolean
    atom_mask = atom_mask_float > 0.5  # (ZN, 14)

    return atom14_pos, atom_mask


def kabsch_align(
    coords_mobile: Float[T, "N 3"],
    coords_target: Float[T, "N 3"],
) -> Float[T, "N 3"]:
    """
    Align mobile coordinates onto target coordinates using the Kabsch algorithm.

    Finds the optimal rotation and translation to minimize RMSD between
    the two point sets. Both inputs must have the same number of points.

    Args:
        coords_mobile: Coordinates to be aligned (N, 3)
        coords_target: Target coordinates to align onto (N, 3)

    Returns:
        aligned_coords: Mobile coordinates after optimal superposition (N, 3)
    """
    # Center both coordinate sets
    centroid_mobile = coords_mobile.mean(dim=0, keepdim=True)
    centroid_target = coords_target.mean(dim=0, keepdim=True)

    mobile_centered = coords_mobile - centroid_mobile
    target_centered = coords_target - centroid_target

    # Compute covariance matrix H = mobile^T @ target
    H = mobile_centered.T @ target_centered  # (3, 3)

    # SVD of covariance matrix
    U, S, Vt = torch.linalg.svd(H)

    # Compute optimal rotation R = V @ U^T
    # Handle reflection case (det(R) = -1)
    d = torch.linalg.det(Vt.T @ U.T)
    sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d], device=coords_mobile.device, dtype=coords_mobile.dtype))
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    aligned = mobile_centered @ R.T + centroid_target

    return aligned


def compute_rmsd(
    coords_pred: Float[T, "ZN 14 3"],
    coords_true: Float[T, "ZN 14 3"],
    atom_mask: Bool[T, "ZN 14"],
) -> float:
    """
    Compute RMSD between predicted and true coordinates with masking.

    Coordinates are aligned using the Kabsch algorithm before computing RMSD,
    since protein coordinates are in arbitrary reference frames.

    Args:
        coords_pred: Predicted atom coordinates (ZN, 14, 3)
        coords_true: True atom coordinates (ZN, 14, 3)
        atom_mask: Boolean mask indicating which atoms to include (ZN, 14)

    Returns:
        rmsd: Root mean square deviation in Angstroms
    """
    # Ensure mask is boolean
    if atom_mask.dtype != torch.bool:
        atom_mask = atom_mask > 0.5

    # Flatten to (N, 3) for masked atoms only
    flat_mask = atom_mask.reshape(-1)  # (ZN * 14,)
    flat_pred = coords_pred.reshape(-1, 3)  # (ZN * 14, 3)
    flat_true = coords_true.reshape(-1, 3)  # (ZN * 14, 3)

    # Extract only valid atoms
    valid_pred = flat_pred[flat_mask]  # (N_valid, 3)
    valid_true = flat_true[flat_mask]  # (N_valid, 3)

    num_atoms = valid_pred.shape[0]
    if num_atoms == 0:
        return 0.0

    # Align coordinates using Kabsch algorithm
    valid_pred = kabsch_align(valid_pred, valid_true)

    # Compute RMSD
    sq_diff = (valid_pred - valid_true).pow(2).sum(dim=-1)  # (N_valid,)
    rmsd = torch.sqrt(sq_diff.mean())

    return rmsd.item()
