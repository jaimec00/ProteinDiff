import torch
import torch.nn.functional as F

from proteus.types import Tuple, Float, Int, Bool, T


def normalize_vec(vec: Float[T, "..."]) -> Float[T, "..."]:
    """normalize vectors along last dim"""
    return F.normalize(vec, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def get_backbone(C: Float[T, "BL 14 3"]) -> Float[T, "BL 4 3"]:
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
def compute_frames(
    C_backbone: Float[T, "B L 4 3"]
) -> Tuple[Float[T, "BL 3"], Float[T, "BL 3 3"]]:

    """compute local reference frames from backbone coords. returns (origin, frames)"""
    n, ca, c, cb = torch.chunk(C_backbone, dim=1, chunks=4)

    # y points from ca to cb
    y = normalize_vec(cb - ca)

    # x is c-n projected onto plane normal to y
    cn = c - n
    x = normalize_vec(cn - y * torch.linalg.vecdot(cn, y, dim=-1).unsqueeze(-1))

    # z is cross product
    z = normalize_vec(torch.linalg.cross(x, y, dim=-1))

    frames = torch.cat([x, y, z], dim=1)  # B,L,3,3
    origin = cb.squeeze(1)  # B,L,3

    return origin, frames