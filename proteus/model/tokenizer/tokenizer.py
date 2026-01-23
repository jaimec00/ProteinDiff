import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from proteus.model.base import Base
from proteus.static.constants import amber_partial_charges, aa_2_lbl
from proteus.types import Tuple, Float, Int, Bool, T
from proteus.utils.struct_utils import get_backbone, compute_frames

@dataclass
class TokenizerCfg:
	voxel_dim: int = 8
	cell_dim: float = 1.0

class Tokenizer(Base):
	
	def __init__(self, cfg: TokenizerCfg) -> None:
		super().__init__()

		voxel_dims = (cfg.voxel_dim,)*3
		x_cells, y_cells, z_cells = voxel_dims
		voxel_x = torch.arange(x_cells, dtype=torch.float32).reshape(-1,1,1).expand(voxel_dims) - x_cells/2
		voxel_y = torch.arange(y_cells, dtype=torch.float32).reshape(1,-1,1).expand(voxel_dims) - y_cells/4 # more cells in positive y direction, since that is where the side chain is located
		voxel_z = torch.arange(z_cells, dtype=torch.float32).reshape(1,1,-1).expand(voxel_dims) - z_cells/2
		voxel = cfg.cell_dim * torch.stack([voxel_x, voxel_y, voxel_z], dim=3) # Vx, Vy, Vz, 3
		self.register_buffer("voxel", voxel)
		self.register_buffer("amber_partial_charges", torch.from_numpy(amber_partial_charges).to(torch.float32))
		self.res = cfg.cell_dim

	@torch.no_grad()
	def forward(
		self,
		C: Float[T, "ZN 14 3"],
		L: Int[T, "ZN"],
		atom_mask: Bool[T, "ZN 14"]
	) -> Tuple[Float[T, "ZN 4 3"], Int[T, "ZN 1 Vx Vy Vz"], Float[T, "ZN 3 3"]]:
		'''
		C (torch.Tensor): full atomic coordinates of shape (ZN,A,3)
		L (torch.Tensor): amino acid class labels of shape (ZN)
		atom_mask (torch.Tensor): mask indicating missing atom coordinates of shape (ZN,A)
		'''

		# get the backbone atoms, using virtual Cb
		C_backbone = get_backbone(C) # ZN,4,3

		# compute unit vectors for each residue's local reference frame
		local_origins, local_frames = compute_frames(C_backbone) # ZN,3 and ZN,3,3

		# create the voxel for each residue by rotating the base voxel to the local frame and translating to local origin, 
		# simply contains the coordinates for the voxels
		local_voxels = self.compute_voxels(local_origins, local_frames) # ZN,Vx,Vy,Vz,3

		# compute electric fields 
		fields = self.compute_fields(C, L, local_voxels, atom_mask) # ZN,Vx,Vy,Vz,3

		# compute divergence of the normed fields, hoping this is easier for diffusion
		divergence = self.compute_divergence(fields) # ZN,1,Vx,Vy,Vz

		return C_backbone, divergence, local_frames

	@torch.no_grad()
	def compute_voxels(
		self,
		origins: Float[T, "ZN 3"],
		frames: Float[T, "ZN 3 3"]
	) -> Float[T, "ZN Vx Vy Vz 3"]:

		ZN, S = origins.shape
		_, U, _ = frames.shape # U is the unit vectors dim
		Vx, Vy, Vz, _ = self.voxel.shape

		# rotate the voxel grid using the local frames, each unit vector (Uxyz) is multipled by the corresponding component and summed across U dim
		rotation = torch.sum(frames.reshape(ZN,1,1,1,U,S) * self.voxel.reshape(1,Vx,Vy,Vz,S,1), dim=4) # ZN,Vx,Vy,Vz,U

		# add the offset so the origin is the beta carbon
		local_voxels = origins.reshape(ZN,1,1,1,S) + rotation # ZN,Vx,Vy,Vz,S

		return local_voxels
		
	@torch.no_grad()
	def compute_fields(
		self,
		C: Float[T, "ZN 14 3"],
		L: Int[T, "ZN"],
		voxels: Float[T, "ZN Vx Vy Vz 3"],
		atom_mask: Bool[T, "ZN 14"]
	) -> Float[T, "ZN 3 Vx Vy Vz"]:

		# prep
		ZN = L.size(0)
		AA, A = self.amber_partial_charges.shape
		_, Vx, Vy, Vz, S = voxels.shape

		# replace invalid labels with X (has the same partial charges as glycine)
		L.masked_fill_(L==-1, aa_2_lbl("X"))

		# compute the electric field, using just basic point charge formula

		# now get distance vectors, as directionality is also used. points from atoms TO voxel cells
		dist_vectors =  voxels.reshape(ZN, Vx, Vy, Vz, 1, S) - C.reshape(ZN, 1, 1, 1, A, S) # ZN,Vx,Vy,Vz,A,S

		# compute magnitudes
		dists = torch.linalg.vector_norm(dist_vectors, dim=-1, keepdim=True) # ZN,Vx,Vy,Vz,A,1

		# the distance term is the r^{hat} / |r|^2 = r / |r|^3. first dist is to make into unit vector, dist**2 is clamped to the cell resolution to avoid singularities
		dists.masked_fill_(dists==0, 1)
		dists_clamped = dists.clamp_(min=self.res)
		dist_term = dist_vectors / (dists * (dists_clamped**2)) # ZN,Vx,Vy,Vz,A,S

		# get partial charges, zero out masked atoms
		partial_charges = self.amber_partial_charges[L].reshape(ZN, A) * atom_mask # ZN, A

		# now compute the final field, sum over atoms.
		# no coulomb constant, since scaling to unit vector anyways
		fields = torch.sum(partial_charges.reshape(ZN, 1, 1, 1, A, 1) * dist_term, dim=4) # ZN,Vx,Vy,Vz,S

		# decidied to norm to unit vectors, thus the models job is to nudge them towards the true direction
		# also works well with latent diffusion, as the compression makes sense, since there is redundancy due to continuous nature
		fields_norm = torch.linalg.vector_norm(fields, dim=-1, keepdim=True)
		fields.div_(fields_norm.masked_fill(fields_norm==0, 1))

		# now reshape to ZN,S,Vx,Vy,Vz to be compatible with conv operatinos later
		fields = fields.permute(0,4,1,2,3)

		return fields

	@torch.no_grad()
	def compute_divergence(self, fields: Float[T, "ZN 3 Vx Vy Vz"]) -> Int[T, "ZN 1 Vx Vy Vz"]:

		'''
		compute divergence of the electric field. field is normed so each cell has unit magnitude
		this helps in smoothing the field and having the model only focus on direction, not magnitude
		hope is that divergence, being a scalar valued function, will be easier to denoise, since unit vectors
		live on S2, requiring brownian motion and riemann manifold denoising
		divergence is computed using finite differences
		'''

		# precompute inverse of resolutions for divisions
		inv_res = 1.0 / self.res
		inv_2res = 0.5 * inv_res

		# div wrt x, also deal with boundaries
		dxc = (fields[:, 0, 2:, :, :] - fields[:, 0, :-2, :, :]) * inv_2res
		dx0 = (fields[:, 0, 1, :, :] - fields[:, 0, 0, :, :]) * inv_res
		dxn = (fields[:, 0, -1, :, :] - fields[:, 0, -2, :, :]) * inv_res
		dx = torch.cat([dx0.unsqueeze(1), dxc, dxn.unsqueeze(1)], dim=1)

		# div wrt y
		dyc = (fields[:, 1, :, 2:, :] - fields[:, 1, :, :-2, :]) * inv_2res
		dy0 = (fields[:, 1, :, 1, :] - fields[:, 1, :, 0, :]) * inv_res
		dyn = (fields[:, 1, :, -1, :] - fields[:, 1, :, -2, :]) * inv_res
		dy = torch.cat([dy0.unsqueeze(2), dyc, dyn.unsqueeze(2)], dim=2)

		# div wrt z
		dzc = (fields[:, 2, :, :, 2:] - fields[:, 2, :, :, :-2]) * inv_2res
		dz0 = (fields[:, 2, :, :, 1] - fields[:, 2, :, :, 0]) * inv_res
		dzn = (fields[:, 2, :, :, -1] - fields[:, 2, :, :, -2]) * inv_res
		dz = torch.cat([dz0.unsqueeze(3), dzc, dzn.unsqueeze(3)], dim=3)

		# sum all divergences and add channel dim
		div = dx + dy + dz
		div.unsqueeze_(1)

		return div