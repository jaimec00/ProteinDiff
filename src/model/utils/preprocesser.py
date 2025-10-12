import torch
import torch.nn as nn
from static.constants import amber_partial_charges, aa_2_lbl
from typing import Tuple

class PreProcesser(nn.Module):
	
	def __init__(self, voxel_dim: int=16, cell_dim: float=1.0) -> None:
		super().__init__()

		voxel_dims = (voxel_dim,)*3
		x_cells, y_cells, z_cells = voxel_dims
		voxel_x = torch.arange(x_cells).reshape(-1,1,1).expand(voxel_dims) - x_cells/2
		voxel_y = torch.arange(y_cells).reshape(1,-1,1).expand(voxel_dims) - y_cells/4 # more cells in positive y direction, since that is where the side chain is located
		voxel_z = torch.arange(z_cells).reshape(1,1,-1).expand(voxel_dims) - z_cells/2
		voxel = cell_dim * torch.stack([voxel_x, voxel_y, voxel_z], dim=3) # Vx, Vy, Vz, 3
		self.register_buffer("voxel", voxel)
		self.register_buffer("amber_partial_charges", torch.from_numpy(amber_partial_charges).to(torch.float32))
		self.res = cell_dim 

	@torch.no_grad()
	def forward(self, C: torch.Tensor, L: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor]:
		'''
		C (torch.Tensor): full atomic coordinates of shape (ZN,A,3)
		L (torch.Tensor): amino acid class labels of shape (ZN)
		atom_mask (torch.Tensor): mask indicating missing atom coordinates of shape (ZN,A)
		'''


		# get the backbone atoms, using virtual Cb
		C_backbone = PreProcesser.get_backbone(C) # ZN,4,3

		# compute unit vectors for each residue's local reference frame
		local_origins, local_frames = self.compute_frames(C_backbone) # ZN,3 and ZN,3,3

		# create the voxel for each residue by rotating the base voxel to the local frame and translating to local origin, 
		# simply contains the coordinates for the voxels
		local_voxels = self.compute_voxels(local_origins, local_frames) # ZN,Vx,Vy,Vz,3

		# compute electric fields 
		fields = self.compute_fields(C, L, local_voxels, atom_mask) # ZN,Vx,Vy,Vz,3

		# compute divergence of the normed fields, hoping this is easier for diffusion
		divergence = self.compute_divergence(fields) # ZN,1,Vx,Vy,Vz

		return C_backbone, divergence, local_frames

	@staticmethod
	def get_backbone(C):

		n = C[:, 0, :]
		ca = C[:, 1, :]
		c = C[:, 2, :]

		b1 = ca - n
		b2 = c - ca
		b3 = torch.linalg.cross(b1, b2, dim=1)

		cb = ca - 0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

		C_backbone = torch.stack([n, ca, c, cb], dim=1)

		return C_backbone

	def compute_frames(self, C_backbone):

		y = C_backbone[:, 3, :] - C_backbone[:, 1, :] # Cb - Ca
		y_unit = y / torch.linalg.vector_norm(y, dim=1, keepdim=True).clamp(min=1e-6)

		x_raw = C_backbone[:, 2, :] - C_backbone[:, 0, :] # C - N
		x_proj = torch.linalg.vecdot(x_raw, y_unit, dim=1) # project x_raw onto y
		x = x_raw - x_proj.unsqueeze(1) # subtract the projection from the original vector to get the projection of x onto the plane perpendicular to y
		x_unit = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp(min=1e-6)

		# already a unit vector, since x and y are already unit, but norm in case of numerical precision
		z = torch.linalg.cross(x_unit, y_unit) 
		z_unit = z / torch.linalg.vector_norm(z, dim=1, keepdim=True).clamp(min=1e-6)

		frames = torch.stack([x_unit, y_unit, z_unit], dim=1) # ZN,U,S

		# the origin is the beta carbon position
		origin = C_backbone[:, 3, :]

		return origin, frames

	def compute_voxels(self, origins, frames):

		ZN, S = origins.shape
		_, U, _ = frames.shape # U is the unit vectors dim
		Vx, Vy, Vz, _ = self.voxel.shape

		# rotate the voxel grid using the local frames, each unit vector (Uxyz) is multipled by the corresponding component and summed across U dim
		rotation = torch.sum(frames.reshape(ZN,1,1,1,U,S) * self.voxel.reshape(1,Vx,Vy,Vz,S,1), dim=4) # ZN,Vx,Vy,Vz,U

		# add the offset so the origin is the beta carbon
		local_voxels = origins.reshape(ZN,1,1,1,S) + rotation # ZN,Vx,Vy,Vz,S

		return local_voxels
		
	def compute_fields(self, C, L, voxels, atom_mask):

		# prep
		ZN = L.size(0)
		AA, A = self.amber_partial_charges.shape
		_, Vx, Vy, Vz, S = voxels.shape
		
		# replace invalid labels with X (has the same partial charges as glycine)
		L = L.masked_fill(L==-1, aa_2_lbl("X")) 

		# compute the electric field, using just basic point charge formula

		# now get distance vectors, as directionality is also used. points from atoms TO voxel cells
		dist_vectors =  voxels.reshape(ZN, Vx, Vy, Vz, 1, S) - C.reshape(ZN, 1, 1, 1, A, S) # ZN,Vx,Vy,Vz,A,S 

		# compute magnitudes
		dists = torch.linalg.vector_norm(dist_vectors, dim=-1, keepdim=True) # ZN,Vx,Vy,Vz,A,1

		# the distance term is the r^{hat} / |r|^2 = r / |r|^3. first dist is to make into unit vector, dist**2 is clamped to the cell resolution to avoid singularities
		dist_term = dist_vectors / (dists.masked_fill(dists==0,1)*(dists.clamp(min=self.res)**2)) # ZN,Vx,Vy,Vz,A,S

		# get partial charges, zero out masked atoms
		partial_charges = torch.gather(self.amber_partial_charges.reshape(1,AA,A).expand(ZN,AA,A), 1, L.reshape(ZN,1,1).expand(ZN,1,A)).squeeze(1) * atom_mask # ZN, A

		# now compute the final field, sum over atoms. 
		# no coulomb constant, since scaling to unit vector anyways
		fields = torch.sum(partial_charges.reshape(ZN, 1, 1, 1, A, 1) * dist_term.reshape(ZN, Vx, Vy, Vz, A, S), dim=4) # ZN,Vx,Vy,Vz,S

		# decidied to norm to unit vectors, thus the models job is to nudge them towards the true direction
		# also works well with latent diffusion, as the compression makes sense, since there is redundancy due to continuous nature
		fields_norm = torch.linalg.vector_norm(fields, dim=-1, keepdim=True)
		fields = fields / fields_norm.masked_fill(fields_norm==0, 1)

		# now reshape to ZN,S,Vx,Vy,Vz to be compatible with conv operatinos later
		fields = fields.permute(0,4,1,2,3)

		return fields

	def compute_divergence(self, fields):

		'''
		compute divergence of the electric field. field is normed so each cell has unit magnitude
		this helps in smoothing the field and having the model only focus on direction, not magnitude
		hope is that divergence, being a scalar valued function, will be easier to denoise, since unit vectors
		live on S2, requiring brownian motion and riemann manifold denoising
		divergence is computed using finite differences
		'''

		# div wrt x, also deal with boundaries
		dxc = (fields[:, 0, 2:, :, :] - fields[:, 0, :-2, :, :]) / (2*self.res)
		dx0 = (fields[:, 0, 1, :, :] - fields[:, 0, 0, :, :] ) / self.res
		dxn = (fields[:, 0, -1, :, :] - fields[:, 0, -2, :, :] ) / self.res
		dx = torch.cat([dx0.unsqueeze(1), dxc, dxn.unsqueeze(1)], dim=1)

		# div wrt y
		dyc = (fields[:, 1, :, 2:, :] - fields[:, 1, :, :-2, :]) / (2*self.res)
		dy0 = (fields[:, 1, :, 1, :] - fields[:, 1, :, 0, :]) / self.res
		dyn = (fields[:, 1, :, -1, :] - fields[:, 1, :, -2, :]) / self.res
		dy = torch.cat([dy0.unsqueeze(2), dyc, dyn.unsqueeze(2)], dim=2)

		# div wrt z
		dzc = (fields[:, 2, :, :, 2:] - fields[:, 2, :, :, :-2]) / (2*self.res)
		dz0 = (fields[:, 2, :, :, 1] - fields[:, 2, :, :, 0]) / self.res
		dzn = (fields[:, 2, :, :, -1] - fields[:, 2, :, :, -2]) / self.res
		dz = torch.cat([dz0.unsqueeze(3), dzc, dzn.unsqueeze(3)], dim=3)

		# sum
		div = dx + dy + dz

		# keep a channel dim
		div = div.unsqueeze(1)

		return div