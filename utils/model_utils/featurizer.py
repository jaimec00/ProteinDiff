import torch
import torch.nn as nn
from data.constants import amber_partial_charges, aa_2_lbl

class Featurizer(nn.Module):
	
	def __init__(self, d_model=128, top_k=30, voxel_dims=(8,8,4), voxel_unit=1.0):
		super().__init__()

		self.register_buffer("amber_partial_charges", amber_partial_charges)

		x_cells, y_cells, z_cells = voxel_dims
		voxel_x = torch.arange(x_cells).view(-1,1,1).expand(voxel_dims) - x_cells/2
		voxel_y = torch.arange(y_cells).view(1,-1,1).expand(voxel_dims) - y_cells/4
		voxel_z = torch.arange(z_cells).view(1,1,-1).expand(voxel_dims) - y_cells/2
		voxel = voxel_unit * torch.stack([voxel_x, voxel_y, voxel_z], dim=3) # Vx, Vy, Vz, 3
		self.register_buffer("voxel", voxel)

		self.top_k = top_k

	def forward(C, L, atom_mask, kp_mask):
		'''
		C (torch.Tensor): full atomic coordinates of shape (Z,N,A,3)
		L (torch.Tensor): amino acid class labels of shape (Z,N)
		atom_mask (torch.Tensor): mask indicating missing atom coordinates of shape (Z,N,A)
		kp_mask (torch.Tensor): mask indicating padded positions of shape (Z,N)
		'''

		# get the backbone atoms, using virtual Cb
		C_backbone = get_backbone_atoms(C) # Z,N,4,3

		# compute unit vectors for each residue's local reference frame
		local_origins, local_frames = self.compute_frames(C_backbone) # Z,N,3

		# create the voxel for each residue, simply contais the coordinates
		local_voxels = self.compute_voxels(local_origins, local_frames) # Z,N,Vx,Vy,Vz,3

		# compute the nearest neighbors
		neighbors, nbr_mask = self.get_neighbors(C_backbone[:, :, 1, :], kp_mask) # use Ca coords

		# compute electric potentials 
		P = self.compute_potentials(C, L, voxels, neighbors, nbr_mask, atom_mask)

	def get_backbone(self, C):

		n = coords[:, :, 0, :]
		ca = coords[:, :, 1, :]
		c = coords[:, :, 2, :]

		b1 = ca - n
		b2 = c - ca
		b3 = torch.linalg.cross(b1, b2, dim=2)

		cb = ca - 0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

		C_backbone = torch.stack([n, ca, c, cb], dim=2)

		return C_backbone

	def compute_frames(self, C_backbone):

		y = C_backbone[:, :, 3, :] - C_backbone[:, :, 1, :] # Cb - Ca
		y_unit = y / torch.linalg.vector_norm(y, dim=2, keepdim=True).clamp(1e-6)

		x_raw = C_backbone[:, :, 2, :] - C_backbone[:, :, 0, :] # C - N
		x_proj = torch.linalg.vecdot(x, y) # project x_raw onto y
		x = x_raw - x_proj.unsqueeze(2) # subtract the projection from the original vector to get the projection of x onto the plane perpendicular to y
		x_unit = x / torch.linalg.vector_norm(x, dim=2, keepdim=True).clamp(1e-6)

		z_unit = torch.linalg.cross(x_unit, y_unit) # already a unit vector, since x and y are already unit

		frames = torch.stack([x_unit, y_unit, z_unit], dim=2)

		# the origin is the beta carbon position
		origin = C_backbone[:, :, 3, :]

		return origin, frames

	def compute_voxels(self, origins, frames):

		Z, N, S = origins.shape
		Vx, Vy, Vz, _ = self.voxel.shape

		local_voxels = origins.view(Z,N,1,1,1,S) + frames.view(Z,N,1,1,1,S)*self.voxel.view(1,1,Vx,Vy,Vz,S) # Z,N,Vx,Vy,Vz,S

		return local_voxels

	def get_neighbors(self, Ca, mask):

		dimZ, dimN, dimS = Ca.shape
		assert dimN>=self.top_k

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(1) - Ca.unsqueeze(2))**2, dim=3)) # Z x N x N
		dists = torch.where(dists==0 | mask.unsqueeze(2), float("inf"), dists) # Z x N x N
		
		# get topk 
		top_k = dists.topk(self.top_k, dim=2, largest=False) # Z x N x K

		# masked nodes have themselves as edges, masked edges are the corresponding node
		node_idxs = torch.arange(dimN, device=dists.device).view(1,-1,1) # 1 x N x 1
		edge_mask = ~(mask.unsqueeze(2) | torch.gather(mask.unsqueeze(2).expand(-1,-1,self.top_k), 1, top_k.indices))
		edge_mask = edge_mask & (top_k.values!=0) # exclude self and distant neighbors
		top_k = torch.where(edge_mask, top_k.indices, node_idxs) # Z x N x K

		return topk, edge_mask
		
	def compute_potentials(self, voxels, C, L, neighbors, atom_mask, nbr_mask):

		Z, N = labels.shape
		AA, A = self.amber_partial_charges.shape
		
		L = torch.masked_fill(L==-1, aa_2_lbl("X")) # replace invalid labels with X

		# get partial charges
		partial_charges = torch.gather(self.amber_partial_charges.view(1,1,AA,A).expand(Z,N,AA,A), 2, L.view(Z,N,1,1).expand(Z,N,1,A)).squeeze(2) # Z, N, A

		