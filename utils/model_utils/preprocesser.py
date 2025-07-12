import torch
import torch.nn as nn
from data.constants import amber_partial_charges, aa_2_lbl, coulomb_constant

class PreProcesser(nn.Module):
	
	def __init__(self, top_k=30, voxel_dims=(6,8,6), cell_size=1.0):
		super().__init__()

		x_cells, y_cells, z_cells = voxel_dims
		voxel_x = torch.arange(x_cells).view(-1,1,1).expand(voxel_dims) - x_cells/2
		voxel_y = torch.arange(y_cells).view(1,-1,1).expand(voxel_dims) - y_cells/4 # more cells in positive y direction, since that is where the side chain is located
		voxel_z = torch.arange(z_cells).view(1,1,-1).expand(voxel_dims) - z_cells/2
		voxel = (cell_size**(1/3)) * torch.stack([voxel_x, voxel_y, voxel_z], dim=3) # Vx, Vy, Vz, 3
		self.register_buffer("voxel", voxel)
		self.res = cell_size

		self.top_k = top_k

	def forward(self, C, L, atom_mask, kp_mask):
		'''
		C (torch.Tensor): full atomic coordinates of shape (Z,N,A,3)
		L (torch.Tensor): amino acid class labels of shape (Z,N)
		atom_mask (torch.Tensor): mask indicating missing atom coordinates of shape (Z,N,A)
		kp_mask (torch.Tensor): mask indicating padded positions of shape (Z,N)
		'''

		# no gradients here, as that would blow everything up, and this is all physics-based preprocessing, no learning
		with torch.no_grad():

			# get the backbone atoms, using virtual Cb
			C_backbone = self.get_backbone(C) # Z,N,4,3

			# compute unit vectors for each residue's local reference frame
			local_origins, local_frames = self.compute_frames(C_backbone) # Z,N,3 and Z,N,3,3

			# create the voxel for each residue by rotating the base voxel to the local frame and translating to local origin, 
			# simply contains the coordinates for the voxels
			local_voxels = self.compute_voxels(local_origins, local_frames) # Z,N,Vx,Vy,Vz,3

			# compute the nearest neighbors
			nbrs, nbr_mask = self.get_neighbors(C_backbone[:, :, 1, :], kp_mask) # use Ca coords

			# compute electric fields 
			fields = self.compute_fields(C, L, local_voxels, nbrs, nbr_mask, atom_mask)

		return C_backbone, fields, nbrs, nbr_mask 

	def get_backbone(self, C):

		n = C[:, :, 0, :]
		ca = C[:, :, 1, :]
		c = C[:, :, 2, :]

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
		x_proj = torch.linalg.vecdot(x_raw, y_unit, dim=2) # project x_raw onto y
		x = x_raw - x_proj.unsqueeze(2) # subtract the projection from the original vector to get the projection of x onto the plane perpendicular to y
		x_unit = x / torch.linalg.vector_norm(x, dim=2, keepdim=True).clamp(1e-6)

		z_unit = torch.linalg.cross(x_unit, y_unit) # already a unit vector, since x and y are already unit

		frames = torch.stack([x_unit, y_unit, z_unit], dim=2) # Z,N,U,S

		# the origin is the beta carbon position
		origin = C_backbone[:, :, 3, :]

		return origin, frames

	def compute_voxels(self, origins, frames):

		Z, N, S = origins.shape
		_, _, U, _ = frames.shape # U is the unit vectors dim
		Vx, Vy, Vz, _ = self.voxel.shape

		# rotate the voxel grid using the local frames, each unit vector (Uxyz) is multipled by the corresponding component
		rotation = torch.sum(frames.view(Z,N,1,1,1,U,S) * self.voxel.view(1,1,Vx,Vy,Vz,S,1), dim=5) # Z,N,Vx,Vy,Vz,U

		# add the offset so the origin is the beta carbon
		local_voxels = origins.view(Z,N,1,1,1,S) + rotation # Z,N,Vx,Vy,Vz,S

		return local_voxels

	def get_neighbors(self, Ca, mask):

		Z, N, S = Ca.shape
		top_k = self.top_k
		# if N<=self.top_k:
		# 	Ca = torch.cat([Ca, torch.zeros(Z, self.top_k-N, S)], dim=1)
		# 	mask = torch.cat([Ca, torch.zeros(Z, self.top_k-N, S)], dim=1)

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(1) - Ca.unsqueeze(2))**2, dim=3)) # Z x N x N
		dists = torch.where((dists==0) | (~mask).unsqueeze(2), float("inf"), dists) # Z x N x N
		
		# get topk 
		nbrs = dists.topk(top_k, dim=2, largest=False) # Z x N x K

		# masked nodes have themselves as edges, masked edges are the corresponding node
		node_idxs = torch.arange(N, device=dists.device).view(1,-1,1) # 1 x N x 1
		nbr_mask = ~(mask.unsqueeze(2) | torch.gather(mask.unsqueeze(2).expand(-1,-1,top_k), 1, nbrs.indices))
		nbr_mask = nbr_mask & (nbrs.values!=0) # exclude self and distant neighbors
		nbrs = torch.where(nbr_mask, nbrs.indices, node_idxs) # Z x N x K

		return nbrs, nbr_mask
		
	def compute_fields(self, C, L, voxels, nbrs, nbr_mask, atom_mask):

		# prep
		Z, N = L.shape
		AA, A = amber_partial_charges.shape
		_, _, Vx, Vy, Vz, S = voxels.shape

		# include self in neighbors
		nbrs = torch.cat([nbrs, torch.arange(N, device=nbrs.device, dtype=nbrs.dtype).view(1,N,1).expand(Z,N,1)], dim=2)
		nbr_mask = torch.cat([nbr_mask, torch.ones(Z,N,1,device=nbr_mask.device, dtype=nbr_mask.dtype)], dim=2)
		_, _, K = nbrs.shape
		
		# replace invalid labels with X
		L = L.masked_fill(L==-1, aa_2_lbl("X")) 

		# compute the electric field, using just basic point charge formula, might move to screened coloumb if i dont like the visualization
		# note that this will be a Vx, Vy, Vz, 3 tensor for each residue, as there is directionality involved. also makes it easier to use 
		# image processing techniques, as there are usually rgb channels, e.g. could downsample from 8x8x4x3 to 4x4x2x4

		# first get the coordinates of the neighbor atoms
		C_nbrs = torch.gather(C.unsqueeze(2).expand(Z, N, K, A, S), 1, nbrs.view(Z, N, K, 1, 1).expand(Z, N, K, A, S))

		# now get distance vectors, as directionality is also used. points from neighbor atoms TO voxel cells
		dist_vectors =  voxels.view(Z, N, 1, Vx, Vy, Vz, 1, S) - C_nbrs.view(Z, N, K, 1, 1, 1, A, S) # Z,N,K,Vx,Vy,Vz,A,S 

		# compute magnitudes, clamp to 2 times the cell resolution (default is 1 A^3), so atoms inside a cell arent overweighted, makes the field look more continuous
		dists = torch.linalg.vector_norm(dist_vectors, dim=7, keepdim=True).clamp(min=self.res*2) # Z,N,K,Vx,Vy,Vz,A,1

		# the distance term is the r^{hat} / |r|^2 = r / |r|^3
		# however, using r^{hat} / |r| = r / |r|^2, since this led to more continuous field, using |r|^3 clustered the field around atoms within the voxel
		dist_term = dist_vectors / (dists**2) # Z,N,K,Vx,Vy,Vz,A,S

		# get partial charges, zero out masked atoms
		partial_charges = torch.gather(amber_partial_charges.view(1,1,AA,A).expand(Z,N,AA,A), 2, L.view(Z,N,1,1).expand(Z,N,1,A)) * atom_mask.unsqueeze(2) # Z, N, 1, A

		# get the partial charges of the neighbors, zero out invalid neighbors
		partial_charges_nbrs = torch.gather(partial_charges.expand(Z, N, K, A), 1, nbrs.unsqueeze(3).expand(Z, N, K, A)) * nbr_mask.unsqueeze(3) # Z,N,K,A

		# now compute the final field, sum over neighbors and atoms. 
		# not an actual field, i tuned it so it is approx continuous
		# using the actual coulomb constant made it less continuous, i want something that is image like, see visualization/compute_voxels.ipynb to see what im talking about
		fields = torch.sum(partial_charges_nbrs.view(Z, N, K, 1, 1, 1, A, 1) * dist_term.view(Z, N, K, Vx, Vy, Vz, A, S), dim=(2,6)) # Z,N,Vx,Vy,Vz,S

		# norm the each voxel independantly
		norm_dims = (2,3,4,5)
		fields_mean = fields.mean(dim=norm_dims, keepdim=True)
		fields = fields - fields_mean
		fields_std = fields.std(dim=norm_dims, keepdim=True)
		fields = fields / fields_std.masked_fill(fields_std==0, 1)

		return fields

		