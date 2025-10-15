

from training.data.data_loader import DataHolder
from model.ProteinDiff import ProteinDiff
from static.constants import canonical_aas
import torch
import unittest
import os

class Test(unittest.TestCase):

	def setUp(self):
		torch.manual_seed(42)
		data_path = os.environ.get("DATA_PATH")
		self.data_holder =   DataHolder(    data_path=data_path, 
											num_train=100, num_val=100, num_test=100, 
											batch_tokens=16384, min_seq_size=16, max_seq_size=16384,
											max_resolution=3.5, homo_thresh=0.70, asymmetric_units_only=False,
											num_workers=8, prefetch_factor=2, rng_seed=42, buffer_size=32
										)
		self.model=ProteinDiff( d_model=256, d_diffusion=256, d_sc_latent=16, d_bb_latent=256,
								top_k=16, voxel_dim=16, cell_dim=0.75,
								sc_enc_layers=1, sc_dec_layers=1, sc_class_layers=1, 
								bb_enc_layers=1, bb_dec_layers=1, bb_dec_heads=8,
								diff_layers=1, diff_heads=8, diff_parameterization="eps", t_max=1000
						) # defaults

		self.loss_func = LossFunction() # defaults

	def test_sc_vae(self):
		for data in self.data_holder.train:

			valid_mask = data.coords_mask
			data.apply_mask(valid_mask)
			data.get_cu_seqlens()
			z_mu, z_logvar, voxels_pred, voxels_true, seq_pred = self._run_model(data, "sc_vae")
			
			toks = len(data)
			self.assertEqual(z_mu.shape, (toks, 16, 1, 1, 1))
			self.assertEqual(z_logvar.shape, (toks, 16, 1, 1, 1))
			self.assertEqual(voxels_pred.shape, (toks, 1, 8, 8, 8))
			self.assertEqual(voxels_true.shape, (toks, 1, 8, 8, 8))
			self.assertEqual(seq.shape, (toks, len(canonical_aas)))

			loss_mask = torch.ones_like(data.labels, dtype=torch.bool) # no loss mask, already pruned out invalid ones through valid_mask
			losses = self.loss_func.sc_vae(latent_mu, latent_logvar, decoded_voxels, voxels, seq, self.labels, loss_mask)
			losses["Full Loss"].backward()

	def _run_model(self, data, mode):

		return self.model(  data.coords, data.labels, 
							seq_pos=data.seq_pos, chain_pos=data.chain_pos, 
							atom_mask=data.atom_mask, cu_seqlens=data.cu_seqlens, max_seqlen=data.max_seqlen,
							no_seq_mask=data.no_seq_mask, sample_idx=data.sample_idx, 
							run_type=mode
						)

		# for data in self.data_holder.val
		#     output = self.model()
		#     self.assertEqual(output.shape, ...)


if __name__=="__main__":
	unittest.main()

