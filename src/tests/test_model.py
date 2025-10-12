import unittest
from model.ProteinDiff import ProteinDiff
from training.losses import LossFunction
from static.constants import alphabet, canonical_aas
import torch

class Tests(unittest.TestCase):

    def setUp(self):
        self.ZN = 16
        self.device = "cuda"
        torch.manual_seed(0)

        self.coords = torch.randn((self.ZN,14,3), device=self.device)
        self.labels = torch.randint(0,len(alphabet),(self.ZN,), dtype=torch.long, device=self.device)
        self.seq_pos = torch.arange(self.ZN, device=self.device)
        self.chain_pos = torch.ones(self.ZN, device=self.device)
        self.atom_mask = torch.rand((self.ZN,14), device=self.device)>0.0 
        self.no_seq_mask = torch.zeros(self.ZN, device=self.device, dtype=torch.bool)
        self.sample_idx = torch.ones(self.ZN, device=self.device)

        _, seqlens = torch.unique_consecutive(self.sample_idx, return_counts=True)
        self.cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(dim=0), (1,0), value=0).to(torch.int32)
        self.max_seqlen = seqlens.max(dim=0).values.item()


        self.d_model=256
        self.d_diffusion=256
        self.d_sc_latent=16
        self.d_bb_latent=256
        self.voxel_dim=8

        self.model = ProteinDiff(   d_model=self.d_model, d_diffusion=self.d_diffusion, d_sc_latent=self.d_sc_latent, d_bb_latent=self.d_bb_latent,
                                    top_k=16, voxel_dim=self.voxel_dim, cell_dim=1.0,
                                    sc_enc_layers=1, sc_dec_layers=1, sc_class_layers=1, 
                                    bb_enc_layers=1, bb_dec_layers=1, bb_dec_heads=8,
                                    diff_layers=1, diff_heads=8, diff_parameterization="eps", t_max=10).to(self.device)

        self.loss_func = LossFunction() # defaults
        self.loss_mask = self.labels < len(canonical_aas)

    def test_sc_vae(self):
        latent_mu, latent_logvar, decoded_voxels, voxels, seq = self.run_model("sc_vae")

        self.assertEqual((self.ZN,self.d_sc_latent,1,1,1), latent_mu.shape)
        self.assertEqual((self.ZN,self.d_sc_latent,1,1,1), latent_logvar.shape)
        self.assertEqual((self.ZN,1,self.voxel_dim, self.voxel_dim, self.voxel_dim), decoded_voxels.shape)
        self.assertEqual((self.ZN,1,self.voxel_dim, self.voxel_dim, self.voxel_dim), voxels.shape)
        self.assertEqual((self.ZN,len(canonical_aas)), seq.shape)

        losses = self.loss_func.sc_vae(latent_mu, latent_logvar, decoded_voxels, voxels, seq, self.labels, self.loss_mask)
        losses["Full Loss"].backward()

    def test_bb_vae(self):
        latent_mu, latent_logvar, distogram, anglogram, seq_pred = self.run_model("bb_vae")

        self.assertEqual((self.ZN,self.d_bb_latent), latent_mu.shape)
        self.assertEqual((self.ZN,self.d_bb_latent), latent_logvar.shape) 
        self.assertEqual((self.ZN,self.ZN,64), distogram.shape)
        self.assertEqual((self.ZN,self.ZN,16), anglogram.shape) 
        self.assertEqual((self.ZN,len(canonical_aas)), seq_pred.shape)

        losses = self.loss_func.bb_vae(latent_mu, latent_logvar, distogram, anglogram, self.coords, seq_pred, self.labels, self.loss_mask)
        losses["Full Loss"].backward()

    def test_diff(self):
        pred, target = self.run_model("diffusion")

        self.assertEqual((self.ZN, self.d_sc_latent), pred.shape)
        self.assertEqual((self.ZN, self.d_sc_latent), target.shape) 

        losses = self.loss_func.diffusion(pred, target, self.loss_mask)
        losses["Mean Squared Error"].backward()

    def test_inference(self):
        seq = self.run_model("inference")

        self.assertEqual((self.ZN, len(canonical_aas)), seq.shape) 

        losses = self.loss_func.inference(seq, self.labels, self.loss_mask)

    def run_model(self, run_type):
        return self.model(self.coords, self.labels, 
                    seq_pos=self.seq_pos, chain_pos=self.chain_pos, 
                    atom_mask=self.atom_mask, no_seq_mask=self.no_seq_mask, 
                    sample_idx=self.sample_idx, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen,
                    run_type=run_type)


if __name__ == "__main__":
    unittest.main()