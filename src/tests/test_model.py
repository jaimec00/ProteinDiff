import unittest
from model.ProteinDiff import ProteinDiff

class Tests(unittest.TestCase):

    def setup(self):
        ZN = 16
        self.device = "cuda"

        self.coords = torch.randn((ZN,14,3), device=self.device)
        self.labels = torch.randn((ZN), device=self.device)
        self.seq_pos = torch.arange(ZN, device=self.device)
        self.chain_pos = torch.ones(ZN, device=self.device)
        self.atom_mask = torch.rand((ZN,), device=self.device)<0.0 
        self.no_seq_mask = torch.zeros(ZN, device=self.device)
        self.sample_idx = torch.ones(ZN, device=self.device)

        _, seqlens = torch.unique_consecutive(self.sample_idx, return_counts=True)
        self.cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(dim=0), (1,0), value=0)
        self.max_seqlen = seqlens.max(dim=0).item()


        self.model = ProteinDiff(d_model=256, d_diffusion=256, d_sc_latent=16, d_bb_latent=256,
                            top_k=16, voxel_dim=8, cell_dim=1.0,
                            sc_enc_layers=1, sc_dec_layers=1, sc_class_layers=1, 
                            bb_enc_layers=1, bb_dec_layers=1, bb_dec_heads=8,
                            diff_layers=1, diff_heads=8, diff_parameterization="eps").to(self.device)

    def test_sc_vae(self):
        (latent_mu, latent_logvar, 
        decoded_voxels, voxels, 
        seq) = self.model(self.coords, self.labels, 
                            seq_pos=self.seq_pos, chain_pos=self.chain_pos, 
                            atom_mask=self.atom_mask, no_seq_mask=self.no_seq_mask, 
                            sample_idx=self.sample_idx, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen
                            run_type="sc_vae")

        print(f"latent_mu.shape: {latent_mu.shape}")
        print(f"latent_logvar.shape: {latent_logvar.shape}")
        print(f"decoded_voxels.shape: {decoded_voxels.shape}")
        print(f"voxels.shape: {voxels.shape}")
        print(f"seq.shape: {seq.shape}")

    def test_bb_vae(self):
        (latent_mu, latent_logvar, 
        distogram, anglogram, 
        seq_pred) = self.model(self.coords, self.labels, 
                    seq_pos=self.seq_pos, chain_pos=self.chain_pos, 
                    atom_mask=self.atom_mask, no_seq_mask=self.no_seq_mask, 
                    sample_idx=self.sample_idx, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen
                    run_type="bb_vae")

        print(f"latent_mu.shape: {latent_mu.shape}")
        print(f"latent_logvar.shape: {latent_logvar.shape}") 
        print(f"distogram.shape: {distogram.shape}")
        print(f"anglogram.shape: {anglogram.shape}") 
        print(f"seq_pred.shape: {seq_pred.shape}")

    def test_diff(self):
        pred, target = self.model(self.coords, self.labels, 
                    seq_pos=self.seq_pos, chain_pos=self.chain_pos, 
                    atom_mask=self.atom_mask, no_seq_mask=self.no_seq_mask, 
                    sample_idx=self.sample_idx, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen
                    run_type="diffusion")

        print(f"pred.shape: {pred.shape}")
        print(f"target.shape: {target.shape}") 


    def test_inference(self):
        seq = self.model(self.coords, self.labels, 
                    seq_pos=self.seq_pos, chain_pos=self.chain_pos, 
                    atom_mask=self.atom_mask, no_seq_mask=self.no_seq_mask, 
                    sample_idx=self.sample_idx, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen
                    run_type="inference")

        print(f"seq.shape: {seq.shape}") 

if __name__ == "__main__":
    unittest.main()