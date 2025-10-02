import unittest
import itertools
from model.utils.base_modules import FlashMHA

class TestCase(unittest.TestCase):

    def setup(self):
        torch.manual_seed(42)
        self.batch_sizes = [1,2,3,4,5,6,7,8]
        self.seq_sizes = [2,4,8,16,32,35]

    def test_8h_256d(self):
        self._run(8, 256)


    def _run(self, H, Dm):
        f = FlashMHA(d_model=Dm, heads=H).to("cuda")
        for Z, N in itertools.product(self.batch_sizes, self.seq_sizes):

            x = torch.randn((Z, N, Dm), device="cuda")
            mask = (x<0.75).any(dim=-1)
            
            out = f(x, mask)
            true_out = ref_mha(x, mask, f)

            self.assertEqual(out, true_out)


    def ref_mha(self, x, mask, module):
        Z, N = mask.shape
        H, Dm, Dk = module.q_proj.shape
        Q = torch.matmul(x.unsqueeze(0), module.q_proj.reshape(1,H,Dm,Dk)) + module.q_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk
		K = torch.matmul(x.unsqueeze(0), module.k_proj.reshape(1,H,Dm,Dk)) + module.k_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk
		V = torch.matmul(x.unsqueeze(0), module.v_proj.reshape(1,H,Dm,Dk)) + module.v_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk

        attn_logits = torch.matmul(Q, K.transpose(-1,-2)) * (Dk**-0.5)
        attn_mask = ~(mask.unsqueeze(0) & mask.unsqueeze(1))
        attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn, V).permute(0,2,1,3).reshape(Z, N, Dm)
        out = module.out_proj(out)

        return out


if __name__ =="__main__":
    unittest.main()