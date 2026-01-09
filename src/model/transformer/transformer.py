
class Transformer(nn.Module):
	def __init__(self, d_model=256, heads=8):
		super().__init__()
		self.attn = FlashMHA(d_model=d_model, heads=heads)
		self.attn_norm = nn.LayerNorm(d_model)
		self.ffn = MLP(d_in=d_model, d_hidden=4*d_model, d_out=d_model, hidden_layers=0, act="silu")
		self.ffn_norm = nn.LayerNorm(d_model)

	def forward(self, x, cu_seqlens, max_seqlen):
		x1 = self.attn(x, cu_seqlens, max_seqlen)
		x = self.attn_norm(x+x1)
		x1 = self.ffn(x)
		x = self.ffn_norm(x+x1)
		return x
