import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from flash_attn_interface import flash_attn_varlen_func

class MLP(nn.Module):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, d_in=512, d_out=512, d_hidden=1024, hidden_layers=0, dropout=0.0, act="gelu", zeros=False):
		super().__init__()

		self.in_proj = nn.Linear(d_in, d_hidden)
		self.hidden_proj = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for layer in range(hidden_layers)])
		self.out_proj = nn.Linear(d_hidden, d_out)

		self.in_dropout = nn.Dropout(dropout)
		self.hidden_dropout = nn.ModuleList([nn.Dropout(dropout) for layer in range(hidden_layers)])

		if act == "gelu":
			self.act = F.gelu 
		elif act == "silu":
			self.act = F.silu
		elif act == "relu":
			self.act = F.relu
		elif act == "sigmoid":
			self.act = F.sigmoid
		else:
			self.act = lambda x: x # no activation if none of the above 

		self.init_linears(zeros=zeros)

	def init_linears(self, zeros=False):

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		if zeros:
			init_zeros(self.out_proj) 
		else:
			init_xavier(self.out_proj)  # Xavier for output layer

	def forward(self, x):
		x = self.in_dropout(self.act(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(self.act(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x

class FlashMHA(nn.Module):
	def __init__(self, d_model, heads):
		super().__init__()

		d_k = d_model // heads
		xavier_scale = (6/(d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_model, d_k) * (2*xavier_scale)) # H x Dm x Dk
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_model, d_k) * (2*xavier_scale)) # H x Dm x Dk
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_model, d_k) * (2*xavier_scale)) # H x Dm x Dk

		self.q_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x Dk
		self.k_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x Dk
		self.v_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x Dk

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, x, mask):

		# convenience
		Z, N, Dm = x.shape
		H, _, Dk = self.q_proj.shape

		# project the tensors, doing reshape for readability
		x = x.unsqueeze(1) # Z x 1 x N x Dm
		Q = torch.matmul(x, self.q_proj.reshape(1,H,Dm,Dk)) + self.q_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk
		K = torch.matmul(x, self.k_proj.reshape(1,H,Dm,Dk)) + self.k_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk
		V = torch.matmul(x, self.v_proj.reshape(1,H,Dm,Dk)) + self.v_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk + 1xHx1xDk->Z,H,N,Dk

		# now need to reshape it for flash attention kernel
		indices = mask.flatten().nonzero(as_tuple=False).flatten().reshape(-1,1,1).expand(-1,H,Dk)
		Q = Q.permute(0,2,1,3).reshape(Z*N, H, Dk).gather(0, indices).to(torch.float16).contiguous() # ZN(valid), H, Dk
		K = K.permute(0,2,1,3).reshape(Z*N, H, Dk).gather(0, indices).to(torch.float16).contiguous()
		V = V.permute(0,2,1,3).reshape(Z*N, H, Dk).gather(0, indices).to(torch.float16).contiguous()

		# compute cu seq lens for kernel
		seq_lens = mask.sum(dim=1) # Z, 
		max_seqlen = seq_lens.max().item()  # 1,
		cu_seqlens = F.pad(seq_lens.cumsum(dim=-1), (1,0)).to(torch.int32) # Z+1,

		# flash attention 3 (hopper)
		flat_out = flash_attn_varlen_func( # ZN(valid) x H x Dk
			Q, K, V,
			cu_seqlens, cu_seqlens, # q_cu_seqlens, k_cu_seqlens
			max_seqlen, max_seqlen, # q_max_seqlen, k_max_seqlen
			softmax_scale=Dk**-0.5,
			deterministic=True # for deterministic bwd
		)

		out = torch.zeros(Z*N, H, Dk, device=x.device, dtype=x.dtype)
		out = torch.scatter(out, 0, indices, flat_out).reshape(Z, N, Dm) # reshapes to split Z and N, and cat heads to get Dm in one go

		out = self.out_proj(out)
		return out



# initializations for linear layers
def init_orthogonal(m):
	if isinstance(m, nn.Linear):
		init.orthogonal_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_kaiming(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, nonlinearity='relu')
		if m.bias is not None:
			init.zeros_(m.bias)
def init_xavier(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_zeros(m):
	if isinstance(m, nn.Linear):
		init.zeros_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)