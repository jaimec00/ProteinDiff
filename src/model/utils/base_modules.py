import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from flash_attn_interface import flash_attn_varlen_qkvpacked_func

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

		self.qkv_proj = nn.Parameter(-xavier_scale + torch.rand(3, heads, d_model, d_k) * (2*xavier_scale)) # 3 x H x Dm x Dk
		self.qkv_bias = nn.Parameter(torch.zeros(3, heads, d_k)) # 3 x H x Dk

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, x, mask):

		# convenience
		Z, N, Dm = x.shape
		_. H, _, Dk = self.qkv_proj.shape

		# get indices of non pad positions
		indices = mask.flatten().nonzero(as_tuple=False).flatten().reshape(-1,1,1,1).expand(-1,3,H,Dk)

		# project the tensors
		QKV = torch.matmul(x.reshape(Z,1,1,N,Dm), self.qkv_proj.reshape(1,3,H,Dm,Dk)) + self.qkv_bias.reshape(1,3,H,1,Dk) # Z,1,1,N,Dm@1,3,H,Dm,Dk + 1x3xHx1xDk->Z,3,H,N,Dk

		# reshape and use indices to unpad
		QKV_unpad = QKV.permute(0,3,1,2,4).reshape(Z*N, 3, H, Dk).gather(0, indices).to(torch.float16).contiguous() # ZN(valid), 3, H, Dk

		# compute cu seq lens and max seq len for kernel
		seq_lens = mask.sum(dim=1) # Z, 
		max_seqlen = seq_lens.max().item()  # 1,
		cu_seqlens = F.pad(seq_lens.cumsum(dim=-1), (1,0)).to(torch.int32) # Z+1,

		# dropout if in training
		dropout_p = self.dropout_p if self.training else 0.0

		# flash attention 2
		out_unpad = flash_attn_varlen_qkvpacked_func( # ZN(valid) x H x Dk
			QKV_unpad,
			cu_seqlens, # q_cu_seqlens, k_cu_seqlens
			max_seqlen, # q_max_seqlen, k_max_seqlen
			dropout_p=dropout_p, # dropout
			softmax_scale=Dk**-0.5, # sm scale
			deterministic=dropout_p>0.0 # for deterministic bwd, only when dropout is used
		)

		# init the output
		out = torch.zeros(Z*N, H, Dk, device=x.device, dtype=x.dtype)

		# scatter the unpadded output to the padded, and reshape
		out = torch.scatter(out, 0, indices, out_unpad).reshape(Z, N, Dm) # reshapes to split Z and N, and cat heads to get Dm in one go

		# output projection
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