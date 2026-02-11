import pytest
import torch
from proteus.utils.test_utils import calculate_error, profile_func, profile_bwd
from proteus.model.tokenizer.wave_func_tokenizer.learn_aa import wf_embedding_learn_aa
from .wf_embed_common import wf_embedding_torch, make_test_data, CORRECTNESS_CONFIGS, STRESS_CONFIGS


# --- helpers ---

def check_correctness(seq_lens, d_model, num_classes, seed=42, atol=1e-2, rtol=1e-2):
	"""run cuda + torch reference, compare fwd + bwd, return error metrics"""
	params = make_test_data(seq_lens, d_model, num_classes, seed, learnable_aa=True)
	coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p = params

	# forward pass
	cuda_out = wf_embedding_learn_aa(*params)
	torch_out = wf_embedding_torch(*params)

	# fwd errors
	fwd_rel, fwd_abs = calculate_error(torch_out, cuda_out)
	fwd_close = torch.allclose(cuda_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)

	# backward pass
	cuda_out.sum().backward(retain_graph=False)
	cuda_dk = aa_magnitudes.grad.clone()
	aa_magnitudes.grad.zero_()

	torch_out.sum().backward(retain_graph=False)
	torch_dk = aa_magnitudes.grad.clone()

	# bwd errors
	bwd_rel, bwd_abs = calculate_error(torch_dk, cuda_dk)
	bwd_close = torch.allclose(cuda_dk, torch_dk, atol=atol, rtol=rtol, equal_nan=False)

	return {
		'fwd_rel': fwd_rel.item(), 'fwd_abs': fwd_abs.item(), 'fwd_close': fwd_close,
		'bwd_rel': bwd_rel.item(), 'bwd_abs': bwd_abs.item(), 'bwd_close': bwd_close,
	}


# --- parametrized correctness tests ---

@pytest.mark.parametrize('config_name', list(CORRECTNESS_CONFIGS.keys()))
def test_correctness(config_name):
	cfg = CORRECTNESS_CONFIGS[config_name]
	result = check_correctness(**cfg)

	print(f"\n[{config_name}] fwd rel={result['fwd_rel']:.5f} abs={result['fwd_abs']:.5f} | "
	      f"bwd rel={result['bwd_rel']:.5f} abs={result['bwd_abs']:.5f}")

	assert result['fwd_close'], (
		f"fwd mismatch [{config_name}]: rel={result['fwd_rel']:.5f}, abs={result['fwd_abs']:.5f}"
	)
	assert result['bwd_close'], (
		f"bwd mismatch [{config_name}]: rel={result['bwd_rel']:.5f}, abs={result['bwd_abs']:.5f}"
	)


# --- gradient check ---

def test_gradcheck():
	"""numerical gradient check on aa_magnitudes with tiny config"""
	device = torch.device('cuda')
	torch.manual_seed(123)

	seq_lens = [4, 5]
	d_model, num_classes = 4, 3
	BL = sum(seq_lens)
	num_wn = d_model // 2

	cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
	for i, sl in enumerate(seq_lens):
		cu_seqlens[i + 1] = cu_seqlens[i] + sl

	coordsA = 20.0 * torch.randn(BL, 3, dtype=torch.float64, device=device)
	coordsB = torch.randn(BL, 3, dtype=torch.float64, device=device)
	coordsB = coordsB / torch.linalg.vector_norm(coordsB, dim=1, keepdim=True)
	aa_labels = torch.randint(0, num_classes, (BL,), dtype=torch.int32, device=device)
	aa_magnitudes = torch.rand(num_wn, num_classes, dtype=torch.float64, device=device, requires_grad=True)
	wavenumbers = torch.randn(num_wn, dtype=torch.float64, device=device)

	# wrap to only differentiate wrt aa_magnitudes
	def func(aa_mag):
		return wf_embedding_learn_aa(
			coordsA.float(), coordsB.float(), aa_labels, aa_mag.float(),
			wavenumbers.float(), cu_seqlens
		).double()

	torch.autograd.gradcheck(func, (aa_magnitudes,), eps=1e-3, atol=5e-2, rtol=5e-2)


# --- stress tests ---

@pytest.mark.stress
@pytest.mark.parametrize('config_name', list(STRESS_CONFIGS.keys()))
def test_stress(config_name):
	cfg = STRESS_CONFIGS[config_name]
	params = make_test_data(**cfg, learnable_aa=True)

	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	# warmup cuda kernel
	for _ in range(3):
		wf_embedding_learn_aa(*params)
	torch.cuda.synchronize()

	# profile cuda fwd
	cuda_out, cuda_time, cuda_mem = profile_func(wf_embedding_learn_aa, params, start_event, end_event)

	# profile torch ref fwd
	torch_out, torch_time, torch_mem = profile_func(wf_embedding_torch, params, start_event, end_event)

	# fwd correctness
	fwd_rel, fwd_abs = calculate_error(torch_out, cuda_out)
	fwd_close = torch.allclose(cuda_out, torch_out, atol=1e-2, rtol=1e-2, equal_nan=False)

	# profile bwd
	cuda_bwd_time, cuda_bwd_mem = profile_bwd(cuda_out.sum(), start_event, end_event)
	cuda_dk = params[3].grad.clone()
	params[3].grad.zero_()

	torch_bwd_time, torch_bwd_mem = profile_bwd(torch_out.sum(), start_event, end_event)
	torch_dk = params[3].grad.clone()

	bwd_rel, bwd_abs = calculate_error(torch_dk, cuda_dk)
	bwd_close = torch.allclose(cuda_dk, torch_dk, atol=1e-2, rtol=1e-2, equal_nan=False)

	BL = sum(cfg['seq_lens'])
	print(f"\n[{config_name}] BL={BL}")
	print(f"  fwd: cuda={cuda_time:.2f}ms torch={torch_time:.2f}ms  "
	      f"rel={fwd_rel:.5f} abs={fwd_abs:.5f}")
	print(f"  bwd: cuda={cuda_bwd_time:.2f}ms torch={torch_bwd_time:.2f}ms  "
	      f"rel={bwd_rel:.5f} abs={bwd_abs:.5f}")
	print(f"  mem: cuda_fwd={cuda_mem/(1024**3):.3f}GB torch_fwd={torch_mem/(1024**3):.3f}GB")

	assert fwd_close, f"fwd mismatch [{config_name}]: rel={fwd_rel:.5f}"
	assert bwd_close, f"bwd mismatch [{config_name}]: rel={bwd_rel:.5f}"


if __name__ == "__main__":
	pytest.main([__file__, "-v", "-s"])
