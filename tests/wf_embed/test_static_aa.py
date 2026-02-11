import pytest
import torch
from proteus.utils.test_utils import calculate_error, profile_func
from proteus.model.tokenizer.wave_func_tokenizer.static_aa import wf_embedding_static_aa
from .wf_embed_common import wf_embedding_torch, make_test_data, CORRECTNESS_CONFIGS, STRESS_CONFIGS


# --- helpers ---

def check_correctness(seq_lens, d_model, num_classes, seed=42, atol=1e-2, rtol=1e-2):
	"""run cuda + torch reference, compare forward only (no gradients for static aa)"""
	params = make_test_data(seq_lens, d_model, num_classes, seed, learnable_aa=False)

	# forward pass
	cuda_out = wf_embedding_static_aa(*params)
	torch_out = wf_embedding_torch(*params)

	# fwd errors
	fwd_rel, fwd_abs = calculate_error(torch_out, cuda_out)
	fwd_close = torch.allclose(cuda_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)

	return {
		'fwd_rel': fwd_rel.item(), 'fwd_abs': fwd_abs.item(), 'fwd_close': fwd_close,
	}


# --- parametrized correctness tests ---

@pytest.mark.parametrize('config_name', list(CORRECTNESS_CONFIGS.keys()))
def test_correctness(config_name):
	cfg = CORRECTNESS_CONFIGS[config_name]
	result = check_correctness(**cfg)

	print(f"\n[{config_name}] fwd rel={result['fwd_rel']:.5f} abs={result['fwd_abs']:.5f}")

	assert result['fwd_close'], (
		f"fwd mismatch [{config_name}]: rel={result['fwd_rel']:.5f}, abs={result['fwd_abs']:.5f}"
	)


# --- stress tests ---

@pytest.mark.stress
@pytest.mark.parametrize('config_name', list(STRESS_CONFIGS.keys()))
def test_stress(config_name):
	cfg = STRESS_CONFIGS[config_name]
	params = make_test_data(**cfg, learnable_aa=False)

	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	# warmup cuda kernel
	for _ in range(3):
		wf_embedding_static_aa(*params)
	torch.cuda.synchronize()

	# profile cuda fwd
	cuda_out, cuda_time, cuda_mem = profile_func(wf_embedding_static_aa, params, start_event, end_event)

	# profile torch ref fwd
	torch_out, torch_time, torch_mem = profile_func(wf_embedding_torch, params, start_event, end_event)

	# fwd correctness
	fwd_rel, fwd_abs = calculate_error(torch_out, cuda_out)
	fwd_close = torch.allclose(cuda_out, torch_out, atol=1e-2, rtol=1e-2, equal_nan=False)

	BL = sum(cfg['seq_lens'])
	print(f"\n[{config_name}] BL={BL}")
	print(f"  fwd: cuda={cuda_time:.2f}ms torch={torch_time:.2f}ms  "
	      f"rel={fwd_rel:.5f} abs={fwd_abs:.5f}")
	print(f"  mem: cuda_fwd={cuda_mem/(1024**3):.3f}GB torch_fwd={torch_mem/(1024**3):.3f}GB")

	assert fwd_close, f"fwd mismatch [{config_name}]: rel={fwd_rel:.5f}"


if __name__ == "__main__":
	pytest.main([__file__, "-v", "-s"])
