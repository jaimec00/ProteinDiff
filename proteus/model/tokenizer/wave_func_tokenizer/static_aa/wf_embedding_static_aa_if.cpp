
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>


// declare the cuda kernel implemented in wf_embedding_static_aa_kernel.cu
void wf_embedding_static_aa_kernel_forward(
    float* coordsA_ptr, int stride_coordsA_S, int stride_coordsA_BL,
    float* coordsB_ptr, int stride_coordsB_S, int stride_coordsB_BL,
    int16_t* aa_labels_ptr, int stride_aa_labels_BL,
    const __half* aa_magnitudes_ptr, int stride_aa_magnitudes_K, int stride_aa_magnitudes_A,
    const float* wavenumbers_ptr, int stride_wavenumbers_K,
    const int* cu_seqlens_ptr, int num_seqs,

    float* out_ptr, int stride_out_BL, int stride_out_D,

    int tot_BL, int d_model, int tot_AA,
    float dropout_p, uint32_t rng_seed,
    cudaStream_t stream
);

void wf_embedding_static_aa_forward(
    torch::Tensor coordsA, torch::Tensor coordsB,
    torch::Tensor aa_labels, torch::Tensor aa_magnitudes,
    torch::Tensor wavenumbers, torch::Tensor cu_seqlens,
    torch::Tensor out,
    float dropout_p, uint32_t rng_seed
) {

    // validate inputs
    TORCH_CHECK(coordsA.device().is_cuda(), "coordsA must be a CUDA tensor");
    TORCH_CHECK(coordsB.device().is_cuda(), "coordsB must be a CUDA tensor");
    TORCH_CHECK(aa_labels.device().is_cuda(), "aa_labels must be a CUDA tensor");
    TORCH_CHECK(aa_magnitudes.device().is_cuda(), "aa_magnitudes must be a CUDA tensor");
    TORCH_CHECK(wavenumbers.device().is_cuda(), "wavenumbers must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");

    TORCH_CHECK(coordsA.dtype() == torch::kFloat32, "coordsA must be of type float32");
    TORCH_CHECK(coordsB.dtype() == torch::kFloat32, "coordsB must be of type float32");
    TORCH_CHECK(aa_labels.dtype() == torch::kInt16, "aa_labels must be of type int16");
    TORCH_CHECK(aa_magnitudes.dtype() == torch::kFloat16, "aa_magnitudes must be of type float16");
    TORCH_CHECK(wavenumbers.dtype() == torch::kFloat32, "wavenumbers must be of type float32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be of type int32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be of type float32");

    // get tensor sizes (coordsA is transposed to [3, BL] for coalesced access)
    int tot_BL = coordsA.size(1);
    int d_model = wavenumbers.size(0) * 2;
    int tot_AA = aa_magnitudes.size(1);
    int num_seqs = cu_seqlens.size(0) - 1;

    TORCH_CHECK(out.size(0) == tot_BL, "out BL size mismatch");
    TORCH_CHECK(out.size(1) == d_model, "out d_model size mismatch");

    // get raw pointers
    float* coordsA_ptr = coordsA.data_ptr<float>();
    float* coordsB_ptr = coordsB.data_ptr<float>();
    int16_t* aa_labels_ptr = aa_labels.data_ptr<int16_t>();
    const __half* aa_magnitudes_ptr = reinterpret_cast<const __half*>(aa_magnitudes.data_ptr<at::Half>());
    const float* wavenumbers_ptr = wavenumbers.data_ptr<float>();
    const int* cu_seqlens_ptr = cu_seqlens.data_ptr<int>();
    float* out_ptr = out.data_ptr<float>();

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    wf_embedding_static_aa_kernel_forward(
        coordsA_ptr,
        coordsA.stride(0), coordsA.stride(1),
        coordsB_ptr,
        coordsB.stride(0), coordsB.stride(1),
        aa_labels_ptr,
        aa_labels.stride(0),
        aa_magnitudes_ptr,
        aa_magnitudes.stride(0), aa_magnitudes.stride(1),
        wavenumbers_ptr,
        wavenumbers.stride(0),
        cu_seqlens_ptr, num_seqs,
        out_ptr,
        out.stride(0), out.stride(1),
        tot_BL, d_model, tot_AA,
        dropout_p, rng_seed,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &wf_embedding_static_aa_forward, "Wavefunction Embedding (Static AA) Forward Method");
}
