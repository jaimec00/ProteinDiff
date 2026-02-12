
/*
author:         jaime cardenas
title:          wf_embedding_static_aa_kernel.cu
description:    cuda kernel for packed wf embedding with static (non-learnable) aa magnitudes.
				removes d_aa gradient computation and storage, reducing smem from ~13.6KB to ~3.3KB.
				operates on packed [BL, ...] tensors with cu_seqlens boundaries.
				one block per packed observer token, binary-searches cu_seqlens
				to find sequence boundaries, then iterates NJ within that sequence.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

__constant__ unsigned FULL_MASK = 0xFFFFFFFF;

// warp reduce via shuffle-down to superpose wavefunctions computed by individual threads
__device__ float superpose(float value) {

	for (int offset = 16; offset > 0; offset /= 2) {
		value += __shfl_down_sync(FULL_MASK, value, offset);
	}
	return value;
}

// broadcast AA magnitude from the thread holding the matching aa index
__device__ float get_AA_scale(int16_t aa_idx, float aa_magnitude) {

	float scale = __shfl_sync(FULL_MASK, aa_magnitude, aa_idx);
	return scale;
}

// binary search cu_seqlens via __ldg (read-only cache) to find which sequence a packed position belongs to
// cu_seqlens lives in global memory and stays in L2 — all blocks read the same small array
__device__ int find_seq_idx(const int* cu_seqlens_ptr, int num_seqs, int pos) {
	int lo = 0, hi = num_seqs;
	while (lo < hi) {
		int mid = (lo + hi) / 2;
		if (__ldg(&cu_seqlens_ptr[mid]) <= pos) lo = mid + 1;
		else hi = mid;
	}
	return lo - 1;
}

// device kernel for wavefunction embedding with static aa magnitudes (no gradient computation)
__launch_bounds__(256, 4)
__global__ void wf_embedding_static_aa_kernel(
    float* coordsA_ptr, int stride_coordsA_S, int stride_coordsA_BL,
    float* coordsB_ptr, int stride_coordsB_S, int stride_coordsB_BL,
    int16_t* aa_labels_ptr, int stride_aa_labels_BL,
    const __half* aa_magnitudes_ptr, int stride_aa_magnitudes_K, int stride_aa_magnitudes_A,
    const float* wavenumbers_ptr, int stride_wavenumbers_K,
    const int* cu_seqlens_ptr, int num_seqs,

    float* out_ptr, int stride_out_BL, int stride_out_D,

    int tot_BL, int d_model, int tot_AA,
    float dropout_p, uint32_t rng_seed
) {

	int thread_id = threadIdx.x;
	int num_threads = blockDim.x;
	int lane_id = thread_id % warpSize;

	// one block per packed observer token — no batch dimension
	int offs_NI = blockIdx.x;

	// shared memory layout: [seq_info | coordsA_NI | wavenumbers | out | aa_magnitudes]
	// seq_info is separate from coordsA_NI to avoid WAR hazard when coords overwrite seq boundaries
	extern __shared__ __align__(4) char smem[];

	int num_wn = d_model / 2;
	int num_aa = tot_AA * num_wn;
	int smem_float_bytes = sizeof(float) * (2 + 3 + num_wn + d_model);
	int smem_half_bytes = sizeof(__half) * num_aa;
	int smem_bytes = smem_float_bytes + smem_half_bytes;

	// partition shared memory — seq_info lives in its own slot, not aliased with coords
	int* seq_info_smem = reinterpret_cast<int*>(smem);
	float* coordsA_NI_smem = reinterpret_cast<float*>(smem + sizeof(int) * 2);
	float* wavenumbers_smem = coordsA_NI_smem + 3;
	float* out_smem = wavenumbers_smem + num_wn;
	__half* aa_magnitudes_smem = reinterpret_cast<__half*>(smem + smem_float_bytes);

	// zero-init smem with 128-bit writes
	int4* smem_int4 = reinterpret_cast<int4*>(smem);
	int smem_int4_count = smem_bytes / 16;
	for (int i = thread_id; i < smem_int4_count; i += num_threads) {
		smem_int4[i] = make_int4(0, 0, 0, 0);
	}
	for (int i = (smem_int4_count * 16) + thread_id; i < smem_bytes; i += num_threads) {
		smem[i] = 0;
	}
	__syncthreads();

	// thread 0 binary-searches cu_seqlens in global memory (L2 cached across all blocks)
	if (thread_id == 0) {
		int seq_idx = find_seq_idx(cu_seqlens_ptr, num_seqs, offs_NI);
		int seq_start = __ldg(&cu_seqlens_ptr[seq_idx]);
		int seq_len = __ldg(&cu_seqlens_ptr[seq_idx + 1]) - seq_start;
		seq_info_smem[0] = seq_start;
		seq_info_smem[1] = seq_len;
	}
	__syncthreads();

	// all threads read sequence boundaries into registers
	int seq_start = seq_info_smem[0];
	int seq_len = seq_info_smem[1];
	int offs_NI_local = offs_NI - seq_start;

	// register variables for observer coordinates and per-aa state
	float coordsA_NI_x;
	float coordsA_NI_y;
	float coordsA_NI_z;
	float aa_magnitude = 1.0f;

	// load wavenumbers and per-wavenumber aa magnitudes to smem
	// stays in smem throughout, moved to registers when looping through corresponding k
	// no sync needed here — covered by j==0 syncthreads before first read
	#pragma unroll 1
	for (int idx = thread_id; idx < num_aa; idx += num_threads) {
		aa_magnitudes_smem[idx] = aa_magnitudes_ptr[idx];
		if (idx < num_wn) {
			wavenumbers_smem[idx] = wavenumbers_ptr[idx];
		}
	}

	// iterate NJ within this sequence using cyclic offset from observer position
	int NJ_iters = (seq_len + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int j = 0; j < NJ_iters; ++j) {

		int thread_offset = j * num_threads + thread_id;
		int offs_NJ_local = (offs_NI_local + thread_offset) % seq_len;
		int offs_NJ = seq_start + offs_NJ_local;
		bool thread_mask = thread_offset < seq_len;

		// load source token data from global memory
		float coordsA_NJ_x = coordsA_ptr[0 * stride_coordsA_S + offs_NJ];
		float coordsA_NJ_y = coordsA_ptr[1 * stride_coordsA_S + offs_NJ];
		float coordsA_NJ_z = coordsA_ptr[2 * stride_coordsA_S + offs_NJ];
		float coordsB_NJ_x = coordsB_ptr[0 * stride_coordsB_S + offs_NJ];
		float coordsB_NJ_y = coordsB_ptr[1 * stride_coordsB_S + offs_NJ];
		float coordsB_NJ_z = coordsB_ptr[2 * stride_coordsB_S + offs_NJ];
		int16_t aa_labels_NJ = aa_labels_ptr[offs_NJ];

		// on first iteration, thread 0's cyclic offset starts at offs_NI_local, so its source IS the observer
		// broadcast observer coords to all threads via smem — one global read serves both NJ and NI
		if (j == 0) {

			if (thread_id == 0) {
				coordsA_NI_smem[0] = coordsA_NJ_x;
				coordsA_NI_smem[1] = coordsA_NJ_y;
				coordsA_NI_smem[2] = coordsA_NJ_z;
			}

			__syncthreads();

			coordsA_NI_x = coordsA_NI_smem[0];
			coordsA_NI_y = coordsA_NI_smem[1];
			coordsA_NI_z = coordsA_NI_smem[2];
		}

		// pairwise distance between observer and source
		float distsA_x = coordsA_NI_x - coordsA_NJ_x;
		float distsA_y = coordsA_NI_y - coordsA_NJ_y;
		float distsA_z = coordsA_NI_z - coordsA_NJ_z;

		// compute distance, exclude out-of-bounds threads and self-interaction
		float distsA_raw = (distsA_x * distsA_x) + (distsA_y * distsA_y) + (distsA_z * distsA_z);
		bool mask_IJ = thread_mask && (distsA_raw != 0);
		float inv_distA = mask_IJ * rsqrtf(distsA_raw + (!mask_IJ));
		float distsA = distsA_raw * inv_distA;

		// anisotropic modulation: project beta carbon unit vector along pairwise direction
		// will scale by per-aa magnitude in wavenumber loop
		float AdotB_unit = inv_distA * ((coordsB_NJ_x * distsA_x) + (coordsB_NJ_y * distsA_y) + (coordsB_NJ_z * distsA_z));

		// loop over wavenumbers, computing wavefunction for each (no gradient computation)
		#pragma unroll 1
		for (int k = 0; k < num_wn; ++k) {

			// first tot_AA threads in each warp load their aa's magnitude for this wavenumber
			if (lane_id < tot_AA) {
				aa_magnitude = __half2float(aa_magnitudes_smem[k * tot_AA + lane_id]);
			}

			// scale anisotropic term by source token's aa magnitude via warp shuffle
			float AdotB_scaled = AdotB_unit * get_AA_scale(aa_labels_NJ, aa_magnitude);

			// compute phase and trig
			float wavenumber = wavenumbers_smem[k];
			float phase = (distsA - AdotB_scaled) * wavenumber;

			float sine, cosine;
			__sincosf(phase, &sine, &cosine);

			float real = cosine * inv_distA;
			float imag = sine * inv_distA;

			// superpose sources within each warp, then atomicAdd across warps to smem
			float real_superposition = superpose(real);
			float imag_superposition = superpose(imag);

			if (lane_id == 0) {
				atomicAdd(&out_smem[2 * k], real_superposition);
				atomicAdd(&out_smem[2 * k + 1], imag_superposition);
			}
		}
	}

	// write output from shared memory to global memory
	out_ptr += offs_NI * stride_out_BL;

	__syncthreads();

	for (int o = thread_id; o < d_model; o += num_threads) {
		out_ptr[o] = out_smem[o];
	}
}

// host function to configure and launch the cuda kernel
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
) {
	// one block per packed token, 256 threads per block for NJ iteration
	dim3 block_size(256, 1, 1);
	dim3 grid_size(tot_BL, 1, 1);

	// configure max shared memory opt-in for the device
	int device;
	cudaGetDevice(&device);
	int maxSharedMemPerBlockOptin = 0;
	cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
	cudaFuncSetAttribute(wf_embedding_static_aa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemPerBlockOptin);
	cudaFuncSetCacheConfig(wf_embedding_static_aa_kernel, cudaFuncCachePreferShared);

	// shared memory: seq_info (2 ints) + float arrays (coords, wavenumbers, out) + half aa_magnitudes
	int num_wn = d_model / 2;
	int shared_mem = sizeof(float) * (2 + 3 + num_wn + d_model)
				   + sizeof(__half) * (num_wn * tot_AA);

	wf_embedding_static_aa_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coordsA_ptr, stride_coordsA_S, stride_coordsA_BL,
		coordsB_ptr, stride_coordsB_S, stride_coordsB_BL,
		aa_labels_ptr, stride_aa_labels_BL,
		aa_magnitudes_ptr, stride_aa_magnitudes_K, stride_aa_magnitudes_A,
		wavenumbers_ptr, stride_wavenumbers_K,
		cu_seqlens_ptr, num_seqs,
		out_ptr, stride_out_BL, stride_out_D,
		tot_BL, d_model, tot_AA,
		dropout_p, rng_seed
	);
}
