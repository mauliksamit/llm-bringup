// Fused RMSNorm forward: y = w * cast(x * rsqrt(mean(x^2) + eps)).
//
// One thread block per row. Each thread accumulates a float32 partial sum of
// squares over a strided slice of the row (vectorized 16-byte loads when the
// row is 16-byte aligned and its length is a multiple of the vector width),
// the block reduces with warp shuffles + shared memory, then every thread
// rescales its slice. Rounding matches the PyTorch composition in
// llm_bringup/model.py::rmsnorm_torch: statistics in float32, the normalized
// value cast to the input dtype, then multiplied by the weight in that dtype
// (computed in float32 and rounded once, as PyTorch does for bf16/fp16 mul).
// Only the summation order differs, so results can differ by rounding of the
// float32 rsqrt factor; tests/test_fused_rmsnorm.py bounds that.
//
// Written from scratch for this repo; the block-reduction pattern is the
// standard warp-shuffle reduction described in the CUDA C++ Programming Guide
// (docs/REFERENCES.md).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }

constexpr int kThreads = 256;

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float warp_sums[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();  // uniform: every thread of the block reaches it
  float total = 0.f;
  if (warp == 0) {
    total = lane < kThreads / 32 ? warp_sums[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

// VEC elements of T moved as one aligned struct: 16 bytes when VEC = 16/sizeof(T)
// (8 for bf16/fp16, 4 for float), so each access is a single 128-bit load/store.
// VEC = 1 is the scalar path for unaligned rows or lengths.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Pack {
  T v[VEC];
};

template <typename T, int VEC>
__global__ void __launch_bounds__(kThreads) rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                                           T* __restrict__ y, int hidden, float eps) {
  using P = Pack<T, VEC>;
  const size_t row = blockIdx.x;
  const T* xr = x + row * hidden;
  T* yr = y + row * hidden;
  float ss = 0.f;
  for (int i = threadIdx.x * VEC; i < hidden; i += kThreads * VEC) {
    const P xv = *reinterpret_cast<const P*>(xr + i);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float v = to_f(xv.v[j]);
      ss += v * v;
    }
  }
  const float r = rsqrtf(block_sum(ss) / hidden + eps);
  for (int i = threadIdx.x * VEC; i < hidden; i += kThreads * VEC) {
    const P xv = *reinterpret_cast<const P*>(xr + i);  // second read of the row: served by L1/L2
    const P wv = *reinterpret_cast<const P*>(w + i);
    P out;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const T normed = from_f<T>(to_f(xv.v[j]) * r);          // cast to input dtype first (matches torch)
      out.v[j] = from_f<T>(to_f(wv.v[j]) * to_f(normed));     // then weight multiply, rounded once
    }
    *reinterpret_cast<P*>(yr + i) = out;
  }
}

template <typename T>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int rows, int hidden, float eps) {
  constexpr int VEC = 16 / sizeof(T);
  const auto* xp = reinterpret_cast<const T*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const T*>(w.data_ptr());
  auto* yp = reinterpret_cast<T*>(y.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool aligned = hidden % VEC == 0 && reinterpret_cast<uintptr_t>(xp) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(wp) % 16 == 0 && reinterpret_cast<uintptr_t>(yp) % 16 == 0;
  // The vector width only changes which elements a thread owns (contiguous
  // runs of VEC); the per-element arithmetic is identical on both paths.
  if (aligned)
    rmsnorm_kernel<T, VEC><<<rows, kThreads, 0, stream>>>(xp, wp, yp, hidden, eps);
  else
    rmsnorm_kernel<T, 1><<<rows, kThreads, 0, stream>>>(xp, wp, yp, hidden, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "fused_rmsnorm: tensors must be on CUDA");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "fused_rmsnorm: x and w must share a dtype");
  TORCH_CHECK(w.dim() == 1 && x.size(-1) == w.size(0), "fused_rmsnorm: w must be [hidden]");
  const at::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  const int hidden = static_cast<int>(x.size(-1));
  const int64_t rows64 = hidden == 0 ? 0 : xc.numel() / hidden;
  TORCH_CHECK(rows64 <= INT32_MAX, "fused_rmsnorm: too many rows");
  auto y = torch::empty_like(xc);
  if (rows64 == 0) return y;
  const int rows = static_cast<int>(rows64);
  const float e = static_cast<float>(eps);
  switch (x.scalar_type()) {
    case at::kFloat: launch<float>(xc, wc, y, rows, hidden, e); break;
    case at::kBFloat16: launch<__nv_bfloat16>(xc, wc, y, rows, hidden, e); break;
    case at::kHalf: launch<__half>(xc, wc, y, rows, hidden, e); break;
    default: TORCH_CHECK(false, "fused_rmsnorm: unsupported dtype ", x.scalar_type());
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward (CUDA)");
}
