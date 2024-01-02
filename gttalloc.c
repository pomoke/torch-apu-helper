// This code is derived from
// 'https://pytorch.org/docs/stable/notes/cuda.html#using-custom-memory-allocators-for-cuda'.

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <sys/types.h>
// Compile with `/opt/rocm/bin/hipcc  alloc.c -o alloc.so --shared -fPIC`

void *gtt_alloc(ssize_t size, int device, hipStream_t stream) {
  void *ptr = NULL;

  if (hipMalloc(&ptr, size) != hipSuccess) {
    hipHostMalloc(&ptr, size, 0);
  }

  return ptr;
}

void gtt_free(void *ptr, ssize_t size, int device, hipStream_t stream) {
  hipPointerAttribute_t attr;

  if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
    return;
  }

  if (attr.memoryType == hipMemoryTypeDevice) {
    hipFree(ptr);
  } else if (attr.memoryType == hipMemoryTypeHost) {
    hipHostFree(ptr);
  }
}
