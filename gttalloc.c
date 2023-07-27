#define __HIP_PLATFORM_AMD__
#include <sys/types.h>
#include <hip/hip_runtime.h>
// Compile with `/opt/rocm/bin/hipcc  alloc.c -o alloc.so --shared -fPIC`

void* gtt_alloc(ssize_t size, int device, hipStream_t stream) {
   void *ptr = NULL;
   hipHostMalloc(&ptr,size,0);
   return ptr;
}

void gtt_free(void* ptr, ssize_t size, int device, hipStream_t stream) {
	hipHostFree(ptr);
}
