# PyTorch Host Allocator for APUs

ROCm does not take GTT into account when calculating usable VRAM on APU platforms.

With this allocator, now we can use GTT (host memory) with PyTorch, and there is no need to tweak VRAM configuration.

## Usage

Compile `gttalloc.c` with `hipcc gttalloc.cc -o alloc.so -shared -fPIC`.

If `hipcc` is not found, it may reside in `/opt/rocm/bin/hipcc`.

Then, for programs using PyTorch, put code below between `import torch` and actual usage of Torch.

```python
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('<path to alloc.so>','gtt_alloc','gtt_free');
torch.cuda.memory.change_current_allocator(new_alloc)
```

## It works!
If you have something work or not work with HIP on APUs, please share it in discussion.

### Summary
To be filled...

## 中文指南
参见 https://typeof.pw/archives/pytorch-on-apu-vram .
