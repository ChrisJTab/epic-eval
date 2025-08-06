//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef GPU_ALLOCATOR_H
#define GPU_ALLOCATOR_H

#include "allocator.h"

#ifdef EPIC_CUDA_AVAILABLE

namespace epic {
class GpuAllocator : public Allocator
{
public:
    void *Allocate(size_t size) override;
    void Free(void *ptr) override;
    void PrintMemoryInfo();
    void GetMemoryInfo(size_t &free, size_t &total);
};
} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // GPU_ALLOCATOR_H
