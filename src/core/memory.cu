#include "core/memory.cuh"

void *memory_device_allocate(const std::size_t size, const bool verbose) {
    void *ptr_ = nullptr;
    
    CUDA_CHECK(cudaMalloc(&ptr_, size), verbose);

    FORMAT_ADDRESS(ptr_, addr_ptr_);
    if (verbose) KERAVNOS_PRINT_CUDA("allocated ", size, " bytes to memory address ", addr_ptr_);
    return ptr_;
}

void *memory_host_allocate(const std::size_t size, const bool verbose) {
    void *ptr_ = nullptr;

    ptr_ = static_cast<void *>(malloc(size));
    if (!ptr_) {
        if (verbose) KERAVNOS_PRINT_CUDA_ERROR("memory allocation for ", size, " bytes failed.");
        exit(EXIT_FAILURE);
    }

    FORMAT_ADDRESS(ptr_, addr_ptr_);
    if (verbose) KERAVNOS_PRINT_CUDA("allocated ", size, " bytes to memory address ", addr_ptr_);
    return ptr_;
}

void memory_device_deallocate(void *ptr, const bool verbose) {
    FORMAT_ADDRESS(ptr, addr_ptr_);
    
    if (!ptr) {
        if (verbose) KERAVNOS_PRINT_CUDA_ERROR("memory address ", addr_ptr_, " is null.");
        return;
    }

    CUDA_CHECK(cudaFree(ptr), verbose);
    if (verbose) KERAVNOS_PRINT_CUDA("deallocated memory address ", addr_ptr_);
}

void memory_host_deallocate(void *ptr, const bool verbose) {
    FORMAT_ADDRESS(ptr, addr_ptr_);

    if (!ptr) {
        if (verbose) KERAVNOS_PRINT_CUDA_ERROR("memory address ", addr_ptr_, " is null.");
        return;
    }
    
    free(ptr);
    if (verbose) KERAVNOS_PRINT_CUDA("deallocated memory address ", addr_ptr_);
}

void memory_copy_host_to_device(void *dvc_dst, const void *hst_src, const std::size_t size, const bool verbose) {
    FORMAT_ADDRESS(dvc_dst, addr_dvc_dst_);
    FORMAT_ADDRESS(hst_src, addr_hst_src_);
    
    if (!dvc_dst || !hst_src) {
        if (verbose) {
            if (!dvc_dst) KERAVNOS_PRINT_CUDA_ERROR("device destination memory address ", addr_dvc_dst_, " is null.");
            else if (!hst_src) KERAVNOS_PRINT_CUDA_ERROR("host source memory address ", addr_hst_src_, " is null.");
        }
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(dvc_dst, hst_src, size, cudaMemcpyHostToDevice), verbose);
    if (verbose) KERAVNOS_PRINT_CUDA("copied ", size, " bytes from host address ", addr_hst_src_, " to device address ", addr_dvc_dst_);
}

void memory_copy_device_to_host(void *hst_dst, const void *dvc_src, const std::size_t size, const bool verbose) {
    FORMAT_ADDRESS(hst_dst, addr_hst_dst_);
    FORMAT_ADDRESS(dvc_src, addr_dvc_src_);
    
    if (!hst_dst || !dvc_src) {
        if (verbose) {
            if (!hst_dst) KERAVNOS_PRINT_CUDA_ERROR("host destination memory address ", addr_hst_dst_, " is null.");
            else if (!dvc_src) KERAVNOS_PRINT_CUDA_ERROR("device source memory address ", addr_dvc_src_, " is null.");
        }
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(hst_dst, dvc_src, size, cudaMemcpyDeviceToHost), verbose);
    if (verbose) KERAVNOS_PRINT_CUDA("copied ", size, " bytes from device address ", addr_dvc_src_, " to host address ", addr_hst_dst_);
}

void memory_copy_device_to_device(void *dvc_dst, const void *dvc_src, const std::size_t size, const bool verbose) {
    FORMAT_ADDRESS(dvc_dst, addr_dvc_dst_);
    FORMAT_ADDRESS(dvc_src, addr_dvc_src_);
    
    if (!dvc_dst || !dvc_src) {
        if (verbose) {
            if (!dvc_dst) KERAVNOS_PRINT_CUDA_ERROR("device destination memory address ", addr_dvc_dst_, " is null.");
            else if (!dvc_src) KERAVNOS_PRINT_CUDA_ERROR("device source memory address ", addr_dvc_src_, " is null.");
        }
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(dvc_dst, dvc_src, size, cudaMemcpyDeviceToDevice), verbose);
    if (verbose) KERAVNOS_PRINT_CUDA("copied ", size, " bytes from device address ", addr_dvc_src_, " to device address ", addr_dvc_dst_);
}
