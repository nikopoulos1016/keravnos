#pragma once

#include "global.cuh"


void *memory_device_allocate(const std::size_t size, const bool verbose);
void *memory_host_allocate(const std::size_t size, const bool verbose);

void memory_device_deallocate(void *ptr, const bool verbose);
void memory_host_deallocate(void *ptr, const bool verbose);

void memory_copy_host_to_device(void *dvc_dst, const void *hst_src, const std::size_t size, const bool verbose);
void memory_copy_device_to_host(void *hst_dst, const void *dvc_src, const std::size_t size, const bool verbose);
void memory_copy_device_to_device(void *dvc_dst, const void *dvc_src, const std::size_t size, const bool verbose);
