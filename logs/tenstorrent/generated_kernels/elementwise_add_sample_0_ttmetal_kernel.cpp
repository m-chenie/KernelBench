// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Kernel for element-wise matrix addition
void elementwise_add(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    IDevice* device) {
    // Setup the device and command queue. Use the first core {0, 0}.
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreCoord core({0, 0});  // single core

    // Calculate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Nt = N / TILE_WIDTH;

    // Setting up DRAM buffers
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = sizeof(bfloat16) * a.size(),
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = sizeof(bfloat16) * b.size(),
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = sizeof(bfloat16) * output.size(),
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    auto src_a_buffer = CreateBuffer(dram_config_A);
    auto src_b_buffer = CreateBuffer(dram_config_B);
    auto dst_c_buffer = CreateBuffer(dram_config_C);

    // Create the compute kernel for element-wise addition
    vector<uint32_t> compute_compile_time_args = {Mt, Nt};
    auto compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "elementwise_add/kernels/compute/add.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src_a_addr = src_a_buffer->address();
    uint32_t src_b_addr = src_b_buffer->address();
    uint32_t dst_c_addr = dst_c_buffer->address();
    SetRuntimeArgs(program, compute_kernel_id, core, {src_a_addr, src_b_addr, dst_c_addr, Mt, Nt});

    // Upload the input data to the DRAM buffers, execute the kernel, and read the result
    EnqueueWriteBuffer(cq, src_a_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src_b_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_c_buffer, output.data(), true);
    Finish(cq);
}

int main() {
    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        constexpr uint32_t M = 640;  // M dimension
        constexpr uint32_t N = 640;  // N dimension

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");

        std::vector<bfloat16> matrix_a(M * N);
        std::vector<bfloat16> matrix_b(M * N);
        std::vector<bfloat16> matrix_c(M * N, 0);
        std::fill(matrix_a.begin(), matrix_a.end(), bfloat16(1.0));
        std::fill(matrix_b.begin(), matrix_b.end(), bfloat16(2.0));

        // Perform element-wise addition
        elementwise_add(matrix_a, matrix_b, matrix_c, M, N, device);

        // Simple check to print the first output value
        fmt::print("Element-wise add result first element: {}\n", float(matrix_c[0]));
        
        CloseDevice(device);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }
    return 0;
}