"""
CUDA Code Generator - Converts IR to CUDA C++ kernels
"""

from ir import *
from typing import Dict, Set


class CUDAGenerator:
    """Generates CUDA C++ code from IR"""
    
    def __init__(self):
        self.kernel_counter = 0
        self.generated_kernels: List[str] = []
    
    def generate(self, module: IRModule) -> str:
        """Generate CUDA code from IR module"""
        code = []
        
        # Add includes
        code.append(self.generate_includes())
        code.append("")
        
        # Generate kernels for each function
        for func in module.functions:
            kernel_code = self.generate_function(func)
            code.append(kernel_code)
            code.append("")
        
        return "\n".join(code)
    
    def generate_includes(self) -> str:
        """Generate necessary includes"""
        return """#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 16
#define CHECK_CUDA(call) { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error in %s:%d: %s\\n", __FILE__, __LINE__, \\
                cudaGetErrorString(err)); \\
        exit(EXIT_FAILURE); \\
    } \\
}"""
    
    def generate_function(self, func: IRFunction) -> str:
        """Generate CUDA function from IR function"""
        code = []
        
        # Analyze operations to determine what kernels we need
        for block in func.blocks:
            for op in block.operations:
                if op.op_type == IROpType.MATMUL:
                    code.append(self.generate_matmul_kernel(op))
                elif op.op_type == IROpType.RELU:
                    code.append(self.generate_relu_kernel(op))
                elif op.op_type == IROpType.SOFTMAX:
                    code.append(self.generate_softmax_kernel(op))
                elif op.op_type == IROpType.ADD:
                    code.append(self.generate_add_kernel(op))
                elif op.op_type == IROpType.TRANSPOSE:
                    code.append(self.generate_transpose_kernel(op))
        
        # Generate host wrapper function
        code.append(self.generate_host_wrapper(func))
        
        return "\n\n".join(code)
    
    def generate_matmul_kernel(self, op: IROperation) -> str:
        """Generate optimized matrix multiplication kernel with tiling"""
        tile_size = op.attributes.get('tile_size', 16)
        
        return f"""// Matrix multiplication kernel with shared memory tiling
// Computes C = A * B where A is MxK and B is KxN
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {{
    // Shared memory for tiles
    __shared__ float As[{tile_size}][{tile_size}];
    __shared__ float Bs[{tile_size}][{tile_size}];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Row and column of C to compute
    int row = by * {tile_size} + ty;
    int col = bx * {tile_size} + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (K + {tile_size} - 1) / {tile_size};
    for (int t = 0; t < num_tiles; t++) {{
        // Load tiles into shared memory
        int a_col = t * {tile_size} + tx;
        int b_row = t * {tile_size} + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < {tile_size}; k++) {{
            sum += As[ty][k] * Bs[k][tx];
        }}
        
        __syncthreads();
    }}
    
    // Write result
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}"""
    
    def generate_relu_kernel(self, op: IROperation) -> str:
        """Generate ReLU activation kernel"""
        return """// ReLU activation: out = max(0, in)
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}"""
    
    def generate_softmax_kernel(self, op: IROperation) -> str:
        """Generate softmax kernel (simplified version)"""
        return """// Softmax activation (per-row for 2D tensors)
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    // Find max value in row (for numerical stability)
    float max_val = -INFINITY;
    for (int col = 0; col < cols; col++) {
        max_val = fmaxf(max_val, input[row * cols + col]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        float exp_val = expf(input[row * cols + col] - max_val);
        output[row * cols + col] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int col = 0; col < cols; col++) {
        output[row * cols + col] /= sum;
    }
}"""
    
    def generate_add_kernel(self, op: IROperation) -> str:
        """Generate element-wise addition kernel"""
        return """// Element-wise addition: C = A + B
__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}"""
    
    def generate_transpose_kernel(self, op: IROperation) -> str:
        """Generate matrix transpose kernel"""
        return """// Matrix transpose with shared memory
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33]; // 33 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load tile
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Write transposed tile
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}"""
    
    def generate_host_wrapper(self, func: IRFunction) -> str:
        """Generate host-side wrapper function"""
        return """// Host wrapper function
extern "C" void launch_kernels() {
    // This would be customized based on actual operations
    // For now, this is a placeholder showing kernel launch pattern
    
    // Example: Launch matmul kernel
    // dim3 blockDim(16, 16);
    // dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    // matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    // CHECK_CUDA(cudaDeviceSynchronize());
}"""


def generate_cuda(module: IRModule) -> str:
    """Convenience function to generate CUDA code from IR"""
    generator = CUDAGenerator()
    return generator.generate(module)


if __name__ == '__main__':
    from lexer import tokenize
    from parser import parse
    from ir_generator import generate_ir
    
    # Test CUDA generation
    test_code = """
    result = matmul(A, B)
    activated = relu(result)
    """
    
    tokens = tokenize(test_code)
    ast = parse(tokens)
    ir_module = generate_ir(ast)
    cuda_code = generate_cuda(ir_module)
    
    print(cuda_code)
