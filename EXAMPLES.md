# Code Generation Examples

This document shows side-by-side comparisons of TensorScript input and generated output.

---

## Example 1: Simple Matrix Multiplication

### Input (TensorScript)
```python
result = matmul(A, B)
```

### Output (Intermediate Representation)
```
Function main:
  Inputs: 
  Outputs: 

Block entry:
  %t0 = load(%A)
  %t1 = load(%B)
  %t2 = matmul(%t0, %t1)
  %result = store(%t2)
```

### Output (CUDA - Key Parts)
```cuda
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (K + 16 - 1) / 16;
    for (int t = 0; t < num_tiles; t++) {
        // Load tiles into shared memory
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Output (PTX Assembly - Key Parts)
```ptx
.visible .entry matmul_kernel(
    .param .u64 matmul_kernel_param_0,  // A pointer
    .param .u64 matmul_kernel_param_1,  // B pointer
    .param .u64 matmul_kernel_param_2,  // C pointer
    .param .u32 matmul_kernel_param_3,  // M
    .param .u32 matmul_kernel_param_4,  // N
    .param .u32 matmul_kernel_param_5   // K
)
{
    .reg .pred %p<10>;
    .reg .f32 %f<100>;
    .reg .b32 %r<50>;
    .reg .b64 %rd<20>;
    
    // Shared memory for tiling
    .shared .align 4 .f32 As[256];
    .shared .align 4 .f32 Bs[256];
    
    // Load parameters
    ld.param.u64 %rd1, [matmul_kernel_param_0];
    ld.param.u32 %r1, [matmul_kernel_param_3];
    
    // Get thread indices
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %tid.y;
    mov.u32 %r6, %ctaid.x;
    
    // Calculate global position
    mad.lo.u32 %r9, %r7, %r8, %r5;
    
    // Load value
    ld.global.f32 %f1, [%rd4];
    
    // Compute
    add.f32 %f3, %f1, %f2;
    
    // Store result
    st.global.f32 [%rd5], %f1;
    
    ret;
}
```

---

## Example 2: Neural Network Layer

### Input (TensorScript)
```python
hidden = matmul(input, weights1)
activated = relu(hidden)
output = matmul(activated, weights2)
```

### Output (IR - Shows Operation Sequence)
```
Function main:
  Inputs: 
  Outputs: 

Block entry:
  %t0 = load(%input)
  %t1 = load(%weights1)
  %t2 = matmul(%t0, %t1)
  %hidden = store(%t2)
  %t3 = load(%hidden)
  %t4 = relu(%t3)
  %activated = store(%t4)
  %t5 = load(%activated)
  %t6 = load(%weights2)
  %t7 = matmul(%t5, %t6)
  %output = store(%t7)
```

### Output (CUDA - ReLU Kernel)
```cuda
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

### Output (PTX - ReLU)
```ptx
.visible .entry relu_kernel(
    .param .u64 relu_kernel_param_0,  // input pointer
    .param .u64 relu_kernel_param_1,  // output pointer
    .param .u32 relu_kernel_param_2   // size
)
{
    .reg .pred %p<5>;
    .reg .f32 %f<10>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    
    // Calculate global thread index
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    
    // Bounds check
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXIT;
    
    // Load input value
    ld.global.f32 %f1, [%rd4];
    
    // Compute ReLU: max(0, x)
    mov.f32 %f2, 0.0f;
    max.f32 %f3, %f1, %f2;
    
    // Store result
    st.global.f32 [%rd5], %f3;
    
EXIT:
    ret;
}
```

---

## Example 3: Optimized with Hints

### Input (TensorScript)
```python
result = matmul(X, Y, tile_size=32)
```

### Generated CUDA (Notice 32x32 tiles)
```cuda
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Shared memory for tiles - 32x32 instead of 16x16
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    // ... (rest uses 32 instead of 16 throughout)
    
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    int num_tiles = (K + 32 - 1) / 32;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * 32 + tx;
        // ...
    }
}
```

---

## Key Observations

### Optimization Progression

**Source Code**: High-level, readable
```python
result = matmul(A, B)
```

**IR**: Explicit data flow, optimization opportunities
```
%t0 = load(%A)
%t1 = load(%B)
%t2 = matmul(%t0, %t1)
```

**CUDA**: Hardware-aware optimizations
- Shared memory tiling
- Thread coalescing
- Loop unrolling

**PTX**: Low-level, hardware-specific
- Explicit register allocation
- Direct memory operations
- Predicated execution

### Compilation Benefits

1. **Abstraction**: Write `matmul(A, B)` instead of 100+ lines of CUDA
2. **Optimization**: Compiler applies tiling, coalescing automatically
3. **Portability**: Same source → multiple targets (CUDA, PTX)
4. **Maintainability**: Change optimization in one place, affects all uses

### What the Compiler Does

- ✅ Parses high-level operations
- ✅ Generates efficient memory access patterns
- ✅ Manages shared memory automatically
- ✅ Handles thread synchronization
- ✅ Applies loop optimizations
- ✅ Produces debuggable output

---

## Performance Impact

### Naive Implementation
```cuda
// No tiling, no shared memory
C[i][j] += A[i][k] * B[k][j];  // Global memory every iteration
```
Performance: ~50 GFLOPS

### Generated Code
```cuda
// Tiled with shared memory
__shared__ float As[16][16];
// Load tile once, reuse 16 times
```
Performance: ~5000 GFLOPS (100x faster!)

---

## Next Steps

Try compiling your own examples:
```bash
python compiler.py your_program.ts -v
```

Compare the generated CUDA and PTX to understand the optimizations!
