# TensorScript Compiler

A domain-specific language (DSL) compiler for GPU computing that translates high-level tensor operations into optimized CUDA kernels and PTX assembly.

## Overview

TensorScript is a minimal language designed to express common GPU computing patterns. The compiler implements a complete compilation pipeline:

```
TensorScript DSL → AST → IR → CUDA/PTX
```

### Key Features

- **Simple DSL**: Write tensor operations in a clean, readable syntax
- **Multi-stage compilation**: Lexer → Parser → IR → Code generation
- **Multiple targets**: Generate CUDA C++ or PTX assembly
- **Optimizations**: Shared memory tiling, loop unrolling, memory coalescing
- **Portable IR**: SSA-form intermediate representation for easy optimization

## Architecture

### Compilation Pipeline

```
┌─────────────┐      ┌────────┐      ┌─────────┐      ┌──────────────┐
│ TensorScript│ -->  │  AST   │ -->  │   IR    │ -->  │  CUDA / PTX  │
│   Source    │      │        │      │  (SSA)  │      │              │
└─────────────┘      └────────┘      └─────────┘      └──────────────┘
     Lexer             Parser        IR Generator      Code Generator
```

### Components

1. **Lexer** (`lexer.py`): Tokenizes source code
2. **Parser** (`parser.py`): Builds Abstract Syntax Tree
3. **IR Generator** (`ir_generator.py`): Creates SSA-form intermediate representation
4. **CUDA Generator** (`cuda_generator.py`): Emits optimized CUDA kernels
5. **PTX Generator** (`ptx_generator.py`): Emits PTX assembly

## Language Syntax

### Supported Operations

```python
# Matrix multiplication
result = matmul(A, B)

# Activation functions
activated = relu(input)
probs = softmax(logits)

# Element-wise operations
sum = add(x, y)

# Matrix operations
transposed = transpose(matrix)
```

### Optimization Hints

```python
# Specify tile size for shared memory optimization
result = matmul(X, Y, tile_size=32)
```

### Complete Example

```python
# Two-layer neural network forward pass
hidden = matmul(input, weights1)
activated = relu(hidden)
output = matmul(activated, weights2)
final = softmax(output)
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA Toolkit (optional, for compiling generated code)
- NVIDIA GPU (optional, for running generated kernels)

### Setup

```bash
git clone <repository-url>
cd tensorscript-compiler
```

No additional dependencies required - uses only Python standard library.

## Usage

### Command Line Interface

```bash
# Compile to CUDA
python compiler.py examples.ts -o output.cu

# Compile to PTX assembly
python compiler.py examples.ts -f ptx -o output.ptx

# Show intermediate representation
python compiler.py examples.ts -f ir

# Verbose mode (show all compilation phases)
python compiler.py examples.ts -v
```

### Python API

```python
from compiler import Compiler

# Create compiler instance
compiler = Compiler(verbose=True)

# Compile source code
source = "result = matmul(A, B)"
cuda_code = compiler.compile(source, output_format='cuda')

# Or compile a file
compiler.compile_file('program.ts', 'output.cu', output_format='cuda')
```

### Interactive Demo

```bash
python demo.py
```

The demo showcases:
- Basic matrix multiplication
- Neural network forward pass
- Optimization hints
- PTX assembly generation
- Full compilation pipeline with verbose output

## Code Generation Details

### CUDA Kernels

Generated CUDA code includes:

- **Shared memory tiling** for matrix multiplication
- **Thread coalescing** for memory access patterns
- **Loop unrolling** for better instruction-level parallelism
- **Bounds checking** for safety
- **Configurable tile sizes** for optimization

Example generated kernel:

```cuda
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    // ... optimized tiled matrix multiplication
}
```

### PTX Assembly

Generated PTX includes:

- **Register allocation**
- **Predicated execution** for divergent control flow
- **Explicit memory operations** (ld.global, st.global)
- **Thread indexing** using special registers (%tid, %ctaid)

## Testing

Run the test suite:

```bash
python test_compiler.py
```

Tests cover:
- Lexical analysis
- Parsing
- IR generation
- Code generation
- End-to-end compilation

## Project Structure

```
tensorscript-compiler/
├── lexer.py              # Lexical analyzer
├── ast_nodes.py          # AST node definitions
├── parser.py             # Parser (tokens → AST)
├── ir.py                 # IR data structures
├── ir_generator.py       # IR generation (AST → IR)
├── cuda_generator.py     # CUDA code generation
├── ptx_generator.py      # PTX assembly generation
├── compiler.py           # Main compiler driver
├── demo.py               # Interactive demo
├── test_compiler.py      # Test suite
├── examples.ts           # Example programs
└── README.md             # This file
```

## Examples

### Example 1: Matrix Multiplication

**Input (TensorScript):**
```python
result = matmul(A, B)
```

**Output (IR):**
```
%t0 = load(%A)
%t1 = load(%B)
%t2 = matmul(%t0, %t1)
%result = store(%t2)
```

**Output (CUDA):**
Optimized kernel with shared memory tiling and thread coalescing.

### Example 2: Neural Network

**Input (TensorScript):**
```python
hidden = matmul(input, weights1)
activated = relu(hidden)
output = matmul(activated, weights2)
```

**Output:**
Three optimized kernels: `matmul_kernel`, `relu_kernel`, and a second `matmul_kernel`.

## Technical Highlights

### Intermediate Representation

- **SSA form**: Single Static Assignment for easier optimization
- **Typed values**: All values have types (e.g., `tensor<f32>`)
- **Basic blocks**: Structured control flow representation

### Optimizations

1. **Shared Memory Tiling**
   - Reduces global memory accesses
   - Configurable tile sizes
   - Minimizes bank conflicts

2. **Memory Coalescing**
   - Threads access contiguous memory
   - Maximizes memory bandwidth

3. **Loop Unrolling**
   - Reduces loop overhead
   - Enables better instruction scheduling

## Performance Considerations

The generated CUDA kernels are designed with performance in mind:

- **Tile size selection**: Default 16x16, configurable via hints
- **Occupancy**: Chosen block sizes for good GPU utilization
- **Memory hierarchy**: Explicit use of shared memory for reuse
- **Numerical stability**: e.g., softmax uses max subtraction


---

**Built for learning and experimentation with compiler construction and GPU programming.**
