"""
PTX Assembly Generator - Converts IR to PTX assembly
PTX (Parallel Thread Execution) is NVIDIA's virtual ISA
"""

from ir import *
from typing import List


class PTXGenerator:
    """Generates PTX assembly from IR"""
    
    def __init__(self):
        self.register_counter = 0
        self.label_counter = 0
    
    def new_register(self, type_suffix='f32') -> str:
        """Generate new register name"""
        name = f"%r{self.register_counter}"
        self.register_counter += 1
        return name
    
    def new_label(self, prefix='L') -> str:
        """Generate new label"""
        name = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return name
    
    def generate(self, module: IRModule) -> str:
        """Generate PTX assembly from IR module"""
        code = []
        
        # PTX version and target
        code.append(self.generate_header())
        code.append("")
        
        # Generate code for each function
        for func in module.functions:
            ptx = self.generate_function(func)
            code.append(ptx)
            code.append("")
        
        return "\n".join(code)
    
    def generate_header(self) -> str:
        """Generate PTX header directives"""
        return """.version 8.0
.target sm_75  // Compute capability 7.5 (Turing)
.address_size 64"""
    
    def generate_function(self, func: IRFunction) -> str:
        """Generate PTX function from IR function"""
        code = []
        
        # Analyze operations to generate appropriate PTX
        for block in func.blocks:
            for op in block.operations:
                if op.op_type == IROpType.MATMUL:
                    code.append(self.generate_matmul_ptx(op))
                elif op.op_type == IROpType.RELU:
                    code.append(self.generate_relu_ptx(op))
                elif op.op_type == IROpType.ADD:
                    code.append(self.generate_add_ptx(op))
        
        return "\n\n".join(code)
    
    def generate_matmul_ptx(self, op: IROperation) -> str:
        """Generate PTX for matrix multiplication using tensor cores if available"""
        return """.visible .entry matmul_kernel(
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
    ld.param.u64 %rd1, [matmul_kernel_param_0];  // A
    ld.param.u64 %rd2, [matmul_kernel_param_1];  // B
    ld.param.u64 %rd3, [matmul_kernel_param_2];  // C
    ld.param.u32 %r1, [matmul_kernel_param_3];   // M
    ld.param.u32 %r2, [matmul_kernel_param_4];   // N
    ld.param.u32 %r3, [matmul_kernel_param_5];   // K
    
    // Get thread indices
    mov.u32 %r4, %tid.x;      // Thread index in block (x)
    mov.u32 %r5, %tid.y;      // Thread index in block (y)
    mov.u32 %r6, %ctaid.x;    // Block index (x)
    mov.u32 %r7, %ctaid.y;    // Block index (y)
    mov.u32 %r8, %ntid.x;     // Block dimension (x)
    
    // Calculate global row and column
    mad.lo.u32 %r9, %r7, %r8, %r5;   // row = blockIdx.y * blockDim.x + threadIdx.y
    mad.lo.u32 %r10, %r6, %r8, %r4;  // col = blockIdx.x * blockDim.x + threadIdx.x
    
    // Initialize accumulator
    mov.f32 %f1, 0.0f;
    
    // Main computation loop (simplified - actual tiling would be more complex)
    // For brevity, showing the structure rather than full tiling logic
    
    // ... (tile loading and computation would go here)
    
    // Store result
    mad.lo.u32 %r11, %r9, %r2, %r10;  // offset = row * N + col
    mul.wide.u32 %rd4, %r11, 4;        // byte offset (float = 4 bytes)
    add.u64 %rd5, %rd3, %rd4;          // result address
    st.global.f32 [%rd5], %f1;         // store result
    
    ret;
}"""
    
    def generate_relu_ptx(self, op: IROperation) -> str:
        """Generate PTX for ReLU activation"""
        return """.visible .entry relu_kernel(
    .param .u64 relu_kernel_param_0,  // input pointer
    .param .u64 relu_kernel_param_1,  // output pointer
    .param .u32 relu_kernel_param_2   // size
)
{
    .reg .pred %p<5>;
    .reg .f32 %f<10>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    
    // Load parameters
    ld.param.u64 %rd1, [relu_kernel_param_0];  // input
    ld.param.u64 %rd2, [relu_kernel_param_1];  // output
    ld.param.u32 %r1, [relu_kernel_param_2];   // size
    
    // Calculate global thread index
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;  // idx = blockIdx.x * blockDim.x + threadIdx.x
    
    // Bounds check
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXIT;
    
    // Load input value
    mul.wide.u32 %rd3, %r5, 4;      // byte offset
    add.u64 %rd4, %rd1, %rd3;       // input address
    ld.global.f32 %f1, [%rd4];      // load value
    
    // Compute ReLU: max(0, x)
    mov.f32 %f2, 0.0f;
    max.f32 %f3, %f1, %f2;
    
    // Store result
    add.u64 %rd5, %rd2, %rd3;       // output address
    st.global.f32 [%rd5], %f3;      // store result
    
EXIT:
    ret;
}"""
    
    def generate_add_ptx(self, op: IROperation) -> str:
        """Generate PTX for element-wise addition"""
        return """.visible .entry add_kernel(
    .param .u64 add_kernel_param_0,  // A pointer
    .param .u64 add_kernel_param_1,  // B pointer
    .param .u64 add_kernel_param_2,  // C pointer
    .param .u32 add_kernel_param_3   // size
)
{
    .reg .pred %p<5>;
    .reg .f32 %f<10>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    
    // Load parameters
    ld.param.u64 %rd1, [add_kernel_param_0];  // A
    ld.param.u64 %rd2, [add_kernel_param_1];  // B
    ld.param.u64 %rd3, [add_kernel_param_2];  // C
    ld.param.u32 %r1, [add_kernel_param_3];   // size
    
    // Calculate global thread index
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;  // idx
    
    // Bounds check
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXIT;
    
    // Calculate byte offset
    mul.wide.u32 %rd4, %r5, 4;
    
    // Load A[idx]
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    
    // Load B[idx]
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd6];
    
    // Compute A[idx] + B[idx]
    add.f32 %f3, %f1, %f2;
    
    // Store to C[idx]
    add.u64 %rd7, %rd3, %rd4;
    st.global.f32 [%rd7], %f3;
    
EXIT:
    ret;
}"""


def generate_ptx(module: IRModule) -> str:
    """Convenience function to generate PTX from IR"""
    generator = PTXGenerator()
    return generator.generate(module)


if __name__ == '__main__':
    from lexer import tokenize
    from parser import parse
    from ir_generator import generate_ir
    
    # Test PTX generation
    test_code = """
    result = matmul(A, B)
    activated = relu(result)
    """
    
    tokens = tokenize(test_code)
    ast = parse(tokens)
    ir_module = generate_ir(ast)
    ptx_code = generate_ptx(ir_module)
    
    print(ptx_code)
