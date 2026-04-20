#!/usr/bin/env python3
"""
Demo script for TensorScript Compiler
Shows the complete compilation pipeline with examples
"""

from compiler import Compiler


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_matmul():
    """Demonstrate basic matrix multiplication"""
    print_section("Example 1: Basic Matrix Multiplication")
    
    source = """
# Simple matrix multiplication
result = matmul(A, B)
"""
    
    print("TensorScript Source:")
    print(source)
    
    compiler = Compiler(verbose=False)
    
    # Show IR
    print("\nIntermediate Representation (IR):")
    ir_code = compiler.compile(source, output_format='ir')
    print(ir_code)
    
    # Show CUDA
    print("\nGenerated CUDA Code (excerpt):")
    cuda_code = compiler.compile(source, output_format='cuda')
    # Print first 30 lines
    print('\n'.join(cuda_code.split('\n')[:35]))
    print("... (full kernel continues)")


def demo_neural_network():
    """Demonstrate neural network forward pass"""
    print_section("Example 2: Neural Network Forward Pass")
    
    source = """
# Two-layer neural network
hidden = matmul(input, weights1)
activated = relu(hidden)
output = matmul(activated, weights2)
"""
    
    print("TensorScript Source:")
    print(source)
    
    compiler = Compiler(verbose=False)
    
    # Show IR
    print("\nIntermediate Representation (IR):")
    ir_code = compiler.compile(source, output_format='ir')
    print(ir_code)


def demo_optimized():
    """Demonstrate optimization hints"""
    print_section("Example 3: Optimized Matrix Multiplication")
    
    source = """
# Matrix multiplication with custom tile size
result = matmul(X, Y, tile_size=32)
"""
    
    print("TensorScript Source:")
    print(source)
    
    compiler = Compiler(verbose=False)
    
    # Show CUDA with tile size
    print("\nGenerated CUDA (note TILE_SIZE parameter):")
    cuda_code = compiler.compile(source, output_format='cuda')
    # Find and print the matmul kernel
    lines = cuda_code.split('\n')
    for i, line in enumerate(lines):
        if 'matmul_kernel' in line:
            print('\n'.join(lines[i:min(i+25, len(lines))]))
            break


def demo_ptx():
    """Demonstrate PTX assembly generation"""
    print_section("Example 4: PTX Assembly Generation")
    
    source = """
activated = relu(input)
"""
    
    print("TensorScript Source:")
    print(source)
    
    compiler = Compiler(verbose=False)
    
    # Show PTX
    print("\nGenerated PTX Assembly:")
    ptx_code = compiler.compile(source, output_format='ptx')
    print(ptx_code)


def demo_full_pipeline():
    """Show complete compilation pipeline with all phases"""
    print_section("Example 5: Complete Compilation Pipeline")
    
    source = """
# Softmax classifier
logits = matmul(features, weights)
probs = softmax(logits)
"""
    
    print("TensorScript Source:")
    print(source)
    
    print("\n" + "-" * 70)
    print("Compiling with verbose output...\n")
    
    compiler = Compiler(verbose=True)
    result = compiler.compile(source, output_format='cuda')


def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              TensorScript Compiler - Demo                           ║
║              DSL → IR → CUDA/PTX Pipeline                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("Basic Matrix Multiplication", demo_basic_matmul),
        ("Neural Network Forward Pass", demo_neural_network),
        ("Optimized Compilation", demo_optimized),
        ("PTX Assembly Output", demo_ptx),
        ("Full Pipeline (Verbose)", demo_full_pipeline),
    ]
    
    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. Run all demos")
    
    try:
        choice = input("\nSelect demo (1-{}): ".format(len(demos) + 1))
        choice = int(choice)
        
        if 1 <= choice <= len(demos):
            demos[choice - 1][1]()
        elif choice == len(demos) + 1:
            for name, demo_func in demos:
                demo_func()
        else:
            print("Invalid choice")
    except (ValueError, KeyboardInterrupt):
        print("\nDemo cancelled")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
