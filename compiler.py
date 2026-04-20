"""
TensorScript Compiler - Main driver
Orchestrates the compilation pipeline: DSL → AST → IR → CUDA/PTX
"""

import argparse
import sys
from pathlib import Path

from lexer import tokenize
from parser import parse
from ast_nodes import pretty_print
from ir_generator import generate_ir
from ir import print_ir
from cuda_generator import generate_cuda
from ptx_generator import generate_ptx


class Compiler:
    """Main compiler class"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def compile(self, source: str, output_format='cuda'):
        """
        Compile TensorScript source code
        
        Args:
            source: TensorScript source code string
            output_format: 'cuda', 'ptx', or 'ir'
        
        Returns:
            Generated code as string
        """
        try:
            # Phase 1: Lexical Analysis
            if self.verbose:
                print("=== Phase 1: Lexical Analysis ===")
            tokens = tokenize(source)
            if self.verbose:
                for token in tokens:
                    print(f"  {token}")
                print()
            
            # Phase 2: Parsing
            if self.verbose:
                print("=== Phase 2: Parsing ===")
            ast = parse(tokens)
            if self.verbose:
                print(pretty_print(ast))
                print()
            
            # Phase 3: IR Generation
            if self.verbose:
                print("=== Phase 3: IR Generation ===")
            ir_module = generate_ir(ast)
            if self.verbose:
                print(print_ir(ir_module))
                print()
            
            # Phase 4: Code Generation
            if self.verbose:
                print(f"=== Phase 4: Code Generation ({output_format.upper()}) ===")
            
            if output_format == 'ir':
                return print_ir(ir_module)
            elif output_format == 'cuda':
                return generate_cuda(ir_module)
            elif output_format == 'ptx':
                return generate_ptx(ir_module)
            else:
                raise ValueError(f"Unknown output format: {output_format}")
        
        except SyntaxError as e:
            print(f"Syntax Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Compilation Error: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def compile_file(self, input_path: str, output_path: str = None, 
                     output_format='cuda'):
        """
        Compile a TensorScript file
        
        Args:
            input_path: Path to input .ts file
            output_path: Path to output file (optional)
            output_format: 'cuda', 'ptx', or 'ir'
        """
        # Read source file
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        source = input_file.read_text()
        
        # Compile
        result = self.compile(source, output_format)
        
        # Determine output path
        if output_path is None:
            extensions = {'cuda': '.cu', 'ptx': '.ptx', 'ir': '.ir'}
            output_path = input_file.with_suffix(extensions[output_format])
        
        # Write output
        output_file = Path(output_path)
        output_file.write_text(result)
        
        if self.verbose:
            print(f"Compilation successful: {input_path} -> {output_path}")
        
        return str(output_file)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='TensorScript Compiler - DSL for GPU computing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile to CUDA
  python compiler.py program.ts -o output.cu
  
  # Compile to PTX
  python compiler.py program.ts -f ptx -o output.ptx
  
  # Show IR
  python compiler.py program.ts -f ir
  
  # Verbose mode
  python compiler.py program.ts -v
        """
    )
    
    parser.add_argument('input', help='Input TensorScript file (.ts)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', 
                       choices=['cuda', 'ptx', 'ir'],
                       default='cuda',
                       help='Output format (default: cuda)')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output showing all compilation phases')
    
    args = parser.parse_args()
    
    # Create compiler and compile
    compiler = Compiler(verbose=args.verbose)
    compiler.compile_file(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
