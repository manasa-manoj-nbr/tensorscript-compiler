"""
Test suite for TensorScript Compiler
"""

import unittest
from lexer import tokenize, TokenType
from parser import parse
from ast_nodes import *
from ir_generator import generate_ir
from ir import IROpType
from cuda_generator import generate_cuda
from ptx_generator import generate_ptx


class TestLexer(unittest.TestCase):
    """Test lexical analysis"""
    
    def test_simple_matmul(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        
        # Check token types
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        expected = [
            TokenType.IDENTIFIER,  # result
            TokenType.ASSIGN,      # =
            TokenType.MATMUL,      # matmul
            TokenType.LPAREN,      # (
            TokenType.IDENTIFIER,  # A
            TokenType.COMMA,       # ,
            TokenType.IDENTIFIER,  # B
            TokenType.RPAREN,      # )
        ]
        self.assertEqual(types, expected)
    
    def test_numbers(self):
        source = "x = add(A, B, tile_size=32)"
        tokens = tokenize(source)
        
        # Find the number token
        numbers = [t for t in tokens if t.type == TokenType.NUMBER]
        self.assertEqual(len(numbers), 1)
        self.assertEqual(numbers[0].value, 32)


class TestParser(unittest.TestCase):
    """Test parsing"""
    
    def test_assignment(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        ast = parse(tokens)
        
        self.assertIsInstance(ast, Program)
        self.assertEqual(len(ast.statements), 1)
        
        stmt = ast.statements[0]
        self.assertIsInstance(stmt, Assignment)
        self.assertEqual(stmt.target, 'result')
        self.assertIsInstance(stmt.value, FunctionCall)
        self.assertEqual(stmt.value.name, 'matmul')
    
    def test_function_arguments(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        ast = parse(tokens)
        
        call = ast.statements[0].value
        self.assertEqual(len(call.args), 2)
        self.assertIsInstance(call.args[0], Variable)
        self.assertEqual(call.args[0].name, 'A')
        self.assertIsInstance(call.args[1], Variable)
        self.assertEqual(call.args[1].name, 'B')
    
    def test_kwargs(self):
        source = "result = matmul(A, B, tile_size=32)"
        tokens = tokenize(source)
        ast = parse(tokens)
        
        call = ast.statements[0].value
        self.assertEqual(call.kwargs['tile_size'], 32)


class TestIRGeneration(unittest.TestCase):
    """Test IR generation"""
    
    def test_simple_ir(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        ast = parse(tokens)
        ir_module = generate_ir(ast)
        
        self.assertEqual(len(ir_module.functions), 1)
        func = ir_module.functions[0]
        self.assertEqual(func.name, 'main')
        
        # Check operations
        ops = func.blocks[0].operations
        self.assertTrue(any(op.op_type == IROpType.MATMUL for op in ops))
    
    def test_multiple_operations(self):
        source = """
        x = matmul(A, B)
        y = relu(x)
        """
        tokens = tokenize(source)
        ast = parse(tokens)
        ir_module = generate_ir(ast)
        
        ops = ir_module.functions[0].blocks[0].operations
        op_types = [op.op_type for op in ops]
        
        self.assertIn(IROpType.MATMUL, op_types)
        self.assertIn(IROpType.RELU, op_types)


class TestCodeGeneration(unittest.TestCase):
    """Test CUDA and PTX code generation"""
    
    def test_cuda_generation(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        ast = parse(tokens)
        ir_module = generate_ir(ast)
        cuda_code = generate_cuda(ir_module)
        
        # Check for expected CUDA constructs
        self.assertIn('__global__', cuda_code)
        self.assertIn('matmul_kernel', cuda_code)
        self.assertIn('__shared__', cuda_code)
    
    def test_ptx_generation(self):
        source = "result = matmul(A, B)"
        tokens = tokenize(source)
        ast = parse(tokens)
        ir_module = generate_ir(ast)
        ptx_code = generate_ptx(ir_module)
        
        # Check for expected PTX constructs
        self.assertIn('.version', ptx_code)
        self.assertIn('.target', ptx_code)
        self.assertIn('.entry', ptx_code)


class TestEndToEnd(unittest.TestCase):
    """End-to-end compilation tests"""
    
    def test_neural_network_pass(self):
        source = """
        hidden = matmul(input, weights1)
        activated = relu(hidden)
        output = matmul(activated, weights2)
        """
        
        tokens = tokenize(source)
        ast = parse(tokens)
        ir_module = generate_ir(ast)
        cuda_code = generate_cuda(ir_module)
        
        # Should generate multiple kernels
        self.assertIn('matmul_kernel', cuda_code)
        self.assertIn('relu_kernel', cuda_code)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
