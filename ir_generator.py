"""
IR Generator - Converts AST to Intermediate Representation
"""

from ast_nodes import *
from ir import *


class IRGenerator:
    """Generates IR from AST"""
    
    def __init__(self):
        self.builder = IRBuilder()
        self.current_block: Optional[IRBasicBlock] = None
    
    def generate(self, ast: Program) -> IRModule:
        """Generate IR module from AST"""
        module = IRModule()
        
        # Create main function
        func = IRFunction(
            name="main",
            inputs=[],
            outputs=[]
        )
        
        # Create entry block
        entry_block = IRBasicBlock(name="entry")
        func.add_block(entry_block)
        self.current_block = entry_block
        
        # Generate IR for each statement
        for stmt in ast.statements:
            self.generate_statement(stmt)
        
        module.add_function(func)
        return module
    
    def generate_statement(self, stmt: ASTNode) -> Optional[IRValue]:
        """Generate IR for a statement"""
        if isinstance(stmt, Assignment):
            return self.generate_assignment(stmt)
        elif isinstance(stmt, Expression):
            return self.generate_expression(stmt)
        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")
    
    def generate_assignment(self, assign: Assignment) -> IRValue:
        """Generate IR for assignment"""
        # Generate IR for the value expression
        value = self.generate_expression(assign.value)
        
        # Store to variable
        target = self.builder.get_or_create_value(assign.target)
        store_op = IROperation(
            op_type=IROpType.STORE,
            result=target,
            operands=[value]
        )
        self.current_block.add_op(store_op)
        
        return target
    
    def generate_expression(self, expr: Expression) -> IRValue:
        """Generate IR for an expression"""
        if isinstance(expr, FunctionCall):
            return self.generate_function_call(expr)
        elif isinstance(expr, Variable):
            return self.generate_variable(expr)
        elif isinstance(expr, Number):
            return self.generate_number(expr)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")
    
    def generate_function_call(self, call: FunctionCall) -> IRValue:
        """Generate IR for function call"""
        # Map function name to IR op type
        op_map = {
            'matmul': IROpType.MATMUL,
            'add': IROpType.ADD,
            'relu': IROpType.RELU,
            'softmax': IROpType.SOFTMAX,
            'transpose': IROpType.TRANSPOSE,
        }
        
        if call.name not in op_map:
            raise ValueError(f"Unknown function: {call.name}")
        
        # Generate IR for arguments
        operands = [self.generate_expression(arg) for arg in call.args]
        
        # Create result value
        result = self.builder.new_temp()
        
        # Create operation
        op = IROperation(
            op_type=op_map[call.name],
            result=result,
            operands=operands,
            attributes=call.kwargs
        )
        
        self.current_block.add_op(op)
        return result
    
    def generate_variable(self, var: Variable) -> IRValue:
        """Generate IR for variable reference"""
        # Load from variable
        value = self.builder.get_or_create_value(var.name)
        result = self.builder.new_temp()
        
        load_op = IROperation(
            op_type=IROpType.LOAD,
            result=result,
            operands=[value]
        )
        self.current_block.add_op(load_op)
        
        return result
    
    def generate_number(self, num: Number) -> IRValue:
        """Generate IR for number literal"""
        # For simplicity, we'll create a constant value
        result = self.builder.new_temp("const")
        result.type = f"const<{num.value}>"
        return result


def generate_ir(ast: Program) -> IRModule:
    """Convenience function to generate IR from AST"""
    generator = IRGenerator()
    return generator.generate(ast)


if __name__ == '__main__':
    from lexer import tokenize
    from parser import parse
    
    # Test IR generation
    test_code = """
    result = matmul(A, B)
    activated = relu(result)
    """
    
    tokens = tokenize(test_code)
    ast = parse(tokens)
    ir_module = generate_ir(ast)
    
    print(ir_module)
